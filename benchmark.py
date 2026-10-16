"""Benchmark sur OGBench Cube : tâches tirées des 200 épisodes de validation de cube_single_expert
(jamais vus à l'entraînement, ni par nos modèles ni par ceux du papier H-JEPA, arXiv 2610.06805,
5 octobre 2026), pour comparer notre modèle à leurs modèles, que l'on refait tourner nous-mêmes.

Protocole (tâches : make_val_tasks.py) :
  - écarts au but K = 25, 50, 100, 200 pas env ; 50 tâches par écart, départ uniforme dans un
    épisode de validation, but = frame départ + K ; tâches où le cube bouge de <= 4 cm écartées ;
  - réussite : critère de l'env, cube à <= 4 cm de sa position dans la frame but ;
    budget 1,5 x K pas env (comme le papier : 30 pas pour un but à 20) ;
  - seeds 42 / 43 / 44 : seed du modèle = seed du planificateur ; moyenne ± erreur standard (SE)
    sur les seeds, par écart et en moyenne sur les écarts. Seul le taux de réussite compte.
La boucle d'éval (World, WorldModelPolicy, reseeding par tâche, métriques) est celle du code H-JEPA
(third_party/H-JEPA) ; seuls les fichiers de tâches et le budget changent.

Modèles de référence (poids publiés par le papier, chacun avec son planificateur publié) :
  ref-lewm                   -> LeWM, planification à plat (config cube_flat)
  ref-hjepa-l2 / l3 / l4     -> H-JEPA 2 / 3 / 4 niveaux (config cube_l2 / l3 / l4)
  ref-hwm-l2 / l3 / l4       -> HWM 2 / 3 / 4 niveaux (mêmes configs)

Nos méthodes (results/<exp>/seed<s>/weights.pt, python train.py exp=<exp> seed=<s>) :
  hwm         -> notre niveau 1 seul : CEM sur le skill m, actions décodées par pi(z0, m*)
  hwm-hybrid  -> GradientSolver de LeWM (cube_flat), candidat 0 initialisé par pi(z0, m*)
Diagnostics (« est-ce de la planification ou du behavior cloning ? ») :
  hwm-noplan        -> sans CEM : m = moyenne de posterior(z1, z1_goal), actions = pi(z0, m)
  hwm-hybrid-nogd   -> hwm-hybrid avec lr = 0 (1 itération) : seul le choix final par le coût
                       LeWM agit ; --hybrid-samples 1 => pi seul via la plomberie de l'hybride
  hwm-hybrid-noplan -> hwm-hybrid, candidat 0 = pi(z0, m) de hwm-noplan (sans CEM sur m)
Ablation « sans hiérarchie » (results/<exp>/seed<s>, python train.py model=flat_policy ...) :
  flat-noplan       -> politique à plat conditionnée par le but seule : actions = pi(z0, z0_goal)
  flat-hybrid       -> GradientSolver de LeWM, candidat 0 initialisé par pi(z0, z0_goal)

Sorties : results/<exp>/seed<s>/eval_<protocole>/<méthode>/offset<K>/ (chunks + metrics.yaml du code
H-JEPA), results/<exp>/benchmark.json, benchmark.png ; idem sous results/baselines/ pour les références.
Leaderboard : results/leaderboard.json, rempli par --baselines (références) puis par chaque
benchmark complet (4 écarts x 50 tâches) de nos expériences.

  python benchmark.py --baselines                            # 7 modèles du papier x 3 seeds (long)
  python benchmark.py <exp>                                  # nos méthodes x 3 seeds
  python benchmark.py <exp> --methods hwm-hybrid --seeds 42 --num-eval 5   # test rapide
  python benchmark.py <exp> --plot-only
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from make_val_tasks import OFFSETS, PER_OFFSET, task_path

ROOT = Path(__file__).resolve().parent
HJEPA_REPO = ROOT / "third_party/H-JEPA"
HJEPA_HOME = Path(os.environ.setdefault("HJEPA_HOME", str(Path.home() / "hjepa_home")))

warnings.filterwarnings("ignore", module="gymnasium")

FULL_EVAL = PER_OFFSET           # seuls les benchmarks complets entrent dans le leaderboard
PROTOCOL = f"val{len(OFFSETS)}x{PER_OFFSET}"
# protocole du papier (benchmark_audit.py) : leurs 50 fenêtres de 20 pas centrées sur la saisie, budget 30
PAPER_PROTOCOL = "paper20"
PAPER_TASKS = HJEPA_REPO / "h_jepa/assets/eval_trajs/ogbench/goal_offset_20_pickup_val.pt"
PAPER_OFFSET = 20
LEADERBOARD = ROOT / "results/leaderboard.json"
BASELINES_DIR = ROOT / "results/baselines"


def budget(offset):
    return int(np.ceil(1.5 * offset))   # pas env, comme le papier (30 pas pour un but à 20)


# départ = état complet du dataset, cible = position du cube dans la frame but (comme le papier)
CALLABLES = [
    {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
    {"method": "set_target_pos", "args": {
        "cube_id": {"value": 0, "in_dataset": False},
        "target_pos": {"value": "goal_privileged_block_0_pos"},
        "target_quat": {"value": "goal_privileged_block_0_quat"}}},
]

# modèles du papier : clé -> (dossier dans $HJEPA_HOME/ckpts/cube, config d'éval H-JEPA, libellé)
BASELINES = {
    "ref-lewm": ("cube_lewm", "cube_flat", "LeWM"),
    "ref-hjepa-l2": ("cube_hjepa_l2", "cube_l2", "H-JEPA (2 levels)"),
    "ref-hjepa-l3": ("cube_hjepa_l3", "cube_l3", "H-JEPA (3 levels)"),
    "ref-hjepa-l4": ("cube_hjepa_l4", "cube_l4", "H-JEPA (4 levels)"),
    "ref-hwm-l2": ("cube_hwm_l2", "cube_l2", "HWM (2 levels)"),
    "ref-hwm-l3": ("cube_hwm_l3", "cube_l3", "HWM (3 levels)"),
    "ref-hwm-l4": ("cube_hwm_l4", "cube_l4", "HWM (4 levels)"),
}
METHODS = ("hwm", "hwm-hybrid", "hwm-noplan", "hwm-hybrid-nogd", "hwm-hybrid-noplan",
           "flat-noplan", "flat-hybrid")
LABELS = {"hwm": "LeHWM (level 1)", "hwm-hybrid": "LeHWM (hybrid)",
          "hwm-noplan": "LeHWM (level 1, no CEM)", "hwm-hybrid-nogd": "LeHWM (hybrid, no GD)",
          "hwm-hybrid-noplan": "LeHWM (hybrid, no CEM)",
          "flat-noplan": "Flat GC policy", "flat-hybrid": "Flat GC policy (hybrid)",
          **{k: v[2] for k, v in BASELINES.items()}}
# une couleur par famille (palette catégorielle validée, slots 1-4 dans l'ordre)
FAMILY_COLORS = {"LeHWM (ours)": "#2a78d6", "LeWM": "#eb6834", "H-JEPA": "#1baf7a", "HWM": "#eda100"}


def family(method):
    if method in METHODS:
        return "LeHWM (ours)"
    return {"ref-lewm": "LeWM"}.get(method, "H-JEPA" if method.startswith("ref-hjepa") else "HWM")


##############################
##   politique hiérarchique  ##
##############################


class SkillPlanner:
    """1) CEM dans le prior (espace blanchi eps) sur un saut de niveau 1,
       2) re-classement ancré : les k meilleurs m sont décodés par pi et déroulés dans P0 ;
          le sous-but est l'état RÉELLEMENT atteint, pas la prédiction de P1."""

    def __init__(self, model, n, num_samples=300, n_steps=30, topk=30,
                 rerank_k=16, lam_eps=0.01, lam_exploit=1.0, history=3,
                 device="cuda", seed=0, plan=True):
        self.model = model
        self.do_plan = plan               # False = ablation : m = moyenne du posterior, sans CEM
        self.n = n
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.rerank_k = rerank_k          # 0 = sans re-classement (ablation)
        self.lam_eps = lam_eps            # pénalité ||eps||^2 : reste dans le prior
        self.lam_exploit = lam_exploit    # pénalité ||atteint - prédit par P1||^2
        self.history = history
        self.device = device
        self.m_dim = model.posterior.mlp.net[-1].out_features // 2
        self.use_prior = getattr(model, "prior", None) is not None
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.last_subgoal = None          # (B, D1)
        self.last_exploit = None          # (B,) diagnostic "Mind the Gap"

    def _prior(self, z1):
        if self.use_prior:
            mu, logvar = self.model.prior(z1)
            return mu, (0.5 * logvar).exp()
        zeros = torch.zeros(z1.size(0), self.m_dim, device=self.device)
        return zeros, torch.ones_like(zeros)

    def _p1(self, z1, m):
        """z1: (N, D1), m: (N, m_dim) -> (N, D1)"""
        model = self.model
        out = model.predictor1(z1[:, None], model.m_encoder(m[:, None]))
        return model.pred_proj1(out[:, 0])

    def _rollout_p0(self, z0, acts):
        """z0: (N, D0), acts: (N, n, A) normalisées -> z0 prédit après n blocs (N, D0)"""
        model = self.model
        roll = z0[:, None]
        act_emb = model.action_encoder(acts)
        for k in range(acts.size(1)):
            L = min(roll.size(1), self.history)
            pred = model.predict(roll[:, -L:], act_emb[:, k + 1 - L:k + 1])
            roll = torch.cat([roll, pred[:, -1:]], dim=1)
        return roll[:, -1]

    @torch.inference_mode()
    def plan(self, pixels, goal):
        """pixels, goal : (B, T, C, H, W). Retourne les actions normalisées (B, n, A)."""
        model, S, dev = self.model, self.num_samples, self.device
        z0 = model.encode({"pixels": pixels[:, -1:].to(dev)})["emb"][:, 0]    # (B, D0)
        zg = model.encode({"pixels": goal[:, -1:].to(dev)})["emb"][:, 0]
        z1, z1g = model.elevator(z0), model.elevator(zg)                       # (B, D1)
        B, D1 = z1.shape
        b = torch.arange(B, device=dev)
        rows = b[:, None]

        if not self.do_plan:          # ablation : behavior cloning conditionné par le but
            m = model.posterior(z1, z1g)[1]
            self.last_subgoal = self._p1(z1, m)
            self.last_exploit = None
            return model.pi(torch.cat([z0, m], -1)).reshape(B, self.n, -1)

        mu_p, std_p = self._prior(z1)                                          # (B, m_dim)

        # ---- 1) CEM dans l'espace blanchi : m = mu_p + std_p * eps ----
        mean = torch.zeros(B, self.m_dim, device=dev)
        std = torch.ones_like(mean)
        z1_rep = z1.repeat_interleave(S, 0)
        for _ in range(self.n_steps):
            eps = mean[:, None] + std[:, None] * torch.randn(
                B, S, self.m_dim, generator=self.torch_gen, device=dev)
            eps[:, 0] = mean
            m = mu_p[:, None] + std_p[:, None] * eps
            pred = self._p1(z1_rep, m.reshape(B * S, -1)).reshape(B, S, D1)
            cost = (pred - z1g[:, None]).pow(2).sum(-1) + self.lam_eps * eps.pow(2).sum(-1)
            elite = eps[rows, cost.topk(self.topk, dim=1, largest=False).indices]
            mean, std = elite.mean(1), elite.std(1)

        if self.rerank_k <= 0:        # ablation : comportement d'avant
            m = mu_p + std_p * mean
            self.last_subgoal = self._p1(z1, m)
            self.last_exploit = None
            return model.pi(torch.cat([z0, m], -1)).reshape(B, self.n, -1)

        # ---- 2) re-classement ancré des k meilleurs ----
        k = self.rerank_k
        top_eps = eps[rows, cost.topk(k, dim=1, largest=False).indices]       # (B, k, m_dim)
        m_top = (mu_p[:, None] + std_p[:, None] * top_eps).reshape(B * k, -1)
        z0_k = z0.repeat_interleave(k, 0)
        acts = model.pi(torch.cat([z0_k, m_top], -1)).reshape(B * k, self.n, -1)
        reached = model.elevator(self._rollout_p0(z0_k, acts)).reshape(B, k, D1)
        predicted = self._p1(z1.repeat_interleave(k, 0), m_top).reshape(B, k, D1)

        exploit = (reached - predicted).pow(2).sum(-1)                          # (B, k)
        g_cost = (reached - z1g[:, None]).pow(2).sum(-1) + self.lam_exploit * exploit
        best = g_cost.argmin(1)

        self.last_subgoal = reached[b, best]                                    # atteignable
        self.last_exploit = exploit[b, best]
        return acts.reshape(B, k, self.n, -1)[b, best]                          # (B, n, A)


class FlatPlanner:
    """Ablation sans hiérarchie : actions = pi(z0, z0_goal) d'une FlatPolicy (pas de niveau 1)."""

    def __init__(self, model, n, device="cuda", seed=0):
        self.model = model
        self.n = n
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)  # inutilisé, reseedé par tâche

    @torch.inference_mode()
    def plan(self, pixels, goal):
        """pixels, goal : (B, T, C, H, W). Retourne les actions normalisées (B, n, A)."""
        model, dev = self.model, self.device
        z0 = model.encode({"pixels": pixels[:, -1:].to(dev)})["emb"][:, 0]
        zg = model.encode({"pixels": goal[:, -1:].to(dev)})["emb"][:, 0]
        return model.pi(torch.cat([z0, zg], -1)).reshape(z0.size(0), self.n, -1)


class SkillSolver:
    """SkillPlanner derrière l'interface Solver de stable_worldmodel (version H-JEPA) :
    WorldModelPolicy l'appelle comme leur GradientSolver et exécute `receding_horizon` blocs."""

    def __init__(self, planner):
        self.planner = planner
        self.torch_gen = planner.torch_gen    # reseedé par tâche par _run_chunked_eval
        self.solve_records = []

    def configure(self, *, action_space, n_envs, config):
        self._n_envs, self._config = n_envs, config
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self.planner.n

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None, steps_taken=None, eval_budget=None):
        return {"actions": self.planner.plan(info_dict["pixels"], info_dict["goal"]).float().cpu()}


class HybridSolver(SkillSolver):
    """GradientSolver de LeWM, candidat 0 initialisé par les actions de pi (les autres = pi + bruit)."""

    def __init__(self, planner, gd):
        super().__init__(planner)
        self.gd = gd
        # _run_chunked_eval reseede par tâche les torch_gen de level_solvers (et lit leurs solve_records)
        self.level_solvers = {1: gd, 2: SkillSolver(planner)}
        self.diag = []                    # par solve : (candidat de pi retenu ?, ||a_retenue - a_pi||)

    def configure(self, *, action_space, n_envs, config):
        super().configure(action_space=action_space, n_envs=n_envs, config=config)
        self.gd.configure(action_space=action_space, n_envs=n_envs, config=config)

    @property
    def horizon(self):
        return self.gd.horizon

    def solve(self, info_dict, init_action=None, steps_taken=None, eval_budget=None):
        init = self.planner.plan(info_dict["pixels"], info_dict["goal"]).float()
        out = self.gd.solve(info_dict, init_action=init, steps_taken=steps_taken, eval_budget=eval_budget)
        # gd.init = candidats optimisés (B, S, H, A) ; le retenu en est une copie exacte
        with torch.no_grad():
            acts = out["actions"].to(self.gd.init.device)
            H = acts.size(1)
            idx = (self.gd.init - acts[:, None]).abs().flatten(2).sum(-1).argmin(1)
            dist = (acts - init[:, :H].to(acts.device)).flatten(1).norm(dim=1)
        self.diag += [(int(i) == 0, float(d)) for i, d in zip(idx, dist)]
        return out





##############################
##        simulation        ##
##############################


def setup_hjepa_imports():
    """Le stable_worldmodel modifié du papier masque celui installé, pour ce processus seulement.
    Notre modèle est construit avant que h_jepa/ (utils.py, data.py...) n'entre dans sys.path."""
    sys.path.insert(0, str(HJEPA_REPO))
    import stable_pretraining  # noqa: F401  enregistre ${eval:...} -> avant hjepa
    import hjepa  # noqa: F401
    import stable_worldmodel as swm

    assert Path(swm.__file__).is_relative_to(HJEPA_REPO), f"mauvais stable_worldmodel : {swm.__file__}"
    if str(HJEPA_REPO / "h_jepa") not in sys.path:
        sys.path.insert(0, str(HJEPA_REPO / "h_jepa"))
    missing = [k for k in OFFSETS if not task_path(k, PER_OFFSET).exists()]
    assert not missing, f"tâches manquantes pour les écarts {missing} : python make_val_tasks.py"


def load_hwm(run_dir):
    """Notre modèle (config.json + weights.pt écrits par RunDirCallback)."""
    import hydra

    cfg = json.loads((run_dir / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)
    model.load_state_dict(torch.load(run_dir / "weights.pt", map_location="cpu"))
    model = model.to("cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def train_cfg(run_dir):
    from omegaconf import OmegaConf

    return OmegaConf.load(run_dir / ".hydra/config.yaml")


def ckpt_path(ckpt_dir, seed):
    return HJEPA_HOME / f"ckpts/cube/{ckpt_dir}/seed{seed}/{ckpt_dir}_object.ckpt"


def eval_cfg(config_name, seed, offset, num_eval, policy, protocol=PROTOCOL):
    """Config d'éval publiée du papier pour ce planificateur, sur nos tâches et notre budget
    (ou, avec protocol=PAPER_PROTOCOL, sur leurs tâches et leur budget, sans rien changer)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(HJEPA_REPO / f"h_jepa/config/eval/{config_name}.yaml")
    cfg.cache_dir = str(HJEPA_HOME)
    cfg.seed = seed                       # les solveurs lisent ${seed}
    cfg.policy = str(policy)
    if protocol == PAPER_PROTOCOL:
        assert offset == cfg.eval.goal_offset_steps == PAPER_OFFSET
        cfg.load_eval_trajs_path = str(PAPER_TASKS)
        cfg.eval.num_eval = num_eval
        return cfg
    assert protocol == PROTOCOL, protocol
    cfg.load_eval_trajs_path = str(task_path(offset, PER_OFFSET))
    cfg.eval.num_eval = num_eval
    cfg.eval.goal_offset_steps = offset
    cfg.eval.eval_budget = budget(offset)
    cfg.eval.callables = CALLABLES
    return cfg


def run_eval(cfg, out_dir, rerun, build_policy=None):
    """planning_eval.run_planning_eval du code H-JEPA ; build_policy remplace leur _build_policy."""
    import planning_eval

    if rerun and out_dir.exists():
        shutil.rmtree(out_dir)
    n = int(cfg.eval.num_eval)
    done = len(list((out_dir / "chunks").glob("tasks_*.json"))) if out_dir.exists() else 0
    original = planning_eval._build_policy
    if build_policy is not None:
        planning_eval._build_policy = build_policy
    try:
        t0 = time.time()
        metrics = planning_eval.run_planning_eval(cfg, results_dir=out_dir)
        elapsed = time.time() - t0
    finally:
        planning_eval._build_policy = original

    new = n - done   # tâches reprises depuis des chunks existants : temps non mesuré
    return {
        "offset": int(cfg.eval.goal_offset_steps), "num_eval": n,
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": [bool(x) for x in metrics["episode_successes"]],
        "time_per_episode_s": elapsed / new if new else None,
    }


def run_baseline(key, seed, offset, args, protocol=PROTOCOL):
    ckpt_dir, config_name, _ = BASELINES[key]
    cfg = eval_cfg(config_name, seed, offset, args.num_eval, ckpt_path(ckpt_dir, seed), protocol)
    out_dir = BASELINES_DIR / key / f"seed{seed}" / f"eval_{protocol}" / f"offset{offset}"
    return {"method": key, "seed": seed, **run_eval(cfg, out_dir, args.rerun)}


def run_ours(method, seed, offset, run_dir, hwm, args, protocol=PROTOCOL):
    """Une de nos méthodes ; le GradientSolver des hybrides est celui de LeWM (cube_flat, même seed)."""
    import stable_worldmodel as swm
    import planning_eval

    cfg = eval_cfg("cube_flat", seed, offset, args.num_eval, ckpt_path("cube_lewm", seed), protocol)
    variant = None
    if "hybrid" in method and args.hybrid_samples is not None:
        variant = f"{method}-s{args.hybrid_samples}"   # diagnostic : ni écrasement, ni leaderboard
    out_dir = run_dir / f"eval_{protocol}" / (variant or method) / f"offset{offset}"
    built = {}
    n = train_cfg(run_dir).n

    def planner():
        if method.startswith("flat"):
            return FlatPlanner(hwm, n=n, seed=seed)
        return SkillPlanner(hwm, n=n, num_samples=args.num_samples, n_steps=args.cem_steps,
                            topk=args.topk, seed=seed, plan=method not in ("hwm-noplan", "hwm-hybrid-noplan"))

    def build_policy(cfg, process, transform, model=None):
        if method in ("hwm", "hwm-noplan", "flat-noplan"):
            config = swm.PlanConfig(horizon=n, receding_horizon=cfg.plan_config.receding_horizon,
                                    action_block=cfg.plan_config.action_block)
            return swm.policy.WorldModelPolicy(solver=SkillSolver(planner()), config=config, process=process,
                                               transform=transform, eval_budget=int(cfg.eval.eval_budget))
        policy = lewm_policy(cfg, process, transform, model)
        gd = policy.solver
        if args.hybrid_samples is not None:
            gd.num_samples = args.hybrid_samples
        if method == "hwm-hybrid-nogd":
            # 1 itération à lr = 0 (AdamW : decay ∝ lr) => candidats inchangés, argmin du coût LeWM
            gd.n_steps = 1
            gd.optimizer_kwargs = {**gd.optimizer_kwargs, "lr": 0.0}
        policy.solver = built["hybrid"] = HybridSolver(planner(), gd)
        return policy

    lewm_policy = planning_eval._build_policy
    r = run_eval(cfg, out_dir, args.rerun, build_policy)
    return {"method": method, "seed": seed, "variant": variant, **r, **hybrid_diag(built.get("hybrid"))}


def hybrid_diag(solver):
    """Sur les solves de ce lancement (tâches reprises exclues)."""
    if solver is None or not solver.diag:
        return {}
    chosen, dist = zip(*solver.diag)
    frac, d = float(np.mean(chosen)), float(np.mean(dist))
    print(f"diag hybride : candidat pi retenu {100 * frac:.1f}% des solves, ||a - a_pi|| moyen = {d:.3f}")
    return {"pi_chosen_frac": frac, "pi_action_dist": d, "n_solves": len(chosen)}


def save(r, out_path):
    """Relit le JSON avant d'écrire : plusieurs processus (seeds / modèles différents) peuvent tourner."""
    key = lambda x: (x["method"], x["seed"], x["offset"], x.get("variant"))  # noqa: E731
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    results = [x for x in results if key(x) != key(r)] + [r]
    out_path.write_text(json.dumps(results, indent=2))  # rien n'est perdu si ça plante
    print(f"-> {r['method']} seed={r['seed']} écart={r['offset']} : succès {r['success_rate']:.1f}%")
    return results


def run_benchmark(args):
    setup_hjepa_imports()
    out_path = args.exp_dir / "benchmark.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    for seed in args.seeds:
        hwm = None if args.baselines else load_hwm(args.exp_dir / f"seed{seed}")
        for m in args.models if args.baselines else args.methods:
            for offset in args.offsets:
                print(f"\n=== {m} | seed={seed} | écart={offset} ===")
                r = (run_baseline(m, seed, offset, args) if args.baselines
                     else run_ours(m, seed, offset, args.exp_dir / f"seed{seed}", hwm, args))
                results = save(r, out_path)

    print(f"\nRésultats : {out_path}")
    return results


##############################
##       leaderboard        ##
##############################


def summarize(results):
    """{method: entrée}. Par écart : moyenne ± SE sur les seeds (std(ddof=1) / sqrt(n), comme le
    papier). « mean » : moyenne sur les écarts, calculée par seed puis moyennée ± SE sur les seeds
    (seulement les seeds évaluées sur tous les écarts présents)."""
    def mean_se(x):
        x = np.asarray(x, dtype=float)
        return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None)

    summary = {}
    for m in dict.fromkeys(r["method"] for r in results):
        runs = [r for r in results if r["method"] == m and not r.get("variant")]
        if not runs:
            continue
        offsets = sorted({r["offset"] for r in runs})
        per = {}
        for k in offsets:
            rates = [r["success_rate"] for r in runs if r["offset"] == k]
            mu, se = mean_se(rates)
            per[str(k)] = {"success": mu, "se": se, "n_seeds": len(rates)}
        by_seed = {}
        for r in runs:
            by_seed.setdefault(r["seed"], {})[r["offset"]] = r["success_rate"]
        complete = {s: v for s, v in by_seed.items() if set(v) == set(offsets)}
        mu, se = mean_se([np.mean(list(v.values())) for v in complete.values()]) if complete else (np.nan, None)
        summary[m] = {
            "success": mu, "se": se, "offsets": per, "seeds": sorted(complete),
            "num_eval": min(r["num_eval"] for r in runs),
            "complete": set(offsets) == set(OFFSETS),
        }
    return summary


def board_key(exp, method):
    return f"baselines/{method}" if method in BASELINES else f"{exp}/{method}"


def load_leaderboard():
    board = json.loads(LEADERBOARD.read_text()) if LEADERBOARD.exists() else {}
    assert all(e["protocol"] == PROTOCOL for e in board.values()), f"{LEADERBOARD} : autre protocole"
    return board


def record(board, exp, results):
    """Ajoute / remplace les méthodes de `exp` évaluées sur toutes les tâches de tous les écarts."""
    for m, s in summarize(results).items():
        if s["num_eval"] != FULL_EVAL or not s["complete"]:
            print(f"leaderboard : {m} ignoré (benchmark incomplet)")
            continue
        board[board_key(exp, m)] = {
            "protocol": PROTOCOL, "label": LABELS[m], "family": family(m),
            "exp": None if m in BASELINES else exp, "method": m, **s,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(json.dumps(board, indent=2))
    print(f"Leaderboard : {LEADERBOARD}")


##############################
##          graphe          ##
##############################


def plot(results, exp, board, path):
    """Expérience courante (couleurs pleines), anciennes expériences du leaderboard (atténuées,
    plus récentes d'abord), puis les modèles du papier ; à droite, la réussite par écart au but."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.patches import Patch

    text, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    nan = lambda x: np.nan if x is None else x  # noqa: E731

    def tag(e):
        if e["num_eval"] != FULL_EVAL or not e["complete"]:
            return f" [partial: {e['num_eval']} tasks/offset]"
        return ""

    rows = []   # (libellé, entrée, couleur)
    current = summarize(results)
    if exp != "baselines":
        for m, s in current.items():
            rows.append((f"{exp} · {LABELS[m]}{tag(s)}", s, FAMILY_COLORS[family(m)]))
    shown = {board_key(exp, m) for m in current} if exp != "baselines" else set()
    older = [e for k, e in board.items() if e["exp"] is not None and k not in shown]
    for e in sorted(older, key=lambda e: e["updated"], reverse=True):
        rows.append((f"{e['exp']} · {e['label']}", e, to_rgba(FAMILY_COLORS[e["family"]], 0.45)))
    refs = {e["method"]: e for e in board.values() if e["exp"] is None}
    if exp == "baselines":   # résultats partiels : pas encore dans le leaderboard
        refs.update(current)
    for m in BASELINES:
        if m in refs:
            rows.append((f"{LABELS[m]} (paper weights){tag(refs[m])}", refs[m], FAMILY_COLORS[family(m)]))

    cols = [str(k) for k in OFFSETS]
    print(f"\n{'méthode':<56}{'moy. %':>8}{'± SE':>7}" + "".join(f"{'K=' + k:>8}" for k in cols))
    for label, e, _ in rows:
        per = "".join(f"{nan(e['offsets'].get(k, {}).get('success')):>8.1f}" for k in cols)
        print(f"{label:<56}{e['success']:>8.1f}{nan(e['se']):>7.1f}{per}")
    rows = rows[::-1]  # barh : premier en haut

    fig, (ax, hm) = plt.subplots(1, 2, figsize=(13, 0.42 * len(rows) + 2.2), facecolor=surface,
                                 gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)
    y = np.arange(len(rows))
    vals = [r[1]["success"] for r in rows]
    errs = [nan(r[1]["se"]) for r in rows]
    ax.set_facecolor(surface)
    bars = ax.barh(y, vals, 0.72, xerr=errs, color=[r[2] for r in rows],
                   error_kw={"ecolor": muted, "elinewidth": 1, "capsize": 3})
    for b, v, se in zip(bars, vals, errs):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", (v + np.nan_to_num(se), b.get_y() + b.get_height() / 2), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=muted)
    ax.set_title("Success rate (%), mean over goal offsets, ± SE over seeds", color=text, fontsize=11, loc="left")
    ax.set_xlim(0, 100)
    ax.set_yticks(y, [r[0] for r in rows], color=text)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()],
              loc="lower right", frameon=False, fontsize=8, labelcolor=muted)

    # réussite par écart : rampe séquentielle bleue (100 -> 650), valeur écrite dans chaque case
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
    grid_vals = np.array([[nan(r[1]["offsets"].get(k, {}).get("success")) for k in cols] for r in rows])
    hm.imshow(grid_vals, cmap=cmap, vmin=0, vmax=100, aspect="auto",
              extent=(-0.5, len(cols) - 0.5, len(rows) - 0.5, -0.5))
    hm.set_ylim(-0.5, len(rows) - 0.5)
    for i, row in enumerate(grid_vals):
        for j, v in enumerate(row):
            if not np.isnan(v):
                hm.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color=text if v < 55 else surface)
    hm.set_xticks(range(len(cols)), cols, color=muted, fontsize=8)
    hm.set_xlabel("goal offset (env steps)", color=muted, fontsize=9)
    hm.set_title("Per goal offset (%)", color=text, fontsize=11, loc="left")
    for a in (ax, hm):
        a.tick_params(colors=muted, length=0)
        for s in ("top", "right", "bottom"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_color(grid)

    fig.suptitle(f"OGBench Cube — held-out episodes, {PER_OFFSET} tasks per goal offset, budget 1.5 × offset",
                 color=text, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(w_pad=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Graphe : {path}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", nargs="?", help="expérience : results/<exp>/seed<s>/ (python train.py exp=<exp> seed=<s>)")
    p.add_argument("--baselines", action="store_true", help="évalue les modèles du papier au lieu d'une expérience")
    p.add_argument("--models", nargs="+", default=list(BASELINES), choices=list(BASELINES),
                   help="avec --baselines : modèles du papier à évaluer")
    p.add_argument("--methods", nargs="+", default=["hwm", "hwm-hybrid", "hwm-noplan"], choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="seed modèle = seed planificateur = seed du LeWM gelé")
    p.add_argument("--offsets", nargs="+", type=int, default=list(OFFSETS), choices=list(OFFSETS),
                   help="écarts au but évalués (pas env)")
    p.add_argument("--num-samples", type=int, default=300, help="CEM sur m : nb d'échantillons")
    p.add_argument("--cem-steps", type=int, default=30, help="CEM sur m : nb d'itérations")
    p.add_argument("--topk", type=int, default=30, help="CEM sur m : nb d'élites")
    p.add_argument("--hybrid-samples", type=int, default=None,
                   help="hybrides : nb de candidats du GradientSolver (défaut : celui de la config, 800)")
    p.add_argument("--num-eval", type=int, default=FULL_EVAL,
                   help=f"tâches par écart (les N premières ; < {FULL_EVAL} = test rapide, hors leaderboard)")
    p.add_argument("--rerun", action="store_true", help="efface les chunks existants au lieu de reprendre")
    p.add_argument("--plot-only", action="store_true", help="ne simule pas, retrace le graphe depuis le JSON")
    args = p.parse_args()
    if args.baselines == (args.exp is not None):
        p.error("donner soit <exp>, soit --baselines")

    exp = "baselines" if args.baselines else args.exp
    args.exp_dir = BASELINES_DIR if args.baselines else ROOT / "results" / exp
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        results = json.loads((args.exp_dir / "benchmark.json").read_text())
    else:
        results = run_benchmark(args)

    board = load_leaderboard()
    record(board, exp, results)
    plot(results, exp, board, args.exp_dir / "benchmark.png")


if __name__ == "__main__":
    main()
