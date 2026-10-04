"""Benchmark : LeWM vs LeHWM (ton modèle hiérarchique) en simulation.

Lance les mêmes épisodes (même état initial, même but) pour chaque méthode,
sauvegarde les résultats en JSON puis trace un graphe comparatif.

Méthodes disponibles :
  lewm        -> LeWM de référence (quentinll/lewm-cube) + CEM sur les actions (comme eval.py)
  hwm         -> LeHWM, planification niveau 1 : CEM sur le skill m, coût = ||P1(z1, m) - z1_goal||²,
                 actions décodées par pi(z0, m*)
  hwm-hybrid  -> LeHWM : pi(z0, m*) initialise le CEM de niveau 0 (warm start), puis CEM classique

Le modèle LeHWM est lu dans results/<run>/weights.pt (sortie de train.py) et les résultats
sont écrits à côté : results/<run>/benchmark.json, benchmark.png (et videos/).

Exemples :
  python benchmark.py mon_run                                     # lewm vs hwm, offset 25, 3 seeds
  python benchmark.py mon_run --methods lewm hwm hwm-hybrid --goal-offsets 25 50 --num-eval 50
  python benchmark.py mon_run --plot-only                         # retrace le graphe depuis le JSON
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

warnings.filterwarnings("ignore", module="gymnasium")

METHODS = ("lewm", "hwm", "hwm-hybrid")
LABELS = {"lewm": "LeWM", "hwm": "LeHWM (niveau 1)", "hwm-hybrid": "LeHWM (hybride)"}
# palette catégorielle validée (slots 1-3), ordre fixe par méthode
COLORS = {"lewm": "#2a78d6", "hwm": "#eb6834", "hwm-hybrid": "#1baf7a"}


##############################
##   politique hiérarchique  ##
##############################


class SkillPlanner:
    """1) CEM dans le prior (espace blanchi eps) sur un saut de niveau 1,
       2) re-classement ancré : les k meilleurs m sont décodés par pi et déroulés dans P0 ;
          le sous-but est l'état RÉELLEMENT atteint, pas la prédiction de P1."""

    def __init__(self, model, n, action_block, num_samples=300, n_steps=30, topk=30,
                 rerank_k=16, lam_eps=0.01, lam_exploit=1.0, history=3,
                 device="cuda", seed=0):
        self.model = model
        self.n = n
        self.action_block = action_block
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
        self.gen = torch.Generator(device=device).manual_seed(seed)
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
        mu_p, std_p = self._prior(z1)                                          # (B, m_dim)

        # ---- 1) CEM dans l'espace blanchi : m = mu_p + std_p * eps ----
        mean = torch.zeros(B, self.m_dim, device=dev)
        std = torch.ones_like(mean)
        z1_rep = z1.repeat_interleave(S, 0)
        for _ in range(self.n_steps):
            eps = mean[:, None] + std[:, None] * torch.randn(
                B, S, self.m_dim, generator=self.gen, device=dev)
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

class HierarchicalPolicy:
    """Politique MPC : replanifie un skill toutes les `receding` blocs d'actions."""

    type = "hierarchical"

    def __init__(self, planner, receding, process, transform):
        from stable_worldmodel.policy import BasePolicy

        self._base = BasePolicy(process=process, transform=transform)
        self.planner = planner
        self.receding = receding
        self.process = process
        self.env = None

    def set_env(self, env):
        self.env = env
        self._buffer = [deque() for _ in range(env.num_envs)]

    def get_action(self, info_dict, **kwargs):
        info = self._base._prepare_info(info_dict)
        n_envs = self.env.num_envs

        needs_flush = info.pop("_needs_flush", None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._buffer[i].clear()

        terminated = info.get("terminated")
        dead = np.asarray(terminated, dtype=bool) if terminated is not None else np.zeros(n_envs, bool)
        replan = [i for i in range(n_envs) if not self._buffer[i] and not dead[i]]

        if replan:
            idx = torch.as_tensor(replan)
            blocks = self.planner.plan(info["pixels"][idx], info["goal"][idx])  # (k, n, block*raw)
            steps = blocks[:, : self.receding].reshape(len(replan), self.receding * self.planner.action_block, -1)
            for row, i in enumerate(replan):
                self._buffer[i].extend(steps[row].cpu())

        raw_dim = self.env.single_action_space.shape[-1]
        action = torch.full((n_envs, raw_dim), float("nan"))
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._buffer[i].popleft()

        action = action.reshape(*self.env.action_space.shape).float().numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


class HybridCostModel(torch.nn.Module):
    """Coût = celui de LeWM (niveau 0 de LeHWM) ; init du CEM = actions proposées par pi.
    Le CEM de stable_worldmodel appelle get_action pour le warm start si le modèle est Actionable."""

    def __init__(self, hwm, planner, horizon):
        super().__init__()
        self.hwm = hwm
        self.planner = planner
        self.horizon = horizon

    def get_cost(self, info_dict, action_candidates):
        return self.hwm.get_cost(info_dict, action_candidates)

    def get_action(self, info, horizon=1, prefix_actions=None):
        blocks = self.planner.plan(info["pixels"], info["goal"])  # (B, n, action_dim)
        B, n, D = blocks.shape
        if n >= horizon:
            return blocks[:, :horizon]
        pad = torch.zeros(B, horizon - n, D, device=blocks.device, dtype=blocks.dtype)
        return torch.cat([blocks, pad], dim=1)


##############################
##        simulation        ##
##############################


def load_models(args):
    import stable_worldmodel as swm

    models = {}

    def prep(model):
        model = model.to("cuda").eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        return model

    if "lewm" in args.methods:
        models["lewm"] = prep(swm.wm.utils.load_pretrained(args.lewm))
    if any(m.startswith("hwm") for m in args.methods):
        models["hwm"] = prep(swm.wm.utils.load_pretrained(str(args.run_dir / "weights.pt")))
    return models


def build_policy(method, models, cfg, args, process, transform, seed):
    import hydra
    import stable_worldmodel as swm

    action_block = cfg.plan_config.action_block
    plan_config = swm.PlanConfig(**cfg.plan_config)

    def skill_planner():
        return SkillPlanner(models["hwm"], n=args.n, action_block=action_block,
                            num_samples=args.num_samples, n_steps=args.cem_steps,
                            topk=args.topk, seed=seed)

    if method == "lewm":
        solver = hydra.utils.instantiate(cfg.solver, model=models["lewm"], seed=seed)
        return swm.policy.WorldModelPolicy(solver=solver, config=plan_config,
                                           process=process, transform=transform)
    if method == "hwm":
        return HierarchicalPolicy(skill_planner(), receding=args.hwm_receding,
                                  process=process, transform=transform)
    if method == "hwm-hybrid":
        cost_model = HybridCostModel(models["hwm"], skill_planner(), cfg.plan_config.horizon)
        solver = hydra.utils.instantiate(cfg.solver, model=cost_model, seed=seed)
        return swm.policy.WorldModelPolicy(solver=solver, config=plan_config,
                                           process=process, transform=transform)
    raise ValueError(method)


def sample_episodes(dataset, goal_offset, num_eval, seed):
    """Même tirage que eval.py -> épisodes identiques pour toutes les méthodes."""
    from eval import get_episodes_length

    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col))
    max_start = get_episodes_length(dataset, ep_indices) - goal_offset - 1
    max_start = dict(zip(ep_indices, max_start))
    max_start_per_row = np.array([max_start[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]

    g = np.random.default_rng(seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    data = dataset.get_row_data(rows)
    return data[col].tolist(), data["step_idx"].tolist()


def run_benchmark(args):
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from sklearn import preprocessing
    from eval import get_dataset, img_transform

    cfg = OmegaConf.load(ROOT / "config/eval" / f"{args.env}.yaml")
    cfg.solver = OmegaConf.load(ROOT / "config/eval/solver/cem.yaml")
    cfg.solver.n_steps = args.cem_steps_l0
    cfg.cache_dir = None
    cfg.eval.num_eval = args.num_eval

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        data = dataset.get_col_data(col)
        process[col] = preprocessing.StandardScaler().fit(data[~np.isnan(data).any(axis=1)])
        if col != "action":
            process[f"goal_{col}"] = process[col]
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    models = load_models(args)
    out_path = args.run_dir / "benchmark.json"
    results = json.loads(out_path.read_text()) if out_path.exists() and args.append else []

    for goal_offset in args.goal_offsets:
        budget = args.budget_ratio * goal_offset
        cfg.world.max_episode_steps = 2 * budget
        callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)

        for seed in args.seeds:
            episodes, starts = sample_episodes(dataset, goal_offset, args.num_eval, seed)

            for method in args.methods:
                print(f"\n=== {method} | goal_offset={goal_offset} | seed={seed} ===")
                world = swm.World(**cfg.world, image_shape=(224, 224))
                world.set_policy(build_policy(method, models, cfg, args, process, transform, seed))

                video = args.run_dir / "videos" / f"{method}_off{goal_offset}_s{seed}"
                t0 = time.time()
                metrics = world.evaluate(
                    dataset=dataset, start_steps=starts, goal_offset=goal_offset,
                    eval_budget=budget, episodes_idx=episodes, callables=callables,
                    video=video if args.video else None,
                )
                elapsed = time.time() - t0
                print(f"-> success_rate={metrics['success_rate']:.1f}%  ({elapsed:.1f}s)")

                results.append({
                    "method": method, "env": args.env, "goal_offset": goal_offset,
                    "eval_budget": budget, "seed": seed, "num_eval": args.num_eval,
                    "success_rate": float(metrics["success_rate"]),
                    "episode_successes": np.asarray(metrics["episode_successes"]).astype(bool).tolist(),
                    "time_s": elapsed,
                })
                # sauvegarde après chaque run : rien n'est perdu si ça plante
                out_path.write_text(json.dumps(results, indent=2))

    print(f"\nRésultats : {out_path}")
    return results


##############################
##          graphe          ##
##############################


def summarize(results):
    """{(method, offset): (mean, std, n_seeds, mean_time_per_episode)}"""
    summary = {}
    keys = {(r["method"], r["goal_offset"]) for r in results}
    for key in sorted(keys, key=lambda k: (METHODS.index(k[0]), k[1])):
        runs = [r for r in results if (r["method"], r["goal_offset"]) == key]
        rates = np.array([r["success_rate"] for r in runs])
        per_ep = np.mean([r["time_s"] / r["num_eval"] for r in runs])
        summary[key] = (rates.mean(), rates.std(), len(runs), per_ep)
    return summary


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize(results)
    methods = [m for m in METHODS if any(k[0] == m for k in summary)]
    offsets = sorted({k[1] for k in summary})

    print(f"\n{'méthode':<18}{'offset':>8}{'succès %':>12}{'± std':>8}{'seeds':>7}{'s/épisode':>12}")
    for (m, off), (mu, sd, n, t) in summary.items():
        print(f"{LABELS[m]:<18}{off:>8}{mu:>12.1f}{sd:>8.1f}{n:>7}{t:>12.2f}")

    text, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="#fcfcfb",
                             gridspec_kw={"width_ratios": [2, 1]})
    width = 0.8 / len(methods)
    x = np.arange(len(offsets))

    for ax, col, title, ylabel in (
        (axes[0], 0, "Taux de succès (moyenne ± écart-type sur les seeds)", "succès (%)"),
        (axes[1], 3, "Temps de planification", "secondes / épisode"),
    ):
        ax.set_facecolor("#fcfcfb")
        for i, m in enumerate(methods):
            vals = [summary.get((m, o), (np.nan,) * 4) for o in offsets]
            heights = [v[col] for v in vals]
            errs = [v[1] for v in vals] if col == 0 else None
            pos = x + (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(pos, heights, width * 0.92, yerr=errs, label=LABELS[m], color=COLORS[m],
                          error_kw={"ecolor": muted, "elinewidth": 1, "capsize": 3})
            for b, h in zip(bars, heights):
                if not np.isnan(h):
                    ax.annotate(f"{h:.0f}" if col == 0 else f"{h:.1f}",
                                (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=8, color=muted)
        ax.set_xticks(x, [f"offset {o}" for o in offsets])
        ax.set_title(title, color=text, fontsize=11, loc="left")
        ax.set_ylabel(ylabel, color=muted)
        ax.tick_params(colors=muted, length=0)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)

    axes[0].set_ylim(0, 120)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].legend(frameon=False, labelcolor=text, loc="upper right")
    env = results[0]["env"] if results else ""
    fig.suptitle(f"LeWM vs LeHWM — {env}", color=text, fontsize=13, x=0.02, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Graphe : {path}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="nom du run : results/<run>/ (contient weights.pt de train.py)")
    p.add_argument("--methods", nargs="+", default=["lewm", "hwm"], choices=METHODS)
    p.add_argument("--env", default="cube", help="nom du fichier dans config/eval/ (sans .yaml)")
    p.add_argument("--lewm", default="quentinll/lewm-cube", help="checkpoint LeWM (relatif à $STABLEWM_HOME/checkpoints)")
    p.add_argument("--goal-offsets", nargs="+", type=int, default=[25], help="distance au but (pas env) ; plusieurs = difficulté croissante")
    p.add_argument("--budget-ratio", type=int, default=2, help="eval_budget = ratio * goal_offset")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--num-eval", type=int, default=50, help="épisodes par (méthode, offset, seed)")
    p.add_argument("--n", type=int, default=4, help="nb de blocs prédits par pi (cfg.n à l'entraînement)")
    p.add_argument("--hwm-receding", type=int, default=1, help="blocs exécutés avant de replanifier (hwm)")
    p.add_argument("--num-samples", type=int, default=300, help="CEM sur m : nb d'échantillons")
    p.add_argument("--cem-steps", type=int, default=30, help="CEM sur m : nb d'itérations")
    p.add_argument("--cem-steps-l0", type=int, default=10,
                   help="CEM sur les actions (lewm, hwm-hybrid) : nb d'itérations (papier LeWM : 10 hors PushT)")
    p.add_argument("--topk", type=int, default=30, help="CEM sur m : nb d'élites")
    p.add_argument("--append", action="store_true", help="ajoute aux résultats existants au lieu d'écraser")
    p.add_argument("--video", action="store_true", help="enregistre les vidéos des épisodes")
    p.add_argument("--plot-only", action="store_true", help="ne simule pas, retrace le graphe depuis le JSON")
    args = p.parse_args()

    args.run_dir = ROOT / "results" / args.run
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        results = json.loads((args.run_dir / "benchmark.json").read_text())
    else:
        results = run_benchmark(args)

    plot(results, args.run_dir / "benchmark.png")


if __name__ == "__main__":
    main()
