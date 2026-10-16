"""Benchmark d'audit, à part du benchmark principal (n'écrit ni results/leaderboard.json ni les
benchmark.json / benchmark.png des expériences). Trois parties :

  main       nos méthodes sur 3 seeds, protocole val4x50 (K = 25, 50, 100, 200)
               1_hwm_base : hwm, hwm-hybrid, hwm-noplan
  ablations  d'où vient le gain de l'hybride ? protocole val4x50
               1_hwm_base    : hwm-hybrid-noplan (pi sans CEM sur m), hwm-hybrid-nogd (sans GD)
               1_flat_policy : flat-noplan, flat-hybrid (politique à plat conditionnée par le but,
                               même LeWM gelé / données / budget, sans niveau 1)
  paper      protocole du papier H-JEPA (leurs 50 tâches de 20 pas centrées sur la saisie, budget 30,
             leurs configs) : leurs 7 modèles, puis toutes nos méthodes ci-dessus

Les chunks d'éval vont aux mêmes endroits que ceux de benchmark.py (results/<exp>/seed<s>/eval_<protocole>/,
results/baselines/<modèle>/seed<s>/eval_<protocole>/) : les tâches déjà évaluées sont reprises, pas refaites.
Résultats : results/audit/seed<s>.json (un fichier par seed : plusieurs seeds en parallèle sans conflit),
puis `report` -> results/audit/report.md + audit.png.

  python benchmark_audit.py run --part paper --seeds 42           # un processus par seed en parallèle
  python benchmark_audit.py run --part main ablations --seeds 43
  python benchmark_audit.py run --part paper ablations --seeds 42 --exps 1_flat_policy
  python benchmark_audit.py report
"""

import argparse
import json
from pathlib import Path

import numpy as np

import benchmark as bm

AUDIT_DIR = bm.ROOT / "results/audit"
HWM_EXP, FLAT_EXP = "1_hwm_base", "1_flat_policy"
MAIN = {HWM_EXP: ["hwm", "hwm-hybrid", "hwm-noplan"]}
ABLATIONS = {HWM_EXP: ["hwm-hybrid-noplan", "hwm-hybrid-nogd"], FLAT_EXP: ["flat-noplan", "flat-hybrid"]}
PROTOCOL_OFFSETS = {bm.PROTOCOL: list(bm.OFFSETS), bm.PAPER_PROTOCOL: [bm.PAPER_OFFSET]}

# comparaisons appariées (mêmes tâches, même LeWM, même seed de planificateur) : (A, B, question)
COMPARISONS = [
    ("hwm-hybrid", "ref-lewm", "l'hybride bat-il LeWM ?"),
    ("hwm-hybrid", "flat-hybrid", "la hiérarchie apporte-t-elle plus qu'une politique à plat ?"),
    ("hwm-hybrid", "hwm-hybrid-noplan", "le CEM sur m aide-t-il ?"),
    ("hwm-hybrid", "hwm-hybrid-nogd", "la descente de gradient aide-t-elle ?"),
    ("flat-hybrid", "ref-lewm", "une politique à plat suffit-elle à battre LeWM ?"),
    ("hwm-noplan", "flat-noplan", "pi hiérarchique vs pi à plat, sans planification"),
    ("hwm-hybrid", "ref-hjepa-l3", "l'hybride bat-il H-JEPA 3 niveaux ?"),
    ("hwm-hybrid", "ref-hjepa-l4", "l'hybride bat-il H-JEPA 4 niveaux ?"),
]


def jobs(parts, exps):
    """[(protocole, exp ou None, méthode)] dans l'ordre d'exécution ; exps filtre nos expériences
    (None = les modèles du papier)."""
    out = []
    for part in parts:
        if part == "paper":
            out += [(bm.PAPER_PROTOCOL, None, m) for m in bm.BASELINES]
            out += [(bm.PAPER_PROTOCOL, exp, m) for d in (MAIN, ABLATIONS) for exp, ms in d.items() for m in ms]
        else:
            out += [(bm.PROTOCOL, exp, m) for exp, ms in {"main": MAIN, "ablations": ABLATIONS}[part].items()
                    for m in ms]
    return [j for j in out if j[1] in exps]


def save(r, seed):
    key = lambda x: (x["protocol"], x["method"], x["offset"])  # noqa: E731
    path = AUDIT_DIR / f"seed{seed}.json"
    results = json.loads(path.read_text()) if path.exists() else []
    results = [x for x in results if key(x) != key(r)] + [r]
    path.write_text(json.dumps(results, indent=2))
    print(f"-> [{r['protocol']}] {r['method']} seed={seed} écart={r['offset']} : succès {r['success_rate']:.1f}%")


def run(args):
    bm.setup_hjepa_imports()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    args.rerun, args.num_eval, args.hybrid_samples = False, bm.FULL_EVAL, None
    for seed in args.seeds:
        models = {}
        for protocol, exp, m in jobs(args.part, {None if e == 'baselines' else e for e in args.exps}):
            if exp is not None and exp not in models:
                models[exp] = bm.load_hwm(bm.ROOT / "results" / exp / f"seed{seed}")
            for offset in PROTOCOL_OFFSETS[protocol]:
                print(f"\n=== [{protocol}] {m} | seed={seed} | écart={offset} ===")
                if exp is None:
                    r = bm.run_baseline(m, seed, offset, args, protocol)
                else:
                    r = bm.run_ours(m, seed, offset, bm.ROOT / "results" / exp / f"seed{seed}",
                                    models[exp], args, protocol)
                save({"protocol": protocol, "exp": exp, **r}, seed)


##############################
##         rapport          ##
##############################


def load_all():
    rows = []
    for p in sorted(AUDIT_DIR.glob("seed*.json")):
        rows += json.loads(p.read_text())
    return rows


def table(rows, protocol):
    """{méthode: {"offsets": {K: (moy, se, n)}, "mean": (moy, se, n)}} sur les seeds complètes."""
    offsets = PROTOCOL_OFFSETS[protocol]
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows if r["protocol"] == protocol):
        rs = [r for r in rows if r["protocol"] == protocol and r["method"] == m]
        per = {}
        for k in offsets:
            x = [r["success_rate"] for r in rs if r["offset"] == k]
            per[k] = mean_se(x)
        by_seed = {}
        for r in rs:
            by_seed.setdefault(r["seed"], {})[r["offset"]] = r["success_rate"]
        full = [np.mean([v[k] for k in offsets]) for v in by_seed.values() if set(v) >= set(offsets)]
        out[m] = {"offsets": per, "mean": mean_se(full)}
    return out


def mean_se(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, 0)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan, len(x))


def paired(rows, protocol, a, b):
    """Tâche par tâche, même seed et même écart : gains / pertes de a sur b, test de signe exact
    (McNemar) sur toutes les paires, et différence moyenne par seed ± SE."""
    from scipy.stats import binomtest

    idx = {(r["method"], r["seed"], r["offset"]): np.array(r["episode_successes"])
           for r in rows if r["protocol"] == protocol}
    wins = losses = 0
    diffs = {}
    for (m, s, k), xa in idx.items():
        if m != a or (b, s, k) not in idx:
            continue
        xb = idx[(b, s, k)]
        wins += int((xa & ~xb).sum())
        losses += int((~xa & xb).sum())
        diffs.setdefault(s, []).append(100 * (xa.mean() - xb.mean()))
    if not diffs:
        return None
    p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
    seeds = sorted(diffs)
    d = mean_se([np.mean(diffs[s]) for s in seeds])
    return {"wins": wins, "losses": losses, "p": p, "diff": d, "seeds": seeds}


def fmt(t):
    mu, se, n = t
    if np.isnan(mu):
        return "—"
    return f"{mu:.1f} ± {se:.1f} ({n})" if n > 1 else f"{mu:.1f} ({n})"


def report(args):
    rows = load_all()
    lines = ["# Audit benchmark", "",
             "Moyenne ± SE sur les seeds (nombre de seeds entre parenthèses). Comparaisons appariées : "
             "mêmes tâches, même seed (même LeWM, même seed de planificateur) ; p = test de signe exact "
             "(McNemar) sur les paires de tâches discordantes, toutes seeds et tous écarts confondus.", ""]
    tables = {}
    for protocol, title in [(bm.PROTOCOL, f"Protocole {bm.PROTOCOL} : val, K = {list(bm.OFFSETS)}, budget 1,5 K"),
                            (bm.PAPER_PROTOCOL, "Protocole du papier : 50 tâches de 20 pas centrées sur la saisie, budget 30")]:
        t = tables[protocol] = table(rows, protocol)
        if not t:
            continue
        offsets = PROTOCOL_OFFSETS[protocol]
        lines += [f"## {title}", ""]
        head = ["méthode"] + ([f"K={k}" for k in offsets] if len(offsets) > 1 else []) + ["moyenne"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for m in sorted(t, key=lambda m: -np.nan_to_num(t[m]["mean"][0], nan=-1)):
            cells = [bm.LABELS.get(m, m)] + ([fmt(t[m]["offsets"][k]) for k in offsets] if len(offsets) > 1 else [])
            lines.append("| " + " | ".join(cells + [fmt(t[m]["mean"])]) + " |")
        lines += ["", "| comparaison | question | gains / pertes | p | écart moyen (points) | seeds |",
                  "|---|---|---|---|---|---|"]
        for a, b, q in COMPARISONS:
            c = paired(rows, protocol, a, b)
            if c:
                lines.append(f"| {bm.LABELS.get(a, a)} vs {bm.LABELS.get(b, b)} | {q} | {c['wins']} / {c['losses']} "
                             f"| {c['p']:.2g} | {fmt(c['diff'])} | {c['seeds']} |")
        lines.append("")
    if tables.get(bm.PAPER_PROTOCOL):
        ref = {"ref-lewm": 32.0, "ref-hjepa-l2": 47.3, "ref-hjepa-l3": 60.0, "ref-hjepa-l4": 62.7,
               "ref-hwm-l2": 46.7, "ref-hwm-l3": 53.3, "ref-hwm-l4": 34.0}   # docs/RESULTS.md (Figure 6)
        lines += ["## Reproduction des chiffres du papier (protocole du papier)", "",
                  "| modèle | notre run | papier |", "|---|---|---|"]
        for m, v in ref.items():
            if m in tables[bm.PAPER_PROTOCOL]:
                lines.append(f"| {bm.LABELS[m]} | {fmt(tables[bm.PAPER_PROTOCOL][m]['mean'])} | {v} |")
        lines.append("")
    (AUDIT_DIR / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    plot(tables, AUDIT_DIR / "audit.png")


def plot(tables, path):
    """Une ligne par méthode, une colonne par écart (val4x50) + moyenne, + protocole du papier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    text, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    val, pap = tables.get(bm.PROTOCOL, {}), tables.get(bm.PAPER_PROTOCOL, {})
    methods = [m for m in (*bm.METHODS, *bm.BASELINES) if m in val or m in pap]
    cols = [f"K={k}" for k in bm.OFFSETS] + ["mean", "paper\nprotocol"]
    grid = np.full((len(methods), len(cols)), np.nan)
    ses = np.full_like(grid, np.nan)
    for i, m in enumerate(methods):
        cells = [val.get(m, {}).get("offsets", {}).get(k) for k in bm.OFFSETS]
        cells += [val.get(m, {}).get("mean"), pap.get(m, {}).get("mean")]
        for j, c in enumerate(cells):
            if c is not None:
                grid[i, j], ses[i, j] = c[0], c[1]

    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(methods) + 1.8), facecolor=surface)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(methods)):
        for j in range(len(cols)):
            v, se = grid[i, j], ses[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}" + ("" if np.isnan(se) else f" ±{se:.0f}"), ha="center", va="center",
                        fontsize=8, color=text if v < 55 else surface)
    ax.set_yticks(range(len(methods)), [bm.LABELS[m] for m in methods], color=text, fontsize=9)
    ax.set_xticks(range(len(cols)), cols, color=muted, fontsize=9)
    ax.axvline(len(bm.OFFSETS) - 0.5, color=surface, lw=3)
    ax.axvline(len(bm.OFFSETS) + 0.5, color=surface, lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Audit — success rate (%), mean ± SE over seeds", color=text, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    print(f"Graphe : {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", nargs="+", required=True, choices=["main", "ablations", "paper"])
    r.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    r.add_argument("--exps", nargs="+", default=["baselines", HWM_EXP, FLAT_EXP],
                   help="restreint les jobs : baselines (modèles du papier), 1_hwm_base, 1_flat_policy")
    r.add_argument("--num-samples", type=int, default=300)
    r.add_argument("--cem-steps", type=int, default=30)
    r.add_argument("--topk", type=int, default=30)
    sub.add_parser("report")
    args = p.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
