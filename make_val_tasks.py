"""Génère les tâches d'éval du benchmark à partir des 200 épisodes de validation de
cube_single_expert (split H-JEPA : aucun modèle, ni du papier ni à nous, ne s'est entraîné dessus).

Pour chaque écart au but K (25, 50, 100, 200 pas env), N tâches :
  - épisode de validation tiré au hasard (sans remise), départ uniforme dans [0, longueur - 1 - K] ;
  - but = frame départ + K (image, et position du cube = cible de set_target_pos) ;
  - tâche écartée si le cube bouge de <= 4 cm entre départ et but (réussie sans rien faire) :
    on retire un autre départ, puis on passe à l'épisode suivant.
Format : celui des tâches chargées par la boucle d'éval H-JEPA (load_eval_trajs_path), 2 frames
par tâche (départ, but).

  python make_val_tasks.py        # -> $HJEPA_HOME/assets/eval_trajs/ogbench/val_offset<K>_n<N>.pt
"""

import argparse
import os
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  filtres de compression des pixels
import numpy as np
import torch

HJEPA_HOME = Path(os.environ.get("HJEPA_HOME", Path.home() / "hjepa_home"))
VAL = HJEPA_HOME / "cube_single_expert_val.h5"
OFFSETS = (25, 50, 100, 200)
PER_OFFSET = 50
SUCCESS_DIST = 0.04   # seuil de réussite de l'env (m)
SEED = 0
COLUMNS = ["pixels", "qpos", "qvel", "privileged_block_0_pos", "privileged_block_0_quat", "action"]


def task_path(offset, n):
    return HJEPA_HOME / f"assets/eval_trajs/ogbench/val_offset{offset}_n{n}.pt"


def sample(f, offset, n, rng, tries=10):
    """[(épisode, départ)] : n tâches non triviales, au plus une par épisode."""
    lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
    pos = f["privileged_block_0_pos"]
    tasks = []
    for ep in rng.permutation(len(lengths)):
        if lengths[ep] - 1 < offset:
            continue
        for _ in range(tries):
            start = int(rng.integers(0, lengths[ep] - offset))
            a, b = offsets[ep] + start, offsets[ep] + start + offset
            if np.linalg.norm(pos[b] - pos[a]) > SUCCESS_DIST:
                tasks.append((int(ep), start))
                break
        if len(tasks) == n:
            return tasks
    raise ValueError(f"écart {offset} : seulement {len(tasks)} tâches non triviales < {n}")


def build(f, tasks, offset):
    offsets = f["ep_offset"][:]
    data = []
    for ep, start in tasks:
        rows = [offsets[ep] + start, offsets[ep] + start + offset]
        e = {c: torch.from_numpy(np.stack([f[c][r] for r in rows])) for c in COLUMNS}
        if e["pixels"].shape[-1] == 3:   # (T, H, W, C) -> (T, C, H, W) comme les tâches H-JEPA
            e["pixels"] = e["pixels"].permute(0, 3, 1, 2).contiguous()
        data.append(e)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offsets", nargs="+", type=int, default=list(OFFSETS))
    p.add_argument("--n", type=int, default=PER_OFFSET, help="tâches par écart")
    args = p.parse_args()

    with h5py.File(VAL) as f:
        for offset in args.offsets:
            rng = np.random.default_rng(SEED + offset)
            tasks = sample(f, offset, args.n, rng)
            out = task_path(offset, args.n)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "dataset_eval_trajs_v1",
                "data": build(f, tasks, offset),
                "columns": COLUMNS,
                "episodes_idx": [ep for ep, _ in tasks],
                "start_steps": [s for _, s in tasks],
                "goal_offset_steps": offset,
                "task_info": {"source": str(VAL), "sampling": "uniform start, cube moved > 4 cm",
                              "seed": SEED + offset, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
            }, out)
            print(f"écart {offset} : {len(tasks)} tâches -> {out}")


if __name__ == "__main__":
    main()
