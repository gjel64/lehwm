#!/usr/bin/env bash
# Prépare l'entraînement (split et LeWM du papier H-JEPA, arXiv 2610.06805) et le benchmark sur OGBench Cube :
#   1. code H-JEPA (commit figé) dans third_party/H-JEPA : boucle d'éval + stable_worldmodel modifié
#   2. leurs 7 modèles cube (LeWM, H-JEPA / HWM 2-4 niveaux ; seeds 42-44, entraînés sur le split train)
#   3. split train (9 800 épisodes) / val (200) de cube_single_expert.h5
#   4. tâches d'éval tirées des 200 épisodes val (make_val_tasks.py)
#   5. conversion des LeWM vers notre disposition (niveau 0 gelé de train.py)
set -euo pipefail
cd "$(dirname "$0")"

HJEPA_COMMIT=97054ecaff83e13839195ec4b92d74c08a0127c5
export HJEPA_HOME=${HJEPA_HOME:-$HOME/hjepa_home}
CUBE_H5=${CUBE_H5:-$HOME/.stable_worldmodel/datasets/ogbench/cube_single_expert.h5}  # dataset LeWM
mkdir -p "$HJEPA_HOME"

if [[ ! -d third_party/H-JEPA ]]; then
  git clone https://github.com/kevinghst/H-JEPA third_party/H-JEPA
fi
git -C third_party/H-JEPA checkout -q "$HJEPA_COMMIT"

hf download jepa-world-models/h-jepa --local-dir "$HJEPA_HOME/ckpts" --include "cube/*"

if [[ ! -f "$HJEPA_HOME/cube_single_expert_val.h5" ]]; then
  ln -sf "$CUBE_H5" "$HJEPA_HOME/cube_single_expert.h5"
  python -I third_party/H-JEPA/scripts/data/split_cube.py
fi
python make_val_tasks.py

for s in 42 43 44; do
  python convert_lewm.py "$HJEPA_HOME/ckpts/cube/cube_lewm/seed$s/cube_lewm_object.ckpt"
done
