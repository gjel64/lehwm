#!/usr/bin/env bash
# Audit complet (benchmark_audit.py), relançable : entraînements terminés et tâches déjà évaluées sont repris.
#   - entraînements dans UNE file (mémoire unifiée du GB10 : plusieurs entraînements en parallèle = OOM,
#     et le GPU est déjà saturé par un seul) : flat 42, hwm 43, flat 43, hwm 44, flat 44
#     (1_hwm_base : recette inchangée ; 1_flat_policy : model=flat_policy)
#   - par seed, en parallèle : modèles du papier sur leur protocole, puis nos méthodes et ablations
#     dès que les modèles de cette seed sont entraînés
#   - puis python benchmark_audit.py report -> results/audit/report.md, audit.png
# Logs : results/audit/logs/
set -uo pipefail
cd "$(dirname "$0")"
source /home/mattin/Desktop/.venv/bin/activate
export HJEPA_HOME=${HJEPA_HOME:-$HOME/hjepa_home}
LOGS=results/audit/logs
mkdir -p "$LOGS" results/1_hwm_base/logs results/1_flat_policy/logs
rm -f "$LOGS"/queue_done

log() { echo "$(date '+%F %T') $*" | tee -a "$LOGS/pipeline.log"; }

trained() {  # weights.pt + log Lightning « max_epochs reached » (pas de log : run antérieur, considéré fini)
  local log=results/$1/logs/train_seed$2.log
  [[ -f results/$1/seed$2/weights.pt ]] && { [[ ! -f $log ]] || grep -q "max_epochs=.*reached" "$log"; }
}

train() {  # exp seed [overrides...]
  local exp=$1 s=$2; shift 2
  trained "$exp" "$s" && return
  log "train $exp seed=$s"
  python train.py exp="$exp" seed="$s" "$@" > "results/$exp/logs/train_seed$s.log" 2>&1
  if trained "$exp" "$s"; then log "entraîné : $exp seed=$s"
  else log "ÉCHEC entraînement $exp seed=$s (voir results/$exp/logs/train_seed$s.log)"; fi
}

wait_trained() {  # attend que la file ait entraîné exp/seed ; échoue si la file a fini sans
  until trained "$1" "$2"; do
    [[ -f $LOGS/queue_done ]] && { trained "$1" "$2" || return 1; }
    sleep 60
  done
}

audit() {  # seed args...
  local s=$1; shift
  log "audit seed=$s $*"
  python benchmark_audit.py run --seeds "$s" "$@" >> "$LOGS/seed$s.log" 2>&1 \
    || { log "ÉCHEC audit seed=$s $*"; return 1; }
  log "fini : audit seed=$s $*"
}

seed_chain() {
  local s=$1
  audit "$s" --part paper --exps baselines
  wait_trained 1_hwm_base "$s" && audit "$s" --part main ablations paper --exps 1_hwm_base
  wait_trained 1_flat_policy "$s" && audit "$s" --part ablations paper --exps 1_flat_policy
}

log "=== début de l'audit"
( train 1_flat_policy 42 model=flat_policy
  train 1_hwm_base 43
  train 1_flat_policy 43 model=flat_policy
  train 1_hwm_base 44
  train 1_flat_policy 44 model=flat_policy
  touch "$LOGS/queue_done" ) &
for s in 42 43 44; do seed_chain $s & done
wait
python benchmark_audit.py report >> "$LOGS/report.log" 2>&1
log "=== fin de l'audit : results/audit/report.md"
