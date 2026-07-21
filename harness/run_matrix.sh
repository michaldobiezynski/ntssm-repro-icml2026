#!/usr/bin/env bash
# Generalises run_lastfm.sh to any bundled dataset: reproduce the LightGCN 2x2 objective
# matrix (BPR, NT-BPR, SSM, NT-SSM) for <dataset>, with that dataset's Table 5 alphas.
# Safe by default: nothing trains unless --execute is passed.
#
#   run_matrix.sh <dataset> [--dry-run | --execute]
#   <dataset> in {lastfm, ml-1m}
set -euo pipefail

DATASET="${1:-}"
MODE="${2:-}"
SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-200}"

# Per-dataset optimal NT alphas (paper Appendix Table 5, LightGCN) + a CPU wall-clock budget.
# NT-BPR and NT-SSM have DIFFERENT alphas; tau/lr/reg/layers are fixed across datasets.
case "$DATASET" in
  lastfm)
    TAU=0.2; BUDGET_DEFAULT=45
    NT_SSM_ALPHAS="--alpha_uu 1.2 --alpha_ii 0.8 --alpha_ui 0.8 --alpha_iu 0.9"
    NT_BPR_ALPHAS="--alpha_uu 1.3 --alpha_ii 1.5 --alpha_ui 0.9 --alpha_iu 1.3" ;;
  ml-1m)
    TAU=0.2; BUDGET_DEFAULT=90
    NT_SSM_ALPHAS="--alpha_uu 1.2 --alpha_ii 0.8 --alpha_ui 0.8 --alpha_iu 1.0"
    NT_BPR_ALPHAS="--alpha_uu 1.4 --alpha_ii 1.3 --alpha_ui 0.8 --alpha_iu 1.3" ;;
  "")
    echo "usage: run_matrix.sh <dataset> [--dry-run|--execute]   (dataset: lastfm, ml-1m)" >&2; exit 2 ;;
  *)
    echo "error: unsupported dataset '$DATASET' (supported: lastfm, ml-1m)" >&2; exit 2 ;;
esac

if ! [[ "$SEED" =~ ^[0-9]+$ ]] || ! [[ "$EPOCHS" =~ ^[0-9]+$ ]]; then
  echo "error: SEED and EPOCHS must be integers" >&2; exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CLONE="${NTSSM_DIR:-$REPO/NT-SSM}"
VENV="$REPO/.venv"

COMMON="--dataset $DATASET --model_type graph --item_ranking 10,20,40 --embedding_size 64 --epoch $EPOCHS --batch_size 2048 --learning_rate 0.001 --reg_lambda 0.0001 --n_layer 2 --seed $SEED"
args_ntssm="--model_name LightGCN_NT $COMMON --loss_type ssm --tau $TAU $NT_SSM_ALPHAS"
args_ssm="--model_name LightGCN $COMMON --loss_type ssm --tau $TAU"
args_ntbpr="--model_name LightGCN_NT $COMMON --loss_type bpr $NT_BPR_ALPHAS"
args_bpr="--model_name LightGCN $COMMON --loss_type bpr"

print_matrix() {
  echo "# NT-SSM"; echo "python main.py $args_ntssm"; echo
  echo "# SSM";    echo "python main.py $args_ssm";   echo
  echo "# NT-BPR"; echo "python main.py $args_ntbpr"; echo
  echo "# BPR";    echo "python main.py $args_bpr";   echo
}

dry_run() {
  echo "=== DRY RUN: $DATASET / LightGCN pipeline (no training executed) ==="
  echo "seed=$SEED epochs=$EPOCHS tau=$TAU  NT-SSM[$NT_SSM_ALPHAS]  NT-BPR[$NT_BPR_ALPHAS]"
  echo
  print_matrix
  echo "=== DRY RUN complete ==="
}

execute() {
  [ -x "$VENV/bin/python" ] || { echo "error: no .venv; run harness/setup_env.sh first" >&2; exit 2; }
  grep -rq "_NTSSM_DEVICE" "$CLONE/model/graph/LightGCN.py" 2>/dev/null || {
    echo "error: clone not device-patched; run harness/setup_env.sh first" >&2; exit 2; }
  grep -q "# trackio: run init" "$CLONE/main.py" 2>/dev/null || {
    echo "error: clone missing the trackio metric patch (dashboards would be empty); run harness/setup_env.sh" >&2; exit 2; }
  local max_min="${MAX_MINUTES:-$BUDGET_DEFAULT}"
  echo "Executing $DATASET matrix on CPU (budget ${max_min} min)."
  cd "$CLONE"
  local start=$SECONDS
  for pair in "NT-SSM=$args_ntssm" "SSM=$args_ssm" "NT-BPR=$args_ntbpr" "BPR=$args_bpr"; do
    local label="${pair%%=*}" args="${pair#*=}"
    local elapsed_min=$(( (SECONDS - start) / 60 ))
    if [ "$elapsed_min" -ge "$max_min" ]; then
      echo "budget ${max_min}m reached before $label; stopping" >&2; break
    fi
    echo ">>> training $label ($DATASET)"
    # shellcheck disable=SC2086  # intentional word-splitting of the arg string
    "$VENV/bin/python" main.py $args
  done
  echo "queue finished; run harness/ordering_check.py on logs/ for the verdict"
}

case "$MODE" in
  --dry-run) dry_run ;;
  --execute) execute ;;
  -h|--help|"") echo "usage: run_matrix.sh <dataset> [--dry-run|--execute]"; echo "datasets: lastfm, ml-1m" ;;
  *) echo "unknown option: $MODE" >&2; exit 2 ;;
esac
