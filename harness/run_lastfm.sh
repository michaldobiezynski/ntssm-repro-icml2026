#!/usr/bin/env bash
# Reproduce the LastFM / LightGCN 2x2 objective matrix (BPR, NT-BPR, SSM, NT-SSM).
# Default is safe: nothing trains unless --execute is passed explicitly.
set -euo pipefail

SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-200}"
# LastFM neighbour-type weights, from the paper's Appendix Table 5 (LightGCN / LastFM).
# NT-BPR and NT-SSM have DIFFERENT optimal alphas per Table 5 ("Optimal hyperparameter
# values of NT-BPR and NT-SSM") -- do not share them. Only these vary per dataset;
# tau/lr/reg/layers/epochs are fixed. Override via SSM_ALPHA_* / BPR_ALPHA_* env vars.
SA_UU="${SSM_ALPHA_UU:-1.2}"; SA_II="${SSM_ALPHA_II:-0.8}"; SA_UI="${SSM_ALPHA_UI:-0.8}"; SA_IU="${SSM_ALPHA_IU:-0.9}"
BA_UU="${BPR_ALPHA_UU:-1.3}"; BA_II="${BPR_ALPHA_II:-1.5}"; BA_UI="${BPR_ALPHA_UI:-0.9}"; BA_IU="${BPR_ALPHA_IU:-1.3}"

# Validate owner-settable numeric overrides. These are interpolated unquoted into the
# training arg string, so a non-numeric value could word-split or glob-expand; reject it.
for _pair in "SEED=$SEED" "EPOCHS=$EPOCHS" \
             "SSM_ALPHA_UU=$SA_UU" "SSM_ALPHA_II=$SA_II" "SSM_ALPHA_UI=$SA_UI" "SSM_ALPHA_IU=$SA_IU" \
             "BPR_ALPHA_UU=$BA_UU" "BPR_ALPHA_II=$BA_II" "BPR_ALPHA_UI=$BA_UI" "BPR_ALPHA_IU=$BA_IU"; do
  if ! [[ "${_pair#*=}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "error: ${_pair%%=*} must be numeric, got '${_pair#*=}'" >&2; exit 2
  fi
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CLONE="${NTSSM_DIR:-$REPO/NT-SSM}"
VENV="$REPO/.venv"

COMMON="--dataset lastfm --model_type graph --item_ranking 10,20,40 --embedding_size 64 --epoch $EPOCHS --batch_size 2048 --learning_rate 0.001 --reg_lambda 0.0001 --n_layer 2 --seed $SEED"
NT_SSM_ALPHAS="--alpha_uu $SA_UU --alpha_ii $SA_II --alpha_ui $SA_UI --alpha_iu $SA_IU"
NT_BPR_ALPHAS="--alpha_uu $BA_UU --alpha_ii $BA_II --alpha_ui $BA_UI --alpha_iu $BA_IU"

# Per objective: the arguments after `main.py`.
args_ntssm="--model_name LightGCN_NT $COMMON --loss_type ssm --tau 0.2 $NT_SSM_ALPHAS"
args_ssm="--model_name LightGCN $COMMON --loss_type ssm --tau 0.2"
args_ntbpr="--model_name LightGCN_NT $COMMON --loss_type bpr $NT_BPR_ALPHAS"
args_bpr="--model_name LightGCN $COMMON --loss_type bpr"

usage() {
  cat <<EOF
usage: run_lastfm.sh [--dry-run | --execute]

Reproduce the LastFM / LightGCN 2x2 objective matrix (BPR, NT-BPR, SSM, NT-SSM).

  --dry-run    print the four training commands and exit 0 (no training)
  --execute    actually run training (owner-triggered; needs .venv + patched ./NT-SSM)
  -h, --help   show this help

Env overrides: SEED (2026), EPOCHS (200),
               SSM_ALPHA_UU/II/UI/IU (1.2/0.8/0.8/0.9), BPR_ALPHA_UU/II/UI/IU (1.3/1.5/0.9/1.3),
               NTSSM_DIR (<repo>/NT-SSM), MAX_MINUTES (45).
Budget (CLAUDE.md Section 5): MAX_MINUTES gates the START of each objective; an
already-running objective is not interrupted (LastFM early-stops, so runs stay short).
EOF
}

print_matrix() {
  echo "# NT-SSM  (neighbour-type-aware sampled softmax)"
  echo "python main.py $args_ntssm"; echo
  echo "# SSM     (standard sampled softmax)"
  echo "python main.py $args_ssm"; echo
  echo "# NT-BPR  (neighbour-type-aware BPR)"
  echo "python main.py $args_ntbpr"; echo
  echo "# BPR     (standard BPR)"
  echo "python main.py $args_bpr"; echo
}

dry_run() {
  echo "=== DRY RUN: planned LastFM pipeline (no training executed) ==="
  echo "seed=$SEED epochs=$EPOCHS  NT-SSM alphas=$SA_UU/$SA_II/$SA_UI/$SA_IU  NT-BPR alphas=$BA_UU/$BA_II/$BA_UI/$BA_IU"
  echo
  print_matrix
  echo "# then compute the reproduction verdict from the four logs:"
  echo "python $HERE/ordering_check.py --logs \\"
  echo "  NT-SSM=<log> SSM=<log> NT-BPR=<log> BPR=<log> \\"
  echo "  --reference $HERE/paper_reference.json"
  echo "=== DRY RUN complete ==="
}

execute() {
  [ -x "$VENV/bin/python" ] || { echo "error: no .venv at $VENV; run harness/setup_env.sh first" >&2; exit 2; }
  [ -d "$CLONE" ] || { echo "error: no clone at $CLONE; run harness/setup_env.sh first" >&2; exit 2; }
  if ! grep -rq "_NTSSM_DEVICE" "$CLONE/model/graph/LightGCN.py" 2>/dev/null; then
    echo "error: clone not device-patched; run harness/setup_env.sh" >&2
    exit 2
  fi
  if ! grep -q "# trackio: run init" "$CLONE/main.py" 2>/dev/null; then
    echo "error: clone missing the trackio metric patch (dashboards would be empty); run harness/setup_env.sh" >&2
    exit 2
  fi
  local max_min="${MAX_MINUTES:-45}"
  echo "Executing LastFM matrix on CPU (budget ${max_min} min; CLAUDE.md Section 5)."
  cd "$CLONE"
  local start=$SECONDS
  for pair in "NT-SSM=$args_ntssm" "SSM=$args_ssm" "NT-BPR=$args_ntbpr" "BPR=$args_bpr"; do
    local label="${pair%%=*}" args="${pair#*=}"
    # Budget gate is checked BEFORE starting each objective, so it caps how many
    # objectives start, not the runtime of one already in flight. macOS lacks GNU
    # `timeout`, so a hard per-run cap is intentionally not imposed here.
    local elapsed_min=$(( (SECONDS - start) / 60 ))
    if [ "$elapsed_min" -ge "$max_min" ]; then
      echo "budget ${max_min}m reached before $label; stopping (CLAUDE.md Section 9)" >&2
      break
    fi
    echo ">>> training $label"
    # shellcheck disable=SC2086  # intentional word-splitting of the arg string
    "$VENV/bin/python" main.py $args
  done
  echo "training queue finished; run harness/ordering_check.py on logs/ for the verdict"
}

case "${1:-}" in
  --dry-run) dry_run ;;
  --execute) execute ;;
  -h|--help) usage ;;
  "") usage ;;
  *) echo "unknown option: $1" >&2; usage; exit 2 ;;
esac
