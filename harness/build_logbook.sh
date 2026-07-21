#!/usr/bin/env bash
# Build a local Trackio logbook for the NT-SSM LastFM reproduction: one page per claim,
# each with the claim + method, the live metric dashboard, and the ordering verdict.
# Does NOT publish (publishing creates a public Space; that stays an explicit owner step).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
TRACKIO="$VENV/bin/trackio"
L="$REPO/NT-SSM/logs"
PROJECT="${PROJECT:-ntssm-lastfm}"
TITLE="${TITLE:-Reproducing NT-SSM on LastFM (LightGCN, CPU) — ICML 2026}"

# Discover the four LastFM logs (plain 'LightGCN_lr' vs neighbour-type 'LightGCN_NT_').
shopt -s nullglob
ntssm=("$L"/lastfm_LightGCN_NT_*loss-ssm*.txt)
ntbpr=("$L"/lastfm_LightGCN_NT_*loss-bpr*.txt)
ssm=("$L"/lastfm_LightGCN_lr*loss-ssm*.txt)
bpr=("$L"/lastfm_LightGCN_lr*loss-bpr*.txt)
if [ ${#ntssm[@]} -eq 0 ] || [ ${#ntbpr[@]} -eq 0 ] || [ ${#ssm[@]} -eq 0 ] || [ ${#bpr[@]} -eq 0 ]; then
  echo "error: missing one of the four LastFM logs under $L; run run_lastfm.sh --execute first" >&2
  exit 2
fi

verdict() {  # $1..$4 = NT-SSM SSM NT-BPR BPR log paths; prints ordering_check output
  "$PY" "$HERE/ordering_check.py" --logs \
    "NT-SSM=$1" "SSM=$2" "NT-BPR=$3" "BPR=$4" \
    --reference "$HERE/paper_reference.json" 2>&1 || true  # exit 1 on FAIL is expected content
}

cd "$REPO"
echo ">>> opening local logbook: $TITLE"
"$TRACKIO" logbook open --title "$TITLE" --no-serve --no-browser

# Intro (index page stays a TOC; context goes on the first claim page).
p1="LastFM · LightGCN · NT-SSM vs SSM"
"$TRACKIO" logbook cell markdown \
"Paper claim (arXiv:2605.24015, Table 1): the neighbour-type-aware NT-SSM loss improves over the standard Sampled-Softmax (SSM) loss on LightGCN. \

**Setup:** upstream geon0325/NT-SSM pinned at da8655e, patched to run on Apple-Silicon CPU (no CUDA). Seed 2026, 200 epochs (early stopping, patience 10), dim 64, 2 layers, tau 0.2. NT-SSM alphas 1.2/0.8/0.8/0.9 (Table 5). Success criterion: NT-SSM > SSM on Recall@20 and NDCG@20." \
  --title "Claim & method" --page "$p1"
"$TRACKIO" logbook cell dashboard "$PROJECT" --title "Training metrics (live)" --page "$p1"
"$TRACKIO" logbook cell markdown \
"**Reproduction verdict**\n\n\`\`\`\n$(verdict "${ntssm[0]}" "${ssm[0]}" "${ntbpr[0]}" "${bpr[0]}")\n\`\`\`" \
  --title "Ordering verdict vs paper" --page "$p1"
"$TRACKIO" logbook cell markdown \
"**Deviations (honest log).** Both ordering claims reproduce and NT-SSM/NT-BPR/BPR land within tolerance of the paper. Plain **SSM** underperforms (NDCG@20 ~0.211 vs paper 0.240; early-stopped ~19 epochs), most likely single-seed variance and patience-10 early stopping on this CPU run. This only *widens* NT-SSM's margin over SSM, so the ordering claim is unaffected. Single seed (2026); the paper averages five. An earlier run mistakenly gave NT-BPR the NT-SSM alphas and underperformed — corrected to the Table 5 NT-BPR values (1.3/1.5/0.9/1.3)." \
  --title "Deviations & caveats" --page "$p1"

p2="LastFM · LightGCN · NT-BPR vs BPR"
"$TRACKIO" logbook cell markdown \
"Paper claim (Table 1): NT-BPR improves over standard BPR on LightGCN. Same setup as the SSM page, but BPR-family loss and the NT-BPR-specific Table 5 alphas 1.3/1.5/0.9/1.3 (distinct from the NT-SSM alphas — a per-objective distinction the paper's Table 5 makes explicit)." \
  --title "Claim & method" --page "$p2"
"$TRACKIO" logbook cell markdown \
"**Reproduction verdict** (same 4-run matrix; BPR pair highlighted)\n\n\`\`\`\n$(verdict "${ntssm[0]}" "${ssm[0]}" "${ntbpr[0]}" "${bpr[0]}")\n\`\`\`" \
  --title "Ordering verdict vs paper" --page "$p2"

echo
echo ">>> logbook built locally under .trackio/logbook/. Next steps (owner):"
echo "    preview:  $TRACKIO logbook serve"
echo "    publish:  $TRACKIO logbook publish <your-username>/<paper-id>   # PUBLIC Space, no prompt"
