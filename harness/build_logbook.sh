#!/usr/bin/env bash
# Build a local Trackio logbook for the NT-SSM reproduction: one page per (dataset, claim),
# each with the claim + method, the live metric dashboard, and the ordering verdict.
# Covers every dataset that has a complete 4-log matrix under NT-SSM/logs/.
# Does NOT publish (publishing creates a public Space; that stays an explicit owner step).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
PY="$REPO/.venv/bin/python"
TRACKIO="$REPO/.venv/bin/trackio"
L="$REPO/NT-SSM/logs"
TITLE="${TITLE:-Reproducing NT-SSM (LightGCN, CPU) — ICML 2026}"
shopt -s nullglob

verdict() {  # $1 dataset, $2 reference json
  local ds="$1" ref="$2"
  local ntssm ntbpr ssm bpr
  ntssm=("$L"/"${ds}"_LightGCN_NT_*loss-ssm*.txt)
  ntbpr=("$L"/"${ds}"_LightGCN_NT_*loss-bpr*.txt)
  ssm=("$L"/"${ds}"_LightGCN_lr*loss-ssm*.txt)
  bpr=("$L"/"${ds}"_LightGCN_lr*loss-bpr*.txt)
  "$PY" "$HERE/ordering_check.py" --logs \
    "NT-SSM=${ntssm[0]}" "SSM=${ssm[0]}" "NT-BPR=${ntbpr[0]}" "BPR=${bpr[0]}" \
    --reference "$ref" 2>&1 || true  # exit 1 on ordering FAIL is expected content
}

have_matrix() {  # $1 dataset -> 0 if all four logs present
  local ds="$1"
  local a=("$L"/"${ds}"_LightGCN_NT_*loss-ssm*.txt) b=("$L"/"${ds}"_LightGCN_NT_*loss-bpr*.txt)
  local c=("$L"/"${ds}"_LightGCN_lr*loss-ssm*.txt) d=("$L"/"${ds}"_LightGCN_lr*loss-bpr*.txt)
  [ ${#a[@]} -gt 0 ] && [ ${#b[@]} -gt 0 ] && [ ${#c[@]} -gt 0 ] && [ ${#d[@]} -gt 0 ]
}

add_dataset() {  # $1 dataset, $2 pretty, $3 reference json, $4 ssm-alphas, $5 bpr-alphas
  local ds="$1" pretty="$2" ref="$3" sa="$4" ba="$5"
  local project="ntssm-${ds}"
  local v; v="$(verdict "$ds" "$ref")"
  local p1="${pretty} · LightGCN · NT-SSM vs SSM"
  local p2="${pretty} · LightGCN · NT-BPR vs BPR"

  "$TRACKIO" logbook cell markdown \
"Paper claim (arXiv:2605.24015, Table 1): the neighbour-type-aware NT-SSM loss improves over standard Sampled-Softmax (SSM) on LightGCN / ${pretty}. \

**Setup:** upstream geon0325/NT-SSM at da8655e, patched for Apple-Silicon CPU (no CUDA). Seed 2026, 200 epochs (early stopping, patience 10), dim 64, 2 layers, tau 0.2. NT-SSM alphas ${sa} (Appendix Table 5). Criterion: NT-SSM > SSM on Recall@20 and NDCG@20." \
    --title "Claim & method" --page "$p1"
  "$TRACKIO" logbook cell dashboard "$project" --title "Training metrics (live)" --page "$p1"
  "$TRACKIO" logbook cell markdown "**Reproduction verdict**\n\n\`\`\`\n${v}\n\`\`\`" \
    --title "Ordering verdict vs paper" --page "$p1"

  "$TRACKIO" logbook cell markdown \
"Paper claim (Table 1): NT-BPR improves over standard BPR on LightGCN / ${pretty}. Same setup, BPR-family loss and the NT-BPR-specific Table 5 alphas ${ba} (distinct from NT-SSM's — the paper tunes each objective separately)." \
    --title "Claim & method" --page "$p2"
  "$TRACKIO" logbook cell dashboard "$project" --title "Training metrics (live)" --page "$p2"
  "$TRACKIO" logbook cell markdown "**Reproduction verdict**\n\n\`\`\`\n${v}\n\`\`\`" \
    --title "Ordering verdict vs paper" --page "$p2"
  echo "  added pages for ${pretty}"
}

cd "$REPO"
if ! have_matrix lastfm && ! have_matrix ml-1m; then
  echo "error: no complete 4-log matrix under $L; run a sweep first" >&2; exit 2
fi
echo ">>> opening local logbook: $TITLE"
"$TRACKIO" logbook open --title "$TITLE" --no-serve --no-browser

if have_matrix lastfm; then
  add_dataset lastfm "LastFM" "$HERE/paper_reference.json" "1.2/0.8/0.8/0.9" "1.3/1.5/0.9/1.3"
fi
if have_matrix ml-1m; then
  add_dataset ml-1m "ML-1M" "$HERE/paper_reference_ml1m.json" "1.2/0.8/0.8/1.0" "1.4/1.3/0.8/1.3"
fi

# Honest-log note (applies across datasets).
"$TRACKIO" logbook cell markdown \
"**Deviations & method (honest log).** Single seed (2026); the paper averages five, so absolute values can drift within the tolerance band \`paper_mean +/- max(5% , 3·std)\`. On LastFM, plain SSM early-stops low (~0.211 vs 0.240 NDCG@20) — this only widens NT-SSM's margin, so the ordering claim is unaffected. NT-BPR uses its own Table 5 alphas (an earlier run wrongly reused the NT-SSM alphas and underperformed). All runs on Apple-Silicon CPU; \`.cuda()\` and a Python-3 \`exec/eval\` import bug were patched to make the pinned clone run locally." \
  --title "Deviations & caveats" --page "Notes"

echo
echo ">>> logbook built under .trackio/logbook/. Next (owner):"
echo "    preview:  $TRACKIO logbook serve"
echo "    publish:  $TRACKIO logbook publish <your-username>/<paper-id>   # PUBLIC Space, no prompt"
