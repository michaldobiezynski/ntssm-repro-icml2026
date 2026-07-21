# ntssm-repro-icml2026

Reproduction of **Table 1** from *"Rethinking Contrastive Learning in Graph Collaborative
Filtering: Limitations and a Simple Remedy"* (NT-SSM; Lee, Kim, Kim, Shin, KAIST;
[arXiv:2605.24015](https://arxiv.org/abs/2605.24015)) on Apple Silicon (CPU, no CUDA), for
the ICML 2026 agent reproducibility challenge.

Upstream code: [`geon0325/NT-SSM`](https://github.com/geon0325/NT-SSM) (built on
[`Coder-Yu/SELFRec`](https://github.com/Coder-Yu/SELFRec)), pinned at commit
`da8655e` (`PINNED_COMMIT_NTSSM.txt`).

## Scope

- **Local reproduction only** at this stage: no HuggingFace challenge submission, no
  `space_id`, no HF token. Trackio logs to a local project. The challenge pre-flight and
  Placeholders A/B in `CLAUDE.md` Section 1 are deliberately unfilled.
- **Target:** LastFM full reproduction, LightGCN backbone, the 2x2 objective matrix
  (BPR / NT-BPR / SSM / NT-SSM). ML-1M is a stretch goal; Yelp and Amazon-Book are skipped
  as infeasible on CPU.
- **Success criterion:** ordering (NT-SSM > SSM and NT-BPR > BPR), per `CLAUDE.md` Section 7.

## Layout

| Path | What |
| --- | --- |
| `CLAUDE.md` | Working instruction pack. Verified-facts banner on top, original pack below. |
| `docs/phase0-verification.md` | Verify-on-clone results: the source of truth for run commands and gotchas. |
| `docs/adversarial-review.md` | Provenance: the adversarial review that produced the pack. |
| `PINNED_COMMIT_NTSSM.txt` | The upstream commit this reproduction pins. |
| `harness/` | Reproduction harness (device patch, CPU run scripts, ordering checker, Trackio wrapper). Added by the feature branch. |

The upstream `NT-SSM/` clone, the Python `.venv/`, datasets, and run outputs are
git-ignored: this repo holds the reproduction *harness and record*, not vendored code or
data.

## Getting started

Read `docs/phase0-verification.md` first (verified commands), then `CLAUDE.md`.

```bash
# 1. one-off environment: venv (python3.13), deps, pinned + patched NT-SSM clone
bash harness/setup_env.sh
bash harness/setup_env.sh --check         # verify readiness without installing

# 2. preview the LastFM pipeline (no training)
bash harness/run_lastfm.sh --dry-run

# 3. run the harness tests (no ML stack needed)
.venv/bin/python -m pytest

# 4. OWNER-TRIGGERED: train the 2x2 matrix on CPU (~45 min budget)
bash harness/run_lastfm.sh --execute

# 5. reproduction verdict from the four run logs
.venv/bin/python harness/ordering_check.py \
  --logs NT-SSM=NT-SSM/logs/<run>.txt SSM=... NT-BPR=... BPR=... \
  --reference harness/paper_reference.json
```

Multi-hour training is triggered by the owner (step 4), never by an unattended build.

## Status

**LastFM / LightGCN reproduced on CPU (both ordering claims hold).** Running the harness
end-to-end confirmed the paper's Table 1 claims on Apple-Silicon CPU:

| Objective | ndcg@20 (ours / paper) | recall@20 (ours / paper) |
| --- | --- | --- |
| NT-SSM | 0.271 / 0.271 | 0.295 / 0.295 |
| NT-BPR | 0.266 / 0.265 | 0.288 / 0.290 |
| BPR | 0.247 / 0.253 | 0.267 / 0.276 |
| SSM | 0.211 / 0.240 | 0.234 / 0.262 |

Ordering: **NT-SSM > SSM** (+0.060 ndcg) and **NT-BPR > BPR** (+0.019) both hold. 3/4
objectives match the paper within tolerance; plain SSM early-stops low (noted in the
logbook).

**ML-1M / LightGCN reproduces the same way** (`run_matrix.sh ml-1m`, ~36 s/epoch):

| Objective | ndcg@20 (ours / paper) | recall@20 (ours / paper) |
| --- | --- | --- |
| NT-SSM | 0.322 / 0.322 | 0.254 / 0.254 |
| NT-BPR | 0.315 / 0.315 | 0.248 / 0.249 |
| BPR | 0.296 / 0.295 | 0.233 / 0.233 |
| SSM | 0.202 / 0.265 | 0.182 / 0.213 |

Both ML-1M ordering claims hold (NT-SSM > SSM +0.119, NT-BPR > BPR +0.019); NT-SSM/NT-BPR/BPR
within tolerance, SSM low (the same reproducible early-stopping pattern as LastFM).

Harness: `setup_env.sh`, three reproducible clone patches (`.cuda()`→CPU, `exec/eval`→
`importlib`, injected Trackio metric logging), `run_matrix.sh`/`run_lastfm.sh`,
`ordering_check.py`, `build_logbook.sh`, and a local-only Trackio wrapper, backed by **71 unit tests**.
Three bugs were found *by running the reproduction* (device, import scoping, NT-BPR alphas).
Training and publishing remain owner-triggered.
