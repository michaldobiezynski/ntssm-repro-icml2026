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

Phase 0 (verify-on-clone) complete. Reproduction harness built and unit-tested (40 tests):
CUDA->CPU device patch, dry-runnable LastFM driver, ordering + tolerance verdict, and a
local-only Trackio wrapper. Training runs remain owner-triggered.
