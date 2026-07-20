# NT-SSM Reproduction: Working CLAUDE.md

> **Reproduction status (updated 2026-07-20, post-clone).** The instruction pack below
> was written before `geon0325/NT-SSM` could be read, so its run commands were flagged
> "must-verify-on-clone". The repo has since been cloned and verified at commit `da8655e`
> (see `PINNED_COMMIT_NTSSM.txt`). Verified facts and corrections live in
> **`docs/phase0-verification.md`**: read that first; it overrides any conflicting guess
> in the pack. The pack is preserved verbatim below as the spec of record.

## Verified quick-reference (supersedes the pack's provisional §3 commands)

- **Entry point:** `python main.py` with CLI args (no interactive menu; `run.sh` is a thin wrapper). There is **no `conf/` dir**: all configuration is argparse in `main.py`.
- **Objective matrix (LightGCN backbone):**

  | Objective | Command shape |
  | --- | --- |
  | SSM    | `--model_name LightGCN    --loss_type ssm` |
  | NT-SSM | `--model_name LightGCN_NT --loss_type ssm --alpha_uu/ii/ui/iu ...` |
  | BPR    | `--model_name LightGCN    --loss_type bpr` |
  | NT-BPR | `--model_name LightGCN_NT --loss_type bpr` |

  The `_NT` suffix selects the neighbour-type-aware variant; `--alpha_{uu,ii,ui,iu}` weight the four neighbour-pair types.
- **Datasets are bundled:** `dataset/{lastfm,ml-1m,yelp2018,amazon-book}/{train,valid,test}.txt` (7:1:2 pre-split). LastFM train = 64,315 interactions.
- **Device blocker:** 20 `.cuda()` sites; the LightGCN path needs 4 patched to a `DEVICE` constant. The core op is `torch.sparse.mm`, so run on **CPU** (MPS sparse is unreliable). `base/torch_interface.py:12` uses the deprecated `torch.sparse.FloatTensor`, which may need `torch.sparse_coo_tensor` on modern torch.
- **No `requirements.txt`** despite the README. Minimal deps: `torch numpy scipy numba` (plus `trackio` for logging). Skip `tensorflow` (SELFRec's unused TF branch) and `faiss` (NCL-only).
- **Defaults (main.py / run.sh):** seed 2026, epoch 200, batch 2048, lr 1e-3, reg 1e-4, dim 64, 2 layers, τ0.2, `--item_ranking 10,20,40`. Metrics are logged per-epoch to `logs/*.txt` (CSV: `epoch,split,,<6 metrics>`).
- **Python:** build the venv with `python3.13` (torch wheels reliable; 3.14 is risky).

## Active scope for this repo (owner decision, 2026-07-20)

- **Local reproduction only.** No HuggingFace challenge submission yet: no `space_id`, no HF write token, Trackio logs to a LOCAL project. Section 1 (HUMAN pre-flight) and Placeholders A/B stay unfilled and out of scope.
- **Target:** LastFM full reproduction, LightGCN backbone, the 2x2 objective matrix above. ML-1M is a stretch; Yelp/Amazon-Book are skipped.
- **Success = ordering** (NT-SSM > SSM and NT-BPR > BPR) per Section 7, with the widened tolerance band.
- **Do not launch multi-hour training in an unattended harness build.** Build and dry-run the harness; the owner triggers real training separately.

---

# IMPROVED CLAUDE.md — NT-SSM Reproduction Pack (self-contained, paste-ready)

## 0. Mission
Reproduce **Table 1** of *"Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy"* (NT-SSM; Geon Lee, Sunwoo Kim, Kyungho Kim, Kijung Shin, KAIST; arXiv:2605.24015; code geon0325/NT-SSM, built on Coder-Yu/SELFRec) for the ICML 2026 agent reproducibility challenge, on an Apple Silicon MacBook (64 GB unified memory, **no CUDA** — CPU/MPS/MLX only), and publish a Trackio logbook. Fallback paper: **LatentTSF** (Muyiiiii/LatentTSF, arXiv:2602.00297).

## 1. Pre-flight (HUMAN — do these yourself, once)
1. Join the org: `https://huggingface.co/organizations/ICML-2026-agent-repro/share/arHUbfnWoYUJXjwdpzKgfjifqnpFoffnSf`
2. Request credit **before 31 July 2026**: `https://icml-2026-agent-repro-collab-api.hf.space/credit` (first 750 participants → $20 HF GPU credit).
3. Create a **fine-grained *write* token** at huggingface.co/settings/tokens, then `hf auth login`. A write token can push to any repo in your namespace, so scope it to the logbook repo if possible and **revoke it after the challenge**.
4. Open the Space's "Add your agent" modal, choose the paper, select the "Your Own Agent" tab, and **copy two things verbatim** into the placeholders below.

```
<<<PLACEHOLDER A — OFFICIAL AGENT PROMPT
Paste the Step-4 prompt from the modal here verbatim. It curls the full challenge
guide from a Hub dataset; that guide is NOT in the Space's static files.>>>

<<<PLACEHOLDER B — OFFICIAL GUIDE CONTENT (curled by the prompt)
Paste the guide here verbatim, especially: (a) the scoring rubric — do NOT assume
2/1/0; (b) the logbook judging rule; (c) the mandated Trackio project/run naming;
(d) whether the logbook Space must live under the ICML-2026-agent-repro org or your
own namespace.>>>
```

## 2. Environment setup
```bash
python3 -m venv .venv && source .venv/bin/activate    # Python >=3.10 (required by trackio)
pip install --upgrade pip
pip install --upgrade trackio                          # per challenge README
git clone https://github.com/geon0325/NT-SSM && cd NT-SSM
git rev-parse HEAD > ../PINNED_COMMIT_NTSSM.txt        # pin commit for reproducibility
# CPU/MPS torch build — do NOT install a CUDA wheel:
pip install torch torchvision torchaudio
pip install -r requirements.txt                        # verify torch not re-pinned to CUDA
```

## 3. VERIFY-ON-CLONE (mandatory before any training run)
The commands below are **provisional** — the repo could not be inspected remotely. Confirm each:
- **Entry point:** expected either `python main.py` (interactive menu, per SELFRec) **or** `./run.sh [DATASET]` (per sibling repo LightGCNpp). Read `main.py` / any `run.sh`.
- **Loss/variant selection:** find how BPR / NT-BPR / SSM / NT-SSM are chosen (a `conf/<Model>.yaml` field, a CLI flag, or the menu).
- **Datasets:** check for a bundled `data/` or `dataset/` folder holding LastFM, MovieLens-1M, Yelp, Amazon-Book. If absent, download per SELFRec's dataset links and verify interaction counts.
- **Device code:** `grep -rn "cuda" .` — list every `.cuda()` / `device='cuda'` (SELFRec is known to use bare `.cuda()`).
- **Determinism/limits:** locate the seed, the max-epoch value, and the early-stopping patience in `conf`.
- **Protocol match:** confirm the split is **7:1:2** and evaluation is **Recall@20 / NDCG@20** under the all-ranking protocol (both stated in the paper §7.1 / §B).

## 4. MPS/CPU patch guidance
Run on **CPU**. LightGCN's core is a large sparse propagation (`torch.sparse.mm`), which is not reliably implemented on MPS and thrashes under `PYTORCH_ENABLE_MPS_FALLBACK`. Insert a single device constant and replace hardcoded CUDA:
```python
import torch
DEVICE = torch.device("cpu")   # sparse GCN ops unreliable/slow on MPS; 64GB RAM is ample
```
- Replace every `.cuda()` with `.to(DEVICE)`.
- Do **not** set `PYTORCH_ENABLE_MPS_FALLBACK=1` for the sparse path — CPU-only is faster than ping-ponging.
- Never cast to `float64` (unsupported on MPS).
- MPS is only worth trying if you confirm a dense-only backbone with no `torch.sparse.mm`.

## 5. Phased plan with consistent wall-clock budgets
Gate on **total wall-clock per dataset** (per-epoch × configured epochs, accounting for early stopping), never on a bare per-epoch threshold.

- **Phase 1 — LastFM — FULL reproduction target.** Budget **45 min total**. Train LightGCN with NT-SSM and SSM (and NT-BPR/BPR). RTX 3090 Ti reference: 0.47–0.73 s/epoch → seconds/epoch on CPU. This is the primary deliverable.
- **Phase 2 — ML-1M — TOY-SCALE target.** Budget **90 min total**. Reference 6.16–8.70 s/epoch. If projected run time > budget, cap max epochs, and label the logbook **"toy-scale (reduced epochs)"**.
- **Phase 3 — Yelp / Amazon-Book — SKIP.** Reference 16.81–22.68 s/epoch (Yelp) and 39.91–50.05 s/epoch (Amazon-Book) on GPU; infeasible on CPU (Amazon-Book reaches billions of neighbour pairs at 3 hops). Record the reason in the logbook rather than attempting.
- **Overall attempt time-box: 4 hours.**

## 6. Trackio logging (verified API)
```python
import trackio

trackio.init(
    project = <<<PLACEHOLDER: guide-mandated project name, else "icml2026-ntssm">>>,
    name    = "lastfm-lightgcn-ntssm-seed0",
    space_id= <<<PLACEHOLDER: guide-mandated namespace, else "your-username/icml-ntssm-logbook">>>,
    config  = {"dataset":"lastfm","backbone":"lightgcn","loss":"nt-ssm",
               "seed":0,"commit":open("../PINNED_COMMIT_NTSSM.txt").read().strip()},
)

for ep in range(max_epochs):
    # ... train one epoch, evaluate on validation ...
    trackio.log({"recall@20": r20, "ndcg@20": n20, "epoch": ep})
    if diverged:
        trackio.alert(title="Diverged",
                      text=f"loss NaN at epoch {ep}",
                      level=trackio.AlertLevel.ERROR)

trackio.finish()
```
- Note the alert argument is `text=`, not `message=`; levels are `INFO` / `WARN` / `ERROR`.
- Setting `space_id` is what **publishes** the logbook (it also persists metrics if the process dies).
- Inspect afterwards: `trackio query project --project <name> --sql "SELECT ..."` or `trackio list alerts --project <name> --json`.

## 7. Tolerance policy
- Metric counts as reproduced if the single-seed value lies within `paper_mean ± max(5% of mean, 3 × reported_std)` (the paper reports mean ± std over five runs — a bare single-seed ±5% is unjustified).
- **Primary success criterion is ordering:** NT-SSM > SSM and NT-BPR > BPR on the same backbone and dataset.
- Average **3 seeds** where the time budget allows; otherwise log the single seed explicitly.

## 8. Honest-logbook rules
- Log every command, seed, pinned commit hash, config diff and deviation.
- **Never** adjust numbers to match the paper. Report raw metrics.
- If you skip a dataset or run toy-scale, state it plainly in the run name and config.
- Log failures, `.cuda()` patches and missing datasets as Trackio alerts so the trace is self-explanatory.

## 9. Guardrails
- **No CUDA.** Assume CPU throughout; patch device code before running.
- Use the **scoped write token** from §1 and revoke it after.
- Enforce the **per-dataset wall-clock budgets** in §5 and the 4-hour overall box.
- **Stop and report** if: datasets are missing and cannot be downloaded; `.cuda()` cannot be patched cleanly; the entry point differs materially from §3; or LastFM ordering (NT-SSM > SSM) does not hold after 3 seeds.

## 10. Fallback — LatentTSF (verified)
Use only if NT-SSM is blocked (e.g. unpatchable device code or missing datasets).
```bash
git clone https://github.com/Muyiiiii/LatentTSF && cd LatentTSF
git rev-parse HEAD > ../PINNED_COMMIT_LATENTTSF.txt
pip install torch torchvision torchaudio          # CPU/MPS; upstream leaves torch unpinned
pip install -r requirements.txt                    # mirrors THUML Time-Series-Library
python download_datasets.py                        # -> ./dataset/ (ETTh1/2, ETTm1/2, weather,
                                                   #    electricity, traffic, exchange_rate, solar)
# China mirror if needed: export HF_ENDPOINT=https://hf-mirror.com  (before the line above)
bash run_train.sh                                  # single example: DLinear, latent mode, ETTh1, pred_len=96
```
- Latent mode requires `--use_latent --label_len 0`; log metrics with `--result_csv results.csv`; point `--autoencoder_path` at the matching shipped checkpoint, e.g.
  `checkpoints/AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0/checkpoint.pth`.
- `run_train.sh` is a **single** run, not the full table; extend its loops (9 datasets × backbones × horizons {96,192,336,720}) only if time allows.
- Light targets on this hardware: **ETTh1 / ETTm1** (7 variables). Avoid electricity/traffic (321/862 variables).
- Licence note: `utils/losses.py`, `utils/m4_summary.py`, `data_provider/m4.py` are **CC BY-NC 4.0** (non-commercial).