# Adversarial Review & Corrected Instruction Pack — NT-SSM Reproduction (ICML 2026 Challenge)

## TL;DR
- The pack's single biggest liability is that its NT-SSM run commands are **guesses**: the repository exists (geon0325/NT-SSM, Python, last updated 3 July 2026) but its files could not be read, so entry point, loss selection and `.cuda()` handling must be verified on clone — and hardcoded `.cuda()` is near-certain because the base framework SELFRec uses bare `.cuda()` calls with no device fallback.
- Several "facts" the pack states as settled — the 2/1/0 scoring rubric, the automated logbook judge, the guide-dataset URL and any Trackio naming convention — are **not present in the public Space files** and must be copied live from the guide/modal; by contrast the Trackio API and the LatentTSF fallback are verified and only lightly corrected.
- On Apple Silicon the right call is **CPU** for LightGCN's sparse propagation (not MPS); only **LastFM** is a realistic full reproduction, **ML-1M** is a toy-scale target, and **Yelp/Amazon-Book should be skipped** — and the time gates must be rewritten as total wall-clock budgets, because a per-epoch ">2 min → skip" gate plus a "2 h max" cap are mutually inconsistent once early stopping runs for hundreds of epochs.

## Key Findings

### CRITICAL
1. **Unverified run commands.** geon0325/NT-SSM could not be fetched (raw and API endpoints blocked); every command guessed from SELFRec conventions and the LightGCNpp `run.sh` pattern must be treated as *must-verify-on-clone*, not fact.
2. **`.cuda()` hardcoding almost certain.** The base framework's graph models contain bare `.cuda()` calls; NT-SSM inherits this and will crash on a CUDA-less Mac unless patched.
3. **Scoring rubric and logbook judge unverified.** The public Space files contain no 2/1/0 points scheme and no description of an automated judge; the README explicitly says the leaderboard is only "a starting point" and that winners are confirmed by organisers reviewing the actual logbooks.

### MAJOR
4. **The guide dataset is a genuine placeholder.** The official agent prompt is injected into an empty `<pre>` element by `repro.js` at runtime; the guide URL it curls is not in the static files and must be copied from the live modal.
5. **Time gates are incoherent.** A per-epoch threshold and a per-run wall-clock cap contradict each other; replace with per-dataset total wall-clock budgets.
6. **Tolerance too tight and mis-framed.** ±5% relative ignores the paper's five-run mean ± std; ordering (NT-SSM > SSM) should be the primary criterion.

### MINOR
7. Missing failure modes: dataset-not-bundled handling, an explicit evaluation-protocol check, epoch/patience specification, commit-hash pinning, a token-scope security note, and clarification of whether the logbook Space sits in the user's or the org's namespace.

## Details

### 1. Repository assumptions (CRITICAL — must-verify-on-clone)
The repository **geon0325/NT-SSM exists** — it is listed under the GitHub `icml2026` topic as the official code for the paper, is written in Python, and was last updated **3 July 2026** with a small number of stars. Its existence is also asserted by the paper itself, which states verbatim: *"Code and datasets are available at https://github.com/geon0325/NT-SSM."* The paper is **"Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy"** by **Geon Lee, Sunwoo Kim, Kyungho Kim and Kijung Shin (all KAIST, Seoul)**, published in PMLR vol. 306, ICML 2026 (arXiv:2605.24015v1 [cs.IR], 20 May 2026; correspondence geonlee0325@kaist.ac.kr, kijungs@kaist.ac.kr).

However, neither the lead nor the sub-agent could read the repository's file contents — the raw GitHub and API endpoints were blocked. Therefore **every run-level detail is unverified** and the pack's original commands (guessed from `python main.py` + `conf/*.yaml` and LightGCNpp's `./run.sh [DATASET]`) are provisional.

What is confirmed from the paper and the framework NT-SSM is built on:
- NT-SSM is implemented on **SELFRec** (Coder-Yu). SELFRec's README states verbatim: *"Configure the xx.yaml file in ./conf. (xx is the name of the model you want to run) Run main.py and choose the model you want to run."* — i.e. the entry point is `python main.py` with an **interactive model-selection menu**, and per-model configuration lives in `conf/*.yaml`. NCL and SimGCL both appear in that menu (SELFRec's "Self-Supervised Graph-Based Models" list: SGL, SimGCL, SEPT, MHCN, BUIR, SelfCF, SSL4Rec, XSimGCL, NCL, MixGCF).
- The sibling repo geon0325/LightGCNpp ships a SELFRec variant driven by `./run.sh [DATASET]` (e.g. `./run.sh lastfm`) and bundles datasets in a `data/` folder. NT-SSM plausibly follows the same convention, but this is unconfirmed.
- BPR, NT-BPR, SSM and NT-SSM are distinct optimisation objectives applied across LightGCN, SimGCL and NCL backbones; the selection mechanism (YAML field vs CLI flag vs menu) is unverified.
- **`.cuda()` risk (confirmed pattern):** SELFRec's graph models contain hardcoded, device-less `.cuda()` — e.g. in `model/graph/SimGCL.py` and `XSimGCL.py`: `self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()`. SELFRec's README also notes it is "compatible with Python 3.9+" and that models "run on GPUs". This is the single most likely blocker on Apple Silicon.
- Seed handling and the exact Table 1 reproduction script are unverified; the paper averages metrics over five runs.

**Fix embedded in the pack:** clone first, pin the commit, then read README.md / requirements.txt / main.py / `conf/` / any `run.sh`, and `grep -rn "cuda"` before running anything. All command snippets are flagged provisional.

### 2. Challenge mechanics
Confirmed from the Space README and index.html:
- Challenge runs **Wednesday 15 July → Sunday 2 August 2026**.
- The first **750** participants who join the org and request credit before **31 July 2026** receive **$20 in HF GPU credit**; awards total **$4,000** ($2,000 / $1,000 / two × $500).
- Join-org URL: `https://huggingface.co/organizations/ICML-2026-agent-repro/share/arHUbfnWoYUJXjwdpzKgfjifqnpFoffnSf`
- Request-credit URL: `https://icml-2026-agent-repro-collab-api.hf.space/credit`
- API base: `https://icml-2026-agent-repro-collab-api.hf.space`
- The modal's Step 4 says verbatim: *"Paste this into your coding agent. It curls the full challenge guide from the Hub dataset."* The prompt itself lives in an empty `<pre id="paste-prompt">` populated by `repro.js?v=26` at runtime — it is **not** in the static files. There are two harness tabs ("OpenResearch" and "Your Own Agent (Claude Code, Codex, etc.)"), each with its own dynamically-injected setup snippet.

**Unverified — must be copied live from the modal/guide:** the 2/1/0 scoring rubric, the automated logbook judge, the exact guide-dataset URL, and any Trackio project/run naming convention. The README does not confirm any points scheme; it states the leaderboard "is a starting point, and final placements are confirmed by our team reviewing the actual logbooks, not by leaderboard points alone," and that "everyone with at least one verified logbook receives a certificate of participation."

### 3. Trackio API (verified)
Current Trackio (requires **Python ≥3.10**) confirms every call the pack uses:
- `trackio.init(project=, name=, config=, space_id=, group=, webhook_url=, webhook_min_level=, ...)`. `space_id` may be `"username/repo"`, `"org/repo"`, or just `"repo"` (created in the logged-in user's namespace); if a `space_id` is given the metrics persist in an auto-created HF Dataset/Bucket.
- `trackio.log({...})` and `trackio.finish()` — confirmed, API-compatible with `wandb.init/log/finish`.
- `trackio.alert(title=, text=, level=trackio.AlertLevel.WARN)` — confirmed; levels are INFO / WARN / ERROR. **Correction:** the message argument is `text=`, not `message=`.
- CLI: `trackio list`, `trackio get`, and `trackio query project --project <name> --sql "SELECT ..."` all exist; append `--json` for machine-readable output (alerts: `trackio list alerts --project <name> --json`).
- For remote/ephemeral compute always pass `space_id`, since local SQLite is lost when the instance terminates. (On this task the run is local, so `space_id` is what publishes the logbook.)

### 4. Apple Silicon technical soundness
- `torch.sparse.mm` and related sparse ops are **not fully implemented on MPS**; sparse CSR support is beta and gaps persist through the 2.5–2.8 era (open PyTorch issues confirm missing sparse `addmv`/matmul paths on Apple Silicon).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` routes unsupported ops to CPU, but for a LightGCN whose core is one large sparse propagation this causes constant CPU↔MPS tensor ping-pong that is **slower** than staying on CPU.
- MPS does not support float64.
- **Verdict:** run NT-SSM on **CPU**, not MPS. 64 GB of unified memory is ample for these graphs. MPS is only worth attempting for a dense-only backbone.
- Paper per-epoch times on an RTX 3090 Ti: LastFM 0.47–0.73 s, ML-1M 6.16–8.70 s, Yelp 16.81–22.68 s, Amazon-Book 39.91–50.05 s. On CPU expect roughly one to two orders of magnitude slower. LastFM stays well under a minute/epoch; ML-1M becomes tens of seconds to a couple of minutes; Yelp and Amazon-Book become impractical. This tracks the paper's own scaling observation (§4.1): *"on Amazon-Book, the number of neighbor pairs involved in computing a prediction score reaches billions on average at three hops, while on MovieLens, more than 90% of all possible neighbor pairs are already included."*

### 5. Tolerance policy
The paper reports **mean ± std over five runs** for NDCG@20 and Recall@20. A single-seed ±5% band is both too tight (single-seed variance on LastFM/ML-1M can exceed 5%) and epistemically wrong (it discards the reported std). **Better policy:** treat a metric as reproduced if the single-seed value falls within `paper_mean ± max(5% of mean, 3 × reported_std)`, and make **ordering** the primary criterion — NT-SSM > SSM and NT-BPR > BPR on the same backbone/dataset. Average 3 seeds where time allows.

### 6. Logic/process gaps
The improved pack adds: what to do if datasets are not bundled (download via SELFRec's links and verify interaction counts); an explicit check that the split is 7:1:2 and evaluation is Recall@20 / NDCG@20 under the all-ranking protocol (both paper-confirmed) and match the code; reading epoch count and early-stopping patience from `conf`; pinning the commit hash on clone; an overall time-box; using a scoped write token and revoking it afterwards; and clarifying that the logbook Space defaults to the user's own namespace unless the guide mandates the org namespace.

### 7. Fallback — LatentTSF (verified and corrected)
Confirmed from the repo README (Muyiiiii/LatentTSF, "From Observations to States: Latent Time Series Forecasting", arXiv:2602.00297; authors Jie Yang, Yifan Hu, Yuante Li, Kexin Zhang, Kaize Ding, Philip S. Yu):
- `python download_datasets.py` pulls ETTh1/2, ETTm1/2, weather, electricity, traffic, exchange_rate and solar into `./dataset/`. Mainland-China mirror: `export HF_ENDPOINT=https://hf-mirror.com` first.
- `bash run_train.sh` runs a **single** example: DLinear in latent mode on ETTh1, `pred_len=96` (not a full-table sweep — the README says so explicitly).
- Flags confirmed: `--use_latent`; `--label_len 0` (required in latent mode); `--result_csv results.csv` to log MSE/MAE per run; `--autoencoder_path` pointing at a shipped checkpoint, e.g. `checkpoints/AutoEncoder_MLP_MAE_ETTh1_AE_ETTh1_ftM_sl24_dm32_dff64_lradj0_Exp-sl24-lr0.0005-500-32bs_0/checkpoint.pth`.
- `torch` is intentionally unpinned (install a CPU/MPS build on Mac); `requirements.txt` mirrors THUML's Time-Series-Library. Note the CC BY-NC 4.0 (non-commercial) files: `utils/losses.py`, `utils/m4_summary.py`, `data_provider/m4.py`.

## Recommendations
Staged, with the benchmarks that would change each step:

1. **Clone and verify NT-SSM before any run** (Phase 0). Do not trust guessed commands. *Change trigger:* if the README documents a different entry point or a CLI loss flag, update §3/§5 accordingly.
2. **Run on CPU; patch every `.cuda()` to a device variable.** *Change trigger:* only revert to MPS if the backbone is confirmed dense-only (no `torch.sparse.mm`).
3. **Target LastFM for full reproduction, ML-1M for toy-scale, skip Yelp/Amazon-Book.** *Change trigger:* if a LastFM run completes well under its budget, attempt ML-1M full-scale before declaring it toy-only.
4. **Copy the official prompt, guide URL and scoring rules live from the modal** into Placeholders A/B before starting.
5. **Apply the ordering-first tolerance policy;** declare success on ordering even if absolute numbers drift within the widened band.
6. **Publish by setting `space_id`** and confirm the logbook appears on the leaderboard; if not, re-check the guide's naming/namespace requirement.

## Caveats
NT-SSM's internal files (exact entry point, loss-selection mechanism, `conf/` field names, dataset bundling, `requirements.txt` pins, seed, Table-1 script, commit hash) remain **unread** — all are flagged must-verify-on-clone. The challenge scoring rubric, the logbook judge, the guide-dataset URL and any Trackio naming convention are **not in the public Space files** and are left as marked placeholders to be filled from the live modal. The CPU-vs-GPU slowdown factor is an estimate; measure the real LastFM per-epoch time before committing to ML-1M.

---
