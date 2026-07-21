# Phase 0: Verify-on-clone results

Cloned `geon0325/NT-SSM` at commit `da8655ef0b331e5fb2b6e5ceb05756b2f8c8aa05` (HEAD of `main`, 2026-07-20) and inspected the source. This resolves every "must-verify-on-clone" item in the instruction pack (`CLAUDE.md` Section 3) and records gotchas the pack did not anticipate. **Where this document conflicts with the pack, this document wins** (the pack's commands were written blind).

## Pack guess vs verified reality

| Item | Pack's guess | Verified reality (commit `da8655e`) |
| --- | --- | --- |
| Entry point | `python main.py` menu **or** `./run.sh [DATASET]` | `python main.py` with **CLI args**; no interactive menu. `run.sh` just sets vars and calls `main.py`. |
| Config mechanism | `conf/<Model>.yaml` | **No `conf/` dir.** Everything is argparse in `main.py`. |
| Loss / variant selection | unknown (yaml / flag / menu) | `--model_name LightGCN{,_NT}` picks backbone + NT variant; `--loss_type {bpr,ssm}` picks objective; `--alpha_{uu,ii,ui,iu}` weight neighbour-pair types. |
| Datasets bundled? | maybe not; may need download | **Yes.** `dataset/{lastfm,ml-1m,yelp2018,amazon-book}/{train,valid,test}.txt` ship in-repo (7:1:2 pre-split). |
| LastFM train size | unknown | 64,315 interactions (train.txt lines). Small: fast on CPU. |
| `.cuda()` sites | "near-certain, bare `.cuda()`" | **Confirmed: 20 sites** across `model/graph/*.py`. LightGCN path = 4. |
| Core sparse op | `torch.sparse.mm` predicted | **Confirmed** in every backbone. CPU is correct; MPS sparse is unreliable. |
| `requirements.txt` | `pip install -r requirements.txt` | **Missing.** README references it but no file exists. Deps must be synthesised. |
| Seed | unverified | `--seed` default **2026**. |
| Epochs / patience | "read from conf" | `--epoch` default **200**. Early stopping with **patience=10** on best validation NDCG@20 (`LightGCN.py:45` / `LightGCN_NT.py:49`). The `bestPerformance` path in `base/graph_recommender.py` is dead code for these models. |
| Eval protocol | Recall@20 / NDCG@20 | `--item_ranking 10,20,40`. Per-epoch metrics to `logs/*.txt`, CSV `epoch,split,'',R@10,N@10,R@20,N@20,R@40,N@40` (**Recall@20 = field 5, NDCG@20 = field 6**; no header; field 2 always empty; `round(x,5)` so trailing zeros are stripped). |

## Device patch surface (the one real blocker)

20 `.cuda()` calls; the LightGCN + LightGCN_NT path (our target) needs 4:

- `model/graph/LightGCN.py:42`: `model = self.model.cuda()`
- `model/graph/LightGCN.py:150`: `self.sparse_norm_adj = ...convert_sparse_mat_to_tensor(self.norm_adj).cuda()`
- `model/graph/LightGCN_NT.py:46`: `model = self.model.cuda()`
- `model/graph/LightGCN_NT.py:232`: `self.sparse_norm_adj = ...cuda()`

Full inventory also covers `SimGCL{,_NT}.py` (5 each, incl. `torch.rand_like(...).cuda()` and `.cuda()` on unique-index tensors) and `NCL{,_NT}.py` (2-4 each, incl. `centroids`/`node2cluster` from faiss clustering). `main.py:94` sets `os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu` but that is harmless on a CUDA-less box.

**Patch approach:** introduce one `DEVICE = torch.device("cpu")` constant and replace `.cuda()` -> `.to(DEVICE)`. Apply as a reproducible patch/script against the pinned clone (not hand edits to an untracked checkout).

**Second gotcha:** `base/torch_interface.py:12` builds the sparse adjacency with the **deprecated** `torch.sparse.FloatTensor(i, v, coo.shape)`. Modern torch may warn or error; the fix is `torch.sparse_coo_tensor(i, v, coo.shape)`. Verify at harness-build time.

## Dependencies (synthesise; there is no requirements.txt)

Third-party imports across the torch path: `torch` (23), `numpy` (8), `scipy` (4, sparse matrices), `numba` (1), `pickle`/`os`/`random`/`heapq`/`math`/`time` (stdlib).

- **Install for LightGCN + LastFM:** `torch numpy scipy numba` (+ `trackio` for logging).
- **Skip `tensorflow`** (2 imports): only `base/tf_interface.py` / `util/loss_tf.py`, SELFRec's unused TF branch. Not needed for torch graph models.
- **Skip `faiss`** (1 import): only `NCL.py` clustering. Not needed for LightGCN.
- **Numba wheel risk** on Python 3.13/3.14: verify `import numba` works in the venv early; it is used in the sampler path.

## Reproduction commands (verified, LastFM, LightGCN, CPU)

After the device patch, from inside the patched `NT-SSM/` clone (2x2 objective matrix, seed 2026):

```bash
# NT-SSM  (neighbour-type-aware sampled softmax)
python main.py --dataset lastfm --model_name LightGCN_NT --model_type graph \
  --loss_type ssm --tau 0.2 --epoch 200 --batch_size 2048 --learning_rate 0.001 \
  --reg_lambda 0.0001 --embedding_size 64 --n_layer 2 --item_ranking 10,20,40 \
  --alpha_uu 1.2 --alpha_ii 0.8 --alpha_ui 0.8 --alpha_iu 0.9 --seed 2026

# SSM     (standard sampled softmax; drop the _NT suffix)
python main.py --dataset lastfm --model_name LightGCN --model_type graph \
  --loss_type ssm --tau 0.2 --epoch 200 --batch_size 2048 --learning_rate 0.001 \
  --reg_lambda 0.0001 --embedding_size 64 --n_layer 2 --item_ranking 10,20,40 --seed 2026

# NT-BPR / BPR: same as above with --loss_type bpr on LightGCN_NT / LightGCN respectively.
```

Note: the LastFM `--alpha_*` values above come from the paper's Appendix Table 5 (LightGCN / LastFM / NT-SSM: uu1.2/ii0.8/ui0.8/iu0.9), not from `run.sh` (which targets ml-1m with iu1.0). Only these four weights vary per dataset; tau/lr/reg/layers/epochs are held fixed. NT-BPR uses a different Table 5 row (uu1.3/ii1.5/ui0.9/iu1.3), so `harness/run_lastfm.sh` keeps the two objectives' alphas separate: override via `SSM_ALPHA_UU/II/UI/IU` (NT-SSM) and `BPR_ALPHA_UU/II/UI/IU` (NT-BPR).

## Open items for the harness build

1. RESOLVED. Log column order is `epoch,split,'',R@10,N@10,R@20,N@20,R@40,N@40`; Recall@20 = field 5, NDCG@20 = field 6 (`LightGCN_NT.py` logging block). The harness parser derives field indices from `--item_ranking` rather than hardcoding them.
2. RESOLVED. Early stopping exists: patience 10 on best validation NDCG@20 (`LightGCN.py:45` / `LightGCN_NT.py:49`); the reported figures are the test row at the best-valid epoch. The `graph_recommender.bestPerformance` path is dead code for these backbones.
3. Verify `numba` and the deprecated `torch.sparse.FloatTensor` call in the target venv before any run (checked by `harness/setup_env.sh`).
