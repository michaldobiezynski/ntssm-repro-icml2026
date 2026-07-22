"""Retention-ratio study reproduction (Section 4.2, Figures 2-3; registry claim 3).

Training-free heuristic from the paper: keep only the top-q% structurally most
similar neighbours (per-column threshold on S), replace the learnable pair weight
with the co-occurrence count omega (Eq. 3), score with Eq. 4, and evaluate
NDCG@20 over a (q, q') grid - aggregated and per neighbour-pair type (UU/II/UI/IU).

Implementation notes:
  - S = mean_{l=0..L} A_norm^l, dense float32 (datasets here are small enough).
  - Retained-neighbour matrix K_q[x, v] = 1 iff S_xv >= (k-th largest of column v),
    k = max(1, round(q% * n)), restricted to v in N_x (S_xv > 0).
  - omega_{q,q'} = K_q[users].T @ R @ K'_{q'}[items]  (Eq. 3 factorised).
  - Score matrix (Eq. 4) = S_U @ omega @ S_I.T, computed per type block by
    restricting v (columns/rows) to user- or item-type nodes; blocks sum to the
    aggregate score by the Eq. 2 partition (verified in audit_decomposition.py).
  - NDCG@20 on the test split, training positives masked.

Run:  .venv/bin/python harness/audit_retention.py --dataset ml-1m --out results.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_decomposition import load_graph  # noqa: E402


def build_s_dense(adj, layers):
    n = adj.shape[0]
    s = np.eye(n, dtype=np.float32)
    acc = np.eye(n, dtype=np.float32)
    for _ in range(layers):
        acc = adj.dot(acc.T).T.astype(np.float32)  # adj @ acc, keep float32
        s += acc
    return s / (layers + 1)


def retained_mask(s_mat, q_percent):
    """K[x, v] = 1 iff S_xv >= k-th largest of column v (and S_xv > 0)."""
    n = s_mat.shape[0]
    k = max(1, int(round(q_percent / 100.0 * n)))
    # k-th largest per column via partition on the negated matrix.
    thresh = -np.partition(-s_mat, k - 1, axis=0)[k - 1, :]
    return (s_mat >= thresh[None, :]) & (s_mat > 0)


def ndcg_at_k(scores, test_lists, k=20):
    """Mean NDCG@k over users with >=1 test item; scores already train-masked."""
    idx_top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    row_scores = np.take_along_axis(scores, idx_top, axis=1)
    order = np.argsort(-row_scores, axis=1)
    ranked = np.take_along_axis(idx_top, order, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    vals = []
    for u, truth in test_lists.items():
        if not truth:
            continue
        hits = np.isin(ranked[u], list(truth))
        dcg = float((hits * discounts).sum())
        idcg = float(discounts[: min(k, len(truth))].sum())
        vals.append(dcg / idcg)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m", choices=["lastfm", "ml-1m"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--grid", default="0.01,0.1,1,10,100")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    grid = [float(x) for x in args.grid.split(",")]

    data = load_graph(args.dataset)
    n_u, n_i = data.user_num, data.item_num
    n = n_u + n_i
    adj = sp.csr_matrix(data.norm_adj, dtype=np.float64)
    print(f"dataset={args.dataset} users={n_u} items={n_i} nodes={n} grid={grid}")

    t0 = time.time()
    s_mat = build_s_dense(adj, args.layers)
    s_users, s_items = s_mat[:n_u], s_mat[n_u:]
    r_mat = sp.csr_matrix(data.ui_adj, dtype=np.float32)[:n_u, n_u:].toarray()

    # Test ground truth + train mask, via the clone's own id mappings.
    test_lists = {}
    for uname, items in data.test_set.items():
        if uname in data.user:
            test_lists[data.user[uname]] = {
                data.item[i] for i in items if i in data.item
            }
    train_mask = r_mat > 0
    print(f"S built in {time.time() - t0:.0f}s; "
          f"{sum(1 for v in test_lists.values() if v)} test users")

    masks = {q: retained_mask(s_mat, q) for q in grid}
    print(f"retention masks built ({time.time() - t0:.0f}s)")

    type_slices = {"U": slice(0, n_u), "I": slice(n_u, n)}
    results = {t: {} for t in ("UU", "II", "UI", "IU", "ALL")}
    for q in grid:
        ku = masks[q][:n_u].astype(np.float32)  # retained sets of edge users
        for qp in grid:
            ki = masks[qp][n_u:].astype(np.float32)  # retained sets of edge items
            omega = ku.T @ r_mat @ ki  # Eq. 3: n x n co-occurrence counts
            per_type = {}
            for t, tsl in type_slices.items():
                left = s_users[:, tsl] @ omega[tsl]  # (U x n), v restricted to type t
                for tp, tpsl in type_slices.items():
                    per_type[t + tp] = left[:, tpsl] @ s_items[:, tpsl].T
            agg = per_type["UU"] + per_type["II"] + per_type["UI"] + per_type["IU"]
            for name, mat in list(per_type.items()) + [("ALL", agg)]:
                sc = mat.copy()
                sc[train_mask] = -np.inf
                results[name][f"{q}/{qp}"] = round(ndcg_at_k(sc, test_lists), 5)
            print(f"  q={q} q'={qp} ALL={results['ALL'][f'{q}/{qp}']:.5f} "
                  f"({time.time() - t0:.0f}s)")

    base_key = "100.0/100.0"
    if base_key not in results["ALL"]:
        base_key = max(results["ALL"])  # partial grid: fall back to any cell
        print(f"\n(note: q=q'=100 not in grid; using {base_key} as comparison base)")
    base = results["ALL"][base_key]
    print(f"\nBaseline ({base_key}): NDCG@20 = {base:.5f}")
    for name in ("ALL", "UU", "II", "UI", "IU"):
        best_key = max(results[name], key=results[name].get)
        best = results[name][best_key]
        rel = (best - base) / base * 100 if base > 0 else float("nan")
        print(f"{name}: best NDCG@20 = {best:.5f} at (q, q') = {best_key} "
              f"({rel:+.2f}% vs all-neighbour aggregate baseline)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"dataset": args.dataset, "grid": grid, "results": results}, f,
                      indent=1)
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
