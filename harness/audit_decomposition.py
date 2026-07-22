"""Numerical audit of the Section 3 forward-pass decomposition (registry claim 1).

Verifies, on the real dataset graphs and the clone's own LightGCN encoder, that:
  A. the structural similarity matrix S = mean_{l=0..L} A_norm^l is symmetric,
     non-negative, and parameter-free (A_norm = D^-1/2 A D^-1/2 on the bipartite graph);
  B. the closed form E = S @ E0 equals LightGCN's iterative mean-pooled propagation,
     and matches the clone's LGCN_Encoder within float32 tolerance;
  C. the prediction score r_ui = e_u . e_i equals the paper's Eq. 1 double sum
     over neighbour pairs, computed the expensive way via an explicit Gram block;
  D. the four neighbour-pair-type sub-scores (UU, II, UI, IU) of Eq. 2 are computed
     from a disjoint, complete partition and sum exactly to r_ui;
  E. (context, Figure 1) multi-hop neighbour-pair coverage at 2 and 3 hops;
  F. control: with random-walk normalisation D^-1 A instead of the paper's symmetric
     normalisation, the closed-form/iterative equivalence to LightGCN breaks.

Run from the repo root:  .venv/bin/python harness/audit_decomposition.py --dataset ml-1m
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

CLONE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NT-SSM")
sys.path.insert(0, CLONE)

from data.loader import FileIO  # noqa: E402
from data.ui_graph import Interaction  # noqa: E402

FAILURES = []


def check(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def load_graph(dataset):
    conf = argparse.Namespace(
        dataset=dataset,
        model_name="LightGCN",
        model_type="graph",
        training_set=f"./dataset/{dataset}/train.txt",
        valid_set=f"./dataset/{dataset}/valid.txt",
        test_set=f"./dataset/{dataset}/test.txt",
    )
    cwd = os.getcwd()
    os.chdir(CLONE)
    try:
        train = FileIO.load_data_set(conf.training_set, conf.model_type)
        valid = FileIO.load_data_set(conf.valid_set, conf.model_type)
        test = FileIO.load_data_set(conf.test_set, conf.model_type)
        data = Interaction(conf, train, valid, test)
    finally:
        os.chdir(cwd)
    return data


def s_row(adj, x, layers):
    """Row x of S = mean_{l=0..L} adj^l, via iterated sparse mat-vecs (exact, float64)."""
    v = np.zeros(adj.shape[0])
    v[x] = 1.0
    acc = v.copy()
    cur = v
    for _ in range(layers):
        cur = adj.T.dot(cur)  # e_x^T adj^l  ==  (adj^T)^l e_x
        acc += cur
    return acc / (layers + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m", choices=["lastfm", "ml-1m"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = load_graph(args.dataset)
    n_u, n_i = data.user_num, data.item_num
    n = n_u + n_i
    adj = sp.csr_matrix(data.norm_adj, dtype=np.float64)  # D^-1/2 A D^-1/2, no self-loops
    raw = sp.csr_matrix(data.ui_adj, dtype=np.float64)  # unnormalised bipartite A
    L = args.layers
    print(
        f"dataset={args.dataset} users={n_u} items={n_i} nodes={n} "
        f"edges={raw.nnz // 2} layers={L} dim={args.dim} seed={args.seed}"
    )

    # Fixed random initial embeddings E0 (float64 ground truth).
    e0 = rng.standard_normal((n, args.dim)) * 0.1

    # Iterative LightGCN propagation, float64: mean over layer outputs.
    layer = e0.copy()
    pooled = e0.copy()
    for _ in range(L):
        layer = adj.dot(layer)
        pooled += layer
    pooled /= L + 1

    users = rng.choice(n_u, size=args.samples, replace=False)
    items = rng.choice(n_i, size=args.samples, replace=False) + n_u

    # A. symmetry / non-negativity of S on sampled node pairs.
    nodes = np.concatenate([users, items])
    rows = {int(x): s_row(adj, int(x), L) for x in nodes}
    sym_err = max(
        abs(rows[int(x)][int(y)] - rows[int(y)][int(x)])
        for x in nodes
        for y in nodes
    )
    min_entry = min(rows[int(x)].min() for x in nodes)
    check("A1 S symmetric (sampled)", sym_err < 1e-12, f"max |S_xy - S_yx| = {sym_err:.2e}")
    check("A2 S non-negative (sampled)", min_entry >= 0.0, f"min entry = {min_entry:.2e}")
    print("[note] S is parameter-free by construction: built solely from the adjacency, "
          "no learnable tensors involved.")

    # B. closed form S @ E0 == iterative mean-pooled propagation (per sampled node).
    closed_err = max(
        np.max(np.abs(rows[int(x)].dot(e0) - pooled[int(x)]))
        / max(np.max(np.abs(pooled[int(x)])), 1e-30)
        for x in nodes
    )
    check("B1 closed form == iterative (float64)", closed_err < 1e-10,
          f"max rel err = {closed_err:.2e}")

    # B2. the clone's own encoder reproduces the same propagation (float32 tolerance).
    import torch  # noqa: E402  (after sys.path setup; heavy import kept local)
    from model.graph.LightGCN import LGCN_Encoder  # noqa: E402

    enc = LGCN_Encoder(data, args.dim, L)
    with torch.no_grad():
        enc.embedding_dict["user_emb"].copy_(torch.tensor(e0[:n_u], dtype=torch.float32))
        enc.embedding_dict["item_emb"].copy_(torch.tensor(e0[n_u:], dtype=torch.float32))
        u_out, i_out = enc()
    enc_out = np.vstack([u_out.numpy(), i_out.numpy()])
    enc_err = np.max(np.abs(enc_out - pooled)) / np.max(np.abs(pooled))
    check("B2 clone LGCN_Encoder == closed form", enc_err < 1e-4,
          f"max rel err = {enc_err:.2e} (float32 encoder)")

    # C + D. Eq. 1 double sum and Eq. 2 four-type partition on sampled (u, i) pairs.
    eq1_worst, part_worst, disjoint_ok = 0.0, 0.0, True
    for u, i in zip(users, items):
        su, si = rows[int(u)], rows[int(i)]
        direct = float(pooled[int(u)].dot(pooled[int(i)]))
        v_idx = np.nonzero(su)[0]
        vp_idx = np.nonzero(si)[0]
        # Explicit neighbour-pair double sum over the learnable-weight Gram block,
        # accumulated in row chunks so the block never exceeds ~200MB.
        chunk = max(1, int(2.5e7) // max(len(vp_idx), 1))
        double_sum = 0.0
        for s in range(0, len(v_idx), chunk):
            blk = v_idx[s : s + chunk]
            double_sum += float(su[blk] @ (e0[blk] @ e0[vp_idx].T) @ si[vp_idx])
        eq1_worst = max(eq1_worst, abs(direct - double_sum) / max(abs(direct), 1e-30))

        v_u, v_i = v_idx[v_idx < n_u], v_idx[v_idx >= n_u]
        vp_u, vp_i = vp_idx[vp_idx < n_u], vp_idx[vp_idx >= n_u]
        disjoint_ok &= len(v_u) + len(v_i) == len(v_idx)
        disjoint_ok &= len(vp_u) + len(vp_i) == len(vp_idx)
        parts = {}
        for t, vs in (("U", v_u), ("I", v_i)):
            for tp, vps in (("U", vp_u), ("I", vp_i)):
                parts[t + tp] = float((su[vs] @ e0[vs]).dot(si[vps] @ e0[vps]))
        part_sum = parts["UU"] + parts["II"] + parts["UI"] + parts["IU"]
        part_worst = max(part_worst, abs(part_sum - direct) / max(abs(direct), 1e-30))
    check("C  Eq.1: score == neighbour-pair double sum", eq1_worst < 1e-9,
          f"worst rel err over {args.samples} (u,i) pairs = {eq1_worst:.2e}")
    check("D1 Eq.2: type partition disjoint + complete", disjoint_ok,
          "user/item split of every neighbour set is a partition")
    check("D2 Eq.2: UU+II+UI+IU == full score", part_worst < 1e-9,
          f"worst rel err = {part_worst:.2e}")

    # E. multi-hop neighbour-pair coverage (Figure 1 context; reachability within k hops).
    for hops in (2, 3):
        cover = []
        for u, i in zip(users[:10], items[:10]):
            reach_u = np.zeros(n, dtype=bool)
            reach_u[int(u)] = True
            reach_i = np.zeros(n, dtype=bool)
            reach_i[int(i)] = True
            fu, fi = reach_u.astype(np.float64), reach_i.astype(np.float64)
            for _ in range(hops):
                fu = fu + adj.dot(fu)
                fi = fi + adj.dot(fi)
            cover.append((fu > 0).sum() * (fi > 0).sum() / float(n) ** 2)
        print(f"[note] E coverage at {hops} hops: mean neighbour-pair coverage over 10 "
              f"sampled (u,i) = {np.mean(cover):.1%} of all node pairs")

    # F. control: random-walk normalisation D^-1 A breaks the equivalence to LightGCN.
    deg = np.asarray(raw.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    rw = sp.diags(1.0 / deg).dot(raw).tocsr()
    ctrl_err = max(
        np.max(np.abs(s_row(rw, int(x), L).dot(e0) - pooled[int(x)]))
        / max(np.max(np.abs(pooled[int(x)])), 1e-30)
        for x in nodes[:10]
    )
    check("F  control: D^-1 A closed form != LightGCN", ctrl_err > 1e-2,
          f"max rel err = {ctrl_err:.2e} (large by design: identity requires "
          f"the paper's symmetric normalisation)")

    print()
    if FAILURES:
        print(f"AUDIT RESULT: FAIL ({len(FAILURES)} failed: {', '.join(FAILURES)})")
        return 1
    print("AUDIT RESULT: PASS - Section 3 decomposition verified numerically "
          f"on {args.dataset} at L={L} (float64).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
