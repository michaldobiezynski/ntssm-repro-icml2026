"""Numerical audit of the Section 5 SSM gradient analysis (registry claim 2).

The paper's Eq. 5 (Appendix A) states, for the SSM loss L(i;u) with negatives B_u:

    dL / d(e_v^(0).e_v'^(0))  =  (S_uv / tau) * ( E_{x~pi_u}[S_xv'] - S_iv' )

so the UPDATE DIRECTION of a neighbour-pair weight depends only on ITEM-side
structural similarity (S_iv' vs its softmax-expected value), while USER-side
similarity S_uv >= 0 enters only as magnitude (Limitation 1), and the same
criterion applies to every neighbour-pair type (Limitation 2).

Audit, on the real dataset graph (float64, autograd):
  A. autograd gradient == Eq. 5 closed form, entrywise over a sampled V x V' grid;
  B. sign is decided by the item side: for fixed v', sign identical across all v
     with S_uv > 0; entries with S_uv = 0 get exactly zero gradient;
  C. sign flips exactly at the threshold S_iv' = E_{x~pi_u}[S_xv'], i.e.
     (v,v') is upweighted (grad < 0) iff S_iv' > E_pi[S_xv'];
  D. type-uniformity: the per-column ratio grad * tau / S_uv is constant across v
     (user- and item-type alike) - the criterion carries no pair-type dependence;
  E. control: negating a subset of S_u entries (violating S >= 0) makes the sign
     vary within columns - non-negativity is the load-bearing premise.

Run from the repo root:  .venv/bin/python harness/audit_ssm_gradient.py --dataset ml-1m
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_decomposition import load_graph, s_row  # noqa: E402

import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

FAILURES = []


def check(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def sample_support(row, n_u, cap, rng):
    """Sample up to cap indices from the row's support, keeping both node types."""
    idx = np.nonzero(row)[0]
    users, items = idx[idx < n_u], idx[idx >= n_u]
    take_u = min(len(users), cap // 2)
    take_i = min(len(items), cap - take_u)
    pick = np.concatenate([
        rng.choice(users, size=take_u, replace=False) if take_u else users[:0],
        rng.choice(items, size=take_i, replace=False) if take_i else items[:0],
    ])
    return np.sort(pick)


def eq5_closed_form(a_u, b_stack, pi, tau):
    """(S_uv/tau) * (E_pi[S_xv'] - S_iv'); b_stack rows ordered [i, negatives...]."""
    expected = pi @ b_stack
    return np.outer(a_u, expected - b_stack[0]) / tau


def run_audit(a_u, b_stack, w0, tau, label):
    """Autograd loss gradient wrt W under scores s(u,x) = a_u^T W b_x; compare to Eq. 5."""
    a = torch.tensor(a_u, dtype=torch.float64)
    b = torch.tensor(b_stack, dtype=torch.float64)  # (1+|B_u|, |V'|), row 0 = positive i
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    scores = b @ (w.T @ a)  # s(u,x) for x in {i} u B_u
    loss = -(scores[0] / tau - torch.logsumexp(scores / tau, dim=0))
    loss.backward()
    grad = w.grad.numpy()
    pi = torch.softmax(scores / tau, dim=0).detach().numpy()
    closed = eq5_closed_form(a_u, b_stack, pi, tau)
    scale = max(np.abs(grad).max(), 1e-30)
    err = np.abs(grad - closed).max() / scale
    check(f"A  [{label}] autograd == Eq.5 closed form", err < 1e-10,
          f"max rel err = {err:.2e} over {grad.shape[0]}x{grad.shape[1]} pair grid")
    return grad, pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m", choices=["lastfm", "ml-1m"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--negatives", type=int, default=128)
    ap.add_argument("--support", type=int, default=400)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--taus", default="0.2,0.1")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = load_graph(args.dataset)
    n_u = data.user_num
    n = n_u + data.item_num
    adj = sp.csr_matrix(data.norm_adj, dtype=np.float64)
    L = args.layers
    print(f"dataset={args.dataset} users={n_u} items={data.item_num} layers={L} "
          f"negatives={args.negatives} support_cap={args.support} seed={args.seed}")

    # A real (u, i) training pair plus uniformly sampled negative items.
    inter = sp.csr_matrix(data.ui_adj, dtype=np.float64)[:n_u, n_u:]
    u = int(rng.choice(np.nonzero(inter.getnnz(axis=1) >= 20)[0]))
    pos_items = inter[u].indices
    i_global = int(rng.choice(pos_items)) + n_u
    neg_pool = np.setdiff1d(np.arange(data.item_num), pos_items)
    negs = rng.choice(neg_pool, size=args.negatives, replace=False) + n_u
    xs = np.concatenate([[i_global], negs])

    # Real structural-similarity rows; restricted supports V (user side), V' (item side).
    s_u = s_row(adj, u, L)
    s_x = np.vstack([s_row(adj, int(x), L) for x in xs])
    # V: user-side support plus a few explicit ZERO-similarity rows, so the
    # "S_uv = 0 gives exactly zero gradient" branch is genuinely exercised.
    V = sample_support(s_u, n_u, args.support, rng)
    zero_pool = np.setdiff1d(np.arange(n), np.nonzero(s_u)[0])
    V = np.sort(np.concatenate([V, rng.choice(zero_pool, size=8, replace=False)]))
    # V': half from the positive item's support, half from negatives' supports
    # OUTSIDE it (S_iv' = 0 there), so both upweighted AND downweighted columns
    # occur and the check-C boundary is crossed in both directions.
    Vp_pos = sample_support(s_x[0], n_u, args.support // 2, rng)
    neg_union = np.unique(np.concatenate([np.nonzero(s_x[k])[0] for k in range(1, 9)]))
    off_pool = np.setdiff1d(neg_union, np.nonzero(s_x[0])[0])
    take = min(len(off_pool), args.support - len(Vp_pos))
    Vp = np.sort(np.concatenate([Vp_pos, rng.choice(off_pool, size=take, replace=False)]))
    a_u = s_u[V]
    b_stack = s_x[:, Vp]
    types_v = np.where(V < n_u, "U", "I")
    types_vp = np.where(Vp < n_u, "U", "I")
    e0 = rng.standard_normal((n, args.dim)) * 0.1
    w0 = e0[V] @ e0[Vp].T  # real initial pair weights e_v^(0) . e_v'^(0)
    print(f"anchor user={u} positive item={i_global - n_u} |V|={len(V)} "
          f"(U/I {np.sum(types_v == 'U')}/{np.sum(types_v == 'I')}) |V'|={len(Vp)} "
          f"(U/I {np.sum(types_vp == 'U')}/{np.sum(types_vp == 'I')})")

    for tau in [float(t) for t in args.taus.split(",")]:
        grad, pi = run_audit(a_u, b_stack, w0, tau, f"tau={tau}")

        pos_rows = a_u > 0
        zero_rows = ~pos_rows
        col_sign = np.sign(pi @ b_stack - b_stack[0])  # sign of (E_pi[S_xv'] - S_iv')
        sign_ok = all(
            np.all(np.sign(grad[pos_rows, c]) == col_sign[c])
            for c in range(len(Vp))
            if col_sign[c] != 0
        )
        zeros_ok = np.all(grad[zero_rows] == 0.0) if zero_rows.any() else True
        check(f"B  [tau={tau}] item side decides sign", sign_ok and zeros_ok,
              f"sign uniform down every column over {int(pos_rows.sum())} rows with "
              f"S_uv>0; {int(zero_rows.sum())} rows with S_uv=0 have exactly zero grad")

        thresh = pi @ b_stack
        up = grad[pos_rows][:, :] < 0
        should_up = (b_stack[0] > thresh)[None, :]
        flip_ok = np.array_equal(up, np.broadcast_to(should_up, up.shape))
        n_up = int(should_up.sum())
        check(f"C  [tau={tau}] upweighted iff S_iv' > E_pi[S_xv']", flip_ok,
              f"{n_up}/{len(Vp)} columns upweighted, boundary exact")

        ratio = grad[pos_rows] * tau / a_u[pos_rows, None]
        ratio_spread = np.abs(ratio - ratio[0]).max()
        check(f"D  [tau={tau}] criterion uniform across pair types", ratio_spread < 1e-12,
              f"grad*tau/S_uv constant per column across user- AND item-type v "
              f"(spread {ratio_spread:.2e}); no pair-type term exists")

    # E. control: signed S_u (premise violation) lets the user side flip signs.
    tau = 0.2
    a_signed = a_u.copy()
    flip = rng.random(len(a_signed)) < 0.3
    a_signed[flip] *= -1.0
    a = torch.tensor(a_signed, dtype=torch.float64)
    b = torch.tensor(b_stack, dtype=torch.float64)
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    scores = b @ (w.T @ a)
    loss = -(scores[0] / tau - torch.logsumexp(scores / tau, dim=0))
    loss.backward()
    g = w.grad.numpy()
    mixed_cols = sum(
        1 for c in range(len(Vp))
        if len({s for s in np.sign(g[np.abs(a_signed) > 0, c]) if s != 0}) > 1
    )
    check("E  control: signed S_u breaks item-side-only rule", mixed_cols > 0,
          f"{mixed_cols}/{len(Vp)} columns now mix signs once S >= 0 is violated")

    print()
    if FAILURES:
        print(f"AUDIT RESULT: FAIL ({len(FAILURES)} failed: {', '.join(FAILURES)})")
        return 1
    print("AUDIT RESULT: PASS - Eq.5 gradient identity and Limitation 1 (item-side-"
          f"dominated updates) verified numerically on {args.dataset} (float64 autograd).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
