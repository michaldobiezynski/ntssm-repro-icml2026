"""Summarise the SSM-baseline tau grid against the paper's Table 1 (claim 4).

For each SSM log of the given dataset: report test Recall@20 / NDCG@20 at the
best-validation epoch (parse_epoch_log's rule, matching the models' own early
stopping), the tolerance verdict vs the paper's SSM row, and the NT-SSM relative
gain both vs the paper's baseline and vs that tau's baseline.

Run:  .venv/bin/python harness/tau_summary.py --dataset ml-1m
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ordering_check import parse_epoch_log, within_tolerance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m", choices=["lastfm", "ml-1m"])
    args = ap.parse_args()

    ref_file = "paper_reference_ml1m.json" if args.dataset == "ml-1m" else "paper_reference.json"
    ref = json.load(open(os.path.join(HERE, ref_file)))
    ssm_ref = ref["SSM"]
    nt_ref = ref["NT-SSM"]

    logs = sorted(glob.glob(
        os.path.join(REPO, "NT-SSM", "logs",
                     f"{args.dataset}_LightGCN_lr*_loss-ssm_tau*.txt")))
    nt_logs = sorted(glob.glob(
        os.path.join(REPO, "NT-SSM", "logs",
                     f"{args.dataset}_LightGCN_NT_*_loss-ssm_tau*.txt")))
    if not logs:
        raise SystemExit("no SSM logs found")
    nt = parse_epoch_log(nt_logs[-1]) if nt_logs else None


    print(f"dataset={args.dataset}  paper SSM: recall@20 {ssm_ref['recall@20']['mean']}"
          f" +/- {ssm_ref['recall@20']['std']}, ndcg@20 {ssm_ref['ndcg@20']['mean']}"
          f" +/- {ssm_ref['ndcg@20']['std']};  paper NT-SSM ndcg@20 {nt_ref['ndcg@20']['mean']}")
    if nt:
        print(f"reproduced NT-SSM (from {os.path.basename(nt_logs[-1])}): "
              f"recall@20 {nt['recall@20']:.5f}, ndcg@20 {nt['ndcg@20']:.5f}\n")

    print("| tau | test Recall@20 | test NDCG@20 | vs paper SSM (band) "
          "| NT-SSM gain vs this baseline |")
    print("| --- | --- | --- | --- | --- |")
    for log in logs:
        tau = re.search(r"tau([0-9.]+)_", os.path.basename(log)).group(1)
        res = parse_epoch_log(log)
        r20 = res["recall@20"]
        n20 = res["ndcg@20"]
        ok_n, band_n = within_tolerance(
            n20, ssm_ref["ndcg@20"]["mean"], ssm_ref["ndcg@20"]["std"])
        ok_r, _ = within_tolerance(
            r20, ssm_ref["recall@20"]["mean"], ssm_ref["recall@20"]["std"])
        verdict = "IN band" if (ok_n and ok_r) else (
            "ndcg in band" if ok_n else ("above band" if n20 >
            ssm_ref["ndcg@20"]["mean"] else "below band"))
        gain = (nt["ndcg@20"] - n20) / n20 * 100 if nt else float("nan")
        print(f"| {tau} | {r20:.5f} | {n20:.5f} "
              f"| {verdict} (ndcg band +/- {band_n:.4f}) | {gain:+.2f}% |")

    paper_gain = (nt_ref["ndcg@20"]["mean"] - ssm_ref["ndcg@20"]["mean"]) / \
        ssm_ref["ndcg@20"]["mean"] * 100
    if nt:
        repro_gain_paperbase = (nt["ndcg@20"] - ssm_ref["ndcg@20"]["mean"]) / \
            ssm_ref["ndcg@20"]["mean"] * 100
        print(f"\npaper relative gain (NDCG@20): {paper_gain:.2f}%  |  reproduced NT-SSM "
              f"vs paper baseline {ssm_ref['ndcg@20']['mean']}: {repro_gain_paperbase:.2f}%")


if __name__ == "__main__":
    main()
