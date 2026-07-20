"""Reproduction verdict for NT-SSM (ICML 2026), LightGCN backbone.

Two independent checks, per CLAUDE.md Section 7:

1. Ordering (PRIMARY): NT-SSM > SSM and NT-BPR > BPR on the same backbone/dataset.
   This is the criterion that decides reproduction success.
2. Tolerance (SECONDARY, reported not gating): a single-seed value counts as matching
   the paper if it lies within `paper_mean +/- max(5% of mean, 3 * std)`.

The comparison and band logic are pure functions so they can be unit-tested without
torch, a trained model, or the challenge infrastructure.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

# The two ordering claims the paper makes on a fixed backbone + dataset.
COMPARISONS = (("NT-SSM", "SSM"), ("NT-BPR", "BPR"))


def within_tolerance(value, mean, std, rel=0.05, std_mult=3.0):
    """Return (is_within, band).

    band = max(rel * |mean|, std_mult * |std|); membership is inclusive of the edge.
    A zero/unknown std collapses the band to the relative term.
    """
    band = max(rel * abs(mean), std_mult * abs(std or 0.0))
    return abs(value - mean) <= band, band


@dataclass
class Comparison:
    better: str
    worse: str
    metric: str
    better_value: float
    worse_value: float
    margin: float
    passed: bool


@dataclass
class OrderingResult:
    comparisons: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    passed: bool = False


def check_ordering(metrics, primary=("recall@20", "ndcg@20")):
    """Check NT-SSM > SSM and NT-BPR > BPR across the primary metrics.

    `metrics` is {objective: {metric: value}}. A comparison is only made when both
    sides expose the metric; otherwise it is recorded in `missing` (never silently
    treated as a pass). The overall verdict requires at least one comparison and
    every made comparison to hold (strict greater-than: a tie is not an improvement).
    """
    comparisons: list[Comparison] = []
    missing: list[list] = []
    for better, worse in COMPARISONS:
        for metric in primary:
            bval = metrics.get(better, {}).get(metric)
            wval = metrics.get(worse, {}).get(metric)
            if bval is None or wval is None:
                missing.append([better, worse, metric])
                continue
            comparisons.append(
                Comparison(
                    better=better,
                    worse=worse,
                    metric=metric,
                    better_value=bval,
                    worse_value=wval,
                    margin=bval - wval,
                    passed=bval > wval,
                )
            )
    passed = bool(comparisons) and all(c.passed for c in comparisons)
    return OrderingResult(comparisons=comparisons, missing=missing, passed=passed)


def tolerance_report(metrics, reference, primary=("recall@20", "ndcg@20")):
    """Compare each objective's metrics against the paper reference means/stds.

    `reference` is {objective: {metric: {"mean": m, "std": s}}}. Returns a list of
    dicts (objective, metric, value, mean, std, band, within). Reported, not gating.
    """
    rows = []
    for objective, mvals in metrics.items():
        for metric in primary:
            value = mvals.get(metric)
            ref = reference.get(objective, {}).get(metric)
            if value is None or ref is None:
                continue
            within, band = within_tolerance(value, ref["mean"], ref.get("std", 0.0))
            rows.append(
                {
                    "objective": objective,
                    "metric": metric,
                    "value": value,
                    "mean": ref["mean"],
                    "std": ref.get("std", 0.0),
                    "band": band,
                    "within": within,
                }
            )
    return rows


def format_report(result: OrderingResult, tolerance=None) -> str:
    lines = []
    verdict = "PASS" if result.passed else "FAIL"
    lines.append(f"ORDERING (primary): {verdict}")
    for c in result.comparisons:
        mark = "ok" if c.passed else "XX"
        lines.append(
            f"  [{mark}] {c.better} > {c.worse} on {c.metric}: "
            f"{c.better_value:.5f} vs {c.worse_value:.5f} (margin {c.margin:+.5f})"
        )
    for better, worse, metric in result.missing:
        lines.append(f"  [--] {better} > {worse} on {metric}: not evaluated (missing data)")
    if tolerance:
        lines.append("TOLERANCE (secondary, reported not gating):")
        for r in tolerance:
            mark = "ok" if r["within"] else "~~"
            lines.append(
                f"  [{mark}] {r['objective']} {r['metric']}: {r['value']:.5f} "
                f"vs {r['mean']:.5f} +/- {r['band']:.5f}"
            )
    return "\n".join(lines)


def parse_epoch_log(path, item_ranking=(10, 20, 40), select_metric="ndcg@20", report_split="test"):
    """Return the `report_split` metrics at the best-validation epoch.

    Mirrors NT-SSM's own model selection: the model early-stops on the best validation
    NDCG@20 and the reported figures are the test row at that epoch. Each log row is
    `epoch,split,'',` followed by (recall, ndcg) pairs for each cutoff in `item_ranking`
    order. There is no header/footer; values are round(x,5)-then-str (trailing zeros
    stripped), so they are parsed as floats.
    """
    cutoffs = [int(c) for c in item_ranking]
    field_index = {}
    for pos, k in enumerate(cutoffs):
        field_index[f"recall@{k}"] = 3 + 2 * pos
        field_index[f"ndcg@{k}"] = 3 + 2 * pos + 1

    if select_metric not in field_index:
        raise ValueError(
            f"selection metric {select_metric!r} not available for item_ranking {cutoffs} "
            f"(known: {sorted(field_index)})"
        )

    per_epoch = {}  # epoch -> {split -> {metric: value}}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            fields = line.split(",")
            try:
                epoch = int(fields[0])
            except (ValueError, IndexError):
                continue  # defensively skip any non-data line
            split = fields[1]
            row = {}
            for metric, idx in field_index.items():
                if idx < len(fields) and fields[idx] != "":
                    row[metric] = float(fields[idx])
            per_epoch.setdefault(epoch, {})[split] = row

    scored = [
        (epoch, splits["valid"][select_metric])
        for epoch, splits in per_epoch.items()
        if "valid" in splits and select_metric in splits["valid"]
    ]
    if not scored:
        raise ValueError(f"no validation rows with {select_metric} in {path}")
    best_epoch = max(scored, key=lambda t: t[1])[0]

    chosen = per_epoch[best_epoch].get(report_split)
    if chosen is None:
        raise ValueError(f"no {report_split!r} row at best epoch {best_epoch} in {path}")
    return chosen


def collect_run_metrics(log_paths, **kwargs):
    """Map {objective: logfile_path} -> {objective: {metric: value}} via parse_epoch_log."""
    return {obj: parse_epoch_log(path, **kwargs) for obj, path in log_paths.items()}


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    parser = argparse.ArgumentParser(description="NT-SSM reproduction verdict")
    parser.add_argument(
        "results",
        nargs="?",
        help='JSON of measured metrics: {"NT-SSM": {"ndcg@20": 0.27, ...}, ...}',
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        default=None,
        metavar="OBJ=PATH",
        help="Build the matrix from NT-SSM logs, e.g. --logs NT-SSM=logs/a.txt SSM=logs/b.txt",
    )
    parser.add_argument(
        "--item-ranking",
        default="10,20,40",
        help="Cutoffs used for the run's --item_ranking (default: 10,20,40)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional paper_reference.json for the tolerance report",
    )
    parser.add_argument(
        "--primary",
        default="recall@20,ndcg@20",
        help="Comma-separated primary metrics (default: recall@20,ndcg@20)",
    )
    args = parser.parse_args(argv)

    primary = tuple(m.strip() for m in args.primary.split(",") if m.strip())
    if args.logs:
        item_ranking = tuple(int(c) for c in args.item_ranking.split(","))
        log_paths = dict(pair.split("=", 1) for pair in args.logs)
        metrics = collect_run_metrics(log_paths, item_ranking=item_ranking)
    elif args.results:
        metrics = _load_json(args.results)
    else:
        parser.error("provide a results JSON path or --logs OBJ=PATH pairs")
    result = check_ordering(metrics, primary=primary)
    tol = None
    if args.reference:
        ref = _load_json(args.reference)
        tol = tolerance_report(metrics, ref, primary=primary)
    print(format_report(result, tolerance=tol))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
