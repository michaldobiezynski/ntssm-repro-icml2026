"""Behaviour tests for the reproduction verdict logic.

Maps to the acceptance criteria:
- AC1a: ordering PASS iff NT-SSM > SSM and NT-BPR > BPR on each primary metric.
- AC1b: tolerance band = |value - mean| <= max(5% of mean, 3 * std), boundary inclusive.
- AC1c: partial objective matrices check only the pairs that are present.
- AC1d: ordering is the PRIMARY verdict; tolerance is reported, not gating.
"""
import math

import pytest

from harness import ordering_check as oc


# --- AC1b: tolerance band maths -------------------------------------------------

def test_within_tolerance_uses_five_percent_when_std_small():
    # 5% of 0.20 = 0.010; 3*std = 0.003 -> band = 0.010
    within, band = oc.within_tolerance(0.208, mean=0.20, std=0.001)
    assert math.isclose(band, 0.010, rel_tol=1e-9)
    assert within is True


def test_within_tolerance_uses_three_std_when_std_large():
    # 5% of 0.20 = 0.010; 3*std = 0.030 -> band = 0.030
    within, band = oc.within_tolerance(0.225, mean=0.20, std=0.010)
    assert math.isclose(band, 0.030, rel_tol=1e-9)
    assert within is True


def test_within_tolerance_boundary_is_inclusive():
    # value exactly at mean + band counts as within
    within, band = oc.within_tolerance(0.21, mean=0.20, std=0.0)  # band = 0.010
    assert math.isclose(band, 0.010, rel_tol=1e-9)
    assert within is True


def test_within_tolerance_just_outside_is_rejected():
    within, _ = oc.within_tolerance(0.2101, mean=0.20, std=0.0)  # band 0.010
    assert within is False


def test_within_tolerance_zero_std_falls_back_to_percent():
    within, band = oc.within_tolerance(0.20, mean=0.20, std=0.0)
    assert math.isclose(band, 0.010, rel_tol=1e-9)
    assert within is True


# --- AC1a / AC1c: ordering verdict ---------------------------------------------

def _full_matrix(nt_ssm, ssm, nt_bpr, bpr, metric="ndcg@20"):
    return {
        "NT-SSM": {metric: nt_ssm},
        "SSM": {metric: ssm},
        "NT-BPR": {metric: nt_bpr},
        "BPR": {metric: bpr},
    }


def test_ordering_passes_when_nt_variants_win_on_all_metrics():
    metrics = {
        "NT-SSM": {"recall@20": 0.33, "ndcg@20": 0.27},
        "SSM": {"recall@20": 0.32, "ndcg@20": 0.26},
        "NT-BPR": {"recall@20": 0.30, "ndcg@20": 0.24},
        "BPR": {"recall@20": 0.29, "ndcg@20": 0.23},
    }
    result = oc.check_ordering(metrics, primary=("recall@20", "ndcg@20"))
    assert result.passed is True
    assert all(c.passed for c in result.comparisons)
    # two comparisons x two metrics = four checks
    assert len(result.comparisons) == 4


def test_ordering_fails_when_ntssm_not_greater_than_ssm():
    metrics = _full_matrix(nt_ssm=0.26, ssm=0.26, nt_bpr=0.24, bpr=0.23)  # tie on the SSM pair
    result = oc.check_ordering(metrics, primary=("ndcg@20",))
    assert result.passed is False
    failed = [c for c in result.comparisons if not c.passed]
    assert len(failed) == 1
    assert failed[0].better == "NT-SSM" and failed[0].worse == "SSM"


def test_ordering_strict_greater_tie_is_failure():
    metrics = _full_matrix(nt_ssm=0.27, ssm=0.26, nt_bpr=0.24, bpr=0.24)  # tie on BPR pair
    result = oc.check_ordering(metrics, primary=("ndcg@20",))
    assert result.passed is False
    assert any(c.better == "NT-BPR" and not c.passed for c in result.comparisons)


def test_partial_matrix_checks_only_present_pairs():
    metrics = {
        "NT-SSM": {"ndcg@20": 0.27},
        "SSM": {"ndcg@20": 0.26},
    }  # no BPR pair at all
    result = oc.check_ordering(metrics, primary=("ndcg@20",))
    assert result.passed is True
    assert len(result.comparisons) == 1
    assert ("NT-BPR", "BPR", "ndcg@20") in [tuple(m) for m in result.missing]


def test_empty_matrix_is_not_a_pass():
    result = oc.check_ordering({}, primary=("ndcg@20",))
    assert result.passed is False
    assert result.comparisons == []


def test_missing_metric_on_one_side_is_skipped_not_crashed():
    metrics = {
        "NT-SSM": {"ndcg@20": 0.27},
        "SSM": {"recall@20": 0.32},  # different metric; ndcg@20 absent on SSM
    }
    result = oc.check_ordering(metrics, primary=("ndcg@20",))
    # comparison cannot be made -> recorded as missing, not a pass
    assert result.passed is False
    assert any(tuple(m) == ("NT-SSM", "SSM", "ndcg@20") for m in result.missing)


# --- AC1d: ordering is primary; tolerance reported separately -------------------

def test_report_includes_ordering_verdict_and_is_string():
    metrics = _full_matrix(0.27, 0.26, 0.24, 0.23)
    result = oc.check_ordering(metrics, primary=("ndcg@20",))
    text = oc.format_report(result)
    assert isinstance(text, str)
    assert "NT-SSM" in text and "SSM" in text
    assert "PASS" in text.upper()


# --- Tolerance report + CLI (F13) ----------------------------------------------

def test_tolerance_report_flags_within_and_outside():
    metrics = {"NT-SSM": {"recall@20": 0.2953, "ndcg@20": 0.40}}
    ref = {
        "NT-SSM": {
            "recall@20": {"mean": 0.2953, "std": 0.0012},
            "ndcg@20": {"mean": 0.2709, "std": 0.0013},
        }
    }
    rows = {r["metric"]: r for r in oc.tolerance_report(metrics, ref)}
    assert rows["recall@20"]["within"] is True
    assert rows["ndcg@20"]["within"] is False  # 0.40 far outside ~0.0135 band


def test_tolerance_report_skips_missing_ref():
    assert oc.tolerance_report({"BPR": {"recall@20": 0.1}}, {}, primary=("recall@20",)) == []


def test_format_report_renders_tolerance_section():
    metrics = _full_matrix(0.27, 0.26, 0.24, 0.23, metric="ndcg@20")
    result = oc.check_ordering(metrics, primary=("ndcg@20",))
    ref = {k: {"ndcg@20": {"mean": v[list(v)[0]], "std": 0.001}} for k, v in metrics.items()}
    tol = oc.tolerance_report(metrics, ref, primary=("ndcg@20",))
    text = oc.format_report(result, tolerance=tol)
    assert "TOLERANCE" in text


def _write_matrix_logs(tmp_path, ntssm_n=0.28, ssm_n=0.26):
    def one(n):
        return f"1,valid,,0.1,0.09,0.2,{n},0.3,0.28\n1,test,,0.1,0.09,0.2,{n},0.3,0.28\n"
    paths = {}
    for obj, n in (("NT-SSM", ntssm_n), ("SSM", ssm_n), ("NT-BPR", 0.25), ("BPR", 0.24)):
        p = tmp_path / f"{obj}.txt"
        p.write_text(one(n))
        paths[obj] = str(p)
    return paths


def test_main_logs_pass_returns_zero(tmp_path, capsys):
    p = _write_matrix_logs(tmp_path)
    code = oc.main(["--logs", *[f"{o}={pt}" for o, pt in p.items()], "--primary", "ndcg@20"])
    assert code == 0
    assert "PASS" in capsys.readouterr().out.upper()


def test_main_logs_fail_returns_one(tmp_path, capsys):
    p = _write_matrix_logs(tmp_path, ntssm_n=0.24, ssm_n=0.26)  # NT-SSM below SSM
    code = oc.main(["--logs", *[f"{o}={pt}" for o, pt in p.items()], "--primary", "ndcg@20"])
    assert code == 1


def test_main_missing_equals_errors(tmp_path):
    with pytest.raises(SystemExit) as exc:  # parser.error -> SystemExit(2) (F5)
        oc.main(["--logs", "NT-SSM"])
    assert exc.value.code == 2


def test_main_results_json_input(tmp_path, capsys):
    import json

    results = tmp_path / "r.json"
    results.write_text(json.dumps(_full_matrix(0.27, 0.26, 0.24, 0.23, metric="ndcg@20")))
    code = oc.main([str(results), "--primary", "ndcg@20"])
    assert code == 0
