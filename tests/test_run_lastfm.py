"""Tests for the LastFM run driver's safety contract.

Acceptance criteria (AC3):
- `run_lastfm.sh --dry-run` prints all four objective commands and exits 0 WITHOUT training.
- With no flag it does not train (prints usage and exits cleanly).
Real training is gated behind an explicit --execute flag (not exercised here).
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "harness" / "run_lastfm.sh"


def _run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=30,
    )


def test_dry_run_exits_zero():
    r = _run("--dry-run")
    assert r.returncode == 0, r.stderr


def test_dry_run_prints_all_four_objectives():
    out = _run("--dry-run").stdout
    # the 2x2 objective matrix, by label
    for label in ("NT-SSM", "SSM", "NT-BPR", "BPR"):
        assert label in out, f"missing objective {label} in dry-run output"
    # both model names and both loss types appear
    assert "LightGCN_NT" in out and "--model_name LightGCN " in out
    assert "--loss_type ssm" in out and "--loss_type bpr" in out


def test_dry_run_uses_verified_hyperparameters():
    out = _run("--dry-run").stdout
    for token in (
        "--dataset lastfm",
        "--seed 2026",
        "--embedding_size 64",
        "--n_layer 2",
        "--tau 0.2",
        "--item_ranking 10,20,40",
    ):
        assert token in out, f"missing hyperparameter {token}"


def test_dry_run_does_not_execute_training():
    out = _run("--dry-run").stdout
    assert "DRY RUN" in out.upper()
    # dry-run must not actually invoke python training
    assert "Running time" not in out  # NT-SSM prints this only on a real run


def test_no_flag_prints_usage_without_training():
    r = _run()
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
    # bare invocation must not emit the training commands
    assert "--model_name" not in r.stdout
