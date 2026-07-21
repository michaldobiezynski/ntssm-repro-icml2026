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
    # '>>> training' is emitted only by execute(); its absence proves dry-run did not
    # mis-route to the training path (this can actually go RED on a case-branch regression).
    assert ">>> training" not in out


def test_dry_run_pins_objective_specific_paper_alphas():
    lines = [l for l in _run("--dry-run").stdout.splitlines() if l.startswith("python main.py")]
    assert len(lines) == 4
    ntssm, ssm, ntbpr, bpr = lines  # print_matrix order: NT-SSM, SSM, NT-BPR, BPR
    # Table 5 gives DIFFERENT optimal alphas to NT-SSM vs NT-BPR (LightGCN/LastFM)
    ssm_alphas = "--alpha_uu 1.2 --alpha_ii 0.8 --alpha_ui 0.8 --alpha_iu 0.9"
    bpr_alphas = "--alpha_uu 1.3 --alpha_ii 1.5 --alpha_ui 0.9 --alpha_iu 1.3"
    assert ssm_alphas in ntssm and "LightGCN_NT" in ntssm and "--loss_type ssm" in ntssm
    assert bpr_alphas in ntbpr and "LightGCN_NT" in ntbpr and "--loss_type bpr" in ntbpr
    # NT-SSM must NOT accidentally carry the NT-BPR alphas and vice versa
    assert bpr_alphas not in ntssm and ssm_alphas not in ntbpr
    # plain baselines carry NO alpha flags
    assert "--alpha_uu" not in ssm and "--loss_type ssm" in ssm
    assert "--alpha_uu" not in bpr and "--loss_type bpr" in bpr


def test_no_flag_prints_usage_without_training():
    r = _run()
    assert r.returncode == 0
    assert "usage" in (r.stdout + r.stderr).lower()
    # bare invocation must not emit the training commands
    assert "--model_name" not in r.stdout
