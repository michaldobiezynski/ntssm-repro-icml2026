"""Tests for the generalised dataset runner (run_matrix.sh)."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "harness" / "run_matrix.sh"


def _run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=str(REPO), timeout=30
    )


def test_lastfm_dry_run_uses_lastfm_table5_alphas():
    out = _run("lastfm", "--dry-run").stdout
    lines = [l for l in out.splitlines() if l.startswith("python main.py")]
    ntssm, ssm, ntbpr, bpr = lines
    assert "--alpha_uu 1.2 --alpha_ii 0.8 --alpha_ui 0.8 --alpha_iu 0.9" in ntssm
    assert "--alpha_uu 1.3 --alpha_ii 1.5 --alpha_ui 0.9 --alpha_iu 1.3" in ntbpr
    assert "--dataset lastfm" in ssm


def test_ml1m_dry_run_uses_ml1m_table5_alphas():
    out = _run("ml-1m", "--dry-run").stdout
    lines = [l for l in out.splitlines() if l.startswith("python main.py")]
    ntssm, ssm, ntbpr, bpr = lines
    # ML-1M Table 5: NT-SSM iu=1.0 (not 0.9), NT-BPR 1.4/1.3/0.8/1.3
    assert "--alpha_uu 1.2 --alpha_ii 0.8 --alpha_ui 0.8 --alpha_iu 1.0" in ntssm
    assert "--alpha_uu 1.4 --alpha_ii 1.3 --alpha_ui 0.8 --alpha_iu 1.3" in ntbpr
    assert "--dataset ml-1m" in ssm and "--alpha_uu" not in ssm


def test_dry_run_exits_zero_without_training():
    r = _run("ml-1m", "--dry-run")
    assert r.returncode == 0
    assert "DRY RUN" in r.stdout.upper()
    assert ">>> training" not in r.stdout


def test_unknown_dataset_errors():
    r = _run("cifar", "--dry-run")
    assert r.returncode == 2
    assert "unsupported dataset" in r.stderr


def test_missing_dataset_errors():
    r = _run()
    assert r.returncode == 2
