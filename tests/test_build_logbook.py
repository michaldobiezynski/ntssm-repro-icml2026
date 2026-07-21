"""Tests for build_logbook.sh's matrix-completeness guard (runs before any trackio call)."""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "harness" / "build_logbook.sh"


def _run(logs_dir):
    env = {**os.environ, "NTSSM_LOGS": str(logs_dir)}
    return subprocess.run(
        ["bash", str(SCRIPT)], capture_output=True, text=True, cwd=str(REPO), timeout=30, env=env
    )


def test_errors_when_no_complete_matrix(tmp_path):
    # empty log dir -> have_matrix false for every dataset -> exit 2 before any trackio call
    r = _run(tmp_path)
    assert r.returncode == 2
    assert "no complete 4-log matrix" in r.stderr


def test_errors_when_matrix_incomplete(tmp_path):
    # only the two NT logs present (SSM/BPR baselines missing) -> still incomplete
    (tmp_path / "lastfm_LightGCN_NT_x_loss-ssm_tau0.2.txt").write_text("1,test,,0,0,0,0,0,0\n")
    (tmp_path / "lastfm_LightGCN_NT_x_loss-bpr.txt").write_text("1,test,,0,0,0,0,0,0\n")
    r = _run(tmp_path)
    assert r.returncode == 2
    assert "no complete 4-log matrix" in r.stderr
