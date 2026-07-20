"""Tests for setup_env.sh arg-parsing and the --check readiness logic.

The full install (torch/numba download, clone) is heavy and owner-triggered, so it is
not exercised here; these tests cover the safe control paths.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "harness" / "setup_env.sh"


def _run(*args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=60,
    )


def test_help_exits_zero_with_usage():
    r = _run("--help")
    assert r.returncode == 0
    assert "usage" in r.stdout.lower()


def test_check_flags_missing_clone_nonzero():
    # No ./NT-SSM clone exists in the repo yet -> --check must report it and exit non-zero.
    r = _run("--check")
    assert r.returncode != 0
    assert "MISSING" in r.stdout
    assert "NT-SSM" in r.stdout


def test_unknown_flag_errors():
    r = _run("--frobnicate")
    assert r.returncode == 2
