"""Tests for setup_env.sh arg-parsing and the --check readiness logic.

The full install (torch/numba download, clone) is heavy and owner-triggered, so it is
not exercised here; these tests cover the safe control paths.
"""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "harness" / "setup_env.sh"


def _run(*args, env=None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=60,
        env=env,
    )


def test_help_exits_zero_with_usage():
    r = _run("--help")
    assert r.returncode == 0
    assert "usage" in r.stdout.lower()


def test_check_flags_missing_clone_isolated(tmp_path):
    # Point the clone/venv at empty temp paths so the result does not depend on ambient
    # repo state (once the owner runs setup_env, ./NT-SSM would otherwise appear) (F16).
    env = {
        **os.environ,
        "NTSSM_DIR": str(tmp_path / "noclone"),
        "NTSSM_VENV": str(tmp_path / "novenv"),
    }
    r = _run("--check", env=env)
    assert r.returncode != 0
    assert "MISSING" in r.stdout
    assert "NT-SSM" in r.stdout


def test_check_reports_venv_and_imports_ok_with_stub(tmp_path):
    # Seed a stub venv python that exits 0 -> the venv + imports branches report ok,
    # covering the positive side of check_setup without a heavy real install (F16).
    bind = tmp_path / "venv" / "bin"
    bind.mkdir(parents=True)
    py = bind / "python"
    py.write_text("#!/usr/bin/env bash\nexit 0\n")
    py.chmod(0o755)
    env = {
        **os.environ,
        "NTSSM_VENV": str(tmp_path / "venv"),
        "NTSSM_DIR": str(tmp_path / "noclone"),
    }
    r = _run("--check", env=env)
    assert "[ok] venv" in r.stdout
    assert "[ok] imports" in r.stdout
    assert "[MISSING] NT-SSM clone" in r.stdout
    assert r.returncode != 0  # clone still missing


def test_help_does_not_require_pin_file(tmp_path):
    # --help must not depend on the pin file being readable (F4).
    r = _run("--help")
    assert r.returncode == 0


def test_unknown_flag_errors():
    r = _run("--frobnicate")
    assert r.returncode == 2
