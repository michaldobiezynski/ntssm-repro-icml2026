#!/usr/bin/env bash
# One-shot environment setup for the NT-SSM reproduction:
#   venv (python3.13) + deps + pinned clone + device patch + import verification.
# Idempotent: safe to re-run. Use --check to verify an existing setup without installing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
# Paths are overridable so --check can run against an isolated tree in tests.
VENV="${NTSSM_VENV:-$REPO/.venv}"
CLONE="${NTSSM_DIR:-$REPO/NT-SSM}"
PIN_FILE="$REPO/PINNED_COMMIT_NTSSM.txt"
# torch wheels are reliable on 3.13 and dicey on 3.14; prefer 3.13 explicitly.
PY_BIN="${PYTHON:-$(command -v python3.13 || true)}"
DEPS=(torch numpy scipy numba trackio pytest)

# Read+validate the pinned commit lazily (only the setup/check paths need it), so
# --help does not depend on the pin file existing.
read_pin() {
  [ -f "$PIN_FILE" ] || { echo "error: pin file not found: $PIN_FILE" >&2; exit 2; }
  local pin; pin="$(tr -d '[:space:]' < "$PIN_FILE")"
  [[ "$pin" =~ ^[0-9a-f]{7,40}$ ]] || { echo "error: invalid commit hash in $PIN_FILE: '$pin'" >&2; exit 2; }
  printf '%s' "$pin"
}

usage() {
  cat <<EOF
usage: setup_env.sh [--check | --help]

  (no arg)   full setup: create .venv (python3.13), install deps, clone+pin NT-SSM
             at the pinned commit (PINNED_COMMIT_NTSSM.txt), apply the CPU device
             patch, verify imports.
  --check    verify an existing setup (venv, pinned+patched clone, imports); no install.
  -h,--help  show this help.
EOF
}

check_setup() {
  local ok=0
  local PIN; PIN="$(read_pin)"
  if [ -x "$VENV/bin/python" ]; then echo "[ok] venv: $VENV"; else echo "[MISSING] venv: run setup_env.sh"; ok=1; fi
  if [ -d "$CLONE/.git" ]; then
    local head; head="$(git -C "$CLONE" rev-parse HEAD 2>/dev/null || echo none)"
    if [ "$head" = "$PIN" ]; then echo "[ok] clone pinned at $PIN"; else echo "[MISSING] clone HEAD $head != pinned $PIN"; ok=1; fi
    if grep -rq "_NTSSM_DEVICE" "$CLONE/model/graph/LightGCN.py" 2>/dev/null; then echo "[ok] device patch applied"; else echo "[MISSING] device patch: run patch_device.py --root $CLONE"; ok=1; fi
  else
    echo "[MISSING] NT-SSM clone at $CLONE"; ok=1
  fi
  if [ -x "$VENV/bin/python" ]; then
    if "$VENV/bin/python" - <<'PY' 2>/dev/null
import importlib, sys
missing = [m for m in ("torch","numpy","scipy","numba") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
    then echo "[ok] imports: torch numpy scipy numba"; else echo "[MISSING] one of torch/numpy/scipy/numba not importable"; ok=1; fi
  fi
  return $ok
}

full_setup() {
  local PIN; PIN="$(read_pin)"
  [ -n "$PY_BIN" ] || { echo "error: python3.13 not found; set PYTHON=/path/to/python3.13" >&2; exit 2; }
  echo ">>> python: $("$PY_BIN" --version)"

  if [ ! -x "$VENV/bin/python" ]; then
    echo ">>> creating venv at $VENV"; "$PY_BIN" -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  echo ">>> installing deps: ${DEPS[*]}"
  "$VENV/bin/python" -m pip install --quiet "${DEPS[@]}"

  if [ ! -d "$CLONE/.git" ]; then
    echo ">>> cloning NT-SSM"; git clone --quiet https://github.com/geon0325/NT-SSM "$CLONE"
  fi
  echo ">>> pinning clone at $PIN"
  git -C "$CLONE" fetch --quiet origin
  git -C "$CLONE" checkout --quiet "$PIN"

  echo ">>> applying device patch"
  "$VENV/bin/python" "$HERE/patch_device.py" --root "$CLONE"

  echo ">>> verifying imports and torch build"
  "$VENV/bin/python" - <<'PY'
import torch, numpy, scipy, numba
print(f"    torch {torch.__version__}  numpy {numpy.__version__}  numba {numba.__version__}")
print(f"    mps_available={torch.backends.mps.is_available()} (we run on CPU regardless)")
assert hasattr(torch, "sparse_coo_tensor"), "torch too old for sparse_coo_tensor"
print("    sparse_coo_tensor: present")
PY
  echo ">>> setup complete. Next: bash harness/run_lastfm.sh --dry-run"
}

case "${1:-}" in
  --check) check_setup ;;
  -h|--help) usage ;;
  "") full_setup ;;
  *) echo "unknown option: $1" >&2; usage; exit 2 ;;
esac
