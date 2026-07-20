"""CUDA -> CPU device patcher for the pinned NT-SSM clone.

NT-SSM (via SELFRec) hardcodes `.cuda()` across its graph models and builds sparse
adjacency with the deprecated `torch.sparse.FloatTensor`. On an Apple Silicon Mac with no
CUDA the `.cuda()` calls crash, and the sparse constructor is deprecated in modern torch.
This rewrites both, reproducibly and idempotently:

- `.cuda()`                     -> `.to(_NTSSM_DEVICE)`
- `torch.sparse.FloatTensor(`   -> `torch.sparse_coo_tensor(`

and inserts one `_NTSSM_DEVICE = torch.device("cpu")` constant per file that needs it.
The core `patch_source` is a pure text transform so it is unit-tested without a clone.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field

_DEVICE_CONST = '_NTSSM_DEVICE = torch.device("cpu")  # patched by patch_device.py: CPU-only, no CUDA'


def patch_source(text):
    """Pure transform. Returns (new_text, n_cuda, n_sparse, inserted_device).

    Only the bare `.cuda()` form is rewritten. Every device call in the pinned NT-SSM
    clone (da8655e) is bare `.cuda()` (verified: 20 sites), so an argumented form like
    `.cuda(0)` does not occur; it is intentionally left untouched rather than risk
    corrupting a non-default-device call.
    """
    n_cuda = text.count(".cuda()")
    n_sparse = text.count("torch.sparse.FloatTensor(")
    new = text.replace(".cuda()", ".to(_NTSSM_DEVICE)")
    new = new.replace("torch.sparse.FloatTensor(", "torch.sparse_coo_tensor(")

    inserted_device = False
    if n_cuda > 0 and "_NTSSM_DEVICE" not in text:
        new = _insert_device_constant(new)
        inserted_device = True
    return new, n_cuda, n_sparse, inserted_device


def _insert_device_constant(text):
    lines = text.splitlines(keepends=True)
    # Preferred: right after a top-level `import torch` (which binds the name `torch`).
    for idx, line in enumerate(lines):
        if line.strip() == "import torch":
            lines.insert(idx + 1, _DEVICE_CONST + "\n")
            return "".join(lines)
    # Fallback: no bare `import torch`. Insert a self-contained block, but AFTER any
    # leading `from __future__` imports and a module docstring (both of which must stay
    # first), so the patched file still compiles.
    at = _first_insertable_line(text)
    block = 'import torch as _ntssm_torch\n_NTSSM_DEVICE = _ntssm_torch.device("cpu")\n'
    lines.insert(at, block)
    return "".join(lines)


def _first_insertable_line(text):
    """0-based line index after any leading docstring and `from __future__` imports."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    idx = 0
    for node in tree.body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_docstring or is_future:
            idx = node.end_lineno  # 1-based end line -> insert at this 0-based index (after it)
        else:
            break
    return idx


@dataclass
class PatchSummary:
    files_patched: list = field(default_factory=list)
    cuda_replaced: int = 0
    sparse_fixed: int = 0


def patch_tree(root):
    """Patch every .py under `root` that uses `.cuda()` or the deprecated sparse ctor."""
    summary = PatchSummary()
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as fh:
                text = fh.read()
            if ".cuda()" not in text and "torch.sparse.FloatTensor(" not in text:
                continue
            new, n_cuda, n_sparse, _ = patch_source(text)
            if new != text:
                with open(path, "w") as fh:
                    fh.write(new)
                summary.files_patched.append(os.path.relpath(path, root))
                summary.cuda_replaced += n_cuda
                summary.sparse_fixed += n_sparse
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Patch NT-SSM .cuda() -> CPU device")
    parser.add_argument("--root", default="NT-SSM", help="Path to the NT-SSM clone (default: ./NT-SSM)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        print(f"error: clone not found at {args.root!r}; run harness/setup_env.sh first", file=sys.stderr)
        return 2

    s = patch_tree(args.root)
    if not s.files_patched:
        print(f"already patched: no .cuda() / sparse.FloatTensor under {args.root}")
    else:
        print(
            f"patched {len(s.files_patched)} file(s): "
            f"{s.cuda_replaced} .cuda() -> .to(_NTSSM_DEVICE), "
            f"{s.sparse_fixed} sparse.FloatTensor -> sparse_coo_tensor"
        )
        for f in s.files_patched:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
