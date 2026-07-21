"""Inject Trackio metric logging into the pinned NT-SSM clone.

NT-SSM writes per-epoch metrics only to `logs/*.txt`; it does not log to Trackio. For the
challenge logbook we want live metric dashboards, so this patch adds:

- `trackio.init(...)` / `trackio.finish()` around `rec.execute()` in `main.py` (guarded so a
  Trackio failure never breaks training).
- a per-epoch `trackio.log(...)` at each backbone's best-epoch anchor
  (`ndcg_valid = float(result_valid[9]...)`, uniform across all six graph backbones), reading
  the recall@20 / ndcg@20 values already parsed there.

All transforms are pure functions so they are unit-tested without a clone or Trackio.
Idempotent: re-running is a no-op.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Anchor present verbatim in every model/graph/*.py backbone.
_ANCHOR = re.compile(r"^(?P<indent> +)ndcg_valid = float\(result_valid\[9\]\.split\(':'\)\[1\]\)", re.M)

_LOG_MARKER = "# trackio: per-epoch metrics"


def _log_block(indent):
    # result_valid/result_test hold 'Metric:value' strings; index 8/9 are recall@20/ndcg@20.
    return (
        f"{indent}{_LOG_MARKER}\n"
        f"{indent}try:\n"
        f"{indent}    import trackio as _trackio\n"
        f"{indent}    _trackio.log({{'epoch': epoch + 1,\n"
        f"{indent}                  'valid_recall@20': float(result_valid[8].split(':')[1]),\n"
        f"{indent}                  'valid_ndcg@20': float(result_valid[9].split(':')[1]),\n"
        f"{indent}                  'test_recall@20': float(result_test[8].split(':')[1]),\n"
        f"{indent}                  'test_ndcg@20': float(result_test[9].split(':')[1])}})\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
    )


def patch_backbone_source(text):
    """Inject the guarded per-epoch trackio.log before the best-epoch anchor. Idempotent."""
    if _LOG_MARKER in text:
        return text, False

    def repl(m):
        return _log_block(m.group("indent")) + m.group(0)

    new, n = _ANCHOR.subn(repl, text, count=1)
    return new, n > 0


_MAIN_INIT_MARKER = "# trackio: run init"
_MAIN_INIT = (
    "    " + _MAIN_INIT_MARKER + "\n"
    "    try:\n"
    "        import trackio\n"
    "        trackio.init(project=f'ntssm-{args.dataset}',\n"
    "                     name=f'{args.model_name}-{args.loss_type}-seed{args.seed}',\n"
    "                     config={'dataset': args.dataset, 'model_name': args.model_name,\n"
    "                             'loss_type': args.loss_type, 'tau': args.tau, 'seed': args.seed,\n"
    "                             'n_layer': args.n_layer, 'embedding_size': args.embedding_size})\n"
    "    except Exception as _e:\n"
    "        print(f'[trackio] init skipped: {_e}')\n"
)


def patch_main_source(text):
    """Wrap rec.execute() with guarded trackio.init/finish in main.py. Idempotent."""
    if _MAIN_INIT_MARKER in text:
        return text, False
    if "rec = SELFRec(args)" not in text:
        return text, False
    new = text.replace("    rec = SELFRec(args)\n", _MAIN_INIT + "    rec = SELFRec(args)\n", 1)
    new = new.replace(
        "    rec.execute()\n",
        "    rec.execute()\n"
        "    try:\n"
        "        import trackio\n"
        "        trackio.finish()\n"
        "    except Exception:\n"
        "        pass\n",
        1,
    )
    return new, new != text


def patch_clone_metrics(root):
    """Apply main.py + backbone metric patches under `root`. Returns list of changed files."""
    changed = []
    main_py = os.path.join(root, "main.py")
    if os.path.isfile(main_py):
        with open(main_py) as fh:
            text = fh.read()
        new, did = patch_main_source(text)
        if did:
            with open(main_py, "w") as fh:
                fh.write(new)
            changed.append("main.py")

    graph_dir = os.path.join(root, "model", "graph")
    if os.path.isdir(graph_dir):
        for name in sorted(os.listdir(graph_dir)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(graph_dir, name)
            with open(path) as fh:
                text = fh.read()
            new, did = patch_backbone_source(text)
            if did:
                with open(path, "w") as fh:
                    fh.write(new)
                changed.append(os.path.join("model", "graph", name))
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inject Trackio metric logging into the NT-SSM clone")
    parser.add_argument("--root", default="NT-SSM", help="Path to the NT-SSM clone (default: ./NT-SSM)")
    args = parser.parse_args(argv)
    if not os.path.isdir(args.root):
        print(f"error: clone not found at {args.root!r}; run harness/setup_env.sh first", file=sys.stderr)
        return 2
    changed = patch_clone_metrics(args.root)
    if not changed:
        print(f"metrics already patched under {args.root}")
    else:
        print(f"trackio metric logging injected into {len(changed)} file(s):")
        for f in changed:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
