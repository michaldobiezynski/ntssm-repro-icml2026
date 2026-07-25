"""Insert per-epoch training wall-time prints into LightGCN / LightGCN_NT (claim 6).

Idempotent (marker-guarded), in the style of patch_device.py / patch_metrics.py.
Adds, inside each model's train() epoch loop, a perf_counter start right after the
`for epoch in ...` line and a `[epoch-timing] epoch N train_seconds S` print just
before the post-loop `with torch.no_grad():` evaluation block, so only the training
batch loop is timed (evaluation excluded). CSV metric logs are unaffected.

Run:  .venv/bin/python harness/patch_epoch_timing.py
"""

import os

MARKER = "NTSSM_EPOCH_TIMING"
CLONE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NT-SSM")
TARGETS = ["model/graph/LightGCN.py", "model/graph/LightGCN_NT.py"]

START_LINE = (
    "            _ep_t0 = __import__('time').perf_counter()  # " + MARKER + "\n"
)
PRINT_LINE = (
    "            print('[epoch-timing] epoch', epoch + 1, 'train_seconds',"
    " round(__import__('time').perf_counter() - _ep_t0, 3))  # " + MARKER + "\n"
)


def patch(path):
    with open(path) as f:
        lines = f.readlines()
    if any(MARKER in line for line in lines):
        print(f"already patched: {path}")
        return
    out = []
    in_train = False
    timed = False
    for line in lines:
        if line.strip().startswith("def train(self):"):
            in_train, timed = True, False
        if in_train and not timed and line.strip() == "with torch.no_grad():":
            out.append(PRINT_LINE)
            timed = True
        out.append(line)
        if in_train and line.strip().startswith("for epoch in range(self.maxEpoch):"):
            out.append(START_LINE)
    if not timed:
        raise SystemExit(f"could not find insertion points in {path}")
    with open(path, "w") as f:
        f.writelines(out)
    print(f"patched: {path}")


for rel in TARGETS:
    patch(os.path.join(CLONE, rel))
