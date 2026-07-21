"""Guard the hand-transcribed paper reference numbers.

The paper_reference*.json means/stds were extracted from the paper HTML by agents; a
transcription flip or typo would silently corrupt every tolerance check. Lock the paper's
own ordering and basic sanity so such an error fails a cheap test instead of shipping.
"""
import glob
import json
import os

import pytest

_HARNESS = os.path.join(os.path.dirname(__file__), "..", "harness")
_REFS = sorted(glob.glob(os.path.join(_HARNESS, "paper_reference*.json")))


def test_reference_files_exist():
    assert _REFS, "expected at least one harness/paper_reference*.json"


@pytest.mark.parametrize("path", _REFS, ids=[os.path.basename(p) for p in _REFS])
def test_reference_encodes_paper_ordering_and_sane_stats(path):
    with open(path) as fh:
        ref = json.load(fh)
    for metric in ("recall@20", "ndcg@20"):
        # the paper's own claim must hold in the reference numbers
        assert ref["NT-SSM"][metric]["mean"] > ref["SSM"][metric]["mean"], f"{path} {metric} NT-SSM"
        assert ref["NT-BPR"][metric]["mean"] > ref["BPR"][metric]["mean"], f"{path} {metric} NT-BPR"
        for obj in ("BPR", "NT-BPR", "SSM", "NT-SSM"):
            cell = ref[obj][metric]
            assert 0.0 < cell["mean"] < 1.0, f"{path} {obj} {metric} mean out of range"
            assert cell["std"] > 0.0, f"{path} {obj} {metric} std must be positive"
