"""Tests for the Trackio metric-injection patch."""
import textwrap

from harness import patch_metrics as pm


BACKBONE_SRC = textwrap.dedent(
    """\
    class LightGCN_NT:
        def train(self):
            for epoch in range(self.maxEpoch):
                self.evaluate(self.test('valid'), 'valid')
                result_valid = [r[:-1] for r in self.result]
                self.evaluate(self.test('test'), 'test')
                result_test = [r[:-1] for r in self.result]
                ndcg_valid = float(result_valid[9].split(':')[1])
                if ndcg_valid > best_valid:
                    best_valid = ndcg_valid
    """
)

MAIN_SRC = textwrap.dedent(
    """\
    import time
    if __name__ == '__main__':
        s = time.time()
        rec = SELFRec(args)
        rec.execute()
        e = time.time()
        print("Running time: %f s" % (e - s))
    """
)


def test_backbone_injects_guarded_log_before_anchor():
    new, changed = pm.patch_backbone_source(BACKBONE_SRC)
    assert changed is True
    # log block appears before the anchor line
    assert new.index("trackio") < new.index("ndcg_valid = float(result_valid[9]")
    assert "'test_ndcg@20'" in new
    assert "except Exception" in new  # guarded: never breaks training
    compile(new, "<backbone>", "exec")


def test_backbone_patch_idempotent():
    once, _ = pm.patch_backbone_source(BACKBONE_SRC)
    twice, changed = pm.patch_backbone_source(once)
    assert changed is False
    assert twice == once


def test_backbone_indentation_matches_anchor():
    new, _ = pm.patch_backbone_source(BACKBONE_SRC)
    # the injected try: sits at the same 12-space indent as the anchor
    assert "\n            try:\n" in new


def test_main_wraps_execute_with_init_and_finish():
    new, changed = pm.patch_main_source(MAIN_SRC)
    assert changed is True
    assert "trackio.init(" in new
    assert new.index("trackio.init(") < new.index("rec = SELFRec(args)")
    assert "trackio.finish()" in new
    assert new.index("rec.execute()") < new.index("trackio.finish()")
    compile(new, "<main>", "exec")


def test_main_patch_idempotent():
    once, _ = pm.patch_main_source(MAIN_SRC)
    twice, changed = pm.patch_main_source(once)
    assert changed is False
    assert twice == once


def test_patch_clone_metrics_walks_tree(tmp_path):
    (tmp_path / "model" / "graph").mkdir(parents=True)
    (tmp_path / "main.py").write_text(MAIN_SRC)
    (tmp_path / "model" / "graph" / "LightGCN_NT.py").write_text(BACKBONE_SRC)
    (tmp_path / "model" / "graph" / "SimGCL.py").write_text(BACKBONE_SRC)
    changed = pm.patch_clone_metrics(str(tmp_path))
    assert "main.py" in changed
    assert len(changed) == 3
    # second run is a no-op
    assert pm.patch_clone_metrics(str(tmp_path)) == []
