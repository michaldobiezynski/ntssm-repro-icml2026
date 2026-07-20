"""Tests for parsing NT-SSM per-epoch logs into a final-metrics dict.

Log format (verified against geon0325/NT-SSM @ da8655e, LightGCN_NT.py logging):
    epoch,split,'',R@k1,N@k1,R@k2,N@k2,R@k3,N@k3
where the cutoffs k1,k2,k3 come from --item_ranking (default 10,20,40), field 2 is
always empty, there is no header/footer, and values are round(x,5)-then-str (trailing
zeros stripped). The model early-stops on the BEST validation NDCG@20 and reports the
test row at that epoch, so the parser must mirror that selection.
"""
import textwrap

import pytest

from harness import ordering_check as oc


# Best valid NDCG@20 is at epoch 2 (0.185), not the last epoch (0.180).
SAMPLE_LOG = textwrap.dedent(
    """\
    1,valid,,0.10,0.09,0.20,0.18,0.30,0.28
    1,test,,0.11,0.10,0.21,0.19,0.31,0.29
    2,valid,,0.101,0.091,0.205,0.185,0.305,0.285
    2,test,,0.115,0.105,0.215,0.195,0.315,0.295
    3,valid,,0.102,0.092,0.206,0.180,0.306,0.286
    3,test,,0.116,0.106,0.216,0.196,0.316,0.296
    """
)


def _write(tmp_path, text, name="run.txt"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_parse_selects_test_metrics_at_best_valid_epoch(tmp_path):
    path = _write(tmp_path, SAMPLE_LOG)
    metrics = oc.parse_epoch_log(path)
    # best valid ndcg@20 is epoch 2 -> return the TEST row at epoch 2
    assert metrics["recall@20"] == pytest.approx(0.215)
    assert metrics["ndcg@20"] == pytest.approx(0.195)


def test_parse_reads_all_cutoffs(tmp_path):
    path = _write(tmp_path, SAMPLE_LOG)
    metrics = oc.parse_epoch_log(path)
    assert metrics["recall@10"] == pytest.approx(0.115)
    assert metrics["ndcg@10"] == pytest.approx(0.105)
    assert metrics["recall@40"] == pytest.approx(0.315)
    assert metrics["ndcg@40"] == pytest.approx(0.295)


def test_parse_handles_stripped_trailing_zeros(tmp_path):
    # a real log writes 0.377, not 0.37700
    log = "1,valid,,0.1,0.09,0.2,0.377,0.3,0.28\n1,test,,0.1,0.09,0.2,0.377,0.3,0.28\n"
    path = _write(tmp_path, log)
    metrics = oc.parse_epoch_log(path)
    assert metrics["ndcg@20"] == pytest.approx(0.377)


def test_parse_respects_item_ranking_override(tmp_path):
    # item_ranking=(20,40): fields become epoch,split,'',R@20,N@20,R@40,N@40
    log = "1,valid,,0.20,0.18,0.30,0.28\n1,test,,0.21,0.19,0.31,0.29\n"
    path = _write(tmp_path, log)
    metrics = oc.parse_epoch_log(path, item_ranking=(20, 40))
    assert metrics["recall@20"] == pytest.approx(0.21)
    assert metrics["ndcg@20"] == pytest.approx(0.19)


def test_parse_raises_if_selection_cutoff_absent(tmp_path):
    # item_ranking without @20 cannot be selected by ndcg@20
    log = "1,valid,,0.10,0.09,0.30,0.28\n1,test,,0.11,0.10,0.31,0.29\n"
    path = _write(tmp_path, log)
    with pytest.raises(ValueError):
        oc.parse_epoch_log(path, item_ranking=(10, 40))


def test_collect_run_metrics_maps_objectives(tmp_path):
    ssm = _write(tmp_path, SAMPLE_LOG, "ssm.txt")
    ntssm = _write(
        tmp_path,
        SAMPLE_LOG.replace("0.195", "0.205").replace("0.215", "0.225"),
        "ntssm.txt",
    )
    matrix = oc.collect_run_metrics({"SSM": ssm, "NT-SSM": ntssm})
    assert matrix["SSM"]["ndcg@20"] == pytest.approx(0.195)
    assert matrix["NT-SSM"]["ndcg@20"] == pytest.approx(0.205)
    result = oc.check_ordering(matrix, primary=("ndcg@20",))
    assert result.passed is True
