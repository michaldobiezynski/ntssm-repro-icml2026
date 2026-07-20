"""Tests for the CUDA -> CPU device patcher.

Acceptance criteria (AC4):
- Every `.cuda()` becomes `.to(_NTSSM_DEVICE)`, with a `_NTSSM_DEVICE = torch.device("cpu")`
  constant inserted after `import torch`.
- The deprecated `torch.sparse.FloatTensor(` is rewritten to `torch.sparse_coo_tensor(`.
- The transform is idempotent (re-running changes nothing) and yields valid Python.
"""
import textwrap

from harness import patch_device as pd


CUDA_SRC = textwrap.dedent(
    """\
    import torch
    import torch.nn as nn


    class LightGCN(nn.Module):
        def build(self):
            model = self.model.cuda()
            self.sparse_norm_adj = convert(self.norm_adj).cuda()
            noise = torch.rand_like(self.x).cuda()
            return model
    """
)

SPARSE_SRC = textwrap.dedent(
    """\
    import torch

    def convert(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    """
)


def test_replaces_every_cuda_call():
    new, n_cuda, _, inserted = pd.patch_source(CUDA_SRC)
    assert n_cuda == 3
    assert ".cuda()" not in new
    assert new.count(".to(_NTSSM_DEVICE)") == 3
    assert inserted is True


def test_inserts_device_constant_after_import_torch():
    new, *_ = pd.patch_source(CUDA_SRC)
    lines = new.splitlines()
    ti = lines.index("import torch")
    assert lines[ti + 1].startswith('_NTSSM_DEVICE = torch.device("cpu")')


def test_fixes_deprecated_sparse_tensor():
    new, n_cuda, n_sparse, inserted = pd.patch_source(SPARSE_SRC)
    assert n_sparse == 1
    assert "torch.sparse.FloatTensor(" not in new
    assert "torch.sparse_coo_tensor(" in new
    # no cuda in this file -> no device constant needed
    assert n_cuda == 0
    assert inserted is False
    assert "_NTSSM_DEVICE" not in new


def test_patched_source_is_valid_python():
    new, *_ = pd.patch_source(CUDA_SRC)
    compile(new, "<patched>", "exec")  # raises SyntaxError if malformed


def test_idempotent_on_already_patched_source():
    once, *_ = pd.patch_source(CUDA_SRC)
    twice, n_cuda, n_sparse, inserted = pd.patch_source(once)
    assert twice == once
    assert n_cuda == 0 and n_sparse == 0 and inserted is False


def test_patch_tree_walks_and_reports(tmp_path):
    (tmp_path / "model" / "graph").mkdir(parents=True)
    (tmp_path / "base").mkdir()
    (tmp_path / "model" / "graph" / "LightGCN.py").write_text(CUDA_SRC)
    (tmp_path / "base" / "torch_interface.py").write_text(SPARSE_SRC)
    (tmp_path / "README.md").write_text("not python")

    summary = pd.patch_tree(str(tmp_path))
    assert summary.cuda_replaced == 3
    assert summary.sparse_fixed == 1
    assert len(summary.files_patched) == 2

    # no .cuda() left anywhere under the tree
    for p in (tmp_path / "model" / "graph" / "LightGCN.py",):
        assert ".cuda()" not in p.read_text()

    # second run is a no-op
    summary2 = pd.patch_tree(str(tmp_path))
    assert summary2.cuda_replaced == 0 and summary2.sparse_fixed == 0
    assert summary2.files_patched == []
