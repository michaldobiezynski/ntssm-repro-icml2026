"""Tests for the local-only Trackio logbook wrapper.

Acceptance criteria (AC2):
- init is called with project="icml2026-ntssm" and NO space_id (local-only).
- log forwards the metrics dict.
- alert uses the kwarg `text=` (NOT `message=`) and a level in {INFO, WARN, ERROR}.
- finish is called.
All verified against an injected mock, so trackio need not be installed to test.
"""
from unittest.mock import MagicMock

import pytest

from harness.trackio_logger import LogbookRun


def _fake_trackio():
    fake = MagicMock(name="trackio")
    # AlertLevel.INFO/WARN/ERROR resolve to distinct sentinels
    fake.AlertLevel.INFO = "LEVEL_INFO"
    fake.AlertLevel.WARN = "LEVEL_WARN"
    fake.AlertLevel.ERROR = "LEVEL_ERROR"
    return fake


def test_start_inits_with_default_project_and_no_space_id():
    fake = _fake_trackio()
    run = LogbookRun(name="lastfm-lightgcn-ntssm-seed2026", config={"seed": 2026}, trackio=fake)
    run.start()
    fake.init.assert_called_once()
    kwargs = fake.init.call_args.kwargs
    assert kwargs["project"] == "icml2026-ntssm"
    assert kwargs["name"] == "lastfm-lightgcn-ntssm-seed2026"
    assert kwargs["config"] == {"seed": 2026}
    # local-only: never publish to a Space
    assert "space_id" not in kwargs


def test_custom_project_name_is_honoured():
    fake = _fake_trackio()
    LogbookRun(project="my-proj", trackio=fake).start()
    assert fake.init.call_args.kwargs["project"] == "my-proj"


def test_log_forwards_metrics_dict():
    fake = _fake_trackio()
    run = LogbookRun(trackio=fake)
    run.start()
    run.log({"recall@20": 0.29, "ndcg@20": 0.27, "epoch": 5})
    fake.log.assert_called_once_with({"recall@20": 0.29, "ndcg@20": 0.27, "epoch": 5})


def test_alert_uses_text_kwarg_not_message():
    fake = _fake_trackio()
    run = LogbookRun(trackio=fake)
    run.start()
    run.alert(title="Diverged", text="loss NaN at epoch 3", level="ERROR")
    kwargs = fake.alert.call_args.kwargs
    assert kwargs["title"] == "Diverged"
    assert kwargs["text"] == "loss NaN at epoch 3"
    assert "message" not in kwargs
    assert kwargs["level"] == "LEVEL_ERROR"


def test_alert_default_level_is_warn():
    fake = _fake_trackio()
    run = LogbookRun(trackio=fake)
    run.start()
    run.alert(title="Slow", text="epoch over budget")
    assert fake.alert.call_args.kwargs["level"] == "LEVEL_WARN"


def test_alert_rejects_unknown_level():
    fake = _fake_trackio()
    run = LogbookRun(trackio=fake)
    run.start()
    with pytest.raises(ValueError):
        run.alert(title="x", text="y", level="CRITICAL")


def test_finish_calls_trackio_finish():
    fake = _fake_trackio()
    run = LogbookRun(trackio=fake)
    run.start()
    run.finish()
    fake.finish.assert_called_once()


def test_context_manager_starts_and_finishes():
    fake = _fake_trackio()
    with LogbookRun(trackio=fake) as run:
        run.log({"epoch": 0})
    fake.init.assert_called_once()
    fake.finish.assert_called_once()
