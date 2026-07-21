"""Local-only Trackio logbook wrapper for the NT-SSM reproduction.

Scope: local reproduction. Publishing to a HuggingFace Space is deliberately NOT
supported here (no `space_id`) until the challenge pre-flight is done. The wrapper also
pins the Trackio alert contract the pack verified: the message goes in `text=` (not
`message=`), and levels are INFO / WARN / ERROR.

`trackio` is injectable so the unit tests need neither the package nor a network call.
"""
from __future__ import annotations

_LEVELS = ("INFO", "WARN", "ERROR")


class LogbookRun:
    def __init__(self, project="icml2026-ntssm", name=None, config=None, trackio=None):
        self.project = project
        self.name = name
        self.config = config or {}
        self._trackio = trackio  # injected in tests; lazily imported otherwise

    @property
    def trackio(self):
        if self._trackio is None:
            import trackio  # local import so importing this module never requires trackio

            self._trackio = trackio
        return self._trackio

    def start(self):
        # local-only run: intentionally no space_id, so nothing is published to a Space.
        self.trackio.init(project=self.project, name=self.name, config=self.config)
        return self

    def log(self, metrics):
        self.trackio.log(metrics)

    def alert(self, title, text, level="WARN"):
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
        try:
            resolved = getattr(self.trackio.AlertLevel, level)
        except AttributeError as exc:  # installed trackio uses different level names
            raise ValueError(
                f"trackio.AlertLevel has no {level!r}; this trackio build may name levels "
                f"differently. Available: {[a for a in dir(self.trackio.AlertLevel) if not a.startswith('_')]}"
            ) from exc
        self.trackio.alert(title=title, text=text, level=resolved)

    def finish(self):
        self.trackio.finish()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.finish()
        return False
