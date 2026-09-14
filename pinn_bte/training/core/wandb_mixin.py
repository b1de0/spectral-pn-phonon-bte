"""Inert experiment-tracking mixin."""
from __future__ import annotations

import contextlib
from typing import Optional


class WandbMixin:
    """No-op tracking mixin preserving the trainer-facing API."""

    _wandb_run_active: bool = False
    _wandb_enabled: bool = False

    def __init_wandb__(self, *args, **kwargs) -> None:
        pass

    @contextlib.contextmanager
    def wandb_run(self, run_name: Optional[str] = None):
        yield None

    def log_epoch_metrics(self, *args, **kwargs) -> None:
        pass

    def log_metric(self, name: str, value: float,
                   step: Optional[int] = None) -> None:
        pass

    def log_metrics(self, *args, **kwargs) -> None:
        pass

    def log_training_artifacts(self) -> None:
        pass

    def set_wandb_tag(self, key: str, value: str) -> None:
        pass

    def alert_training_complete(self, *args, **kwargs) -> None:
        pass

    def alert_training_failed(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def disable_wandb(cls) -> None:
        cls._wandb_enabled = False

    @classmethod
    def enable_wandb(cls) -> None:
        cls._wandb_enabled = False  # tracking is inert in this release
