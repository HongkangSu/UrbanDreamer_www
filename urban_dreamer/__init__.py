"""Minimal online world-model training mainline."""

from .workflow.config import load_config
from .workflow.pipeline import run_mainline

__all__ = ["load_config", "run_mainline"]
