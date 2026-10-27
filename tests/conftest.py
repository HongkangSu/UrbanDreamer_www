"""Keep CPU regression checks deterministic and independent of the parent tree."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(1)
