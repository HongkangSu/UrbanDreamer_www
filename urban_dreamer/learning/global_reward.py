"""Causal finite-horizon mean-time accounting shared by all routing methods."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np

REWARD_CONTRACT = 'causal_cumulative_mean_global_v1'
REWARD_SCALE = 1. / 60.
CONTEXT_DIM = 4


def observed_mean_seconds(cumulative_seconds: float, started_count: int) -> float:
    if not math.isfinite(cumulative_seconds) or cumulative_seconds < 0:
        raise ValueError('cumulative vehicle seconds must be finite and nonnegative')
    if isinstance(started_count, bool) or not isinstance(started_count, Integral) or started_count < 0:
        raise ValueError('started count must be a nonnegative integer')
    return float(cumulative_seconds) / max(int(started_count), 1)


def causal_context(timestamp: float, deadline: float, active_count: int,
                   started_count: int, cumulative_seconds: float) -> np.ndarray:
    """Invertible cumulative statistics; no future population or demand law."""
    observed_mean_seconds(cumulative_seconds, started_count)
    if not math.isfinite(timestamp) or timestamp < 0 or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('context requires nonnegative time and a positive finite horizon')
    if active_count < 0 or active_count > started_count:
        raise ValueError('active count must lie within the already-started population')
    return np.asarray([min(timestamp / deadline, 1.), active_count / max(started_count, 1),
        math.log1p(cumulative_seconds / 60.), math.log1p(started_count)], dtype=np.float32)


@dataclass
class RewardAccounting:
    started_count: int = 0
    cumulative_vehicle_seconds: float = 0.

    def __post_init__(self):
        observed_mean_seconds(self.cumulative_vehicle_seconds, self.started_count)

    @property
    def mean_seconds(self) -> float:
        return observed_mean_seconds(self.cumulative_vehicle_seconds, self.started_count)

    def update(self, active_vehicle_seconds: float, started_count: int) -> float:
        if not math.isfinite(active_vehicle_seconds) or active_vehicle_seconds < 0:
            raise ValueError('interval vehicle seconds must be finite and nonnegative')
        if started_count < self.started_count:
            raise ValueError('cumulative started count cannot decrease')
        old = self.mean_seconds
        new_seconds = self.cumulative_vehicle_seconds + float(active_vehicle_seconds)
        new = observed_mean_seconds(new_seconds, started_count)
        self.started_count = int(started_count)
        self.cumulative_vehicle_seconds = new_seconds
        return -REWARD_SCALE * (new - old)


def started_count_from_state(state: dict, *, deadline: float | None = None) -> int:
    if 'started_vehicle_count' in state:
        return int(state['started_vehicle_count'])
    rows = state.get('vehicles', ())
    starts = 0
    for row in rows:
        if not row.get('departed', False):
            continue
        timestamp = row.get('actual_departure_time', row.get('departure_time'))
        if timestamp is None or deadline is None or float(timestamp) < deadline:
            starts += 1
    # Lightweight test/session adapters may only expose public population counts.
    if not rows:
        starts = int(state.get('active_vehicle_count', 0)) + int(state.get('arrived_vehicle_count', 0))
    return starts
