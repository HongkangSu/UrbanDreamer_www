"""Physical-minute bounds for fixed-cohort, finite-horizon critics."""
from __future__ import annotations

import math
from numbers import Real

import torch


def remaining_cost_budget(timestamp: float, deadline: float, active_count: int,
                          started_count: int) -> float:
    """Cost if every currently active vehicle remains until the deadline.

    The denominator is the rollout seed's cumulative started count. Only
    floating point endpoint roundoff (1e-7 seconds) may cross the deadline.
    """
    for name, value in (("timestamp", timestamp), ("deadline", deadline),
                        ("active_count", active_count), ("started_count", started_count)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and numeric")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if int(active_count) != active_count or int(started_count) != started_count:
        raise ValueError("vehicle counts must be integers")
    if active_count > started_count:
        raise ValueError("active_count must not exceed started_count")
    remaining = float(deadline) - float(timestamp)
    if remaining < -1e-7:
        raise ValueError("timestamp exceeds deadline")
    if active_count == 0:
        return 0.0
    budget = max(remaining, 0.) * (float(active_count) / float(started_count)) / 60.
    if not math.isfinite(budget):
        raise ValueError("remaining cost budget must be finite")
    return budget


def bounded_physical_return(raw: torch.Tensor, budget: torch.Tensor | None) -> torch.Tensor:
    """Convert a learned cost fraction into concrete FP32 return units."""
    if (not isinstance(budget, torch.Tensor) or budget.dtype != torch.float32
            or budget.shape != (raw.shape[0],)):
        raise ValueError("budget must be an FP32 tensor of shape (batch,)")
    if budget.requires_grad or budget.device != raw.device:
        raise ValueError("budget must be detached and on the critic device")
    if not bool(torch.isfinite(budget).all()) or bool((budget < 0).any()):
        raise ValueError("budget must be finite and nonnegative")
    return -budget[:, None] * torch.sigmoid(raw.float())


def validate_bounded_contract(module, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
    """Reject accidental reinterpretation of legacy weights even with strict=False."""
    key = prefix + '_bounded_cohort_contract'
    contract = state_dict.get(key)
    if contract is None or not torch.equal(contract.cpu(), torch.tensor(1, dtype=torch.int64)):
        error_msgs.append(f"{prefix}bounded critic requires bounded_cohort_v1 weights; "
                          "initialize fresh critics for an explicit migration")
