"""Canonical SUMO physical settings shared by sessions and artifact identities."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .environment import EnvironmentContractError


def _value(simulation: Any, key: str, default: Any) -> Any:
    return simulation.get(key, default) if isinstance(simulation, Mapping) else getattr(simulation, key, default)


def physics_identity(simulation: Any) -> dict[str, Any]:
    """Describe physical/state semantics; snapshots without model retain micro."""
    model = _value(simulation, "traffic_model", "micro")
    if model not in ("micro", "meso"):
        raise EnvironmentContractError(f"unknown traffic_model {model!r}")
    if model == "micro":
        return {"traffic_model": "micro"}
    settings = {
        "traffic_model": "meso",
        "meso_junction_control": _value(simulation, "meso_junction_control", True),
        "meso_lane_queue": _value(simulation, "meso_lane_queue", False),
        "meso_multi_queue": _value(simulation, "meso_multi_queue", False),
        "meso_edge_length_m": _value(simulation, "meso_edge_length_m", 98.0),
        "meso_interpolate_positions": _value(simulation, "meso_interpolate_positions", True),
        "vehicle_position": "interpolated_external_edge",
        "junction_state": "zero_internal_distance",
    }
    for key, required in (("meso_junction_control", True), ("meso_lane_queue", False),
                          ("meso_multi_queue", False), ("meso_interpolate_positions", True)):
        if settings[key] is not required:
            raise EnvironmentContractError(f"meso contract requires {key}={required}")
    length = settings["meso_edge_length_m"]
    if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0:
        raise EnvironmentContractError("meso_edge_length_m must be finite and positive")
    settings["meso_edge_length_m"] = float(length)
    return settings


def sumo_model_args(simulation: Any) -> list[str]:
    """Return owned model flags, rejecting raw overrides of physical identity."""
    identity = physics_identity(simulation)
    for arg in _value(simulation, "sumo_extra_args", ()):
        option = str(arg).split("=", 1)[0]
        if option in ("--mesosim", "--no-internal-links") or option.startswith("--meso-"):
            raise EnvironmentContractError(f"SUMO extra argument {option!r} overrides the physical simulation contract")
    if identity["traffic_model"] == "micro":
        return []
    return ["--mesosim", "true", "--meso-junction-control", "true",
            "--meso-lane-queue", "false", "--meso-multi-queue", "false",
            "--meso-edgelength", str(identity["meso_edge_length_m"]),
            "--meso-interpolate-pos", "true"]
