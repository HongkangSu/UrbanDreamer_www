from __future__ import annotations

import math
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .topology import NextHopCandidates
from ..learning.global_reward import RewardAccounting

__all__ = ["NextHopDecision", "RouteDecision", "SumoSession", "EnvironmentContractError",
           "SessionClosedError", "WindowResult", "build_session_factory", "safe_microstep_seconds", "to_plain_state"]


class EnvironmentContractError(ValueError):
    """An invalid or unobserved event makes the entire episode unusable."""


class SessionClosedError(RuntimeError):
    pass


def to_plain_state(value: Any) -> Any:
    # Almost every archived vehicle field is already a native immutable scalar.
    # Exact types keep dataclass/subclass and NumPy conversion precedence intact.
    value_type = type(value)
    if value is None or value_type in (str, int, float, bool):
        return value
    if value_type is dict:
        return {str(key): to_plain_state(item) for key, item in value.items()}
    if value_type in (list, tuple, set, frozenset):
        return [to_plain_state(item) for item in value]
    if is_dataclass(value):
        return {item.name: to_plain_state(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_plain_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_plain_state(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value)!r}")


@dataclass(frozen=True)
class NextHopDecision:
    vehicle_id: str
    current_edge: int
    next_edge: int
    decision_time: float


@dataclass(frozen=True)
class RouteDecision(NextHopDecision):
    """A selected complete suffix; next_edge retains its physical turn meaning."""

    route: tuple[int, ...]


@dataclass(frozen=True)
class WindowResult:
    decisions: tuple[NextHopDecision, ...]
    action_counts: np.ndarray
    departure_counts: np.ndarray
    active_vehicle_seconds: float
    next_state: dict[str, Any]
    start_time: float
    end_time: float
    conditioning_decisions: tuple[NextHopDecision, ...] | None = None


def safe_microstep_seconds(lengths: Sequence[float], max_speed: float) -> float:
    values = tuple(float(length) for length in lengths)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise EnvironmentContractError("passenger edge lengths must be finite and positive")
    if not math.isfinite(max_speed) or max_speed <= 0:
        raise EnvironmentContractError("vehicle speed limit must be finite and positive")
    bound_ms = min(1000, math.floor(min(values) / (2 * max_speed) * 1000 + 1e-9))
    divisors = [value for value in range(1, bound_ms + 1) if 1000 % value == 0]
    if not divisors:
        raise EnvironmentContractError("network requires a step below SUMO's one millisecond minimum")
    return max(divisors) / 1000


class SumoSession:
    """Direct TraCI control with external-edge decisions and fixed 30-second windows."""

    def __init__(self, config: Any, prepared_inputs: Any, topology: Any, episode_index: int = 0):
        self.config, self.prepared_inputs, self.topology = config, prepared_inputs, topology
        self.network, self.candidates = topology.network, NextHopCandidates(topology.network)
        self.episode_index = episode_index
        self.max_speed = float(config.simulation.vehicle_max_speed_mps)
        self.microstep_seconds = safe_microstep_seconds(
            tuple(self.network.external_lane_lengths_m.values()) or self.network.lengths_m[self.network.passenger_allowed], self.max_speed)
        self._step_ms = round(self.microstep_seconds * 1000)
        self._connection = None
        self._closed = False
        self._invalid = False
        self._time_ms = 0
        self._vehicles: dict[str, dict[str, Any]] = {}
        self.travel_times: dict[str, float] = {}
        self.reward_accounting = RewardAccounting()
        self.metrics = {"microstep_seconds": self.microstep_seconds, "microsteps": 0,
                        "events": 0, "wall_seconds": 0., "active_vehicle_seconds": 0.}
        self._incident_lanes: dict[str, float] = {}
        self._incident_active = False
        self._active: set[str] = set()
        self._pending: set[str] = set()

    @property
    def step_length_seconds(self) -> float:
        return self.microstep_seconds

    @property
    def microstep_count(self) -> int:
        return self.metrics["microsteps"]

    @property
    def done(self) -> bool:
        return self._connection is not None and len(self.travel_times) == len(self._vehicles)

    @property
    def active_vehicle_count(self) -> int:
        horizon = getattr(self, 'reward_horizon_seconds', math.inf)
        if self._time_ms / 1000. < horizon:
            return len(self._active)
        return sum(self._vehicles[key]['actual_departure_time'] < horizon for key in self._active)

    @property
    def started_vehicle_count(self) -> int:
        accounting = getattr(self, 'reward_accounting', None)
        if accounting is not None:
            return accounting.started_count
        return sum(row.get('departed', False) for row in self._vehicles.values())

    @property
    def cumulative_vehicle_seconds(self) -> float:
        accounting = getattr(self, 'reward_accounting', None)
        return accounting.cumulative_vehicle_seconds if accounting is not None else self.metrics['active_vehicle_seconds']

    def reset(self) -> dict[str, Any]:
        if self._closed or self._connection is not None:
            raise SessionClosedError("a SUMO session can only be reset once")
        import traci
        import traci.constants as tc
        self._tc = tc
        binary = shutil.which("sumo")
        if binary is None:
            import sys
            candidate = Path(sys.executable).parent / "sumo"
            if not candidate.exists():
                raise EnvironmentContractError("SUMO executable is not available")
            binary = str(candidate)
        args = list(self.config.simulation.sumo_extra_args)
        # These overrides invalidate event visibility or inject uncontrolled vehicles.
        forbidden = ("--step-length", "--route-files", "--additional-files", "--net-file", "--begin", "--end", "--scale", "--device.rerouting", "--default.speeddev")
        if any(any(arg == key or arg.startswith(key + "=") or (key == "--device.rerouting" and arg.startswith(key)) for key in forbidden) for arg in args):
            raise EnvironmentContractError("SUMO extra arguments override the controlled simulation contract")
        label = "urban-dreamer-" + uuid.uuid4().hex
        traci.start([binary, "--net-file", str(self.prepared_inputs.network_xml),
                     "--step-length", str(self.microstep_seconds), "--seed", str(self.config.run.seed + self.episode_index),
                     "--no-step-log", "true", "--duration-log.disable", "true", "--time-to-teleport", "-1", *args], label=label)
        self._connection = traci.getConnection(label)
        self._started_at = time.monotonic()
        try:
            self._connection.vehicletype.copy("DEFAULT_VEHTYPE", "urban_cav")
            self._connection.vehicletype.setMaxSpeed("urban_cav", self.max_speed)
            self._connection.vehicletype.setSpeedFactor("urban_cav", 1.)
            self._connection.vehicletype.setSpeedDeviation("urban_cav", 0.)
            for index, demand in enumerate(sorted(self.topology.vehicles, key=lambda vehicle: (vehicle.departure_second, vehicle.vehicle_id))):
                origin = self.network.edge_to_index[self.topology.origins[demand.origin_index]]
                destination = self.network.edge_to_index[self.topology.destinations[demand.destination_index]]
                choices = self.candidates.candidates(origin, destination)
                if choices:
                    costs = self.candidates.costs(destination)
                    initial = min(choices, key=lambda edge: (self.network.free_flow_travel_seconds[edge] + costs[edge], edge))
                    route = self.candidates.temporary_route(origin, initial, destination)
                else:
                    route = (origin,)
                route_id = f"urban_initial_{index}"
                self._connection.route.add(route_id, [self.network.edge_ids[edge] for edge in route])
                self._connection.vehicle.add(demand.vehicle_id, route_id, typeID="urban_cav",
                                             depart=str(demand.departure_second), departPos="0", departSpeed="0")
                self._vehicles[demand.vehicle_id] = {
                    "vehicle_id": demand.vehicle_id, "current_edge": origin, "destination_edge": destination,
                    "departure_time": float(demand.departure_second), "actual_departure_time": None,
                    "observed_departure_time": None, "departed": False, "arrived": False,
                    "remaining_distance": float(self.network.lengths_m[origin]), "committed_next_edge": None,
                    "pending_decision": False, "in_junction": False, "junction_remaining_distance": 0.,
                    "junction_speed": 0., "junction_connection": None,
                }
            for edge in str(self.config.simulation.incident_edge).split(","):
                if not edge.strip():
                    continue
                if edge not in self.network.edge_to_index:
                    raise EnvironmentContractError(f"unknown incident edge {edge!r}")
                for lane_index in range(self._connection.edge.getLaneNumber(edge)):
                    lane_id = f"{edge}_{lane_index}"
                    self._incident_lanes[lane_id] = self._connection.lane.getMaxSpeed(lane_id)
            return self._state()
        except BaseException:
            self.close()
            raise

    def _state(self) -> dict[str, Any]:
        edge_state = {}
        if not self.done:
            for edge in self.network.edge_ids:
                edge_state[edge] = {
                    # SUMO reports percentages even below one percent.
                    "occupancy": self._connection.edge.getLastStepOccupancy(edge) / 100.,
                    "mean_speed": self._connection.edge.getLastStepMeanSpeed(edge),
                    "vehicle_number": self._connection.edge.getLastStepVehicleNumber(edge),
                    "halting_number": self._connection.edge.getLastStepHaltingNumber(edge),
                }
        return {"time": self._time_ms / 1000., "flow": self.network.extract_flow(edge_state),
                "vehicles": [dict(self._vehicles[key]) for key in sorted(self._vehicles)],
                "active_vehicle_count": self.active_vehicle_count,
                "started_vehicle_count": self.started_vehicle_count,
                "cumulative_vehicle_seconds": self.cumulative_vehicle_seconds,
                "arrived_vehicle_count": len(self.travel_times), "done": self.done}

    def _apply_decisions(self, callback: Callable, flow: np.ndarray) -> list[NextHopDecision]:
        pending = [dict(self._vehicles[key]) for key in sorted(self._pending)]
        if not pending:
            return []
        decisions = list(callback(pending, flow, self._time_ms / 1000.))
        expected = [row["vehicle_id"] for row in pending]
        if sorted(decision.vehicle_id for decision in decisions) != expected:
            raise EnvironmentContractError("policy must return exactly one action per pending vehicle")
        successful = []
        for decision in sorted(decisions, key=lambda action: action.vehicle_id):
            vehicle = self._vehicles[decision.vehicle_id]
            if decision.current_edge != vehicle["current_edge"] or abs(decision.decision_time - self._time_ms / 1000.) > 1e-8:
                raise EnvironmentContractError("decision uses stale edge or time")
            route = self.candidates.temporary_route(decision.current_edge, decision.next_edge, vehicle["destination_edge"])
            try:
                self._connection.vehicle.setRoute(decision.vehicle_id, [self.network.edge_ids[edge] for edge in route])
            except Exception as exc:
                raise EnvironmentContractError(f"next-hop execution failed for {decision.vehicle_id}: {exc}") from exc
            vehicle["committed_next_edge"] = decision.next_edge
            vehicle["pending_decision"] = False
            self._pending.discard(decision.vehicle_id)
            successful.append(decision)
            self.metrics["events"] += 1
        return successful

    def _observe(self, departure_counts: np.ndarray) -> None:
        connection, tc = self._connection, self._tc
        now = self._time_ms / 1000.
        if connection.simulation.getStartingTeleportIDList() or connection.simulation.getEndingTeleportIDList():
            raise EnvironmentContractError("SUMO teleported a vehicle; episode is invalid")
        for vehicle_id in connection.simulation.getDepartedIDList():
            if vehicle_id not in self._vehicles:
                raise EnvironmentContractError(f"uncontrolled vehicle departed: {vehicle_id}")
            row = self._vehicles[vehicle_id]
            row["departed"] = True
            self._active.add(vehicle_id)
            row["actual_departure_time"] = float(connection.vehicle.getDeparture(vehicle_id))
            accounting = getattr(self, 'reward_accounting', None)
            if accounting is not None and row['actual_departure_time'] < getattr(self, 'reward_horizon_seconds', math.inf):
                accounting.update(0., accounting.started_count + 1)
            row["departure_time"] = row["actual_departure_time"]
            row["observed_departure_time"] = now
            row["pending_decision"] = row["current_edge"] != row["destination_edge"]
            if row["pending_decision"]:
                self._pending.add(vehicle_id)
            departure_counts[row["current_edge"]] += 1
            connection.vehicle.subscribe(vehicle_id, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID))
        arrived = set(connection.simulation.getArrivedIDList())
        subscriptions = connection.vehicle.getAllSubscriptionResults()
        for vehicle_id in sorted(self._active):
            row = self._vehicles[vehicle_id]
            if vehicle_id in arrived:
                if row["current_edge"] != row["destination_edge"]:
                    raise EnvironmentContractError(f"vehicle {vehicle_id} arrived before its destination edge was observed")
                row["arrived"], row["pending_decision"], row["remaining_distance"] = True, False, 0.
                self._active.discard(vehicle_id)
                self._pending.discard(vehicle_id)
                row["arrival_time"] = now
                self.travel_times[vehicle_id] = now - row["actual_departure_time"]
                continue
            state = subscriptions.get(vehicle_id)
            if state is None:
                raise EnvironmentContractError(f"active vehicle {vehicle_id} disappeared without arrival")
            edge_id = state[tc.VAR_ROAD_ID]
            position = float(state[tc.VAR_LANEPOSITION])
            if edge_id.startswith(":"):
                if row["committed_next_edge"] is None:
                    raise EnvironmentContractError(f"vehicle {vehicle_id} entered junction without a committed action")
                lane = state[tc.VAR_LANE_ID]
                row["in_junction"] = True
                row["remaining_distance"] = 0.
                row["junction_connection"] = (row["current_edge"], row["committed_next_edge"])
                row["junction_speed"] = min(self.max_speed, self.network.internal_lane_speeds_mps.get(lane, self.max_speed))
                suffix = self.network.internal_lane_remaining_lengths_m.get(lane)
                if suffix is None:
                    raise EnvironmentContractError(f"unregistered internal lane {lane!r}")
                row["junction_remaining_distance"] = max(0., suffix - position)
                row["junction_lane"], row["junction_position"] = lane, position
                continue
            if edge_id not in self.network.edge_to_index:
                raise EnvironmentContractError(f"unknown physical road {edge_id!r} for {vehicle_id}")
            edge = self.network.edge_to_index[edge_id]
            if edge != row["current_edge"] or row["in_junction"]:
                if edge != row["committed_next_edge"]:
                    raise EnvironmentContractError(f"vehicle {vehicle_id} missed committed next edge {row['committed_next_edge']}; observed {edge}")
                row["current_edge"], row["committed_next_edge"] = edge, None
                row["pending_decision"] = edge != row["destination_edge"]
                if row["pending_decision"]:
                    self._pending.add(vehicle_id)
            row["in_junction"] = False
            row["junction_remaining_distance"] = 0.
            row["junction_connection"] = None
            row["remaining_distance"] = max(0., self.network.external_lane_lengths_m.get(state[tc.VAR_LANE_ID], float(self.network.lengths_m[edge])) - position)

    def _apply_incident(self) -> None:
        config = self.config.simulation
        active = config.incident_start_seconds <= self._time_ms / 1000. < config.incident_end_seconds
        if active != self._incident_active:
            for lane, speed in self._incident_lanes.items():
                self._connection.lane.setMaxSpeed(lane, config.incident_speed_mps if active else speed)
            self._incident_active = active

    def advance_window(self, callback: Callable) -> WindowResult:
        if self._closed or self._connection is None:
            raise SessionClosedError("SUMO session is not open")
        if self._invalid:
            raise EnvironmentContractError("invalid episode cannot be continued or trained")
        start = self._time_ms
        end = min(start + 30000, round(getattr(self, 'reward_horizon_seconds', (start + 30000) / 1000.) * 1000))
        frozen_flow = self._state()["flow"]
        decisions, departures = [], np.zeros(self.network.edge_count, dtype=np.float32)
        active_seconds = 0.
        try:
            while self._time_ms < end:
                if time.monotonic() - self._started_at > self.config.simulation.episode_wall_timeout_seconds:
                    raise EnvironmentContractError("episode exceeded wall timeout")
                if self.done:
                    self._time_ms = end
                    break
                decisions.extend(self._apply_decisions(callback, frozen_flow))
                self._apply_incident()
                previous_time = self._time_ms / 1000.
                active_before = len(self._active)
                self._connection.simulationStep()
                self._time_ms += self._step_ms
                self.metrics["microsteps"] += 1
                self._observe(departures)
                step_seconds = active_before * self.microstep_seconds
                for vehicle_id in self._connection.simulation.getDepartedIDList():
                    row = self._vehicles[vehicle_id]
                    step_seconds += max(0., self._time_ms / 1000. - max(previous_time, row["actual_departure_time"]))
                active_seconds += step_seconds
                accounting = getattr(self, 'reward_accounting', None)
                if accounting is not None:
                    accounting.update(step_seconds, accounting.started_count)
            counts = np.zeros(self.network.edge_count, dtype=np.float32)
            for decision in decisions:
                counts[decision.next_edge] += 1
            self.metrics["active_vehicle_seconds"] += active_seconds
            self.metrics["wall_seconds"] = time.monotonic() - self._started_at
            return WindowResult(tuple(decisions), counts, departures, active_seconds, self._state(), start / 1000., end / 1000.)
        except BaseException:
            self._invalid = True
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def build_session_factory(*, config: Any, prepared_inputs: Any, topology: Any) -> Callable[..., SumoSession]:
    def factory(*, episode_index: int = 0) -> SumoSession:
        return SumoSession(config, prepared_inputs, topology, episode_index)
    return factory
