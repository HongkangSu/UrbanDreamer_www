"""Event-driven SUMO control: fixed-length steps and fork-only decisions.

Unlike :class:`urban_dreamer.traffic.environment.SumoSession`, which forces a
decision on every traversed edge and microsteps so no edge can be crossed
unobserved, this session advances SUMO with a fixed step (default 1.0 s) and
asks the policy only when a vehicle enters an edge whose
destination-conditioned candidate set has more than one entry. Vehicles always
carry complete routes to their destination (``NextHopCandidates.full_route``),
so an edge crossed inside a single step — or never observed at all — is simply
followed from the existing route instead of invalidating the episode.
With ``forbid_revisit=True``, full routes and policy candidates exclude every
previously entered external edge; route indices also audit unobserved crossings.
``current_to_dest`` instead selects complete routes at 30-second boundaries.
Observed forks along those routes retain the same world-model turn conditions.
"""
from __future__ import annotations

import math
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .environment import (EnvironmentContractError, NextHopDecision, SessionClosedError,
                          WindowResult)
from .topology import NextHopCandidates
from .simulation_settings import physics_identity, sumo_model_args
from ..learning.global_reward import RewardAccounting

__all__ = ["SumoEventSession"]

WINDOW_SECONDS = 30.


def _edge_state_vars() -> tuple[int, ...]:
    import traci.constants as tc
    return (tc.LAST_STEP_OCCUPANCY, tc.LAST_STEP_MEAN_SPEED,
            tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_HALTING_NUMBER)


_EDGE_STATE_VARS = _edge_state_vars()


class SumoEventSession:
    """Fixed-step SUMO control with fork or periodic complete-route decisions."""

    def __init__(self, config: Any, prepared_inputs: Any, topology: Any, episode_index: int = 0,
                 *, step_seconds: float = 1.0, backend: str | None = None, forbid_revisit: bool = False):
        self.config, self.prepared_inputs, self.topology = config, prepared_inputs, topology
        self.physics = physics_identity(config.simulation)
        self.traffic_model = self.physics["traffic_model"]
        self._model_args = sumo_model_args(config.simulation)
        self.forbid_revisit = bool(forbid_revisit)
        self.routing_mode = getattr(config.simulation, "routing_mode", "next_hop")
        if self.routing_mode not in ("next_hop", "current_to_dest"):
            raise EnvironmentContractError(f"unknown routing mode {self.routing_mode!r}")
        candidate_type = NextHopCandidates
        if self.routing_mode == "current_to_dest":
            from .route_candidates import RouteCandidates
            candidate_type = RouteCandidates
        self.network, self.candidates = topology.network, candidate_type(
            topology.network, forbid_revisit=self.forbid_revisit, traffic_model=self.traffic_model)
        self._commanded_routes: dict[str, tuple[int, ...]] = {}
        self._route_cursors: dict[str, int] = {}
        # SUMO owns the accepted route, which may retain a traversed prefix.
        # Only departure and our setRoute calls can change it; route indices
        # remain live observations on every physical step.
        self._actual_routes: dict[str, tuple[str, ...]] = {}
        self._route_commitment_cache: dict[str, tuple] = {}
        self.episode_index = episode_index
        if backend is None:
            backend = getattr(config.simulation, "backend", "traci")
        if backend not in ('traci', 'libsumo'):
            raise EnvironmentContractError(f"unknown SUMO backend {backend!r}")
        self.backend = backend
        # In-process meso getters avoid rebuilding subscription result maps.
        # Read the same road, interpolated position and route index every step.
        self._direct_vehicle_observation = self.traffic_model == 'meso' and backend == 'libsumo'
        if self.traffic_model == 'meso' and step_seconds != 1.0:
            raise EnvironmentContractError('meso event sessions require one-second observations')
        if not math.isfinite(step_seconds) or step_seconds <= 0:
            raise EnvironmentContractError("step_seconds must be finite and positive")
        self._step_ms = round(step_seconds * 1000)
        if self._step_ms < 1 or abs(step_seconds * 1000 - self._step_ms) > 1e-6:
            raise EnvironmentContractError("step_seconds must be a whole number of milliseconds")
        if 30000 % self._step_ms:
            raise EnvironmentContractError("the 30-second window must be a whole number of steps")
        self.microstep_seconds = float(step_seconds)
        self.max_speed = float(config.simulation.vehicle_max_speed_mps)
        self._connection = None
        self._closed = False
        self._invalid = False
        self._time_ms = 0
        self._vehicles: dict[str, dict[str, Any]] = {}
        self.travel_times: dict[str, float] = {}
        self.reward_accounting = RewardAccounting()
        self.metrics = {"microstep_seconds": self.microstep_seconds, "microsteps": 0,
                        "events": 0, "wall_seconds": 0., "active_vehicle_seconds": 0.}
        if self.forbid_revisit:
            self.metrics.update(physical_edge_entries=0, physical_revisit_count=0)
        self._incident_lanes: dict[str, float] = {}
        self._incident_edges: dict[str, float] = {}
        self._incident_active = False
        self._active: set[str] = set()
        self._pending: set[str] = set()
        self._flow_cache: tuple[int, np.ndarray] | None = None

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

    def _legal_candidates(self, edge: int, destination: int, visited_edges=()) -> tuple[int, ...]:
        if getattr(self, "forbid_revisit", False):
            return self.candidates.legal_candidates(edge, destination, visited_edges)
        return self.candidates.candidates(edge, destination)

    def _needs_decision(self, edge: int, destination: int, visited_edges=()) -> bool:
        if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
            return False
        return self._needs_turn(edge, destination, visited_edges)

    def _needs_turn(self, edge: int, destination: int, visited_edges=()) -> bool:
        return edge != destination and len(self._legal_candidates(edge, destination, visited_edges)) > 1

    def reset(self) -> dict[str, Any]:
        if self._closed or self._connection is not None:
            raise SessionClosedError("a SUMO session can only be reset once")
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
        start_args = [binary, "--net-file", str(self.prepared_inputs.network_xml),
                      "--step-length", str(self.microstep_seconds), "--seed", str(self.config.run.seed + self.episode_index),
                      "--no-step-log", "true", "--duration-log.disable", "true", "--time-to-teleport", "-1",
                      *self._model_args, *args]
        if self.backend == 'libsumo':
            # libsumo embeds SUMO in-process: no socket, no connection labels,
            # one simulation at a time per Python process.
            import libsumo
            libsumo.start(start_args)
            self._connection = libsumo
        else:
            import traci
            label = "urban-dreamer-event-" + uuid.uuid4().hex
            traci.start(start_args, label=label)
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
                choices = self._legal_candidates(origin, destination)
                if choices:
                    costs = self.candidates.costs(destination)
                    initial = min(choices, key=lambda edge: (self.network.free_flow_travel_seconds[edge] + costs[edge], edge))
                    route = self.candidates.full_route(origin, initial, destination)
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
                if self.forbid_revisit:
                    self._vehicles[demand.vehicle_id]["visited_edges"] = ()
                    self._commanded_routes[demand.vehicle_id] = tuple(route)
                    self._route_cursors[demand.vehicle_id] = -1
                if self.routing_mode == "current_to_dest":
                    self._vehicles[demand.vehicle_id].update(committed_route=tuple(route), pending_turn=False)
            for edge in str(self.config.simulation.incident_edge).split(","):
                if not edge.strip():
                    continue
                if edge not in self.network.edge_to_index:
                    raise EnvironmentContractError(f"unknown incident edge {edge!r}")
                for lane_index in range(self._connection.edge.getLaneNumber(edge)):
                    lane_id = f"{edge}_{lane_index}"
                    self._incident_lanes[lane_id] = self._connection.lane.getMaxSpeed(lane_id)
                if self.traffic_model == "meso":
                    # Mesoscopic segments are updated by the edge API. Lane
                    # speed changes alone do not update their physical dynamics.
                    self._incident_edges[edge] = self._connection.lane.getMaxSpeed(f"{edge}_0")
            # Socket TraCI benefits from a batched pull. In-process libsumo
            # reads only at window boundaries, avoiding subscription updates
            # for every edge on every simulation step.
            if self.backend == 'traci':
                for edge in self.network.edge_ids:
                    self._connection.edge.subscribe(edge, _EDGE_STATE_VARS)
            return self._state()
        except BaseException:
            self.close()
            raise

    def _flow(self) -> np.ndarray:
        """Read boundary traffic once per simulation time using the backend's API."""
        now = self._time_ms
        if self._flow_cache is not None and self._flow_cache[0] == now:
            return self._flow_cache[1]
        tc = self._tc
        edge_state = {}
        if not self.done and self.backend == 'libsumo':
            edge_api = self._connection.edge
            for edge in self.network.edge_ids:
                edge_state[edge] = {
                    "occupancy": float(edge_api.getLastStepOccupancy(edge)) / 100.,
                    "mean_speed": float(edge_api.getLastStepMeanSpeed(edge)),
                    "vehicle_number": float(edge_api.getLastStepVehicleNumber(edge)),
                    "halting_number": float(edge_api.getLastStepHaltingNumber(edge)),
                }
        elif not self.done:
            results = self._connection.edge.getAllSubscriptionResults()
            for edge in self.network.edge_ids:
                values = results.get(edge)
                if values is None:
                    continue
                edge_state[edge] = {
                    # SUMO reports percentages even below one percent.
                    "occupancy": float(values[tc.LAST_STEP_OCCUPANCY]) / 100.,
                    "mean_speed": float(values[tc.LAST_STEP_MEAN_SPEED]),
                    "vehicle_number": float(values[tc.LAST_STEP_VEHICLE_NUMBER]),
                    "halting_number": float(values[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]),
                }
        flow = self.network.extract_flow(edge_state)
        self._flow_cache = (now, flow)
        return flow

    def _state(self) -> dict[str, Any]:
        return {"time": self._time_ms / 1000., "flow": self._flow(),
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
            if decision.next_edge not in self._legal_candidates(decision.current_edge, vehicle["destination_edge"], vehicle.get("visited_edges", ())):
                raise EnvironmentContractError(f"illegal next edge {decision.next_edge} from {decision.current_edge}")
            if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                route = self.candidates.decision_route(decision, vehicle["destination_edge"],
                    visited_edges=vehicle.get("visited_edges", ()))
            else:
                route = self.candidates.full_route(decision.current_edge, decision.next_edge, vehicle["destination_edge"],
                                                   visited_edges=vehicle.get("visited_edges", ()))
            try:
                self._connection.vehicle.setRoute(decision.vehicle_id, [self.network.edge_ids[edge] for edge in route])
            except Exception as exc:
                raise EnvironmentContractError(f"next-hop execution failed for {decision.vehicle_id}: {exc}") from exc
            actual_route_ids = None
            if hasattr(self, "_actual_routes"):
                actual_route_ids = tuple(self._connection.vehicle.getRoute(decision.vehicle_id))
                self._actual_routes[decision.vehicle_id] = actual_route_ids
            getattr(self, "_route_commitment_cache", {}).pop(decision.vehicle_id, None)
            if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                if actual_route_ids is None:
                    actual_route_ids = tuple(self._connection.vehicle.getRoute(decision.vehicle_id))
                actual_index = int(self._connection.vehicle.getRouteIndex(decision.vehicle_id))
                actual_route = tuple(self.network.edge_to_index[edge] for edge in actual_route_ids)
                if not 0 <= actual_index < len(actual_route) or actual_route[actual_index:] != tuple(route):
                    raise EnvironmentContractError(f"vehicle {decision.vehicle_id} did not accept its selected route")
            if getattr(self, "forbid_revisit", False):
                # SUMO may retain the already traversed route prefix after
                # setRoute. Cache its actual index rather than assuming zero.
                api = self._connection.vehicle
                if actual_route_ids is None:
                    actual_route_ids = api.getRoute(decision.vehicle_id)
                actual_route = tuple(self.network.edge_to_index[edge] for edge in actual_route_ids)
                actual_index = int(api.getRouteIndex(decision.vehicle_id))
                if not 0 <= actual_index < len(actual_route) or actual_route[actual_index:] != tuple(route):
                    raise EnvironmentContractError(f"vehicle {decision.vehicle_id} did not accept its constrained route")
                self._commanded_routes[decision.vehicle_id] = actual_route
                self._route_cursors[decision.vehicle_id] = actual_index
            vehicle["committed_next_edge"] = decision.next_edge
            if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                vehicle["committed_route"] = tuple(route)
                vehicle["pending_turn"] = self._needs_turn(decision.current_edge, vehicle["destination_edge"],
                                                          vehicle.get("visited_edges", ()))
            vehicle["pending_decision"] = False
            self._pending.discard(decision.vehicle_id)
            successful.append(decision)
            self.metrics["events"] += 1
        return successful

    def _observe(self, departure_counts: np.ndarray) -> None:
        connection, tc = self._connection, self._tc
        direct = getattr(self, "_direct_vehicle_observation", False)
        meso = getattr(self, "traffic_model", "micro") == "meso"
        forbid_revisit = getattr(self, "forbid_revisit", False)
        vehicle_api = connection.vehicle
        edge_indices = self.network.edge_to_index
        now = self._time_ms / 1000.
        if connection.simulation.getStartingTeleportIDList() or connection.simulation.getEndingTeleportIDList():
            raise EnvironmentContractError("SUMO teleported a vehicle; episode is invalid")
        for vehicle_id in connection.simulation.getDepartedIDList():
            if vehicle_id not in self._vehicles:
                raise EnvironmentContractError(f"uncontrolled vehicle departed: {vehicle_id}")
            row = self._vehicles[vehicle_id]
            if hasattr(self, "_actual_routes"):
                self._actual_routes[vehicle_id] = tuple(vehicle_api.getRoute(vehicle_id))
            getattr(self, "_route_commitment_cache", {}).pop(vehicle_id, None)
            if forbid_revisit:
                self._record_route_progress(vehicle_id, 0)
            row["departed"] = True
            self._active.add(vehicle_id)
            row["actual_departure_time"] = float(vehicle_api.getDeparture(vehicle_id))
            accounting = getattr(self, 'reward_accounting', None)
            if accounting is not None and row['actual_departure_time'] < getattr(self, 'reward_horizon_seconds', math.inf):
                accounting.update(0., accounting.started_count + 1)
            row["departure_time"] = row["actual_departure_time"]
            row["observed_departure_time"] = now
            row["pending_decision"] = self._needs_decision(row["current_edge"], row["destination_edge"], row.get("visited_edges", ()))
            if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                row["pending_turn"] = self._needs_turn(row["current_edge"], row["destination_edge"], row.get("visited_edges", ()))
            if row["pending_decision"]:
                self._pending.add(vehicle_id)
            departure_counts[row["current_edge"]] += 1
            # MESO reports no microscopic lane. Its live route index replaces
            # that unused field without changing position interpolation cadence.
            if not direct:
                vehicle_api.subscribe(vehicle_id, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION,
                    tc.VAR_ROUTE_INDEX if meso else tc.VAR_LANE_ID))
        arrived = set(connection.simulation.getArrivedIDList())
        subscriptions = None if direct else vehicle_api.getAllSubscriptionResults()
        for vehicle_id in sorted(self._active):
            row = self._vehicles[vehicle_id]
            if vehicle_id in arrived:
                if forbid_revisit:
                    self._record_route_progress(vehicle_id, len(self._commanded_routes[vehicle_id]) - 1)
                    row["current_edge"] = row["destination_edge"]
                # Every assigned route ends at the destination edge, so arrival
                # from a reachable edge is consistent even when a fixed-length
                # step crossed the destination edge without observing it.
                if row["current_edge"] != row["destination_edge"] and not math.isfinite(
                        self.candidates.distances(row["destination_edge"])[row["current_edge"]]):
                    raise EnvironmentContractError(f"vehicle {vehicle_id} arrived from an edge that cannot reach its destination")
                if meso:
                    row["current_edge"] = row["destination_edge"]
                row["arrived"], row["pending_decision"], row["remaining_distance"] = True, False, 0.
                row["committed_next_edge"] = None
                if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                    row["committed_route"], row["pending_turn"] = (row["destination_edge"],), False
                row["in_junction"] = False
                row["junction_remaining_distance"] = 0.
                row["junction_connection"] = None
                self._active.discard(vehicle_id)
                self._pending.discard(vehicle_id)
                getattr(self, "_route_commitment_cache", {}).pop(vehicle_id, None)
                row["arrival_time"] = now
                self.travel_times[vehicle_id] = now - row["actual_departure_time"]
                continue
            if direct:
                try:
                    edge_id = vehicle_api.getRoadID(vehicle_id)
                    position = float(vehicle_api.getLanePosition(vehicle_id))
                    route_index = int(vehicle_api.getRouteIndex(vehicle_id))
                except Exception as exc:
                    raise EnvironmentContractError(f"active vehicle {vehicle_id} disappeared without arrival") from exc
            else:
                state = subscriptions.get(vehicle_id)
                if state is None:
                    raise EnvironmentContractError(f"active vehicle {vehicle_id} disappeared without arrival")
                edge_id = state[tc.VAR_ROAD_ID]
                position = float(state[tc.VAR_LANEPOSITION])
                route_index = (state.get(getattr(tc, "VAR_ROUTE_INDEX", None))
                               if meso else None)
            if edge_id.startswith(":"):
                if meso:
                    raise EnvironmentContractError(f"meso vehicle {vehicle_id} reported a micro internal junction")
                # On internal lanes SUMO's route index still identifies the
                # incoming external edge. It may differ from our last observed
                # edge when a fixed step skipped an entire short edge.
                source, successor = self._route_commitment(vehicle_id)
                if successor is None:
                    raise EnvironmentContractError(f"vehicle {vehicle_id} entered a junction without a route successor")
                row["current_edge"], row["committed_next_edge"] = source, successor
                row["pending_decision"] = False
                if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                    row["pending_turn"] = False
                self._pending.discard(vehicle_id)
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
            if edge_id not in edge_indices:
                raise EnvironmentContractError(f"unknown physical road {edge_id!r} for {vehicle_id}")
            edge = edge_indices[edge_id]
            commitment = None
            if forbid_revisit or meso or getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                commitment = self._route_commitment(vehicle_id, route_index=route_index)
                source, _ = commitment
                if source != edge:
                    raise EnvironmentContractError(f"vehicle {vehicle_id} route index disagrees with its physical road {edge}: source={source}, route={self._connection.vehicle.getRoute(vehicle_id)}, index={self._connection.vehicle.getRouteIndex(vehicle_id)}")
            if edge != row["current_edge"] or row["in_junction"]:
                # Only the edge the vehicle is observed on can produce an event;
                # edges skipped inside one step stay route-followed.
                row["current_edge"], row["committed_next_edge"] = edge, None
                row["pending_decision"] = self._needs_decision(edge, row["destination_edge"], row.get("visited_edges", ()))
                if getattr(self, "routing_mode", "next_hop") == "current_to_dest":
                    row["pending_turn"] = self._needs_turn(edge, row["destination_edge"], row.get("visited_edges", ()))
                if row["pending_decision"]:
                    self._pending.add(vehicle_id)
                else:
                    self._pending.discard(vehicle_id)
            if row["committed_next_edge"] is None and edge != row["destination_edge"]:
                source, successor = commitment if commitment is not None else self._route_commitment(vehicle_id, route_index=route_index)
                if source != edge:
                    raise EnvironmentContractError(f"vehicle {vehicle_id} route index disagrees with its physical road {edge}: source={source}, route={self._connection.vehicle.getRoute(vehicle_id)}, index={self._connection.vehicle.getRouteIndex(vehicle_id)}")
                # A complete SUMO route also commits automatic singleton and
                # skipped-edge movement. A pending fork may replace this route
                # at its next policy event; observing it creates no action.
                row["committed_next_edge"] = successor
            row["in_junction"] = False
            row["junction_remaining_distance"] = 0.
            row["junction_connection"] = None
            if meso:
                if not math.isfinite(position) or position < 0:
                    raise EnvironmentContractError(f"meso vehicle {vehicle_id} has invalid interpolated position {position}")
                # SUMO meso uses the edge (first lane) length for all combined
                # lane segments, rather than an actual microscopic lane.
                row["junction_speed"] = 0.
                length = self.network.external_lane_lengths_m.get(f"{edge_id}_0", float(self.network.lengths_m[edge]))
            else:
                length = self.network.external_lane_lengths_m.get(state[tc.VAR_LANE_ID], float(self.network.lengths_m[edge]))
            row["remaining_distance"] = max(0., length - position)

    def _record_route_progress(self, vehicle_id: str, index: int) -> None:
        """Consume every entered external edge, including edges skipped in a step."""
        route = self._commanded_routes[vehicle_id]
        cursor = self._route_cursors[vehicle_id]
        if not 0 <= index < len(route) or index < cursor:
            raise EnvironmentContractError(f"vehicle {vehicle_id} has an invalid route index {index}")
        if index == cursor:
            return
        row = self._vehicles[vehicle_id]
        history = row["visited_edges"]
        entered = route[cursor + 1:index + 1]
        seen = set(history)
        for edge in entered:
            if edge in seen:
                self.metrics["physical_revisit_count"] += 1
                raise EnvironmentContractError(f"vehicle {vehicle_id} revisited external edge {edge}")
            seen.add(edge)
        row["visited_edges"] = history + entered
        self.metrics["physical_edge_entries"] += len(entered)
        self._route_cursors[vehicle_id] = index

    def _route_commitment(self, vehicle_id: str, *, route_index: int | None = None) -> tuple[int, int | None]:
        """Read the actual external route pair, including while inside a junction."""
        api = self._connection.vehicle
        index = int(api.getRouteIndex(vehicle_id) if route_index is None else route_index)
        if getattr(self, "forbid_revisit", False):
            self._record_route_progress(vehicle_id, index)
            route = tuple(self.network.edge_ids[edge] for edge in self._commanded_routes[vehicle_id])
        else:
            route = getattr(self, "_actual_routes", {}).get(vehicle_id)
            if route is None:
                route = api.getRoute(vehicle_id)
        if not 0 <= index < len(route):
            raise EnvironmentContractError(f"vehicle {vehicle_id} has an invalid route index {index}")
        destination = self._vehicles[vehicle_id]["destination_edge"]
        route_mode = getattr(self, "routing_mode", "next_hop") == "current_to_dest"
        # Accepted routes are immutable; only our setRoute calls and departure
        # can replace them. Keep the route object alive as the identity key and
        # always use the live SUMO index. _observe still checks the physical road.
        cache = getattr(self, "_route_commitment_cache", None)
        cached = cache.get(vehicle_id) if cache is not None and route_mode else None
        if (cached is not None and cached[0] is route and cached[1] == index
                and cached[2] == destination):
            self._vehicles[vehicle_id]["committed_route"] = cached[5]
            return cached[3], cached[4]
        try:
            source = self.network.edge_to_index[route[index]]
            successor = (self.network.edge_to_index[route[index + 1]]
                         if index + 1 < len(route) else None)
        except KeyError as exc:
            raise EnvironmentContractError(f"vehicle {vehicle_id} route contains an unknown external edge") from exc
        if route[-1] != self.network.edge_ids[destination]:
            raise EnvironmentContractError(f"vehicle {vehicle_id} route does not end at its destination")
        if successor is None and source != destination:
            raise EnvironmentContractError(f"vehicle {vehicle_id} route ends before its destination")
        if successor is not None and successor not in self.network.adjacency_list[source]:
            raise EnvironmentContractError(f"vehicle {vehicle_id} route has an illegal successor")
        if route_mode:
            suffix = tuple(self.network.edge_to_index[edge] for edge in route[index:])
            self._vehicles[vehicle_id]["committed_route"] = suffix
            if isinstance(route, tuple):
                if cache is None:
                    cache = self._route_commitment_cache = {}
                cache[vehicle_id] = (route, index, destination, source, successor, suffix)
        return source, successor

    def _conditioning_turns(self) -> list[NextHopDecision]:
        """Expose only observed forks, using the route accepted before this step."""
        turns = []
        for vehicle_id in sorted(self._active):
            row = self._vehicles[vehicle_id]
            if not row.get("pending_turn", False):
                continue
            row["pending_turn"] = False
            if row["in_junction"] or row["current_edge"] == row["destination_edge"]:
                continue
            successor = row["committed_next_edge"]
            if successor is None:
                raise EnvironmentContractError(f"observed fork {vehicle_id} lacks an accepted route successor")
            turns.append(NextHopDecision(vehicle_id, row["current_edge"], successor, self._time_ms / 1000.))
        return turns

    def _apply_incident(self) -> None:
        config = self.config.simulation
        active = config.incident_start_seconds <= self._time_ms / 1000. < config.incident_end_seconds
        if active != self._incident_active:
            for edge, speed in self._incident_edges.items():
                self._connection.edge.setMaxSpeed(edge, config.incident_speed_mps if active else speed)
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
        frozen_flow = self._flow()
        decisions, departures = [], np.zeros(self.network.edge_count, dtype=np.float32)
        route_mode = getattr(self, "routing_mode", "next_hop") == "current_to_dest"
        conditioning = [] if route_mode else None
        active_seconds = 0.
        try:
            if route_mode:
                self._pending = {key for key in self._active if not self._vehicles[key]["in_junction"]
                                 and self._vehicles[key]["current_edge"] != self._vehicles[key]["destination_edge"]}
                for key in self._pending:
                    self._vehicles[key]["pending_decision"] = True
            while self._time_ms < end:
                if time.monotonic() - self._started_at > self.config.simulation.episode_wall_timeout_seconds:
                    raise EnvironmentContractError("episode exceeded wall timeout")
                if self.done:
                    self._time_ms = end
                    break
                if not route_mode or self._time_ms == start:
                    decisions.extend(self._apply_decisions(callback, frozen_flow))
                if route_mode:
                    conditioning.extend(self._conditioning_turns())
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
            for decision in conditioning if route_mode else decisions:
                counts[decision.next_edge] += 1
            self.metrics["active_vehicle_seconds"] += active_seconds
            self.metrics["wall_seconds"] = time.monotonic() - self._started_at
            return WindowResult(tuple(decisions), counts, departures, active_seconds, self._state(),
                                start / 1000., end / 1000.,
                                conditioning_decisions=tuple(conditioning) if route_mode else None)
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
