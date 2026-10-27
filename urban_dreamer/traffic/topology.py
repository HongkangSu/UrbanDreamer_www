from __future__ import annotations

import ast
from collections import OrderedDict
import csv
import json
import heapq
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .routing import PreparedInputs


class TopologyError(ValueError):
    """Raised when a validated artifact cannot form the runtime topology."""


@dataclass(frozen=True)
class VehicleDemand:
    vehicle_id: str
    origin_index: int
    destination_index: int
    departure_second: float


@dataclass(frozen=True)
class EdgeNetwork:
    edge_ids: tuple[str, ...]
    edge_to_index: dict[str, int]
    static_features: np.ndarray
    speed_limits_mps: np.ndarray
    capacities: np.ndarray
    lengths_m: np.ndarray
    adjacency: np.ndarray
    adjacency_list: tuple[tuple[int, ...], ...]
    passenger_allowed: np.ndarray
    free_flow_travel_seconds: np.ndarray

    connection_lengths_m: dict[tuple[int, int], float] = field(default_factory=dict)
    connection_speeds_mps: dict[tuple[int, int], float] = field(default_factory=dict)
    internal_lane_lengths_m: dict[str, float] = field(default_factory=dict)
    internal_lane_speeds_mps: dict[str, float] = field(default_factory=dict)
    internal_lane_remaining_lengths_m: dict[str, float] = field(default_factory=dict)
    external_lane_lengths_m: dict[str, float] = field(default_factory=dict)

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)

    def extract_flow(
        self,
        edge_state: Mapping[str, Mapping[str, Any]],
    ) -> np.ndarray:
        """Return the sole model input layout: four dynamic + four static fields."""
        flow = np.empty((self.edge_count, 8), dtype=np.float32)
        flow[:, :4] = (0., 1., 0., 0.)
        flow[:, 4:] = self.static_features
        present, dynamic_rows = [], []
        for edge_index, edge_id in enumerate(self.edge_ids):
            state = edge_state.get(edge_id)
            if state is None:
                continue
            speed_limit = max(float(self.speed_limits_mps[edge_index]), 0.1)
            capacity = max(float(self.capacities[edge_index]), 1.0)
            raw_occupancy = float(state.get("occupancy", 0.0))
            occupancy = raw_occupancy / 100.0 if raw_occupancy > 1.0 else raw_occupancy
            present.append(edge_index)
            dynamic_rows.append((occupancy,
                float(state.get("mean_speed", speed_limit)) / speed_limit,
                float(state.get("vehicle_number", 0.0)) / capacity,
                float(state.get("halting_number", 0.0)) / capacity))
        if dynamic_rows:
            # Preserve float64 normalization and clipping before float32 output,
            # while avoiding one NumPy scalar dispatch and static copy per edge.
            dynamic = np.asarray(dynamic_rows, dtype=np.float64)
            dynamic[:, 0] = np.clip(dynamic[:, 0], 0., 1.)
            flow[present, :4] = dynamic
        return flow


@dataclass(frozen=True)
class TrafficTopology:
    network: EdgeNetwork
    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    vehicles: tuple[VehicleDemand, ...]


def load_traffic_topology(prepared: PreparedInputs) -> TrafficTopology:
    network = _load_connections(_load_edge_network(prepared.edge_registry), prepared.network_xml)
    origins, destinations = _load_od_edges(prepared.od_file)
    unknown = set((*origins, *destinations)) - set(network.edge_ids)
    if unknown:
        raise TopologyError(f"OD file references unknown edges: {sorted(unknown)}")
    vehicles = _load_vehicle_demand(prepared.network_agents_csv,
                                    origin_count=len(origins), destination_count=len(destinations))
    candidates = NextHopCandidates(network)
    for vehicle in vehicles:
        origin = network.edge_to_index[origins[vehicle.origin_index]]
        destination = network.edge_to_index[destinations[vehicle.destination_index]]
        if not network.passenger_allowed[origin] or not network.passenger_allowed[destination]:
            raise TopologyError(f"vehicle {vehicle.vehicle_id} has a passenger-forbidden OD edge")
        candidates.candidates(origin, destination)
    return TrafficTopology(network, origins, destinations, vehicles)


def _lane_allows_passenger(lane: ET.Element) -> bool:
    allow, disallow = lane.get("allow", "").split(), lane.get("disallow", "").split()
    return (not allow or "all" in allow or "passenger" in allow) and not ({"all", "passenger"} & set(disallow))


def _load_connections(network: EdgeNetwork, path: Path) -> EdgeNetwork:
    """Use actual passenger lane connections, not junction-sharing adjacency."""
    root = ET.parse(path).getroot()
    lanes = {lane.get("id"): lane for edge in root.findall("edge") for lane in edge.findall("lane")}
    allowed_by_edge = {edge.get("id"): any(_lane_allows_passenger(lane) for lane in edge.findall("lane"))
                       for edge in root.findall("edge")}
    passenger_allowed = np.array([allowed_by_edge.get(name, False) for name in network.edge_ids], dtype=np.bool_)
    lane_by_edge = {(edge.get("id"), lane.get("index")): lane
                    for edge in root.findall("edge") for lane in edge.findall("lane")}
    continuation = {(connection.get("from"), connection.get("fromLane")): connection.get("via")
                    for connection in root.findall("connection") if connection.get("from", "").startswith(":")}
    lengths, speeds = {}, {}
    adjacency = np.zeros_like(network.adjacency)
    for connection in root.findall("connection"):
        source, target = connection.get("from"), connection.get("to")
        if source not in network.edge_to_index or target not in network.edge_to_index:
            continue
        source_lane = lane_by_edge.get((source, connection.get("fromLane")))
        target_lane = lane_by_edge.get((target, connection.get("toLane")))
        if source_lane is None or target_lane is None or not all(map(_lane_allows_passenger, (source_lane, target_lane))):
            continue
        i, j = network.edge_to_index[source], network.edge_to_index[target]
        if not passenger_allowed[i] or not passenger_allowed[j]:
            continue
        total, speed, via, seen = 0., float(target_lane.get("speed", "1")), connection.get("via"), set()
        while via and via not in seen:
            seen.add(via)
            lane = lanes[via]
            total += float(lane.get("length", "0"))
            speed = min(speed, float(lane.get("speed", "1")))
            internal_edge, lane_index = via.rsplit("_", 1)
            via = continuation.get((internal_edge, lane_index))
        pair = (i, j)
        if pair not in lengths or total / max(speed, .1) < lengths[pair] / max(speeds[pair], .1):
            lengths[pair], speeds[pair] = total, speed
        adjacency[i, j] = 1.
    internal = {name: lane for name, lane in lanes.items() if name.startswith(":")}
    suffix_lengths = {}
    for name in internal:
        total, via, seen = 0., name, set()
        while via and via not in seen:
            seen.add(via)
            total += float(lanes[via].get("length", "0"))
            internal_edge, lane_index = via.rsplit("_", 1)
            via = continuation.get((internal_edge, lane_index))
        suffix_lengths[name] = total
    return replace(network, adjacency=adjacency, passenger_allowed=passenger_allowed,
                   adjacency_list=tuple(tuple(map(int, np.flatnonzero(row))) for row in adjacency),
                   connection_lengths_m=lengths, connection_speeds_mps=speeds,
                   external_lane_lengths_m={name: float(lane.get("length", "0")) for name, lane in lanes.items()
                                            if not name.startswith(":") and _lane_allows_passenger(lane)},
                   internal_lane_remaining_lengths_m=suffix_lengths,
                   internal_lane_lengths_m={name: float(lane.get("length", "0")) for name, lane in internal.items()},
                   internal_lane_speeds_mps={name: float(lane.get("speed", "1")) for name, lane in internal.items()})


class NextHopCandidates:
    """Destination-conditioned successors and bounded immutable route caches."""

    def __init__(self, network: EdgeNetwork, forbid_revisit: bool = False, *, traffic_model: str = 'micro'):
        if traffic_model not in {'micro', 'meso'}:
            raise TopologyError('traffic_model must be micro or meso')
        self.network = network
        self.traffic_model = traffic_model
        self.forbid_revisit = bool(forbid_revisit)
        self.max_candidates = max(1, max(map(len, network.adjacency_list), default=0))
        self._distances: dict[tuple[int, bool], np.ndarray] = {}
        self._candidate_memo: dict[tuple[int, int], tuple[int, ...]] = {}
        self._legal_candidate_cache: OrderedDict[tuple[int, int, tuple[int, ...]], tuple[int, ...]] = OrderedDict()
        self._legal_candidate_cache_limit = 1024
        # Actor static templates are small, immutable and bounded per network instance.
        self._actor_static_cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._actor_static_cache_limit = 8192
        # Deterministic imagined route suffixes depend only on this immutable
        # topology. Keep initial/suffix rules distinct and memory bounded.
        self._event_route_cache: OrderedDict[tuple[int, int, bool], int] = OrderedDict()
        self._event_route_cache_limit = 8192
        self._full_route_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, tuple[int, ...]]] = OrderedDict()
        self._full_route_cache_limit = 8192
        self._stack_caches: dict[bool, dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {False: {}, True: {}}
        self._predecessors = [[] for _ in network.edge_ids]
        for source, targets in enumerate(network.adjacency_list):
            for target in targets:
                self._predecessors[target].append(source)

    def _table(self, destination: int, seconds: bool) -> np.ndarray:
        if not 0 <= destination < self.network.edge_count:
            raise TopologyError(f"unknown destination edge {destination}")
        key = (destination, seconds)
        if key not in self._distances:
            distances = np.full(self.network.edge_count, np.inf, dtype=np.float64)
            distances[destination] = 0.
            queue = [(0., destination)]
            edge_costs = self.network.free_flow_travel_seconds if seconds else self.network.lengths_m
            while queue:
                distance, target = heapq.heappop(queue)
                if distance != distances[target]:
                    continue
                for source in self._predecessors[target]:
                    length, speed = self.junction(source, target)
                    cost = distance + float(edge_costs[target]) + (length / speed if seconds else length)
                    if cost < distances[source]:
                        distances[source] = cost
                        heapq.heappush(queue, (cost, source))
            distances.setflags(write=False)
            self._distances[key] = distances
        return self._distances[key]

    def distances(self, destination: int) -> np.ndarray:
        return self._table(destination, False)

    def costs(self, destination: int) -> np.ndarray:
        return self._table(destination, True)

    def candidates(self, current: int, destination: int) -> tuple[int, ...]:
        if not 0 <= current < self.network.edge_count:
            raise TopologyError(f"unknown current edge {current}")
        key = (current, destination)
        if key not in self._candidate_memo:
            distance = self.distances(destination)
            if current == destination:
                result: tuple[int, ...] = ()
            else:
                result = tuple(sorted(target for target in self.network.adjacency_list[current]
                                      if self.network.passenger_allowed[target] and math.isfinite(distance[target])))
                if not result:
                    raise TopologyError(f"no destination-reachable successor from edge {current} to {destination}")
            self._candidate_memo[key] = result
        return self._candidate_memo[key]

    def junction(self, current: int, target: int) -> tuple[float, float]:
        # MESO crosses the external-edge connection without traversing the
        # microscopic internal lanes. Keep the network itself unchanged so
        # micro and meso candidate views can coexist without cache pollution.
        return (0. if self.traffic_model == 'meso' else self.network.connection_lengths_m.get((current, target), 0.),
                max(.1, self.network.connection_speeds_mps.get((current, target), float(self.network.speed_limits_mps[target]))))

    def legal_candidates(self, current: int, destination: int,
                         visited_edges: tuple[int, ...] = ()) -> tuple[int, ...]:
        """Successors with a destination path avoiding the vehicle's used edges.

        Static candidates stay history-independent for feature template reuse.
        Reverse traversal is sparse. A bounded FIFO reuses repeated validations
        of the same vehicle history within a decision/window.
        """
        choices = self.candidates(current, destination)
        if not self.forbid_revisit or current == destination:
            return choices
        key = (current, destination, tuple(visited_edges))
        cached = self._legal_candidate_cache.get(key)
        if cached is not None:
            return cached
        blocked = set(visited_edges) | {current}
        reachable = set()
        pending = []
        if destination not in blocked and self.network.passenger_allowed[destination]:
            reachable.add(destination)
            pending.append(destination)
        while pending:
            for source in self._predecessors[pending.pop()]:
                if source not in blocked and source not in reachable and self.network.passenger_allowed[source]:
                    reachable.add(source)
                    pending.append(source)
        result = tuple(edge for edge in choices if edge in reachable)
        if not result:
            raise TopologyError(f"no no-revisit destination-reachable successor from edge {current} to {destination}")
        self._legal_candidate_cache[key] = result
        while len(self._legal_candidate_cache) > self._legal_candidate_cache_limit:
            self._legal_candidate_cache.popitem(last=False)
        return result

    def temporary_route(self, current: int, next_edge: int, destination: int) -> tuple[int, ...]:
        if next_edge not in self.candidates(current, destination):
            raise TopologyError(f"invalid next edge {next_edge} from {current} to {destination}")
        route = (current, next_edge)
        if next_edge != destination:
            costs = self.costs(destination)
            placeholder = min(self.candidates(next_edge, destination),
                              key=lambda target: (self.junction(next_edge, target)[0] / self.junction(next_edge, target)[1]
                                                  + float(self.network.free_flow_travel_seconds[target]) + costs[target], target))
            route += (placeholder,)
        return route

    def full_route(self, current: int, next_edge: int, destination: int,
                   visited_edges: tuple[int, ...] = ()) -> tuple[int, ...]:
        """Complete path from ``current`` through ``next_edge`` to ``destination``.

        The suffix beyond ``next_edge`` follows the free-flow cost-to-go table
        greedily (junction seconds + free-flow seconds + remaining cost, ties to
        the lower edge index), so vehicles can cross later junctions without a
        runtime ``findRoute`` even when no decision event fires there.
        """
        if self.forbid_revisit:
            return self._no_revisit_route(current, next_edge, destination, visited_edges)
        if next_edge not in self.candidates(current, destination):
            raise TopologyError(f"invalid next edge {next_edge} from {current} to {destination}")
        costs = self.costs(destination)
        key = (next_edge, destination)
        cached = self._full_route_cache.get(key)
        if cached is not None and cached[0] is costs and not costs.flags.writeable:
            self._full_route_cache.move_to_end(key)
            return (current,) + cached[1]
        route = [current, next_edge]
        edge = next_edge
        while edge != destination:
            if len(route) > self.network.edge_count + 1:
                raise TopologyError(f"cost-to-go chain from {current} to {destination} exceeds the edge count")
            edge = min(self.candidates(edge, destination),
                       key=lambda target: (self.junction(edge, target)[0] / self.junction(edge, target)[1]
                                           + float(self.network.free_flow_travel_seconds[target]) + costs[target], target))
            route.append(edge)
        result = tuple(route)
        if not costs.flags.writeable:
            self._full_route_cache[key] = (costs, result[1:])
            while len(self._full_route_cache) > self._full_route_cache_limit:
                self._full_route_cache.popitem(last=False)
        return result

    def _no_revisit_route(self, current: int, next_edge: int, destination: int,
                          visited_edges: tuple[int, ...]) -> tuple[int, ...]:
        if next_edge not in self.candidates(current, destination):
            raise TopologyError(f"invalid next edge {next_edge} from {current} to {destination}")
        blocked = set(visited_edges) | {current}
        if next_edge in blocked or destination in blocked:
            raise TopologyError(f"no-revisit route from {current} via {next_edge} to {destination} repeats a visited edge")
        # An unblocked unconstrained optimum is also a constrained optimum.
        # Keep legacy local tie-breaking; a blocked/cyclic suffix falls back to
        # residual Dijkstra (including zero-cost cycles in the static table).
        static_costs = self.costs(destination)
        route = [current, next_edge]
        seen = blocked | {next_edge}
        while route[-1] != destination:
            edge = route[-1]
            target = min(self.candidates(edge, destination),
                         key=lambda target: (self.junction(edge, target)[0] / self.junction(edge, target)[1]
                                             + float(self.network.free_flow_travel_seconds[target]) + static_costs[target], target))
            if target in seen:
                break
            route.append(target)
            seen.add(target)
        else:
            return tuple(route)
        # Reverse Dijkstra stores a successor only toward an already settled node.
        # This also guarantees a simple suffix if zero-cost edges have tied costs.
        costs = {destination: 0.}
        successors: dict[int, int] = {}
        settled: set[int] = set()
        queue = [(0., destination)]
        while queue:
            cost, target = heapq.heappop(queue)
            if target in settled or cost != costs[target]:
                continue
            settled.add(target)
            if target == next_edge:
                route = [current, next_edge]
                while route[-1] != destination:
                    route.append(successors[route[-1]])
                return tuple(route)
            for source in self._predecessors[target]:
                if source in blocked or source in settled or not self.network.passenger_allowed[source]:
                    continue
                length, speed = self.junction(source, target)
                candidate_cost = cost + float(self.network.free_flow_travel_seconds[target]) + length / speed
                old = costs.get(source, math.inf)
                if candidate_cost < old or (candidate_cost == old and target < successors.get(source, self.network.edge_count)):
                    costs[source], successors[source] = candidate_cost, target
                    heapq.heappush(queue, (candidate_cost, source))
        raise TopologyError(f"no no-revisit route from {current} via {next_edge} to {destination}")

    def stacked_distance_tables(self, destinations: np.ndarray, *,
                                seconds: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(unique destinations, inverse map, stacked tables) for a batch of destinations.

        Memoized with a bounded FIFO because consecutive decision batches usually
        share the same destination mix; tables themselves are immutable.
        """
        cache = self._stack_caches[seconds]
        key = destinations.tobytes()
        if key not in cache:
            unique, inverse = np.unique(destinations, return_inverse=True)
            tables = np.stack([self._table(int(edge), seconds) for edge in unique])
            cache[key] = (unique, inverse, tables)
            while len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[key]

    def junction_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense (edge, edge) junction length/speed tables equivalent to ``junction``.

        Lazily built float32 copies of the connection dictionaries (zero length and
        the target edge speed limit, floored at 0.1 m/s, where no connection exists),
        so batch builders can replace per-pair dictionary lookups with fancy indexing.
        First use costs 8 * edge_count**2 bytes.
        """
        cached = getattr(self, "_junction_matrices", None)
        if cached is None:
            network = self.network
            count = network.edge_count
            lengths = np.zeros((count, count), dtype=np.float32)
            speeds = np.broadcast_to(network.speed_limits_mps, (count, count)).copy()
            connection_lengths = {} if self.traffic_model == 'meso' else network.connection_lengths_m
            for table, values in ((lengths, connection_lengths), (speeds, network.connection_speeds_mps)):
                if values:
                    pairs = np.asarray(list(values), dtype=np.int64)
                    table[pairs[:, 0], pairs[:, 1]] = np.fromiter(values.values(), dtype=np.float32, count=len(pairs))
            np.maximum(speeds, 0.1, out=speeds)
            lengths.setflags(write=False)
            speeds.setflags(write=False)
            self._junction_matrices = cached = (lengths, speeds)
        return cached


def _load_edge_network(path: Path) -> EdgeNetwork:
    raw = _read_json_object(path)
    edge_ids = tuple(str(value) for value in _required_list(raw, "edge_ids"))
    edge_count = len(edge_ids)
    if edge_count == 0 or len(set(edge_ids)) != edge_count:
        raise TopologyError("edge registry must contain unique edge IDs")

    static_features = _float_array(
        raw, "static_features", expected_shape=(edge_count, 4)
    )
    speed_limits = _float_array(
        raw, "speed_limits", expected_shape=(edge_count,)
    )
    capacities = _float_array(raw, "capacities", expected_shape=(edge_count,))
    lengths = _float_array(raw, "lengths", expected_shape=(edge_count,))
    adjacency = _float_array(
        raw, "adjacency", expected_shape=(edge_count, edge_count)
    )
    passenger_allowed = np.asarray(
        _required_list(raw, "passenger_allowed"),
        dtype=np.bool_,
    )
    if passenger_allowed.shape != (edge_count,):
        raise TopologyError(
            "edge registry passenger_allowed must have one value per edge"
        )
    if (
        not np.all(np.isfinite(static_features))
        or not np.all(np.isfinite(speed_limits))
        or not np.all(np.isfinite(capacities))
        or not np.all(np.isfinite(lengths))
        or not np.all(np.isfinite(adjacency))
    ):
        raise TopologyError("edge registry numeric arrays must be finite")
    if np.any(speed_limits <= 0.0) or np.any(capacities <= 0.0):
        raise TopologyError("edge speed limits and capacities must be positive")
    if np.any(lengths < 0.0) or np.any(adjacency < 0.0):
        raise TopologyError("edge lengths and adjacency must be non-negative")

    adjacency_list: list[tuple[int, ...]] = []
    for source in range(edge_count):
        if not bool(passenger_allowed[source]):
            adjacency_list.append(())
            continue
        neighbors = np.flatnonzero(adjacency[source] > 0.0)
        adjacency_list.append(
            tuple(
                int(target)
                for target in neighbors
                if bool(passenger_allowed[int(target)])
            )
        )
    free_flow = lengths / np.maximum(speed_limits, 0.1)
    return EdgeNetwork(
        edge_ids=edge_ids,
        edge_to_index={edge_id: index for index, edge_id in enumerate(edge_ids)},
        static_features=static_features,
        speed_limits_mps=speed_limits,
        capacities=capacities,
        lengths_m=lengths,
        adjacency=adjacency,
        adjacency_list=tuple(adjacency_list),
        passenger_allowed=passenger_allowed,
        free_flow_travel_seconds=free_flow.astype(np.float32),
    )


def _load_od_edges(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as exc:
        raise TopologyError(f"cannot parse OD file {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TopologyError("OD file must contain a mapping")
    origins = tuple(str(value) for value in _required_list(raw, "origins"))
    destinations = tuple(str(value) for value in _required_list(raw, "destinations"))
    if not origins or not destinations:
        raise TopologyError("OD file origins and destinations must be non-empty")
    return origins, destinations


def _load_vehicle_demand(
    path: Path,
    *,
    origin_count: int,
    destination_count: int,
) -> tuple[VehicleDemand, ...]:
    vehicles: list[VehicleDemand] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "origin", "destination", "start_time"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise TopologyError(
                "agents.csv must contain id,origin,destination,start_time"
            )
        for row in reader:
            vehicle_id = str(row["id"])
            try:
                origin = int(row["origin"])
                destination = int(row["destination"])
                departure = float(row["start_time"])
            except (TypeError, ValueError) as exc:
                raise TopologyError(
                    f"invalid demand row for vehicle {vehicle_id!r}"
                ) from exc
            if not 0 <= origin < origin_count or not 0 <= destination < destination_count:
                raise TopologyError(
                    f"vehicle {vehicle_id!r} has out-of-range OD indices"
                )
            if not math.isfinite(departure) or departure < 0:
                raise TopologyError(
                    f"vehicle {vehicle_id!r} has a negative departure time"
                )
            vehicles.append(
                VehicleDemand(
                    vehicle_id=vehicle_id,
                    origin_index=origin,
                    destination_index=destination,
                    departure_second=departure,
                )
            )
    if not vehicles:
        raise TopologyError("agents.csv contains no vehicles")
    if len({vehicle.vehicle_id for vehicle in vehicles}) != len(vehicles):
        raise TopologyError("agents.csv contains duplicate vehicle IDs")
    return tuple(vehicles)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TopologyError(f"{path} must contain a JSON object")
    return raw


def _required_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise TopologyError(f"missing or invalid list field {key!r}")
    return value


def _float_array(
    raw: Mapping[str, Any],
    key: str,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    try:
        value = np.asarray(raw[key], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise TopologyError(f"missing or invalid numeric field {key!r}") from exc
    if value.shape != expected_shape:
        raise TopologyError(
            f"edge registry {key} has shape {value.shape}; expected {expected_shape}"
        )
    return value
