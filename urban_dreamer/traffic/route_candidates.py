"""Four current-to-destination routes over the current traffic topology."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import heapq
import os
from typing import Any

import numpy as np

from .topology import EdgeNetwork, NextHopCandidates, TopologyError


class RouteCandidates(NextHopCandidates):
    """Historical penalty-Dijkstra routes with unchanged next-hop topology APIs.

    ``candidates`` continues to describe successor edges for world-model vehicle
    movement; ``generate_paths`` describes the four actor actions. Multiple
    actor actions may share a successor while retaining different route suffixes.
    """

    routing_mode = 'current_to_dest'

    def __init__(self, network: EdgeNetwork, forbid_revisit: bool = False, *,
                 traffic_model: str = 'micro'):
        if forbid_revisit:
            raise TopologyError('current_to_dest routing does not support forbid_revisit')
        costs = np.asarray(network.free_flow_travel_seconds, dtype=np.float64)
        if not np.isfinite(costs).all() or (costs <= 0).any():
            raise TopologyError('route free-flow costs must be finite and positive')
        super().__init__(network, traffic_model=traffic_model)
        self.max_candidates = 4
        self._route_candidate_cache: OrderedDict[tuple[int, int], tuple[tuple[int, ...], ...]] = OrderedDict()
        self._route_candidate_cache_limit = 200000
        self._shared_route_cache = None
        directory = os.environ.get('URB_ROUTE_CACHE_DIR')
        if directory:
            from .shared_route_cache import SharedRouteCache, network_identity
            self._shared_route_cache = SharedRouteCache(directory, network_identity(network),
                                                        self._route_candidate_cache_limit)

    def generate_paths(self, current_edge: int, destination_edge: int) -> tuple[tuple[int, ...], ...]:
        for label, edge in (('current', current_edge), ('destination', destination_edge)):
            if not 0 <= edge < self.network.edge_count:
                raise TopologyError(f'unknown {label} edge {edge}')
            if not self.network.passenger_allowed[edge]:
                raise TopologyError(f'passenger-forbidden {label} edge {edge}')
        if self._shared_route_cache is not None:
            return self._shared_route_cache.get_or_compute(current_edge, destination_edge,
                self._compute_paths, fallback=self._generate_local_paths)
        return self._generate_local_paths(current_edge, destination_edge)

    def _generate_local_paths(self, current_edge: int, destination_edge: int) -> tuple[tuple[int, ...], ...]:
        key = (current_edge, destination_edge)
        cached = self._route_candidate_cache.get(key)
        if cached is not None:
            self._route_candidate_cache.move_to_end(key)
            return cached
        paths = self._compute_paths(current_edge, destination_edge)
        self._route_candidate_cache[key] = paths
        if len(self._route_candidate_cache) > self._route_candidate_cache_limit:
            self._route_candidate_cache.popitem(last=False)
        return paths

    def _compute_paths(self, current_edge: int, destination_edge: int) -> tuple[tuple[int, ...], ...]:
        if current_edge == destination_edge:
            paths = ((current_edge,),) * 4
        else:
            costs = self.network.free_flow_travel_seconds.astype(np.float64, copy=True)
            selected: list[tuple[int, ...]] = []
            seen: set[tuple[int, ...]] = set()
            for _ in range(4):
                path = self._shortest_path(current_edge, destination_edge, costs)
                if path is None:
                    break
                if path in seen:
                    costs[list(path)] *= 3.
                    path = self._shortest_path(current_edge, destination_edge, costs)
                    if path is None or path in seen:
                        break
                selected.append(path)
                seen.add(path)
                costs[list(path)] *= 3.
            if not selected:
                raise TopologyError(f'no destination-reachable route from edge {current_edge} to {destination_edge}')
            paths = tuple(selected + [selected[0]] * (4 - len(selected)))
        return paths

    def _shortest_path(self, current: int, destination: int, costs: np.ndarray) -> tuple[int, ...] | None:
        best = {current: float(costs[current])}
        previous: dict[int, int] = {}
        frontier = [(best[current], 0, current)]
        sequence = 1
        while frontier:
            cost, _, edge = heapq.heappop(frontier)
            if cost > best.get(edge, float('inf')):
                continue
            if edge == destination:
                path = [destination]
                while path[-1] != current:
                    path.append(previous[path[-1]])
                return tuple(reversed(path))
            for target in self.network.adjacency_list[edge]:
                if not self.network.passenger_allowed[target]:
                    continue
                target_cost = cost + float(costs[target])
                if target_cost < best.get(target, float('inf')):
                    best[target] = target_cost
                    previous[target] = edge
                    heapq.heappush(frontier, (target_cost, sequence, target))
                    sequence += 1
        return None

    def decision_route(self, decision: Any, destination: int,
                       visited_edges: tuple[int, ...] = ()) -> tuple[int, ...]:
        route = tuple(decision.route)
        current = int(decision.current_edge)
        if not route or route[0] != current or route[-1] != destination:
            raise TopologyError('route decision endpoints must match current and destination edges')
        if route not in self.generate_paths(current, destination):
            raise TopologyError('route decision must select an exact current_to_dest candidate')
        if any(target not in self.network.adjacency_list[source]
               for source, target in zip(route, route[1:])):
            raise TopologyError('route decision contains an invalid edge connection')
        next_edge = route[1] if len(route) > 1 else destination
        if int(decision.next_edge) != next_edge:
            raise TopologyError('route decision next edge does not match the selected route')
        return route


def build_route_observations(*, policy_edge_features: Any,
                             vehicle_states: Sequence[dict[str, Any]], candidates: RouteCandidates,
                             episode_progress: float, active_vehicle_fraction: float,
                             prior_cost_mode: str = 'free_flow',
                             prior_speed_floor_mps: float = .5,
                             prior_speed_cap_mps: float | None = None):
    """Keep the ten actor features while describing each complete route action."""
    # Local imports keep topology usable without loading the learning package.
    from ..learning.dynamic_prior import DynamicTravelCosts
    from ..learning.observations import CandidateObservations, PreparedActorWindow, _edge_features, _log_distance

    if prior_cost_mode not in {'free_flow', 'dynamic'}:
        raise ValueError('prior_cost_mode must be free_flow or dynamic')
    network = candidates.network
    if isinstance(policy_edge_features, PreparedActorWindow):
        edges = _edge_features(policy_edge_features.edge_features, network.edge_count)
        global_features = policy_edge_features.global_features
    else:
        edges = _edge_features(policy_edge_features, network.edge_count)
        global_features = edges.mean(axis=0)
    dynamic = None
    if prior_cost_mode == 'dynamic':
        if isinstance(policy_edge_features, PreparedActorWindow):
            key = (id(candidates), float(prior_speed_floor_mps), prior_speed_cap_mps)
            cache = policy_edge_features._dynamic_prior_cache
            if key not in cache:
                cache[key] = DynamicTravelCosts(candidates, edges, prior_speed_floor_mps, prior_speed_cap_mps)
            dynamic = cache[key]
        else:
            dynamic = DynamicTravelCosts(candidates, edges, prior_speed_floor_mps, prior_speed_cap_mps)
    shape = (len(vehicle_states), 4)
    features = np.zeros((*shape, 10), dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    indices = np.empty(shape, dtype=np.int64)
    prior_costs = np.zeros(shape, dtype=np.float64) if dynamic is not None else None
    rows = []
    for row, state in enumerate(vehicle_states):
        current, destination = int(state['current_edge']), int(state['destination_edge'])
        paths = candidates.generate_paths(current, destination)
        rows.append(paths)
        remaining = float(state.get('remaining_distance', network.lengths_m[current]))
        if np.isnan(remaining):
            remaining = float(network.lengths_m[current])
        if not np.isfinite(remaining):
            raise ValueError('remaining_distance must be finite')
        remaining = max(remaining, 0.)
        for column, path in enumerate(paths):
            route_indices = list(path)
            next_edge = path[1] if len(path) > 1 else destination
            indices[row, column] = next_edge
            features[row, column, :2] = edges[list(path[1:] or path)].mean(axis=0)
            features[row, column, 2:4] = global_features
            features[row, column, 4] = np.log1p(network.free_flow_travel_seconds[route_indices].sum()) / np.log1p(3600.)
            features[row, column, 5] = _log_distance(network.lengths_m[route_indices].sum())
            features[row, column, 6] = _log_distance(remaining)
            features[row, column, 7:] = (next_edge == destination, episode_progress, active_vehicle_fraction)
            if dynamic is not None:
                current_fraction = min(remaining / max(float(network.lengths_m[current]), .1), 1.)
                seconds = dynamic.edge_seconds[current] * current_fraction
                seconds += dynamic.edge_seconds[list(path[1:])].sum()
                seconds += sum(candidates.junction(source, target)[0] / candidates.junction(source, target)[1]
                               for source, target in zip(path, path[1:]))
                prior_costs[row, column] = seconds
    if prior_costs is not None:
        with np.errstate(over='ignore'):
            prior_costs = prior_costs.astype(np.float32)
        if not np.isfinite(prior_costs).all():
            raise ValueError('dynamic prior route costs exceed finite FP32 range')
    return CandidateObservations(features, mask, indices, prior_costs, routes=tuple(rows))
