from __future__ import annotations

import itertools
import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..traffic.topology import NextHopCandidates, TopologyError
from .dynamic_prior import DynamicTravelCosts

CANDIDATE_FEATURE_DIM = 10

_LOG_DISTANCE_SCALE = float(np.log1p(10000.0))
_LOG_HOUR_SCALE = float(np.log1p(3600.0))
_NAN = float("nan")


@dataclass(frozen=True)
class CandidateObservations:
    features: np.ndarray
    mask: np.ndarray
    edge_indices: np.ndarray
    prior_costs_seconds: np.ndarray | None = None
    routes: tuple[tuple[tuple[int, ...], ...], ...] | None = None


@dataclass(frozen=True)
class PreparedActorWindow:
    """Owned immutable policy traffic snapshot and its global mean for one window."""

    edge_features: np.ndarray
    global_features: np.ndarray
    _dynamic_prior_cache: dict = field(default_factory=dict, compare=False, repr=False)


def selected_decisions(states, observations, choices, timestamp):
    """Translate policy slots to physical first turns and optional full routes."""
    from ..traffic.environment import NextHopDecision, RouteDecision

    decisions = []
    for row, (state, choice) in enumerate(zip(states, choices, strict=True)):
        values = (str(state['vehicle_id']), int(state['current_edge']),
                  int(observations.edge_indices[row, choice]), float(timestamp))
        routes = getattr(observations, 'routes', None)
        if routes is None:
            decisions.append(NextHopDecision(*values))
        else:
            decisions.append(RouteDecision(*values, route=routes[row][choice]))
    return decisions


def prepare_actor_window(policy_edge_features: np.ndarray, edge_count: int) -> PreparedActorWindow:
    original = _edge_features(policy_edge_features, edge_count)
    # Reduction order can depend on C/F strides; preserve the original rounding.
    mean = original.mean(axis=0)
    edges = original.copy(order="K")
    edges.setflags(write=False)
    mean.setflags(write=False)
    return PreparedActorWindow(edges, mean)


def _static_templates(candidates: Any, pairs: np.ndarray) -> list[CandidateObservations]:
    """Fill cold pairs together; retain only immutable topology-derived templates."""
    network, width = candidates.network, candidates.max_candidates
    cache = getattr(candidates, '_actor_static_cache', None)
    keys = [(int(pair // network.edge_count), int(pair % network.edge_count)) for pair in pairs]
    templates = [cache.get(key) if cache is not None else None for key in keys]
    missing = [row for row, template in enumerate(templates) if template is None]
    if not missing:
        return templates
    currents = np.asarray([keys[row][0] for row in missing], dtype=np.int64)
    destinations = np.asarray([keys[row][1] for row in missing], dtype=np.int64)
    choices = list(map(candidates.candidates, currents, destinations))
    counts = np.fromiter(map(len, choices), dtype=np.int64, count=len(choices))
    masks = np.arange(width) < counts[:, None]
    indices = np.full((len(missing), width), -1, dtype=np.int64)
    indices[masks] = np.fromiter(itertools.chain.from_iterable(choices), dtype=np.int64,
                                count=int(counts.sum()))
    safe = np.where(masks, indices, 0)
    stacked = getattr(candidates, 'stacked_distance_tables', None)
    if callable(stacked):
        unique_destinations, _, distances = stacked(destinations, seconds=False)
        costs = stacked(destinations, seconds=True)[2]
    else:
        unique_destinations = np.unique(destinations)
        distances = np.asarray([candidates.distances(int(edge)) for edge in unique_destinations])
        costs = np.asarray([candidates.costs(int(edge)) for edge in unique_destinations])
    destination_rows = np.searchsorted(unique_destinations, destinations)
    if isinstance(candidates, NextHopCandidates):
        # Preserve the former dense tables' float32 lengths and speed-limit dtype.
        junction_lengths = np.zeros_like(safe, dtype=np.float32)
        junction_speeds = network.speed_limits_mps[safe].copy()
        connection_keys = list(zip(np.repeat(currents, counts), indices[masks]))
        if candidates.traffic_model == 'micro':
            junction_lengths[masks] = np.fromiter(
                map(network.connection_lengths_m.get, connection_keys, itertools.repeat(0.)),
                dtype=np.float32, count=int(counts.sum()))
        overrides = np.fromiter(
            map(network.connection_speeds_mps.get, connection_keys, itertools.repeat(_NAN)),
            dtype=np.float32, count=int(counts.sum()))
        junction_speeds[masks] = np.where(np.isnan(overrides), junction_speeds[masks], overrides)
        np.maximum(junction_speeds, .1, out=junction_speeds)
        junction_lengths = junction_lengths.astype(np.float64)
        junction_speeds = junction_speeds.astype(np.float64)
    elif callable(getattr(candidates, 'junction_matrices', None)):
        dense_lengths, dense_speeds = candidates.junction_matrices()
        junction_lengths = dense_lengths[currents[:, None], safe].astype(np.float64)
        junction_speeds = dense_speeds[currents[:, None], safe].astype(np.float64)
    else:
        junction_lengths = np.zeros_like(safe, dtype=np.float64)
        junction_speeds = np.ones_like(safe, dtype=np.float64)
        for row, successors in enumerate(choices):
            for column, edge in enumerate(successors):
                length, speed = candidates.junction(currents[row], edge)
                junction_lengths[row, column], junction_speeds[row, column] = length, speed
    travel_seconds = (junction_lengths / junction_speeds + network.free_flow_travel_seconds[safe]
                      + costs[destination_rows[:, None], safe])
    total_distance = (junction_lengths + network.lengths_m[safe]
                      + distances[destination_rows[:, None], safe])
    features = np.zeros((len(missing), width, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    np.copyto(features[:, :, 4], np.log1p(travel_seconds) / _LOG_HOUR_SCALE, where=masks)
    np.copyto(features[:, :, 5], _log_distance(total_distance), where=masks)
    np.copyto(features[:, :, 7], indices == destinations[:, None], where=masks)
    for array in (features, masks, indices):
        array.setflags(write=False)
    for index, row in enumerate(missing):
        # Own each row: retaining one cache entry must not retain a whole cold batch.
        arrays = (features[index].copy(), masks[index].copy(), indices[index].copy())
        for array in arrays:
            array.setflags(write=False)
        template = CandidateObservations(*arrays)
        templates[row] = template
        if cache is not None:
            cache[keys[row]] = template
    if cache is not None:
        while len(cache) > candidates._actor_static_cache_limit:
            cache.popitem(last=False)
    return templates


def _log_distance(values: Any) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0)) / _LOG_DISTANCE_SCALE


def _edge_features(value: np.ndarray, edge_count: int) -> np.ndarray:
    features = np.asarray(value, dtype=np.float32)
    if features.shape != (edge_count, 2) or not np.isfinite(features).all():
        raise ValueError("policy_edge_features must be finite with shape (E, 2)")
    return features


def _int_column(states: Sequence[dict[str, Any]], key: str, count: int) -> np.ndarray:
    return np.fromiter(map(operator.itemgetter(key), states), dtype=np.int64, count=count)


def _float_column(states: Sequence[dict[str, Any]], key: str, default: float, count: int) -> np.ndarray:
    return np.fromiter(map(dict.get, states, itertools.repeat(key), itertools.repeat(default)),
                       dtype=np.float64, count=count)


def build_actor_observations(*, policy_edge_features: np.ndarray | PreparedActorWindow,
                             vehicle_states: Sequence[dict[str, Any]], candidates: Any,
                             episode_progress: float, active_vehicle_fraction: float,
                             prior_cost_mode: str = 'free_flow',
                             prior_speed_floor_mps: float = .5,
                             prior_speed_cap_mps: float | None = None) -> CandidateObservations:
    """Score candidate traffic, static cost and destination distance without path materialization."""
    if getattr(candidates, 'routing_mode', 'next_hop') == 'current_to_dest':
        from ..traffic.route_candidates import build_route_observations
        return build_route_observations(policy_edge_features=policy_edge_features,
            vehicle_states=vehicle_states, candidates=candidates,
            episode_progress=episode_progress, active_vehicle_fraction=active_vehicle_fraction,
            prior_cost_mode=prior_cost_mode, prior_speed_floor_mps=prior_speed_floor_mps,
            prior_speed_cap_mps=prior_speed_cap_mps)
    network = candidates.network
    if prior_cost_mode not in {"free_flow", "dynamic"}:
        raise ValueError("prior_cost_mode must be free_flow or dynamic")
    if prior_cost_mode == "dynamic" and not isinstance(policy_edge_features, PreparedActorWindow):
        policy_edge_features = prepare_actor_window(policy_edge_features, network.edge_count)
    if isinstance(policy_edge_features, PreparedActorWindow):
        edges = policy_edge_features.edge_features
        if edges.shape != (network.edge_count, 2):
            raise ValueError("prepared policy_edge_features must have shape (E, 2)")
        global_features = policy_edge_features.global_features
    else:
        edges = _edge_features(policy_edge_features, network.edge_count)
        global_features = None
    count, width = len(vehicle_states), candidates.max_candidates
    if count == 0:
        return CandidateObservations(np.zeros((0, width, CANDIDATE_FEATURE_DIM), dtype=np.float32),
                                     np.zeros((0, width), dtype=bool),
                                     np.full((0, width), -1, dtype=np.int64))
    if global_features is None:
        global_features = edges.mean(axis=0)
    edge_count = network.edge_count
    currents = _int_column(vehicle_states, 'current_edge', count)
    destinations = _int_column(vehicle_states, 'destination_edge', count)
    remaining = _float_column(vehicle_states, 'remaining_distance', _NAN, count)
    if ((currents < 0) | (currents >= edge_count)).any() or ((destinations < 0) | (destinations >= edge_count)).any():
        for row in range(count):
            current, destination = int(currents[row]), int(destinations[row])
            if not 0 <= current < edge_count:
                raise TopologyError(f"unknown current edge {current}")
            if not 0 <= destination < edge_count:
                raise TopologyError(f"unknown destination edge {destination}")
    packed = currents * edge_count + destinations
    unique_pairs, pair_inverse = np.unique(packed, return_inverse=True)
    templates = _static_templates(candidates, unique_pairs)
    pair_mask = np.asarray([template.mask for template in templates])
    if (~pair_mask.any(axis=1)).any():
        row = int(np.flatnonzero(~pair_mask[pair_inverse].any(axis=1))[0])
        raise ValueError(f"vehicle {vehicle_states[row].get('vehicle_id', row)!r} has no legal next-hop candidate")
    pair_indices = np.asarray([template.edge_indices for template in templates])
    pair_features = np.asarray([template.features for template in templates])
    mask = pair_mask[pair_inverse]
    indices = pair_indices[pair_inverse]
    pair_safe = np.where(pair_mask, pair_indices, 0)
    np.copyto(pair_features[:, :, 0:2], edges[pair_safe], where=pair_mask[..., None])
    np.copyto(pair_features[:, :, 2:4], global_features, where=pair_mask[..., None])
    np.copyto(pair_features[:, :, 8], episode_progress, where=pair_mask)
    np.copyto(pair_features[:, :, 9], active_vehicle_fraction, where=pair_mask)
    features = pair_features[pair_inverse]
    remaining = np.where(np.isnan(remaining), network.lengths_m[currents], remaining)
    np.copyto(features[:, :, 6], _log_distance(remaining)[:, None], where=mask)
    if getattr(candidates, 'forbid_revisit', False):
        for row, state in enumerate(vehicle_states):
            if 'visited_edges' not in state:
                raise TopologyError(f"vehicle {state.get('vehicle_id', row)!r} is missing visited_edges for no-revisit observations")
            legal = candidates.legal_candidates(int(currents[row]), int(destinations[row]),
                                                state['visited_edges'])
            mask[row] &= np.isin(indices[row], legal)
            if not mask[row].any():
                raise TopologyError(f"vehicle {state.get('vehicle_id', row)!r} has no no-revisit next-hop candidate")
        indices[~mask] = -1
        features[~mask] = 0.
    prior_costs = None
    if prior_cost_mode == "dynamic":
        key = (id(candidates), float(prior_speed_floor_mps), prior_speed_cap_mps)
        cache = policy_edge_features._dynamic_prior_cache
        if key not in cache:
            cache[key] = DynamicTravelCosts(candidates, edges, prior_speed_floor_mps, prior_speed_cap_mps)
        prior_costs = cache[key].candidates_seconds(currents, destinations, indices, mask)
    return CandidateObservations(features, mask, indices, prior_costs)


def aggregate_next_hop_actions(decisions: Sequence[Any], *, edge_count: int, total_vehicle_count: int) -> np.ndarray:
    if edge_count <= 0 or total_vehicle_count <= 0:
        raise ValueError("edge_count and total_vehicle_count must be positive")
    counts = np.zeros(edge_count, dtype=np.float32)
    for decision in decisions:
        edge = int(decision.next_edge)
        if not 0 <= edge < edge_count:
            raise ValueError("next edge is outside the network")
        counts[edge] += 1.
    return counts / float(total_vehicle_count)
