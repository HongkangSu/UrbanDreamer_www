from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .action_events import ActionEvent


@dataclass(frozen=True)
class ActiveCostSegment:
    start_time: float
    end_time: float
    active_count: int


@dataclass(frozen=True)
class ImaginedWindow:
    action: np.ndarray
    departure: np.ndarray
    decisions: tuple[Any, ...]
    arrivals: dict[str, float]
    active_vehicle_seconds: float
    cost_segments: tuple[ActiveCostSegment, ...]
    end_time: float
    action_events: tuple[ActionEvent, ...] = ()
    policy_events: tuple[ActionEvent, ...] | None = None


def _prepare_no_revisit_history(states: Sequence[dict[str, Any]], candidates: Any) -> None:
    """Retain seed history; never infer past real roads from the current road."""
    if not getattr(candidates, 'forbid_revisit', False):
        return
    for state in states:
        if 'visited_edges' not in state:
            raise ValueError('no-revisit imagination requires complete visited_edges seed history')
        history = tuple(int(edge) for edge in state['visited_edges'])
        if len(history) != len(set(history)):
            raise ValueError('no-revisit seed contains repeated physical roads')
        if state.get('departed', False) and (not history or history[-1] != int(state['current_edge'])):
            raise ValueError('no-revisit seed history must end at its current external road')
        state['visited_edges'] = history
        successor = state.get('committed_next_edge')
        if successor is not None and int(successor) not in _legal_candidates(state, candidates):
            raise ValueError('no-revisit seed has an illegal committed next hop')


def _record_physical_entry(state: dict[str, Any], edge: int, candidates: Any) -> None:
    if not getattr(candidates, 'forbid_revisit', False):
        return
    history = state['visited_edges']
    if edge in history:
        raise ValueError('no-revisit imagined vehicle entered a previously visited road')
    state['visited_edges'] = (*history, edge)


def _legal_candidates(state: dict[str, Any], candidates: Any) -> tuple[int, ...]:
    current, destination = int(state['current_edge']), int(state['destination_edge'])
    if getattr(candidates, 'forbid_revisit', False):
        return candidates.legal_candidates(current, destination, state['visited_edges'])
    return candidates.candidates(current, destination)


def clip_road_speed_fractions(speed_fractions: np.ndarray,
                              traffic_model: str = 'micro') -> np.ndarray:
    """Preserve stopped meso roads while retaining the legacy micro speed floor."""
    if traffic_model not in ('micro', 'meso'):
        raise ValueError('traffic_model must be micro or meso')
    return np.clip(speed_fractions, 0. if traffic_model == 'meso' else .05, 1.5)


def advance_imagined_window(
    vehicle_states: Sequence[dict[str, Any]], *, candidates: Any,
    edge_speeds_mps: np.ndarray, frozen_flow: np.ndarray, start_time: float,
    total_vehicle_count: int,
    decide: Callable[[list[dict[str, Any]], np.ndarray, float], Sequence[Any]],
    interval_seconds: float = 30.0,
    observation_step_seconds: float | None = None,
    vehicle_max_speed_mps: float | None = None,
    stop_when_empty: bool = False,
    session_protocol: str = 'legacy',
    event_index_start: int = 0,
    traffic_model: str = 'micro',
    passage_constraint: Any = None,
) -> ImaginedWindow:
    """Advance physical events through a window; policy events at its end stay pending.

    States are mutated in place, with distances (rather than cached traversal
    times) preserving progress under the next window's newly decoded speeds.
    A committed successor remains attached throughout the external edge and its
    junction connection. Only arrival on that successor creates a new decision.
    ``session_protocol='event'`` instead exposes one-second observations and
    calls the policy only for observed multi-candidate external entries.
    Current-to-destination candidates call the policy at 30-second boundaries,
    retain its complete route, and expose observed turns separately from policy
    samples so the existing world-model action interface remains unchanged.
    Meso motion uses external-road distances and skips internal junction links.
    Decoded road speeds still move vehicles independently: this approximation
    does not reproduce SUMO mesoscopic queue or junction-control dynamics.
    ``passage_constraint`` optionally adds a meso stop-line/FIFO gate. It is off
    by default and does not provide segment capacity, headways or merge control.
    """
    if traffic_model not in ('micro', 'meso'):
        raise ValueError('traffic_model must be micro or meso')
    if passage_constraint is not None and (traffic_model != 'meso' or session_protocol != 'event'):
        raise ValueError('passage constraints require meso event imagination')
    if session_protocol not in ('legacy', 'event'):
        raise ValueError('session_protocol must be legacy or event')
    if getattr(candidates, 'routing_mode', 'next_hop') == 'current_to_dest' and session_protocol != 'event':
        raise ValueError('current_to_dest imagination requires the event session protocol')
    if traffic_model == 'meso':
        if session_protocol != 'event':
            raise ValueError('meso imagination requires the event session protocol')
        if any(state.get('in_junction', False) for state in vehicle_states):
            raise ValueError('meso imagination cannot accept microscopic in_junction seeds')
    if getattr(candidates, 'traffic_model', 'micro') != traffic_model:
        raise ValueError('imagination traffic_model must match its candidates traffic_model')
    if observation_step_seconds is None:
        observation_step_seconds = 1. if session_protocol == 'event' else .001
    if event_index_start < 0 or int(event_index_start) != event_index_start:
        raise ValueError('event_index_start must be a non-negative integer')
    event_index_start = int(event_index_start)
    if session_protocol == 'event':
        return _advance_event_window(vehicle_states, candidates=candidates,
            edge_speeds_mps=edge_speeds_mps, frozen_flow=frozen_flow, start_time=start_time,
            total_vehicle_count=total_vehicle_count, decide=decide,
            interval_seconds=interval_seconds, observation_step_seconds=observation_step_seconds,
            vehicle_max_speed_mps=vehicle_max_speed_mps, stop_when_empty=stop_when_empty,
            event_index_start=event_index_start, traffic_model=traffic_model,
            passage_constraint=passage_constraint)
    network = candidates.network
    speeds = np.asarray(edge_speeds_mps, dtype=np.float64)
    if speeds.shape != (network.edge_count,) or not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError('edge_speeds_mps must contain one finite non-negative speed per edge')
    if total_vehicle_count <= 0 or interval_seconds <= 0 or not np.isfinite(start_time + interval_seconds):
        raise ValueError('positive population and finite positive interval required')
    if not np.isfinite(observation_step_seconds) or observation_step_seconds <= 0:
        raise ValueError('observation_step_seconds must be finite and positive')
    max_speed = float('inf') if vehicle_max_speed_mps is None else float(vehicle_max_speed_mps)
    if max_speed <= 0 or (vehicle_max_speed_mps is not None and not np.isfinite(max_speed)):
        raise ValueError('vehicle_max_speed_mps must be finite and positive')
    speeds = np.minimum(speeds, max_speed)
    states = sorted(vehicle_states, key=lambda state: str(state['vehicle_id']))
    _prepare_no_revisit_history(states, candidates)
    for state in states:
        if state.get('in_junction', False):
            junction_speed = float(state['junction_speed'])
            if not np.isfinite(junction_speed) or junction_speed <= 0:
                raise ValueError('in-transit junction speed must be finite and positive')
            state['junction_speed'] = min(junction_speed, max_speed)
    if len({str(s['vehicle_id']) for s in states}) != len(states):
        raise ValueError('vehicle IDs must be unique')
    action = np.zeros(network.edge_count, dtype=np.float32)
    departure = np.zeros_like(action)
    decisions: list[Any] = []
    arrivals: dict[str, float] = {}
    active_seconds = 0.0
    cost_segments: list[ActiveCostSegment] = []
    time, end = float(start_time), float(start_time + interval_seconds)
    tolerance = 1e-9
    # An undeparted seed is physically absent at S_t even when its scheduled
    # departure is overdue (e.g. queued SUMO insertion). Insert it on the next
    # observation, so the resulting appearance is represented in D_t.
    departure_events = {
        str(state['vehicle_id']): (start_time + observation_step_seconds
            if float(state['departure_time']) <= start_time
            else float(state['departure_time']))
        for state in states if not state.get('departed', False)
    }

    def active(state: dict[str, Any]) -> bool:
        return bool(state.get('departed', False)) and not bool(state.get('arrived', False))

    def enter_successor(state: dict[str, Any]) -> None:
        successor = state.get('committed_next_edge')
        if successor is None:
            raise ValueError('vehicle exhausted its edge without a committed next hop')
        _record_physical_entry(state, int(successor), candidates)
        state['current_edge'] = int(successor)
        state['remaining_distance'] = float(network.lengths_m[int(successor)])
        state['committed_next_edge'] = None
        state['in_junction'] = False
        state['junction_remaining_distance'] = 0.0
        state['pending_decision'] = int(successor) != int(state['destination_edge'])

    while True:
        # Complete physical events at the endpoint before assigning the window.
        for state in states:
            if not state.get('departed', False) and not state.get('arrived', False) and departure_events[str(state['vehicle_id'])] <= time + tolerance:
                _record_physical_entry(state, int(state['current_edge']), candidates)
                state['departed'] = True
                state['observed_departure_time'] = time
                state['pending_decision'] = int(state['current_edge']) != int(state['destination_edge'])
                if time > start_time + tolerance:
                    departure[int(state['current_edge'])] += 1.0
            if not active(state):
                continue
            if state.get('in_junction', False):
                if float(state['junction_remaining_distance']) <= tolerance:
                    enter_successor(state)
            elif float(state['remaining_distance']) <= tolerance:
                if int(state['current_edge']) == int(state['destination_edge']):
                    state['arrived'] = True
                    state['remaining_distance'] = 0.0
                    state['pending_decision'] = False
                    state['arrival_time'] = time
                    arrivals[str(state['vehicle_id'])] = time
                elif not state.get('pending_decision', False):
                    successor = state.get('committed_next_edge')
                    if successor is None:
                        raise ValueError('vehicle exhausted its edge without a committed next hop')
                    length, speed = candidates.junction(int(state['current_edge']), int(successor))
                    if length < 0 or speed <= 0 or not np.isfinite(length + speed):
                        raise ValueError('junction length and speed must be finite and physically valid')
                    state['in_junction'] = True
                    state['junction_remaining_distance'] = float(length)
                    state['junction_speed'] = min(float(speed), max_speed)
                    if length <= tolerance:
                        enter_successor(state)
        if time >= end - tolerance:
            break
        if stop_when_empty and all(state.get('arrived', False) for state in states):
            break

        pending = [s for s in states if active(s) and s.get('pending_decision', False)]
        if pending:
            selected = list(decide(pending, frozen_flow, time))
            by_id = {str(item.vehicle_id): item for item in selected}
            if len(by_id) != len(selected) or set(by_id) != {str(s['vehicle_id']) for s in pending}:
                raise ValueError('policy must return exactly one decision per pending vehicle')
            # Validate the complete batch before mutating commitments or counts.
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                legal = _legal_candidates(state, candidates)
                if int(item.current_edge) != int(state['current_edge']) or int(item.next_edge) not in legal:
                    raise ValueError('policy selected a next hop outside the legal candidate set')
                if abs(float(item.decision_time) - time) > tolerance:
                    raise ValueError('decision timestamp must match its physical event')
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                state['committed_next_edge'] = int(item.next_edge)
                state['pending_decision'] = False
                action[int(item.next_edge)] += 1.0
                decisions.append(item)

        duration = end - time
        for state in states:
            if not state.get('departed', False) and not state.get('arrived', False):
                duration = min(duration, max(0., departure_events[str(state['vehicle_id'])] - time))
            elif active(state):
                in_junction = bool(state.get('in_junction', False))
                distance = float(state['junction_remaining_distance'] if in_junction else state['remaining_distance'])
                speed = float(state['junction_speed'] if in_junction else speeds[int(state['current_edge'])])
                if speed > 0:
                    duration = min(duration, max(distance, 0.) / speed)
        if duration <= tolerance:
            # Zero-distance committed edges need another physical event pass.
            if any(active(s) and not s.get('in_junction', False) and float(s['remaining_distance']) <= tolerance for s in states):
                if any(float(network.lengths_m[int(s['current_edge'])]) <= 0 for s in states if active(s)):
                    raise ValueError('zero-length external edges cannot be advanced safely')
                continue
            # A residual distance within one tolerance-step of completing its
            # edge or junction traversal (distance <= speed * tolerance) is
            # invisible to the physical pass but still stalls the clock; snap
            # it to completion. Runs that never raised never enter this branch.
            repaired = False
            for state in states:
                if not active(state):
                    continue
                in_junction = bool(state.get('in_junction', False))
                key = 'junction_remaining_distance' if in_junction else 'remaining_distance'
                speed = float(state['junction_speed'] if in_junction else speeds[int(state['current_edge'])])
                distance = float(state[key])
                if 0. < distance and speed > 0 and distance / speed <= tolerance:
                    state[key] = 0.
                    repaired = True
            if repaired:
                continue
            raise ValueError('imagined vehicle clock failed to advance')
        active_count = sum(active(s) for s in states)
        cost_segments.append(ActiveCostSegment(time, min(time + duration, end), active_count))
        active_seconds += active_count * duration
        for state in states:
            if not active(state):
                continue
            if state.get('in_junction', False):
                state['junction_remaining_distance'] = max(0., float(state['junction_remaining_distance']) - float(state['junction_speed']) * duration)
            else:
                state['remaining_distance'] = max(0., float(state['remaining_distance']) - float(speeds[int(state['current_edge'])]) * duration)
        time = min(time + duration, end)

    return ImaginedWindow(action / total_vehicle_count, departure / total_vehicle_count,
                          tuple(decisions), arrivals, float(active_seconds),
                          tuple(cost_segments), time)


def _advance_event_window(
    vehicle_states: Sequence[dict[str, Any]], *, candidates: Any,
    edge_speeds_mps: np.ndarray, frozen_flow: np.ndarray, start_time: float,
    total_vehicle_count: int,
    decide: Callable[[list[dict[str, Any]], np.ndarray, float], Sequence[Any]],
    interval_seconds: float, observation_step_seconds: float,
    vehicle_max_speed_mps: float | None, stop_when_empty: bool,
    event_index_start: int, traffic_model: str,
    passage_constraint: Any = None,
) -> ImaginedWindow:
    """Follow complete routes continuously and expose only one-second observations.

    Route commitments identify the actual physical pair between observations.
    Pending actions identify the last *observed* external entry. In particular,
    crossing a fork between observations never creates an action. Exposure and
    arrivals follow the session's observation clock, even when an edge finishes
    in the middle of a step. Route-mode policy samples occur only at macro
    boundaries; physical turns continue to use the existing observation clock.
    """
    from .action_events import ActionEvent

    if observation_step_seconds != 1.:
        raise ValueError('event imagination requires one-second observations')
    end = float(start_time + interval_seconds)
    if (not np.isfinite(start_time) or not np.isfinite(end) or interval_seconds <= 0
            or abs(interval_seconds - round(interval_seconds)) > 1e-8
            or abs(start_time - round(start_time)) > 1e-8):
        raise ValueError('event windows require finite, whole-second boundaries')
    network = candidates.network
    speeds = np.asarray(edge_speeds_mps, dtype=np.float64)
    if speeds.shape != (network.edge_count,) or not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError('edge_speeds_mps must contain one finite non-negative speed per edge')
    if total_vehicle_count <= 0:
        raise ValueError('positive population required')
    max_speed = float('inf') if vehicle_max_speed_mps is None else float(vehicle_max_speed_mps)
    if max_speed <= 0 or (vehicle_max_speed_mps is not None and not np.isfinite(max_speed)):
        raise ValueError('vehicle_max_speed_mps must be finite and positive')
    speeds = np.minimum(speeds, max_speed)
    states = sorted(vehicle_states, key=lambda state: str(state['vehicle_id']))
    _prepare_no_revisit_history(states, candidates)
    if len({str(s['vehicle_id']) for s in states}) != len(states):
        raise ValueError('vehicle IDs must be unique')
    if passage_constraint is not None:
        passage_constraint.prepare(states)
    tolerance = 1e-9
    route_mode = getattr(candidates, 'routing_mode', 'next_hop') == 'current_to_dest'

    def active(state: dict[str, Any]) -> bool:
        return bool(state.get('departed', False)) and not bool(state.get('arrived', False))

    def needs_decision(state: dict[str, Any]) -> bool:
        return (int(state['current_edge']) != int(state['destination_edge'])
                and len(_legal_candidates(state, candidates)) > 1)

    route_cache = getattr(candidates, '_event_route_cache', {})
    route_cache_limit = getattr(candidates, '_event_route_cache_limit', None)

    def validate_route(state: dict[str, Any], route: Sequence[int]) -> tuple[int, ...]:
        route = tuple(int(edge) for edge in route)
        if (not route or route[0] != int(state['current_edge'])
                or route[-1] != int(state['destination_edge'])):
            raise ValueError('committed route endpoints disagree with its vehicle seed')
        if any(not 0 <= edge < network.edge_count or not network.passenger_allowed[edge] for edge in route):
            raise ValueError('committed route contains an invalid external road')
        if any(target not in network.adjacency_list[source] for source, target in zip(route, route[1:])):
            raise ValueError('committed route contains an illegal edge connection')
        return route

    def route_successor(state: dict[str, Any], *, initial: bool = False) -> int | None:
        edge, destination = int(state['current_edge']), int(state['destination_edge'])
        if route_mode:
            route = state.get('committed_route')
            if route is None:
                successor = state.get('committed_next_edge')
                # Old replay/checkpoint seeds contain only their first hop.
                route = (candidates.full_route(edge, int(successor), destination)
                         if successor is not None and not initial else candidates.generate_paths(edge, destination)[0])
                route = validate_route(state, route)
            elif route[0] != edge:
                if len(route) < 2 or route[1] != edge:
                    raise ValueError('committed route disagrees with physical entry')
                route = route[1:]
            # Complete routes were validated at seeding/selection. Physical
            # entry only consumes that commitment; it never rebuilds a suffix.
            state['committed_route'] = route
            return int(route[1]) if len(route) > 1 else None
        if edge == destination:
            return None
        if getattr(candidates, 'forbid_revisit', False):
            route = state.get('no_revisit_route', ())
            if initial or not route:
                choices = _legal_candidates(state, candidates)
                costs = candidates.costs(destination)
                successor = min(choices, key=lambda target:
                    (float(network.free_flow_travel_seconds[target]) + costs[target], target))
                route = candidates.full_route(edge, successor, destination, state['visited_edges'])
            elif route[0] != edge:
                if len(route) < 2 or route[1] != edge:
                    raise ValueError('no-revisit route commitment disagrees with physical entry')
                route = route[1:]
            state['no_revisit_route'] = route
            return int(route[1])
        key = (edge, destination, initial)
        cached = route_cache.get(key)
        if cached is not None:
            return cached
        choices, costs = candidates.candidates(edge, destination), candidates.costs(destination)
        # SUMO initialization intentionally omits the first junction cost. Its
        # later full-route suffix includes it; preserve both selection rules.
        def cost(target: int) -> tuple[float, int]:
            length, speed = candidates.junction(edge, target)
            junction_seconds = 0. if initial else length / speed
            return junction_seconds + float(network.free_flow_travel_seconds[target]) + costs[target], target
        successor = int(min(choices, key=cost))
        route_cache[key] = successor
        if route_cache_limit is not None and len(route_cache) > route_cache_limit:
            route_cache.popitem(last=False)
        return successor

    for state in states:
        if traffic_model == 'meso':
            state.update(junction_remaining_distance=0., junction_speed=0.,
                         junction_connection=None)
        if not active(state):
            if route_mode and state.get('committed_route') is not None:
                state['committed_route'] = validate_route(state, state['committed_route'])
            continue
        if state.get('in_junction', False):
            speed = float(state['junction_speed'])
            if not np.isfinite(speed) or speed <= 0:
                raise ValueError('in-transit junction speed must be finite and positive')
            state['junction_speed'] = min(speed, max_speed)
            state['pending_decision'] = False
        elif not needs_decision(state):
            state['pending_decision'] = False
        if route_mode:
            if state.get('committed_route') is not None:
                state['committed_route'] = validate_route(state, state['committed_route'])
                expected = (state['committed_route'][1] if len(state['committed_route']) > 1 else None)
                if state.get('committed_next_edge') is not None and state['committed_next_edge'] != expected:
                    raise ValueError('committed route disagrees with its seed next hop')
                state['committed_next_edge'] = expected
            elif int(state['current_edge']) == int(state['destination_edge']):
                state['committed_route'] = (int(state['current_edge']),)
            state['pending_turn'] = bool(state.get('pending_turn', state.get('pending_decision', False)))
        if int(state['current_edge']) != int(state['destination_edge']):
            successor = state.get('committed_next_edge')
            if successor is None:
                if not state.get('pending_decision', False):
                    raise ValueError('non-pending event seed has no committed next hop')
                state['committed_next_edge'] = route_successor(state)
            elif int(successor) not in _legal_candidates(state, candidates):
                raise ValueError('event seed has an illegal committed next hop')
            if route_mode:
                state['committed_next_edge'] = route_successor(state)
                state['pending_decision'] = False
            if getattr(candidates, 'forbid_revisit', False):
                state['no_revisit_route'] = candidates.full_route(
                    int(state['current_edge']), int(state['committed_next_edge']),
                    int(state['destination_edge']), state['visited_edges'])

    action = np.zeros(network.edge_count, dtype=np.float32)
    departure = np.zeros_like(action)
    decisions: list[Any] = []
    action_events: list[ActionEvent] = []
    policy_events: list[ActionEvent] = []
    arrivals: dict[str, float] = {}
    cost_segments: list[ActiveCostSegment] = []
    time = float(start_time)

    def enter_successor(state: dict[str, Any]) -> None:
        successor = state.get('committed_next_edge')
        if successor is None:
            raise ValueError('vehicle exhausted its edge without a committed next hop')
        source_edge = int(state['current_edge'])
        _record_physical_entry(state, int(successor), candidates)
        state['current_edge'] = int(successor)
        state['remaining_distance'] = float(network.lengths_m[int(successor)])
        if state['remaining_distance'] <= 0:
            raise ValueError('zero-length external edges cannot be advanced safely')
        state['in_junction'] = False
        state['junction_remaining_distance'] = 0.
        state['junction_connection'] = None
        if passage_constraint is not None:
            passage_constraint.entered(state, source_edge, crossing_time)
        state['committed_next_edge'] = route_successor(state)

    def move(state: dict[str, Any], duration: float) -> bool:
        """Return physical arrival without publishing it until observation."""
        nonlocal crossing_time
        initial_duration = duration
        while True:
            crossing_time = time + initial_duration - duration
            in_junction = bool(state.get('in_junction', False))
            key = 'junction_remaining_distance' if in_junction else 'remaining_distance'
            distance = float(state[key])
            if not math.isfinite(distance) or distance < 0:
                raise ValueError('remaining route distances must be finite and non-negative')
            if distance <= tolerance:
                if in_junction:
                    if state.get('committed_next_edge') is None:
                        raise ValueError('junction seed has no committed next hop')
                    enter_successor(state)
                elif int(state['current_edge']) == int(state['destination_edge']):
                    if passage_constraint is not None and not passage_constraint.permits_exit(state, None, time):
                        return False
                    return True
                else:
                    successor = state.get('committed_next_edge')
                    if successor is None:
                        raise ValueError('vehicle exhausted its edge without a committed next hop')
                    if traffic_model == 'meso':
                        if passage_constraint is not None and not passage_constraint.permits_exit(state, successor, time):
                            return False
                        enter_successor(state)
                        continue
                    length, speed = candidates.junction(int(state['current_edge']), int(successor))
                    if length < 0 or speed <= 0 or not math.isfinite(length + speed):
                        raise ValueError('junction length and speed must be finite and physically valid')
                    state['in_junction'] = True
                    state['junction_remaining_distance'] = float(length)
                    state['junction_speed'] = min(float(speed), max_speed)
                    state['junction_connection'] = (int(state['current_edge']), int(successor))
                continue
            if duration <= tolerance:
                return False
            speed = float(state['junction_speed'] if in_junction else speeds[int(state['current_edge'])])
            if speed <= 0:
                return False
            elapsed = min(duration, distance / speed)
            state[key] = max(0., distance - speed * elapsed)
            duration = max(0., duration - elapsed)

    # Frozen speeds allow all cars that stay on their current segment to take
    # their ordinary one-second subtraction together. Dictionary fields remain
    # current before every policy callback; crossing cars retain scalar motion.
    moving = np.array([active(state) for state in states], dtype=bool)
    finished = np.array([bool(state.get('arrived', False)) for state in states], dtype=bool)
    distances = np.zeros(len(states), dtype=np.float64)
    motion_speeds = np.zeros(len(states), dtype=np.float64)
    distance_keys = ['junction_remaining_distance' if state.get('in_junction', False)
                     else 'remaining_distance' for state in states]

    def refresh_motion(index: int, state: dict[str, Any]) -> None:
        in_junction = bool(state.get('in_junction', False))
        distance_keys[index] = 'junction_remaining_distance' if in_junction else 'remaining_distance'
        distances[index] = float(state[distance_keys[index]])
        motion_speeds[index] = float(state['junction_speed'] if in_junction
                                     else speeds[int(state['current_edge'])])

    for index in np.flatnonzero(moving).tolist():
        refresh_motion(index, states[index])

    crossing_time = float(start_time)
    pending = [state for state in states if active(state) and (
        (int(start_time) % 30 == 0 and not state.get('in_junction', False)
         and int(state['current_edge']) != int(state['destination_edge']))
        if route_mode else state.get('pending_decision', False))]
    while time < end - tolerance:
        if stop_when_empty and states and finished.all():
            break
        if pending:
            selected = list(decide(pending, frozen_flow, time))
            by_id = {str(item.vehicle_id): item for item in selected}
            if len(by_id) != len(selected) or set(by_id) != {str(s['vehicle_id']) for s in pending}:
                raise ValueError('policy must return exactly one decision per pending vehicle')
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                if (int(item.current_edge) != int(state['current_edge']) or int(item.next_edge) not in
                        _legal_candidates(state, candidates)):
                    raise ValueError('policy selected a next hop outside the legal candidate set')
                if abs(float(item.decision_time) - time) > tolerance:
                    raise ValueError('decision timestamp must match its observation')
                if route_mode:
                    candidates.decision_route(item, int(state['destination_edge']))
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                state['committed_next_edge'] = int(item.next_edge)
                if route_mode:
                    state['committed_route'] = candidates.decision_route(item, int(state['destination_edge']))
                    policy_events.append(ActionEvent(str(item.vehicle_id), event_index_start + len(policy_events),
                        time, int(item.current_edge), int(item.next_edge), int(state['destination_edge'])))
                    state['pending_turn'] = needs_decision(state)
                if getattr(candidates, 'forbid_revisit', False):
                    state['no_revisit_route'] = candidates.full_route(
                        int(state['current_edge']), int(item.next_edge),
                        int(state['destination_edge']), state['visited_edges'])
                state['pending_decision'] = False
                decisions.append(item)
                if not route_mode:
                    action[int(item.next_edge)] += 1.
                    action_events.append(ActionEvent(str(item.vehicle_id), event_index_start + len(action_events),
                        time, int(item.current_edge), int(item.next_edge), int(state['destination_edge'])))

        if route_mode:
            for state in states:
                if active(state) and state.get('pending_turn', False) and not state.get('in_junction', False):
                    successor = state.get('committed_next_edge')
                    if needs_decision(state) and successor is not None:
                        action[int(successor)] += 1.
                        action_events.append(ActionEvent(str(state['vehicle_id']), event_index_start + len(action_events),
                            time, int(state['current_edge']), int(successor), int(state['destination_edge'])))
                    state['pending_turn'] = False

        pending = []
        if passage_constraint is not None:
            passage_constraint.before_step()
        observed_time = time + 1.
        active_before = int(moving.sum())
        inserted: list[float] = []
        # Strictly above the crossing tolerance reproduces move()'s two tests:
        # consume one second, then observe distance <= tolerance before return.
        fast = moving & np.isfinite(distances) & ((distances - motion_speeds) > tolerance)
        distances[fast] -= motion_speeds[fast]
        for index, distance in zip(np.flatnonzero(fast).tolist(), distances[fast].tolist()):
            states[index][distance_keys[index]] = distance
        movement_indices = np.flatnonzero(~finished & ~fast).tolist()
        if passage_constraint is not None:
            movement_indices = passage_constraint.movement_order(movement_indices, states)
        for index in movement_indices:
            state = states[index]
            previous_edge = int(state['current_edge'])
            previous_junction = bool(state.get('in_junction', False))
            was_departed = bool(state.get('departed', False))
            duration = 1.
            if not was_departed:
                scheduled = float(state['departure_time'])
                if not math.isfinite(scheduled):
                    raise ValueError('departure_time must be finite')
                if scheduled > observed_time + tolerance:
                    continue
                actual = max(time, scheduled)
                _record_physical_entry(state, int(state['current_edge']), candidates)
                state['departed'] = True
                moving[index] = True
                state['actual_departure_time'] = actual
                state['departure_time'] = actual
                state['observed_departure_time'] = observed_time
                state['committed_next_edge'] = route_successor(state, initial=True)
                state['pending_decision'] = False
                departure[int(state['current_edge'])] += 1.
                inserted.append(actual)
                duration = observed_time - actual
            if move(state, duration):
                state.update(arrived=True, arrival_time=observed_time, pending_decision=False,
                    committed_next_edge=None, in_junction=False, remaining_distance=0.,
                    junction_remaining_distance=0., junction_connection=None)
                if route_mode:
                    state['pending_turn'] = False
                arrivals[str(state['vehicle_id'])] = observed_time
                moving[index] = False
                finished[index] = True
            elif state.get('in_junction', False):
                state['pending_decision'] = False
                if route_mode:
                    state['pending_turn'] = False
            elif (previous_edge != int(state['current_edge']) or previous_junction or not was_departed):
                if route_mode:
                    state['pending_decision'] = False
                    state['pending_turn'] = needs_decision(state)
                else:
                    state['pending_decision'] = needs_decision(state)
                    if state['pending_decision']:
                        pending.append(state)
            if moving[index]:
                refresh_motion(index, state)

        # The real session charges all previously active cars until the next
        # observation, plus each new departure from its actual insertion time.
        count, segment_start = active_before, time
        for actual in sorted(inserted):
            if actual > segment_start:
                cost_segments.append(ActiveCostSegment(segment_start, actual, count))
            count += 1
            segment_start = actual
        if segment_start < observed_time:
            cost_segments.append(ActiveCostSegment(segment_start, observed_time, count))
        time = observed_time
    active_seconds = sum((segment.end_time - segment.start_time) * segment.active_count for segment in cost_segments)
    return ImaginedWindow(action / total_vehicle_count, departure / total_vehicle_count,
        tuple(decisions), arrivals, float(active_seconds), tuple(cost_segments), time, tuple(action_events),
        tuple(policy_events) if route_mode else None)
