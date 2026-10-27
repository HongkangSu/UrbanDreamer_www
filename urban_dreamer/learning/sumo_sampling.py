"""Shared SUMO collection loop for baseline and world-model policies.

Policy sampling, replay construction and metrics stay with callers. This module
owns the physical episode and builds the same cached candidate observations for
all collection clients; it never changes the chosen session's traffic semantics.
"""
from __future__ import annotations

import functools
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from ..traffic.environment import WindowResult
from ..traffic.event_session import SumoEventSession
from ..traffic.topology import NextHopCandidates
from .observations import build_actor_observations, prepare_actor_window
from .global_reward import RewardAccounting, causal_context, started_count_from_state


def resolve_session_factory(session, step_seconds, backend='traci', *, forbid_revisit=False):
    """None preserves callers' injectable module-level legacy SumoSession."""
    if forbid_revisit and session != 'event':
        raise ValueError('forbid_revisit requires event session')
    if session == 'event':
        options = {'forbid_revisit': True} if forbid_revisit else {}
        return functools.partial(SumoEventSession, step_seconds=step_seconds, backend=backend, **options)
    if session == 'microstep':
        return None
    raise ValueError(session)


@dataclass(frozen=True)
class SampledWindow:
    start_state: dict[str, Any]
    result: WindowResult
    start_context: np.ndarray
    index: int


class SumoEpisodeSampler:
    """Context-managed physical episode with one observation cache per window.

    ``decide(pending, observations, frozen_flow, now, index, context)`` returns
    next-hop decisions. ``before_window(index, state, context)`` can capture
    replay inputs before an injected session mutates its arrays. SampledWindow
    is a frozen record, but its state arrays retain the session's ownership;
    callers retaining mutable session buffers must copy them in before_window.

    ``pad_empty`` emits one synthetic empty 30-second transition. Existing
    clients that need the session's own terminal transition use
    ``advance_empty=True`` instead. Both leave nonempty sessions unchanged.
    """

    def __init__(self, *, session, candidates, deadline_seconds: float,
                 total_vehicle_count: int, context_mode: str = 'window',
                 sort_pending: bool = False, pad_empty: bool = True,
                 advance_empty: bool = False, max_windows: int | None = None,
                 wall_timeout_seconds: float | None = None,
                 timeout_label: str = 'episode', observation_builder: Callable | None = None,
                 reward_objective: str = 'observed_mean'):
        if context_mode not in {'window', 'observation'}:
            raise ValueError(f'unsupported collection context mode {context_mode!r}')
        if not np.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError('deadline_seconds must be finite and positive')
        if max_windows is not None and max_windows < 0:
            raise ValueError('max_windows must be nonnegative')
        self.session = session
        self.candidates = candidates
        physical_model = getattr(session, 'traffic_model', None)
        candidate_model = getattr(candidates, 'traffic_model', None)
        if physical_model is not None and candidate_model is not None and physical_model != candidate_model:
            raise ValueError('collection session and candidate observations must use the same traffic_model')
        if isinstance(session, SumoEventSession) and isinstance(candidates, NextHopCandidates):
            if session.network is not candidates.network:
                raise ValueError('collection session and candidate observations must use the same network')
            if session.forbid_revisit != candidates.forbid_revisit:
                raise ValueError('collection session and candidate observations must use the same forbid_revisit setting')
            if getattr(session, 'routing_mode', 'next_hop') != getattr(candidates, 'routing_mode', 'next_hop'):
                raise ValueError('collection session and candidate observations must use the same routing_mode')
            # The runner owns this immutable topology across episodes. Reuse
            # its destination tables instead of repeating reverse Dijkstra
            # in every fresh physical session.
            session.candidates = candidates
        self.deadline_seconds = float(deadline_seconds)
        if reward_objective not in ('observed_mean', 'legacy_exposure'):
            raise ValueError('unknown reward objective')
        self.reward_objective = reward_objective
        self.session.reward_horizon_seconds = self.deadline_seconds
        self.accounting = RewardAccounting()
        self.denominator = max(int(total_vehicle_count), 1)
        self.context_mode = context_mode
        self.sort_pending = sort_pending
        self.pad_empty = pad_empty
        self.advance_empty = advance_empty
        self.max_windows = max_windows
        self.wall_timeout_seconds = wall_timeout_seconds
        self.timeout_label = timeout_label
        self.observation_builder = observation_builder or build_actor_observations
        self.state = None
        self._started = None
        self._iterated = False
        self._resources = ExitStack()

    def __enter__(self):
        self._started = time.monotonic()
        try:
            if hasattr(self.session, '__enter__') and hasattr(self.session, '__exit__'):
                self.session = self._resources.enter_context(self.session)
            else:
                self._resources.callback(self.session.close)
            self.state = self.session.reset()
            if hasattr(self.session, 'reward_accounting'):
                self.accounting = self.session.reward_accounting
            else:
                self.accounting = RewardAccounting(started_count_from_state(self.state, deadline=self.deadline_seconds),
                    float(self.state.get('cumulative_vehicle_seconds', 0.)))
            self._write_accounting(self.state)
        except BaseException as error:
            self._resources.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._resources.__exit__(exc_type, exc, tb)

    def _write_accounting(self, state):
        state['started_vehicle_count'] = self.accounting.started_count
        state['cumulative_vehicle_seconds'] = self.accounting.cumulative_vehicle_seconds

    def _context(self, timestamp, active_count):
        if self.reward_objective == 'legacy_exposure':
            return np.asarray([min(timestamp / self.deadline_seconds, 1.),
                active_count / self.denominator], dtype=np.float32)
        return causal_context(timestamp, self.deadline_seconds, active_count,
            self.accounting.started_count, self.accounting.cumulative_vehicle_seconds)

    def windows(self, decide: Callable, *, before_window: Callable | None = None) -> Iterator[SampledWindow]:
        if self.state is None:
            raise RuntimeError('enter the sampler before collecting windows')
        if self._iterated:
            raise RuntimeError('an episode sampler can only be iterated once')
        self._iterated = True
        index = 0
        while self.state['time'] < self.deadline_seconds:
            if self.max_windows is not None and index >= self.max_windows:
                break
            if self.session.done and (index or not (self.pad_empty or self.advance_empty)):
                break
            if (self.wall_timeout_seconds is not None
                    and time.monotonic() - self._started > self.wall_timeout_seconds):
                raise TimeoutError(f'{self.timeout_label} episode exceeded episode_wall_timeout_seconds')
            start_state = self.state
            timestamp = float(start_state['time'])
            active_count = start_state.get('active_vehicle_count')
            if active_count is None:
                active_count = sum(bool(vehicle['departed']) and not bool(vehicle['arrived'])
                                   for vehicle in start_state['vehicles'])
            context = self._context(timestamp, active_count)
            context.setflags(write=False)
            if before_window is not None:
                before_window(index, start_state, context)
            actor_window = None

            def select_observed(pending, frozen, now):
                nonlocal actor_window
                if self.sort_pending:
                    pending = sorted(pending, key=lambda vehicle: str(vehicle['vehicle_id']))
                if actor_window is None:
                    actor_window = prepare_actor_window(frozen[:, [2, 1]], self.candidates.network.edge_count)
                observed_context = context
                if self.context_mode == 'observation' or self.reward_objective == 'observed_mean':
                    observed_context = self._context(now,
                        getattr(self.session, 'active_vehicle_count', active_count))
                observations = self.observation_builder(policy_edge_features=actor_window,
                    vehicle_states=pending, candidates=self.candidates,
                    episode_progress=observed_context[0], active_vehicle_fraction=observed_context[1])
                return decide(pending, observations, frozen, now, index, observed_context)

            if self.session.done and not self.advance_empty:
                next_state = dict(start_state, time=timestamp + 30., done=True)
                counts = np.zeros(self.candidates.network.edge_count, dtype=np.int64)
                result = WindowResult((), counts, counts.copy(), 0., next_state, timestamp, timestamp + 30.)
            else:
                result = self.session.advance_window(select_observed)
            if not hasattr(self.session, 'reward_accounting'):
                next_state = dict(result.next_state)
                # Remove any old copied boundary count before deriving new starts.
                next_state.pop('started_vehicle_count', None)
                starts = started_count_from_state(next_state, deadline=self.deadline_seconds)
                self.accounting.update(result.active_vehicle_seconds, max(starts, self.accounting.started_count))
            self._write_accounting(result.next_state)
            self.state = result.next_state
            yield SampledWindow(start_state, result, context, index)
            index += 1
