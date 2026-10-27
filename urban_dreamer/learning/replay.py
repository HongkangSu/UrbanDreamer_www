from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from urban_dreamer.learning.action_events import ActionEvent, SparseActionEvents, pack_action_events


REPLAY_SCHEMA_VERSION = 8


def _tensor(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'{name} must contain finite values')
    if shape is not None and tuple(result.shape) != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(result.shape)}')
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


@dataclass(frozen=True)
class MacroTransition:
    """One physical 30 second interval; conditions belong to this exact interval."""

    flow: torch.Tensor
    action: torch.Tensor
    departure: torch.Tensor
    context: torch.Tensor
    next_flow: torch.Tensor
    active_vehicle_fraction: float
    vehicles: tuple[dict[str, Any], ...] = ()
    time: float = 0.0
    terminal: bool = False
    episode: int = 0
    step_index: int = 0
    action_events: tuple[ActionEvent, ...] | None = None
    # Intermediate next boundaries reuse the next transition's snapshots.
    # Only the episode's last boundary needs separate active snapshots.
    next_vehicles: tuple[dict[str, Any], ...] | None = None

    started_count: int = 0
    cumulative_vehicle_seconds: float = 0.0
    next_started_count: int = 0
    next_cumulative_vehicle_seconds: float = 0.0
    global_reward: float | None = None

    def __post_init__(self) -> None:
        flow = _tensor(self.flow, name='flow')
        if flow.ndim != 2 or flow.shape[-1] != 8 or flow.shape[0] < 1:
            raise ValueError('flow must have shape (E, 8) with E > 0')
        object.__setattr__(self, 'flow', flow)
        object.__setattr__(self, 'next_flow', _tensor(self.next_flow, name='next_flow', shape=tuple(flow.shape)))
        for name in ('action', 'departure'):
            value = _tensor(getattr(self, name), name=name, shape=(flow.shape[0],))
            if bool((value < 0).any()):
                raise ValueError(f'{name} must be non-negative')
            object.__setattr__(self, name, value)
        context = _tensor(self.context, name='context')
        # Offline traffic-only records use the legacy two-field context; they
        # are never used to seed causal policy reward accounting.
        if context.shape == (2,):
            context = torch.cat((context, context.new_zeros(2)))
        object.__setattr__(self, 'context', _tensor(context, name='context', shape=(4,)))
        for name in ('started_count', 'next_started_count'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
        for name in ('cumulative_vehicle_seconds', 'next_cumulative_vehicle_seconds'):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
            object.__setattr__(self, name, value)
        if self.next_started_count < self.started_count or self.next_cumulative_vehicle_seconds < self.cumulative_vehicle_seconds:
            raise ValueError('reward accounting must be cumulative')
        if self.global_reward is not None:
            reward = float(self.global_reward)
            if not np.isfinite(reward):
                raise ValueError('global_reward must be finite')
            object.__setattr__(self, 'global_reward', reward)
        fraction = float(self.active_vehicle_fraction)
        if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
            raise ValueError('active_vehicle_fraction must be in [0, 1]')
        object.__setattr__(self, 'active_vehicle_fraction', fraction)
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError('time must be finite and non-negative')
        object.__setattr__(self, 'time', float(self.time))
        for name in ('episode', 'step_index'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer')
            object.__setattr__(self, name, int(value))
        if not isinstance(self.terminal, bool):
            raise ValueError('terminal must be bool')
        snapshots = tuple(deepcopy(dict(snapshot)) for snapshot in self.vehicles)
        ids = [snapshot['vehicle_id'] for snapshot in snapshots]
        if len(ids) != len(set(ids)):
            raise ValueError('vehicle snapshots must have unique vehicle_id values')
        for snapshot in snapshots:
            for key in ('departure_time', 'remaining_distance', 'junction_remaining_distance', 'junction_speed'):
                if key in snapshot and (not np.isfinite(snapshot[key]) or snapshot[key] < 0):
                    raise ValueError(f'vehicle snapshot {key} must be finite and non-negative')
        object.__setattr__(self, 'vehicles', snapshots)
        if self.next_vehicles is not None:
            following = tuple(deepcopy(dict(snapshot)) for snapshot in self.next_vehicles)
            if len({str(snapshot['vehicle_id']) for snapshot in following}) != len(following):
                raise ValueError('next boundary snapshots must have unique vehicle_id values')
            object.__setattr__(self, 'next_vehicles', following)
        if self.action_events is not None:
            events = tuple(event if isinstance(event, ActionEvent) else ActionEvent(**event)
                           for event in self.action_events)
            indexes = [event.event_index for event in events]
            if len(set(indexes)) != len(indexes):
                raise ValueError('window action event indexes must be unique')
            for event in events:
                if not self.time <= event.decision_time < self.time + 30.:
                    raise ValueError('action event is outside its half-open macro window')
                if max(event.current_edge, event.next_edge, event.destination_edge) >= flow.shape[0]:
                    raise ValueError('action event edge is outside the road graph')
            object.__setattr__(self, 'action_events', events)

    def state_dict(self) -> dict[str, Any]:
        # Events contain only immutable scalars. Serialize them once instead of
        # deep-copying the tuple only to discard it during dictionary conversion.
        result = {field.name: deepcopy(getattr(self, field.name)) for field in fields(self)
                  if field.name != 'action_events'}
        result['vehicles'] = list(result['vehicles'])
        result['action_events'] = None if self.action_events is None else [
            {'vehicle_id': event.vehicle_id, 'event_index': event.event_index,
             'decision_time': event.decision_time, 'current_edge': event.current_edge,
             'next_edge': event.next_edge, 'destination_edge': event.destination_edge}
            for event in self.action_events]
        return result

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> MacroTransition:
        if set(state) != {field.name for field in fields(cls)}:
            raise ValueError('macro transition has an invalid schema')
        return cls(**state)


@dataclass(frozen=True)
class SequenceBatch:
    flow: torch.Tensor
    action: torch.Tensor
    departure: torch.Tensor
    context: torch.Tensor
    active_vehicle_fraction: torch.Tensor
    vehicles: tuple[tuple[dict[str, Any], ...], ...]
    time: torch.Tensor
    terminal: torch.Tensor
    total_travel_minutes: torch.Tensor
    episode_complete: torch.Tensor
    episode: torch.Tensor
    start_index: torch.Tensor
    total_vehicle_count: torch.Tensor
    action_events: SparseActionEvents | None = None
    boundary_vehicles: tuple[tuple[tuple[dict[str, Any], ...], ...], ...] | None = None
    arrival_times: tuple[dict[str, float], ...] | None = None
    transition_flow: torch.Tensor | None = None
    transition_action: torch.Tensor | None = None
    transition_action_events: SparseActionEvents | None = None
    transition_active_vehicle_fraction: torch.Tensor | None = None
    cohort_count: torch.Tensor | None = None
    transition_volume_scope: str | None = None
    transition_capacity_identity: str | None = None
    # Labels use a newly selected observed population for each interval. They
    # are separate from the full observed encoder inputs in ``flow``.
    transition_target_flow: torch.Tensor | None = None
    transition_cohort_count: torch.Tensor | None = None
    # Decoder-only labels need the real boundary following each endpoint.
    # They never enter the dynamics condition or extend the FM sequence.
    decoder_following_flow: torch.Tensor | None = None
    decoder_following_valid: torch.Tensor | None = None
    started_count: torch.Tensor | None = None
    cumulative_vehicle_seconds: torch.Tensor | None = None
    global_reward: torch.Tensor | None = None


class ReplayBuffer:
    """Validated episode segments with T transitions and T+1 physical states."""

    def __init__(self, sequence_length: int, max_episodes: int | None = 8, *, active_only: bool = False) -> None:
        if not isinstance(active_only, bool):
            raise ValueError('active_only must be bool')
        self.active_only = active_only
        self.sequence_length = _positive_integer(sequence_length, 'sequence_length')
        self.max_episodes = None if max_episodes is None else _positive_integer(max_episodes, 'max_episodes')
        self._episodes: list[dict[str, Any]] = []
        self._next_storage_id = 0
        self._sample_order: list[tuple[int, int]] = []
        self._sample_cursor = 0

    @property
    def episodes(self) -> tuple[tuple[MacroTransition, ...], ...]:
        return tuple(episode['transitions'] for episode in self._episodes)

    @property
    def has_samples(self) -> bool:
        return bool(self._candidates())

    @property
    def has_active_seeds(self) -> bool:
        return bool(self._active_seed_candidates())

    def __len__(self) -> int:
        return sum(len(rows) for rows in self.episodes)

    @staticmethod
    def _validate_episode(rows: tuple[MacroTransition, ...]) -> None:
        if any(row.terminal for row in rows[:-1]):
            raise ValueError('terminal transition must be last')
        if rows[0].step_index != 0:
            raise ValueError('episode must begin at step_index zero')
        for index, row in enumerate(rows):
            if row.episode != rows[0].episode or row.step_index != index:
                raise ValueError('episode transitions must be continuous')
            if tuple(row.flow.shape) != tuple(rows[0].flow.shape):
                raise ValueError('episode edge shapes must agree')
            if index and (not torch.equal(rows[index - 1].next_flow, row.flow) or not np.isclose(row.time, rows[index - 1].time + 30., rtol=0., atol=1e-7)):
                raise ValueError('physical states and 30 second boundaries must be continuous')
        events = [event for row in rows if row.action_events is not None for event in row.action_events]
        indexes = [event.event_index for event in events]
        if any(later <= earlier for earlier, later in zip(indexes, indexes[1:])):
            raise ValueError('episode action event indexes must strictly increase')
        if any(later.decision_time < earlier.decision_time for earlier, later in zip(events, events[1:])):
            raise ValueError('episode action events must be chronological')

    def append_episode(
        self, transitions: Sequence[MacroTransition], *, total_travel_minutes: float,
        total_vehicle_count: int,
        require_complete: bool = False,
        arrival_times: Mapping[str, float] | None = None,
    ) -> bool:
        rows = tuple(row if isinstance(row, MacroTransition) else MacroTransition(**row) for row in transitions)
        if not rows:
            return False
        if (isinstance(total_vehicle_count, bool) or not isinstance(total_vehicle_count, (int, np.integer))
                or total_vehicle_count < 0):
            raise ValueError('total_vehicle_count must be a non-negative integer')
        if total_vehicle_count == 0 and any(self._is_active(row) for row in rows):
            raise ValueError('zero-population episodes cannot contain active traffic')
        self._validate_episode(rows)
        if require_complete and not rows[-1].terminal:
            return False
        total = float(total_travel_minutes)
        if not np.isfinite(total) or total < 0:
            raise ValueError('total_travel_minutes must be finite and non-negative')
        arrivals = None
        if arrival_times is not None:
            arrivals = {str(key): float(value) for key, value in arrival_times.items()}
            if len(arrivals) != len(arrival_times) or any(not key or not np.isfinite(value) or value < 0
                                                       for key, value in arrivals.items()):
                raise ValueError('observed arrival_times must map unique vehicle IDs to finite nonnegative times')
        self._episodes.append({'storage_id': self._next_storage_id, 'transitions': rows,
                               'total_travel_minutes': total, 'total_vehicle_count': int(total_vehicle_count),
                               'arrival_times': arrivals})
        self._next_storage_id += 1
        if self.max_episodes is not None:
            self._episodes = self._episodes[-self.max_episodes:]
        self._sample_order = []
        self._sample_cursor = 0
        return True

    @staticmethod
    def _has_active_snapshot(row: MacroTransition) -> bool:
        return any(snapshot.get('departed', False) and not snapshot.get('arrived', False)
                   for snapshot in row.vehicles)

    @classmethod
    def _is_active(cls, row: MacroTransition) -> bool:
        # The integrated fraction also includes vehicles departing inside the window.
        return bool(row.context[1] > 0 or row.active_vehicle_fraction > 0 or cls._has_active_snapshot(row))

    def _available_length(self, rows: tuple[MacroTransition, ...], start: int) -> int:
        length = min(self.sequence_length, len(rows) - start)
        if self.active_only:
            for offset in range(length):
                if not self._is_active(rows[start + offset]):
                    return offset
        return length

    def _candidates(self) -> list[tuple[int, int]]:
        if self.active_only:
            return [(episode['storage_id'], start) for episode in self._episodes
                    for start, row in enumerate(episode['transitions']) if self._is_active(row)]
        return [(episode['storage_id'], start) for episode in self._episodes
                for start in range(max(1, len(episode['transitions']) - self.sequence_length + 1))]

    def _active_seed_candidates(self, *, after_demand_seconds: float | None = None) -> list[tuple[int, int]]:
        if after_demand_seconds is not None:
            if (isinstance(after_demand_seconds, bool) or not np.isfinite(after_demand_seconds)
                    or after_demand_seconds < 0):
                raise ValueError('after_demand_seconds must be finite and non-negative')
        return [(episode['storage_id'], start) for episode in self._episodes
                for start, row in enumerate(episode['transitions']) if self._has_active_snapshot(row)
                and (after_demand_seconds is None or (
                    row.time >= after_demand_seconds
                    and row.started_count == episode['total_vehicle_count']))]

    def active_seed_summary(self, *, after_demand_seconds: float | None = None) -> dict[str, float]:
        """Describe eligible observed boundaries without consuming any RNG."""
        picks = self._active_seed_candidates(after_demand_seconds=after_demand_seconds)
        by_id = {episode['storage_id']: episode for episode in self._episodes}
        rows = [by_id[key]['transitions'][start] for key, start in picks]
        return dict(imagination_eligible_seed_count=float(len(rows)),
            imagination_eligible_seed_time_min=min((row.time for row in rows), default=0.),
            imagination_eligible_seed_time_max=max((row.time for row in rows), default=0.),
            imagination_eligible_seed_population_min=float(min((row.started_count for row in rows), default=0)),
            imagination_eligible_seed_population_max=float(max((row.started_count for row in rows), default=0)))

    def sample_active_seeds(self, batch_size: int, rng: np.random.Generator,
                            *, initial_only: bool = False,
                            after_demand_seconds: float | None = None,
                            causal_seed_only: bool = False) -> SequenceBatch:
        """Sample observed active start states without advancing the training sampler."""
        if not isinstance(causal_seed_only, bool):
            raise ValueError('causal_seed_only must be boolean')
        batch_size = _positive_integer(batch_size, 'batch_size')
        candidates = self._active_seed_candidates(after_demand_seconds=after_demand_seconds)
        if not candidates:
            raise ValueError('cannot sample replay without eligible active vehicle snapshots')
        picks = [candidates[int(index)] for index in rng.integers(len(candidates), size=batch_size)]
        return self._batch_from_picks(picks, seed_only=initial_only,
            causal_seed_only=causal_seed_only or after_demand_seconds is not None)

    def sample_terminal_seeds(
        self, batch_size: int, rng: np.random.Generator,
        deadline: float, horizon_seconds: float,
    ) -> SequenceBatch | None:
        """Sample observed active states whose imagination can reach the deadline.

        Sampling is uniform with replacement and uses only the supplied RNG;
        the ordinary replay sampler's order and cursor are left untouched.
        No future vehicle snapshot is needed to select or initialize a seed.
        """
        batch_size = _positive_integer(batch_size, 'batch_size')
        for name, value in (('deadline', deadline), ('horizon_seconds', horizon_seconds)):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        by_id = {episode['storage_id']: episode for episode in self._episodes}
        candidates = [
            (key, start) for key, start in self._active_seed_candidates()
            if (by_id[key]['transitions'][start].time < deadline
                and by_id[key]['transitions'][start].time + horizon_seconds >= deadline)
        ]
        if not candidates:
            return None
        picks = [candidates[int(index)] for index in rng.integers(len(candidates), size=batch_size)]
        return self._batch_from_picks(picks, seed_only=True)

    def iter_epoch(self, batch_size: int, rng: np.random.Generator) -> Iterator[SequenceBatch]:
        """Visit each eligible sequence start once, retaining a final partial batch.

        Independent of the cycling online sampler. Epoch-boundary checkpoints
        restore the RNG before creating the next permutation.
        """
        batch_size = _positive_integer(batch_size, 'batch_size')
        candidates = self._candidates()
        if not candidates:
            raise ValueError('cannot iterate an empty replay buffer')
        order = rng.permutation(len(candidates))
        for start in range(0, len(order), batch_size):
            yield self._batch_from_picks([candidates[int(index)]
                                          for index in order[start:start + batch_size]])

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> SequenceBatch:
        batch_size = _positive_integer(batch_size, 'batch_size')
        candidates = self._candidates()
        if not candidates:
            raise ValueError('cannot sample an empty replay buffer')
        picks = []
        while len(picks) < batch_size:
            if self._sample_cursor == len(self._sample_order):
                self._sample_order = [candidates[int(index)] for index in rng.permutation(len(candidates))]
                self._sample_cursor = 0
            take = min(batch_size - len(picks), len(self._sample_order) - self._sample_cursor)
            picks.extend(self._sample_order[self._sample_cursor:self._sample_cursor + take])
            self._sample_cursor += take
        return self._batch_from_picks(picks)

    def _batch_from_picks(self, picks: list[tuple[int, int]], *, seed_only=False,
                          causal_seed_only=False) -> SequenceBatch:
        by_id = {episode['storage_id']: episode for episode in self._episodes}
        if causal_seed_only:
            # Explicit causal imagination admits only observed boundary
            # inputs. Do not even read next_flow, future actions, final costs,
            # completion, or archived arrival times into this collector batch.
            rows = [by_id[key]['transitions'][start] for key, start in picks]
            flow = torch.stack([row.flow for row in rows])[:, None]
            zero = flow.new_zeros(len(rows), 1, flow.shape[2])
            return SequenceBatch(flow=flow, action=zero, departure=zero.clone(),
                context=torch.stack([row.context for row in rows])[:, None],
                active_vehicle_fraction=flow.new_tensor([[sum(bool(s.get('departed')) and not bool(s.get('arrived'))
                    for s in row.vehicles) / max(row.started_count, 1)] for row in rows]),
                vehicles=tuple(deepcopy(row.vehicles) for row in rows),
                time=torch.tensor([row.time for row in rows], dtype=torch.float64),
                terminal=torch.zeros(len(rows), 1, dtype=torch.bool),
                total_travel_minutes=torch.zeros(len(rows)),
                episode_complete=torch.zeros(len(rows), dtype=torch.bool),
                episode=torch.tensor([row.episode for row in rows]),
                start_index=torch.tensor([start for _, start in picks]),
                # At post-demand seeds this equals the episode population. At
                # earlier seeds only observed departures are causal; the model
                # must generate its own future population instead of reading N.
                total_vehicle_count=torch.tensor([row.started_count for row in rows]),
                started_count=torch.tensor([row.started_count for row in rows]),
                cumulative_vehicle_seconds=torch.tensor([row.cumulative_vehicle_seconds for row in rows],
                    dtype=torch.float64))
        length = 1 if seed_only else min(self._available_length(by_id[key]['transitions'], start) for key, start in picks)
        windows = [by_id[key]['transitions'][start:start + length] for key, start in picks]
        boundaries = []
        for (key, start), rows in zip(picks, windows):
            if seed_only:
                # Imagined motion owns its start snapshot and never reads the
                # archived future snapshots. Avoid copying those unused cars.
                boundaries = None
                break
            episode_rows = by_id[key]['transitions']
            endpoint = (episode_rows[start + length].vehicles if start + length < len(episode_rows)
                        else rows[-1].next_vehicles)
            if endpoint is None:
                boundaries = None
                break
            boundaries.append(deepcopy(tuple(row.vehicles for row in rows) + (endpoint,)))
        boundaries = None if boundaries is None else tuple(boundaries)
        def stack(name: str) -> torch.Tensor:
            return torch.stack([torch.stack([getattr(row, name) for row in rows]) for rows in windows])
        return SequenceBatch(
            flow=torch.stack([torch.stack([rows[0].flow, *[row.next_flow for row in rows]]) for rows in windows]),
            action=stack('action'), departure=stack('departure'), context=stack('context'),
            active_vehicle_fraction=torch.tensor([[row.active_vehicle_fraction for row in rows] for rows in windows]),
            vehicles=(tuple(sequence[0] for sequence in boundaries) if boundaries is not None
                      else tuple(deepcopy(rows[0].vehicles) for rows in windows)),
            time=torch.tensor([rows[0].time for rows in windows], dtype=torch.float64),
            terminal=torch.tensor([[row.terminal for row in rows] for rows in windows], dtype=torch.bool),
            total_travel_minutes=torch.tensor([by_id[key]['total_travel_minutes'] for key, _ in picks]),
            episode_complete=torch.tensor([by_id[key]['transitions'][-1].terminal for key, _ in picks], dtype=torch.bool),
            episode=torch.tensor([rows[0].episode for rows in windows]),
            start_index=torch.tensor([start for _, start in picks]),
            total_vehicle_count=torch.tensor([by_id[key]['total_vehicle_count'] for key, _ in picks], dtype=torch.long),
            started_count=torch.tensor([rows[0].started_count for rows in windows], dtype=torch.long),
            cumulative_vehicle_seconds=torch.tensor([rows[0].cumulative_vehicle_seconds for rows in windows], dtype=torch.float64),
            global_reward=(torch.tensor([[row.global_reward for row in rows] for rows in windows])
                           if all(row.global_reward is not None for rows in windows for row in rows) else None),
            action_events=pack_action_events(windows) if all(row.action_events is not None
                                                           for rows in windows for row in rows) else None,
            boundary_vehicles=boundaries,
            arrival_times=(tuple(dict(by_id[key]['arrival_times']) for key, _ in picks)
                           if all(by_id[key]['arrival_times'] is not None for key, _ in picks) else None),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            'schema_version': REPLAY_SCHEMA_VERSION,
            'sequence_length': self.sequence_length, 'max_episodes': self.max_episodes,
            'active_only': self.active_only,
            'episodes': [{'storage_id': episode['storage_id'], 'total_travel_minutes': episode['total_travel_minutes'],
                          'total_vehicle_count': episode['total_vehicle_count'],
                          'arrival_times': deepcopy(episode['arrival_times']),
                          'transitions': [row.state_dict() for row in episode['transitions']]} for episode in self._episodes],
            'next_storage_id': self._next_storage_id,
            'sample_order': [list(key) for key in self._sample_order], 'sample_cursor': self._sample_cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {'schema_version', 'sequence_length', 'max_episodes', 'active_only', 'episodes', 'next_storage_id', 'sample_order', 'sample_cursor'}
        if set(state) != expected or state['schema_version'] != REPLAY_SCHEMA_VERSION:
            raise ValueError('unsupported replay schema; regenerate replay with the current version')
        if (state['sequence_length'] != self.sequence_length or state['max_episodes'] != self.max_episodes
                or not isinstance(state['active_only'], bool) or state['active_only'] != self.active_only):
            raise ValueError('replay configuration mismatch')
        restored = ReplayBuffer(self.sequence_length, self.max_episodes, active_only=self.active_only)
        for raw in state['episodes']:
            if set(raw) != {'storage_id', 'total_travel_minutes', 'total_vehicle_count', 'transitions', 'arrival_times'}:
                raise ValueError('replay episode has an invalid population schema')
            rows = tuple(MacroTransition.from_state_dict(row) for row in raw['transitions'])
            if not restored.append_episode(rows, total_travel_minutes=raw['total_travel_minutes'],
                                           total_vehicle_count=raw['total_vehicle_count'], arrival_times=raw['arrival_times']):
                raise ValueError('replay contains an empty episode')
            restored._episodes[-1]['storage_id'] = raw['storage_id']
        ids = [episode['storage_id'] for episode in restored._episodes]
        if len(ids) != len(set(ids)) or any(not isinstance(value, int) or value < 0 for value in ids):
            raise ValueError('replay storage IDs are invalid')
        restored._next_storage_id = state['next_storage_id']
        if not isinstance(restored._next_storage_id, int) or restored._next_storage_id < max(ids, default=-1) + 1:
            raise ValueError('replay next_storage_id is invalid')
        restored._sample_order = [tuple(key) for key in state['sample_order']]
        restored._sample_cursor = state['sample_cursor']
        if restored._sample_order and (len(restored._sample_order) != len(restored._candidates()) or set(restored._sample_order) != set(restored._candidates())):
            raise ValueError('replay sampler keys are invalid')
        if not isinstance(restored._sample_cursor, int) or not 0 <= restored._sample_cursor <= len(restored._sample_order):
            raise ValueError('replay sampler cursor is invalid')
        self.__dict__.update(restored.__dict__)
