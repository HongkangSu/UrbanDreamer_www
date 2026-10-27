"""Convert real environment decisions into task-independent learning arrays.

Only observation/action data, identities, timestamps and environment rewards
cross this boundary. Window rewards are integrated uniformly over their
recorded intervals; no physical state or counterfactual simulator is consulted.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


ARRAY_FIELDS = (
    'obs', 'actions', 'rewards', 'next_obs', 'terminals', 'masks',
    'next_masks', 'agent_ids', 'frame_indices',
)


def pack_observation(features, masks):
    """Flatten arbitrary candidate features and append their availability mask."""
    features = np.asarray(features, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.bool_)
    if features.ndim != 3 or masks.ndim != 2 or features.shape[:2] != masks.shape:
        raise ValueError('expected features [B,K,F] and masks [B,K]')
    if not np.isfinite(features).all():
        raise ValueError('observation features must be finite')
    return np.concatenate((features.reshape(len(features), features.shape[1] * features.shape[2]),
                           masks.astype(np.float32)), axis=-1)


def _reward_trace(windows):
    intervals = []
    for window in windows:
        start, end, reward = (float(window[key]) for key in ('start_time', 'end_time', 'reward'))
        if not np.isfinite([start, end, reward]).all() or end < start:
            raise ValueError('reward intervals must be finite and have nonnegative duration')
        if end == start:
            if reward != 0.:
                raise ValueError('nonzero reward has no interval over which to integrate')
            continue
        intervals.append((start, end, reward))
    intervals.sort(key=lambda interval: interval[0])
    if not intervals:
        return np.empty((0, 3), dtype=np.float64)
    trace = np.asarray(intervals, dtype=np.float64)
    if np.any(trace[1:, 0] < trace[:-1, 1]):
        raise ValueError('reward intervals must not overlap')
    return trace


def _reward_integral(trace, times):
    """Integral from the beginning of a nonoverlapping piecewise-constant trace."""
    index = np.searchsorted(trace[:, 0], times, side='right') - 1
    valid = index >= 0
    chosen = index.clip(0)
    prefix = np.concatenate(([0.], np.cumsum(trace[:, 2], dtype=np.float64)))
    fraction = np.clip((times - trace[chosen, 0]) / (trace[chosen, 1] - trace[chosen, 0]), 0., 1.)
    return np.where(valid, prefix[chosen] + fraction * trace[chosen, 2], 0.)


def _empty_episode(obs_dim, action_dim, episode_end):
    obs_dim = 0 if obs_dim is None else int(obs_dim)
    action_dim = 0 if action_dim is None else int(action_dim)
    if min(obs_dim, action_dim) < 0:
        raise ValueError('observation and action dimensions must be nonnegative')
    return dict(obs=np.empty((0, obs_dim), dtype=np.float32),
        actions=np.empty(0, dtype=np.int64), rewards=np.empty(0, dtype=np.float32),
        next_obs=np.empty((0, obs_dim), dtype=np.float32), terminals=np.empty(0, dtype=np.bool_),
        masks=np.empty((0, action_dim), dtype=np.bool_),
        next_masks=np.empty((0, action_dim), dtype=np.bool_),
        agent_ids=np.empty(0, dtype=np.int64), frame_indices=np.empty(0, dtype=np.int64),
        frames=[], episode_end=float(episode_end), agent_count=0, vehicle_ids=())


def build_episode(rows, windows, obs_dim=None, action_dim=None):
    """Build every real decision transition and simultaneous-decision frame.

    A successor is the same agent's next actual decision, in timestamp order.
    Its reward integrates the common environment trace over that interval.
    The last decision receives the remaining team reward through episode end,
    and terminates with zero successor observation and no available action.
    Repeated decisions, including repeats within a reward window, are retained.

    Returned arrays own their storage; callers may release the collector's raw
    rows and windows immediately. Input dictionaries and arrays are never edited.
    """
    rows, windows = list(rows), list(windows)
    completion = windows[-1].get('terminal') if windows else None
    if completion is not None and not bool(completion):
        raise ValueError('build_episode requires a completed environment episode')
    trace = _reward_trace(windows)
    episode_end = float(trace[-1, 1]) if len(trace) else 0.
    if not rows:
        return _empty_episode(obs_dim, action_dim, episode_end)
    if not len(trace):
        raise ValueError('decision rows require recorded reward intervals')

    raw_features = np.stack([np.asarray(row['features'], dtype=np.float32) for row in rows])
    masks = np.stack([np.asarray(row['mask'], dtype=np.bool_) for row in rows])
    observations = pack_observation(raw_features, masks)
    if obs_dim is not None and observations.shape[1] != obs_dim:
        raise ValueError('observation dimension differs from requested obs_dim')
    if action_dim is not None and masks.shape[1] != action_dim:
        raise ValueError('action mask dimension differs from requested action_dim')
    actions = np.asarray([row['action'] for row in rows], dtype=np.int64)
    if np.any(actions < 0) or np.any(actions >= masks.shape[1]):
        raise ValueError('recorded action index is out of range')
    if not masks[np.arange(len(rows)), actions].all():
        raise ValueError('recorded action is unavailable under its observation mask')
    times = np.asarray([row['time'] for row in rows], dtype=np.float64)
    if (not np.isfinite(times).all() or np.any(times < trace[0, 0])
            or np.any(times > episode_end)):
        raise ValueError('decision timestamps must lie inside the recorded episode')

    identity_to_slot = {}
    agent_ids = np.empty(len(rows), dtype=np.int64)
    per_agent = defaultdict(list)
    for index, row in enumerate(rows):
        identity = row['vehicle_id']
        if identity not in identity_to_slot:
            identity_to_slot[identity] = len(identity_to_slot)
        agent_ids[index] = identity_to_slot[identity]
        per_agent[agent_ids[index]].append(index)

    next_observations = np.zeros_like(observations)
    next_masks = np.zeros_like(masks)
    terminals = np.ones(len(rows), dtype=np.bool_)
    next_times = np.full(len(rows), episode_end, dtype=np.float64)
    for indices in per_agent.values():
        indices.sort(key=lambda index: (times[index], index))
        for current, successor in zip(indices, indices[1:]):
            next_observations[current] = observations[successor]
            next_masks[current] = masks[successor]
            terminals[current] = False
            next_times[current] = times[successor]
    rewards = (_reward_integral(trace, next_times) - _reward_integral(trace, times)).astype(np.float32)
    frame_times, frame_indices = np.unique(times, return_inverse=True)
    result = dict(obs=observations, actions=actions, rewards=rewards,
        next_obs=next_observations, terminals=terminals, masks=masks,
        next_masks=next_masks, agent_ids=agent_ids, frame_indices=frame_indices.astype(np.int64),
        episode_end=episode_end, agent_count=len(identity_to_slot), vehicle_ids=tuple(identity_to_slot))
    frames = []
    for frame_index, timestamp in enumerate(frame_times):
        indices = np.flatnonzero(frame_indices == frame_index)
        frames.append(dict(time=float(timestamp), frame_index=frame_index,
            **{key: result[key][indices].copy() for key in ARRAY_FIELDS}))
    result['frames'] = frames
    return result
