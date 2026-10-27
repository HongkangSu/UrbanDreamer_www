"""City-independent budgets for the three generic discrete-action learners."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True, slots=True)
class GenericConfig:
    obs_dim: int
    action_dim: int
    device: str = 'cpu'
    seed: int = 42
    hidden_dim: int = 256
    gamma: float = .99
    gae_lambda: float = .95
    model_updates: int = 4
    policy_updates: int = 4
    batch_size: int = 256
    sequence_length: int = 16
    sequence_batch_size: int = 16
    mamba_sequence_batch_size: int = 2
    mamba_imagination_starts: int = 8
    imagination_horizon: int = 15
    imagination_starts: int = 16
    replay_capacity: int = 200000
    replay_episodes: int = 32
    model_rollout_length: int = 1
    rollout_batch_size: int = 512
    model_replay_capacity: int = 50000
    real_ratio: float = .1
    policy_sample_to_insert_ratio: float = 1.
    model_sample_to_insert_ratio: float = .1
    ensemble_size: int = 5
    elite_count: int = 3
    target_entropy_scale: float = .5
    alpha_min: float = 1e-4
    alpha_max: float = 1.
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    model_lr: float = 3e-4
    target_tau: float = .005
    max_agents_per_chunk: int = 32

    def __post_init__(self):
        for name in ('obs_dim', 'action_dim', 'hidden_dim', 'model_updates', 'policy_updates',
                     'batch_size', 'sequence_length', 'sequence_batch_size', 'mamba_sequence_batch_size',
                     'mamba_imagination_starts', 'imagination_horizon',
                     'imagination_starts', 'replay_capacity', 'replay_episodes', 'model_rollout_length',
                     'rollout_batch_size', 'model_replay_capacity', 'ensemble_size', 'elite_count',
                     'max_agents_per_chunk'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.sequence_length < 2 or self.elite_count > self.ensemble_size:
            raise ValueError('sequence_length must be >=2 and elite_count <= ensemble_size')
        for name in ('gamma', 'gae_lambda', 'real_ratio', 'target_tau'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0. < value <= 1.:
                raise ValueError(f'{name} must be in (0, 1]')
        for name in ('actor_lr', 'critic_lr', 'model_lr'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.target_entropy_scale) or not 0. < self.target_entropy_scale <= 1.:
            raise ValueError('target_entropy_scale must be in (0, 1]')
        if not (math.isfinite(self.alpha_min) and math.isfinite(self.alpha_max)
                and 0. < self.alpha_min < self.alpha_max):
            raise ValueError('alpha bounds must satisfy 0 < alpha_min < alpha_max')
        for name in ('policy_sample_to_insert_ratio', 'model_sample_to_insert_ratio'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.:
                raise ValueError(f'{name} must be finite and nonnegative')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')

    def protocol(self):
        values = asdict(self)
        values.pop('device')
        return values
