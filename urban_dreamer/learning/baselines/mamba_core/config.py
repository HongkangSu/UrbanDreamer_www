"""Architecture settings for the vendored MAMBA network core.

Training budgets and optimizer settings belong to GenericConfig, shared by the
three native learned-model baselines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(slots=True)
class MambaConfig:
    obs_dim: int
    action_dim: int = 4
    device: str = 'cpu'
    seed: int = 42
    hidden_size: int = 256
    model_hidden_size: int = 256
    embedding_size: int = 256
    deterministic_size: int = 256
    n_categoricals: int = 32
    n_classes: int = 32
    transformer_layers: int = 3
    transformer_heads: int = 8
    transformer_dropout: float = 0.
    mlp_layers: int = 2
    kl_coef: float = 1.

    def __post_init__(self):
        for name in ('obs_dim', 'action_dim', 'hidden_size', 'model_hidden_size',
                     'embedding_size', 'deterministic_size', 'n_categoricals',
                     'n_classes', 'transformer_layers', 'transformer_heads', 'mlp_layers'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.hidden_size % self.transformer_heads:
            raise ValueError('hidden_size must be divisible by transformer_heads')
        if not 0. <= self.transformer_dropout < 1. or self.kl_coef < 0.:
            raise ValueError('invalid attention dropout or KL coefficient')

    @property
    def stochastic_size(self):
        return self.n_categoricals * self.n_classes

    @property
    def feature_size(self):
        return self.stochastic_size + self.deterministic_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]):
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f'unknown MambaConfig keys: {sorted(unknown)}')
        return cls(**dict(values))
