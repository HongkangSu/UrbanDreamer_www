"""The seven learned baselines and two routing heuristics used in the paper."""
from __future__ import annotations

METHODS: dict[str, type] = {}


def build_method(name, args, context):
    """Construct a model-free learner around the shared physical collector."""
    if name not in METHODS:
        raise ValueError(f'unknown model-free baseline {name!r}; choose {sorted(METHODS)}')
    return METHODS[name](args, context)


from . import gnac, ppo, iql  # noqa: E402,F401
