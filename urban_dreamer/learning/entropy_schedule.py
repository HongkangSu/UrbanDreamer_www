"""Shared entropy coefficients indexed by completed training episodes."""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from urban_dreamer.workflow.config import MethodConfig


SCHEDULE_FIELDS = (
    'actor_entropy_initial_coefficient',
    'actor_entropy_final_coefficient',
    'actor_entropy_hold_fraction',
    'actor_entropy_end_fraction',
)


def entropy_schedule_episodes(args) -> int:
    """Keep the annealing horizon independent of an extended stop budget.

    Old checkpoints have no override and use their saved episode budget.
    Zero-episode evaluation runs still require a positive schedule horizon.
    """
    get = args.get if isinstance(args, Mapping) else lambda key, default=None: getattr(args, key, default)
    override = get('entropy_schedule_episodes')
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, Integral) or override < 1:
            raise ValueError('entropy_schedule_episodes must be a positive integer')
        return int(override)
    return max(1, int(get('episodes', 5000)))


def validate_entropy_schedule_settings(method_config: MethodConfig) -> None:
    """Validate direct dataclass construction as well as parsed configuration."""
    for name in SCHEDULE_FIELDS:
        value = getattr(method_config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    initial = method_config.actor_entropy_initial_coefficient
    final = method_config.actor_entropy_final_coefficient
    hold = method_config.actor_entropy_hold_fraction
    end = method_config.actor_entropy_end_fraction
    if not 0 <= final <= initial:
        raise ValueError('actor entropy coefficients must satisfy 0 <= final <= initial')
    if not 0 <= hold < end <= 1:
        raise ValueError('actor entropy fractions must satisfy 0 <= hold < end <= 1')


def entropy_schedule_profile(
    method_config: MethodConfig,
    *,
    completed_episodes: int,
    total_episodes: int,
) -> dict[str, str | float | int]:
    """Return the coefficient for the next update without changing policy logits.

    Episode counts use the fixed annealing horizon, including episodes restored
    from a checkpoint. All updates within one episode receive the same coefficient.
    ``annealed`` holds the initial coefficient, then follows a cosine decay to
    the final floor. Legacy and population modes retain the historical base
    coefficient; population scaling is applied separately by the loss helper.
    """
    for name, value, minimum in (
        ('completed_episodes', completed_episodes, 0),
        ('total_episodes', total_episodes, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    mode = method_config.actor_entropy_mode
    if mode not in {'annealed', 'legacy', 'population', 'off'}:
        raise ValueError('entropy mode must be annealed, legacy, population, or off')
    progress = min(completed_episodes / total_episodes, 1.)
    coefficient = 0. if mode == 'off' else .002
    if mode == 'annealed':
        validate_entropy_schedule_settings(method_config)
        initial = method_config.actor_entropy_initial_coefficient
        final = method_config.actor_entropy_final_coefficient
        hold = method_config.actor_entropy_hold_fraction
        end = method_config.actor_entropy_end_fraction
        if progress <= hold:
            coefficient = initial
        elif progress >= end:
            coefficient = final
        else:
            phase = (progress - hold) / (end - hold)
            coefficient = final + .5 * (initial - final) * (1. + math.cos(math.pi * phase))
    return {
        'mode': mode,
        'coefficient': coefficient,
        'progress': progress,
        'completed_episodes': int(completed_episodes),
        'total_episodes': int(total_episodes),
    }
