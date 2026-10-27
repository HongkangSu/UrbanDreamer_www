from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..workflow.config import NONARRIVAL_PENALTY
from ..learning.policy_temperature import read_actor_temperature
from ..learning.critic_budget import bounded_physical_return, validate_bounded_contract

CANDIDATE_FEATURE_DIM = 10
LATENT_DIM = 64
CONTEXT_DIM = 4
HIDDEN_DIM = 256
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 0.95


@dataclass(frozen=True)
class ActionSelection:
    candidate_index: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor

    def indices_and_entropy(self) -> tuple[list[int], list[float]]:
        """Read collection statistics with one device-to-host transfer on CUDA."""
        if self.candidate_index.device.type == 'cpu':
            return self.candidate_index.tolist(), self.entropy.tolist()
        # Float64 preserves both integer indices and the original entropy values.
        packed = torch.stack((self.candidate_index.to(torch.float64),
                              self.entropy.to(torch.float64)), dim=-1).cpu().tolist()
        return [int(row[0]) for row in packed], [row[1] for row in packed]


def _make_mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.Tanh(),
                         nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh())


def _initialize(module: nn.Module, head: nn.Linear, gain: float) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(head.weight, gain=gain)


def _validate_prior(weight: float, temperature_seconds: float) -> None:
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("shortest-path prior weight must be finite and nonnegative")
    if not math.isfinite(temperature_seconds) or temperature_seconds <= 0:
        raise ValueError("shortest-path prior temperature_seconds must be finite and positive")


def set_shortest_path_prior(actor: nn.Module, weight: float, temperature_seconds: float) -> None:
    """Set nontrainable behavior settings; callers restore them from episode/config."""
    _validate_prior(weight, temperature_seconds)
    actor.prior_weight = float(weight)
    actor.prior_temperature_seconds = float(temperature_seconds)


def shortest_path_prior_weight(episode: int, *, initial_weight: float = 1.,
                              hold_episodes: int = 200, off_episode: int = 1000) -> float:
    """Hold through ``hold_episodes``, then cosine-decay to zero at ``off_episode``."""
    for name, value in (("episode", episode), ("hold_episodes", hold_episodes),
                        ("off_episode", off_episode)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    _validate_prior(initial_weight, 1.)
    if off_episode <= hold_episodes:
        raise ValueError("off_episode must be greater than hold_episodes")
    if episode <= hold_episodes:
        return float(initial_weight)
    if episode >= off_episode:
        return 0.
    progress = (episode - hold_episodes) / (off_episode - hold_episodes)
    return float(initial_weight) * 0.5 * (1. + math.cos(math.pi * progress))


def apply_shortest_path_prior(logits: torch.Tensor, features: torch.Tensor,
                             mask: torch.Tensor, *, weight: float,
                             temperature_seconds: float,
                             prior_costs_seconds: torch.Tensor | None = None) -> torch.Tensor:
    """Bias legal logits by full via-candidate travel cost, measured in seconds.

    Feature 4 is log1p(junction + candidate edge + downstream seconds) /
    log1p(3600). Costs are detached conditions, not another learned objective.
    Explicit snapshot costs override this static prior without changing features.
    Input logits are already legally masked when this is called by an actor.
    """
    _validate_prior(weight, temperature_seconds)
    if weight == 0:
        return logits
    if logits.ndim != 2 or features.shape != (*logits.shape, CANDIDATE_FEATURE_DIM):
        raise ValueError("prior requires logits (B, K) and candidate features (B, K, 10)")
    if mask.shape != logits.shape or logits.shape[-1] == 0:
        raise ValueError("candidate mask must have shape (B, K) with K positive")
    legal = mask.to(device=logits.device, dtype=torch.bool)
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("every decision must have at least one legal candidate")
    if prior_costs_seconds is None:
        # Mask before exponentiation: padding must not introduce NaNs.
        encoded = features[..., 4].detach().to(logits).masked_fill(~legal, 0.)
        if not bool(torch.isfinite(encoded).all()) or bool((encoded < 0).any()):
            raise ValueError("legal free-flow cost features must be finite and nonnegative")
        costs = torch.expm1(encoded * math.log1p(3600.))
    else:
        if prior_costs_seconds.shape != logits.shape:
            raise ValueError("prior_costs_seconds must have shape (B, K)")
        costs = prior_costs_seconds.detach().to(logits).masked_fill(~legal, 0.)
    if (not bool(torch.isfinite(costs).all()) or bool((costs < 0).any())
            or not bool(torch.isfinite(logits[legal]).all())):
        raise ValueError("legal prior costs and actor logits must be finite with nonnegative costs")
    minimum = costs.masked_fill(~legal, torch.inf).amin(dim=-1, keepdim=True)
    bias = (costs - minimum).masked_fill(~legal, 0.) * (weight / temperature_seconds)
    if not bool(torch.isfinite(bias).all()):
        raise ValueError("shortest-path prior bias must be finite")
    return (logits - bias).masked_fill(~legal, -torch.inf)


class NextHopActor(nn.Module):
    """One shared scorer for any number of destination-reachable successor edges."""
    def __init__(self) -> None:
        super().__init__()
        self.candidate_scorer = _make_mlp(CANDIDATE_FEATURE_DIM)
        self.policy_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.policy_head, 0.01)
        set_shortest_path_prior(self, 0., 30.)

    def forward_logits(self, features: torch.Tensor, mask: torch.Tensor, *,
                       prior_costs_seconds: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != CANDIDATE_FEATURE_DIM:
            raise ValueError(f"expected candidate features (B, K, {CANDIDATE_FEATURE_DIM})")
        if mask.shape != features.shape[:2] or features.shape[1] == 0:
            raise ValueError("candidate mask must have shape (B, K) with K positive")
        mask = mask.to(device=features.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every decision must have at least one legal candidate")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("candidate features must be finite")
        score_gauge = getattr(self, 'score_gauge', 'legacy')
        if score_gauge not in {'legacy', 'pre_head_legal_reference_v1'}:
            raise ValueError('unsupported actor score_gauge')
        hidden = self.candidate_scorer(features)
        if score_gauge == 'pre_head_legal_reference_v1':
            # Subtract before projection: subtracting large raw logits afterward
            # cannot recover the precision lost to their common offset.
            first = mask.to(torch.int64).argmax(dim=-1)
            reference = hidden[torch.arange(len(hidden), device=hidden.device), first]
            logits = torch.nn.functional.linear(
                hidden - reference[:, None, :], self.policy_head.weight, bias=None).squeeze(-1)
        else:
            logits = self.policy_head(hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        logits = apply_shortest_path_prior(logits, features, mask, weight=self.prior_weight,
                                         temperature_seconds=self.prior_temperature_seconds,
                                         prior_costs_seconds=prior_costs_seconds)
        temperature = read_actor_temperature(self)['temperature']
        return logits if temperature == 1. else logits / temperature

    def sample_action(self, features: torch.Tensor, mask: torch.Tensor, *, deterministic: bool = False,
                      prior_costs_seconds: torch.Tensor | None = None,
                      temperature: float = 1.) -> ActionSelection:
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("action temperature must be finite and positive")
        logits = (self.forward_logits(features, mask) if prior_costs_seconds is None else
                  self.forward_logits(features, mask, prior_costs_seconds=prior_costs_seconds))
        distribution = Categorical(logits=logits / temperature)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())


class CentralizedLatentValueCritic(nn.Module):
    def __init__(self, *, bounded: bool = False) -> None:
        super().__init__()
        self.bounded = bounded
        if bounded:
            self.register_buffer("_bounded_cohort_contract", torch.tensor(1, dtype=torch.int64))
            self.register_load_state_dict_pre_hook(validate_bounded_contract)
        self.trunk = _make_mlp(LATENT_DIM + CONTEXT_DIM)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.value_head, 1.0)

    def forward(self, z: torch.Tensor, context: torch.Tensor, *,
                budget: torch.Tensor | None = None) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != LATENT_DIM:
            raise ValueError(f"expected z shape (B, E, {LATENT_DIM})")
        if context.shape != (z.shape[0], CONTEXT_DIM):
            raise ValueError(f"expected context shape (B, {CONTEXT_DIM})")
        pooled = z.mean(dim=1)
        raw = self.value_head(self.trunk(torch.cat((pooled, context.to(pooled)), dim=-1)))
        return bounded_physical_return(raw, budget) if self.bounded else raw


def lambda_return(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    *,
    terminal: torch.Tensor,
) -> torch.Tensor:
    if rewards.dim() != 1 or next_values.dim() != 1:
        raise ValueError(
            f"lambda_return expects 1D tensors, got rewards={tuple(rewards.shape)} next_values={tuple(next_values.shape)}"
        )
    if rewards.shape != next_values.shape:
        raise ValueError(
            f"rewards and next_values must match, got {tuple(rewards.shape)} and {tuple(next_values.shape)}"
        )
    terminal_mask = terminal.to(
        device=rewards.device,
        dtype=torch.bool,
    ).reshape(-1)
    if terminal_mask.shape != rewards.shape:
        raise ValueError(f"expected terminal shape {tuple(rewards.shape)}, got {tuple(terminal_mask.shape)}")

    returns = torch.zeros_like(rewards)
    if rewards.numel() == 0:
        return returns
    running = next_values[-1]
    for idx in reversed(range(int(rewards.shape[0]))):
        if bool(terminal_mask[idx].item()):
            running = rewards[idx]
        else:
            running = rewards[idx] + DEFAULT_GAMMA * (
                (1.0 - DEFAULT_LAMBDA) * next_values[idx]
                + DEFAULT_LAMBDA * running
            )
        returns[idx] = running
    return returns


def attach_timeout_penalties(rows: list[dict], failed_vehicle_ids: set) -> None:
    """Record one terminal cost per failed vehicle and its own gamma=1 returns.

    Rows are chronological within each vehicle's trajectory. The caller must
    supply failures only at the real deadline and call separately per rollout.
    A vehicle without decisions produces no synthetic policy-training row.
    """
    last_actions = {}
    for row in rows:
        failed = row['vehicle_id'] in failed_vehicle_ids
        row['timeout_penalty'] = 0.
        row['timeout_return'] = NONARRIVAL_PENALTY if failed else 0.
        if failed:
            last_actions[row['vehicle_id']] = row
    for row in last_actions.values():
        row['timeout_penalty'] = NONARRIVAL_PENALTY


def individual_actor_advantage(global_advantages: torch.Tensor,
                               timeout_returns: torch.Tensor, *, center: bool = False) -> torch.Tensor:
    """Scale shared traffic advantages before adding vehicle-specific costs.

    Mainline callers center the shared term so a baseline offset cannot push
    the whole batch onto a clipping boundary. The default preserves historical
    baseline algorithms. The scale floor avoids amplifying nearly constant noise.
    Never compute batch statistics over penalties: marking one vehicle failed
    must not rescale the advantage of an arrived vehicle in the same batch.
    """
    timeout_returns = timeout_returns.to(global_advantages)
    if global_advantages.ndim != 1 or timeout_returns.shape != global_advantages.shape:
        raise ValueError('global advantages and individual timeout returns must be matching vectors')
    if not torch.isfinite(global_advantages).all() or not torch.isfinite(timeout_returns).all():
        raise ValueError('actor advantages and timeout returns must be finite')
    if not global_advantages.numel():
        return global_advantages.clone()
    scale = global_advantages.std(unbiased=False).clamp_min(1.)
    shared = global_advantages - global_advantages.mean() if center else global_advantages
    return (shared / scale + timeout_returns).clamp(-2.5, 2.5)


def population_scaled_entropy(
    entropies: torch.Tensor,
    populations: torch.Tensor,
    global_advantages: torch.Tensor,
    *,
    base_coef: float = 0.002,
    mode: str = 'population',
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return per-event entropy bonuses in the shared reward's population units.

    ``population`` divides the coefficient by each event's full episode
    population and the same pass-wide scale used by ``individual_actor_advantage``.
    ``legacy`` keeps the fixed coefficient; ``annealed`` uses the scheduled
    coefficient passed by the caller without population or advantage scaling.
    ``off`` provides a controlled ablation.
    This helper does not change rewards, critic targets, penalties, or advantages.
    Conditions and diagnostic scalars are detached; only entropy carries gradients.
    Callers combine the returned vector with log-probability times advantage
    before taking the event mean. They must derive entropy from legally masked
    action distributions, exactly as for the policy log-probabilities.
    """
    if mode not in {'population', 'legacy', 'off', 'annealed'}:
        raise ValueError('entropy mode must be population, legacy, off, or annealed')
    if not math.isfinite(base_coef) or base_coef < 0:
        raise ValueError('entropy coefficient must be finite and nonnegative')
    if entropies.ndim != 1 or not entropies.is_floating_point():
        raise ValueError('entropies must be a floating point vector')
    advantages = global_advantages.detach().to(entropies)
    population = torch.as_tensor(populations, device=entropies.device, dtype=entropies.dtype).detach()
    if population.ndim == 0:
        population = population.expand_as(entropies)
    if advantages.shape != entropies.shape or population.shape != entropies.shape:
        raise ValueError('entropy, global advantage, and population must align per event')
    if not torch.isfinite(entropies).all() or not torch.isfinite(advantages).all():
        raise ValueError('entropies and global advantages must be finite')
    if (not torch.isfinite(population).all() or (population <= 0).any()
            or (population != population.round()).any()):
        raise ValueError('episode populations must be finite positive integers')
    scale = advantages.std(unbiased=False).clamp_min(1.) if advantages.numel() else entropies.new_ones(())
    coefficients = torch.full_like(entropies, base_coef if mode != 'off' else 0.)
    if mode == 'population':
        coefficients = coefficients / population / scale
    bonuses = entropies * coefficients
    zero = entropies.new_zeros(())
    stats = {'entropy_advantage_scale': scale.detach(),
             'entropy_coefficient_mean': coefficients.mean().detach() if coefficients.numel() else zero,
             'entropy_coefficient_min': coefficients.min().detach() if coefficients.numel() else zero,
             'entropy_coefficient_max': coefficients.max().detach() if coefficients.numel() else zero,
             'entropy_bonus_mean': bonuses.mean().detach() if bonuses.numel() else zero}
    return bonuses, stats


__all__ = ["ActionSelection", "NextHopActor", "CentralizedLatentValueCritic", "lambda_return",
           "attach_timeout_penalties", "individual_actor_advantage", "apply_shortest_path_prior",
           "shortest_path_prior_weight", "set_shortest_path_prior", "population_scaled_entropy"]
