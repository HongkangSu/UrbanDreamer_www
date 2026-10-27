"""Causal candidate values and detached counterfactual policy baselines.

The prepared conditions belong to the decision that produced them. In particular,
they never contain later events, and they are not rebuilt after fitting the Q
critic to the same trajectory's returns.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..learning.action_events import SparseActionEvents
from ..learning.critic_budget import bounded_physical_return, validate_bounded_contract
from .policy import CANDIDATE_FEATURE_DIM, CONTEXT_DIM, LATENT_DIM

EVENT_BACKGROUND_DIM = 64
EVENT_DESTINATION_DIM = 16
EVENT_Q_FEATURE_DIM = (3 * LATENT_DIM + CANDIDATE_FEATURE_DIM
                       + EVENT_DESTINATION_DIM + 1 + CONTEXT_DIM
                       + 3 * EVENT_BACKGROUND_DIM)


def _legal_mask(mask: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if tuple(mask.shape) != shape or len(shape) != 2 or shape[1] == 0:
        raise ValueError("candidate mask must have shape (N, K) with K positive")
    legal = mask.to(device=device, dtype=torch.bool)
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("every decision must have at least one legal candidate")
    return legal


class EventActionCritic(nn.Module):
    """One shared two-layer scorer for destination-reachable event candidates."""

    def __init__(self, input_dim: int = EVENT_Q_FEATURE_DIM, *, bounded: bool = False) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        self.bounded = bounded
        if bounded:
            self.register_buffer("_bounded_cohort_contract", torch.tensor(1, dtype=torch.int64))
            self.register_load_state_dict_pre_hook(validate_bounded_contract)
        self.input_dim = input_dim
        self.scorer = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor, *,
                budget: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"expected event Q features (N, K, {self.input_dim})")
        legal = _legal_mask(mask, tuple(features.shape[:2]), features.device)
        if not bool(torch.isfinite(features[legal]).all()):
            raise ValueError("legal event Q features must be finite")
        # Conditions are fixed for a policy pass, even if a caller forgot to
        # detach a latent or an actor-derived observation before preparation.
        safe_features = features.detach().masked_fill(~legal[..., None], 0.)
        raw = self.scorer(safe_features).squeeze(-1)
        values = bounded_physical_return(raw, budget) if self.bounded else raw
        return values.masked_fill(~legal, 0.)


def detached_counterfactual_baseline(probabilities: torch.Tensor, q_values: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
    """Save the current legal-policy expectation before either critic is updated."""
    if probabilities.shape != q_values.shape or q_values.ndim != 2:
        raise ValueError("actor probabilities and candidate Q values must have matching (N, K) shapes")
    legal = _legal_mask(mask, tuple(q_values.shape), q_values.device)
    probs = probabilities.detach().to(q_values).masked_fill(~legal, 0.)
    values = q_values.detach().masked_fill(~legal, 0.)
    if (not bool(torch.isfinite(probs).all()) or bool((probs < 0).any())
            or not bool(torch.isfinite(values).all())):
        raise ValueError("legal probabilities and Q values must be finite with nonnegative probabilities")
    mass = probs.sum(dim=-1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("every decision must have positive legal probability mass")
    return ((probs / mass) * values).sum(dim=-1).detach().clone()


def executed_action_q_loss(q_values: torch.Tensor, actions: torch.Tensor,
                           returns: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fit only executed candidates to shared traffic return, excluding timeout penalties."""
    if q_values.ndim != 2:
        raise ValueError("candidate Q values must have shape (N, K)")
    legal = _legal_mask(mask, tuple(q_values.shape), q_values.device)
    if actions.shape != (q_values.shape[0],) or returns.shape != actions.shape:
        raise ValueError("executed actions and returns must have shape (N,)")
    if actions.dtype == torch.bool or actions.is_floating_point():
        raise ValueError("executed candidate indices must be integers")
    selected = actions.to(device=q_values.device, dtype=torch.long)
    if bool(((selected < 0) | (selected >= q_values.shape[1])).any()):
        raise ValueError("executed candidate index is out of range")
    if not bool(legal.gather(1, selected[:, None]).all()):
        raise ValueError("executed candidate must be legal")
    targets = returns.detach().to(q_values)
    predictions = q_values.gather(1, selected[:, None]).squeeze(-1)
    if not bool(torch.isfinite(targets).all()) or not bool(torch.isfinite(predictions).all()):
        raise ValueError("executed Q predictions and shared returns must be finite")
    if predictions.numel() == 0:
        return q_values.sum() * 0.
    return F.mse_loss(predictions, targets)


@torch.no_grad()
def build_event_q_inputs(world_model: nn.Module, *, z: torch.Tensor,
                         observations: torch.Tensor, candidate_edges: torch.Tensor,
                         mask: torch.Tensor, current_edges: torch.Tensor,
                         destination_edges: torch.Tensor, decision_times: torch.Tensor,
                         event_indices: torch.Tensor, context: torch.Tensor,
                         events: SparseActionEvents, window_start: float,
                         window_seconds: float = 30.) -> torch.Tensor:
    """Prepare frozen causal conditions without rerunning dynamics or action GNNs.

    ``events`` includes the actions sampled simultaneously with each decision.
    Prefix and simultaneous backgrounds are separate, and only the exact event
    is removed: earlier decisions by the same vehicle remain in the prefix.
    Candidate replacements operate on additive counts and feature sums before
    the small nonlinear readout. All candidates share the same window-start
    latent and the same prefix/background computation.
    """
    if observations.ndim != 3 or observations.shape[-1] != CANDIDATE_FEATURE_DIM:
        raise ValueError("candidate observations must have shape (N, K, 10)")
    count, width = observations.shape[:2]
    legal = _legal_mask(mask, (count, width), z.device)
    if candidate_edges.shape != (count, width):
        raise ValueError("candidate edges must have shape (N, K)")
    if z.ndim == 2:
        z = z.unsqueeze(0)
    if z.ndim != 3 or z.shape[-1] != LATENT_DIM or z.shape[0] not in (1, count):
        raise ValueError("window-start latent must have shape (E, 64), (1, E, 64), or (N, E, 64)")
    if z.shape[1] == 0 or not bool(torch.isfinite(z).all()):
        raise ValueError("window-start latent must contain finite edge features")
    global_latent = z.mean(dim=1).expand(count, -1)
    z = z.expand(count, -1, -1)
    conditions = {}
    for name, value in (("current_edges", current_edges), ("destination_edges", destination_edges),
                        ("decision_times", decision_times), ("event_indices", event_indices)):
        value = torch.as_tensor(value, device=z.device)
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape (N,)")
        conditions[name] = value
    currents = conditions["current_edges"].long()
    destinations = conditions["destination_edges"].long()
    candidates = candidate_edges.to(device=z.device, dtype=torch.long)
    for name, indices in (("current", currents), ("destination", destinations),
                          ("candidate", candidates[legal])):
        if bool(((indices < 0) | (indices >= z.shape[1])).any()):
            raise ValueError(f"{name} edge index is outside the network")
    if not isinstance(window_seconds, (int, float)) or not 0 < window_seconds < float("inf"):
        raise ValueError("window_seconds must be finite and positive")
    phases = (conditions["decision_times"].double() - float(window_start)) / window_seconds
    if not bool(torch.isfinite(phases).all()) or bool(((phases < 0) | (phases >= 1)).any()):
        raise ValueError("decision times must lie in the half-open macro window")
    context = torch.as_tensor(context, device=z.device, dtype=z.dtype)
    if context.shape == (CONTEXT_DIM,):
        context = context.unsqueeze(0)
    if context.shape not in ((1, CONTEXT_DIM), (count, CONTEXT_DIM)):
        raise ValueError("policy context must have shape (4,), (1, 4), or (N, 4)")
    context = context.expand(count, -1)
    observations = observations.to(z)
    if not bool(torch.isfinite(observations[legal]).all()) or not bool(torch.isfinite(context).all()):
        raise ValueError("legal observations and policy context must be finite")
    encoder = world_model.dynamics.event_encoder
    if events.phase.device != z.device:
        events = events.to(z.device)
    # Empty decisions are useful for archive validation and never require an
    # artificial event or an invalid critic sample.
    if count == 0:
        return z.new_zeros((0, width, EVENT_Q_FEATURE_DIM))
    # Episode event IDs are normally unique. Binary search avoids allocating
    # a decision-by-event identity matrix for whole-window batches. Archives
    # may combine episodes with repeated IDs; retain exact identity matching
    # for that general case (including ambiguous duplicate rejection).
    if len(events) == 0:
        raise ValueError("each Q decision must identify exactly one stored action event")
    event_rows = None
    # Small identity matrices cost less than sort/search kernel launches.
    if count * len(events) >= 512 * 512:
        sorted_ids, sorted_rows = events.event_index.sort()
        left = torch.searchsorted(sorted_ids, conditions["event_indices"])
        right = torch.searchsorted(sorted_ids, conditions["event_indices"], right=True)
        if bool((right - left == 1).all()):
            event_rows = sorted_rows[left]
            valid = ((events.current_edge[event_rows] == currents)
                     & (events.destination_edge[event_rows] == destinations)
                     & torch.isclose(events.phase[event_rows], phases, rtol=0., atol=1e-9))
            if not bool(valid.all()):
                raise ValueError("each Q decision must identify exactly one stored action event")
    if event_rows is None:
        matches = ((events.event_index[None] == conditions["event_indices"][:, None])
                   & (events.current_edge[None] == currents[:, None])
                   & (events.destination_edge[None] == destinations[:, None])
                   & torch.isclose(events.phase[None], phases[:, None], rtol=0., atol=1e-9))
        if not bool((matches.sum(-1) == 1).all()):
            raise ValueError("each Q decision must identify exactly one stored action event")
        event_rows = matches.long().argmax(-1)
    # Compute a prefix once per observation group, rather than once per
    # vehicle. Keep the exact phase comparison used by the original path.
    groups, group_for_row = torch.unique(torch.stack((
        events.batch_index[event_rows].double(),
        events.time_index[event_rows].double(), events.phase[event_rows]), -1),
        dim=0, return_inverse=True)
    same_window = ((groups[:, 0, None] == events.batch_index[None])
                   & (groups[:, 1, None] == events.time_index[None]))
    prefix_mask = same_window & (events.phase[None] < groups[:, 2, None])
    simultaneous = same_window & (events.phase[None] == groups[:, 2, None])
    other_mask = simultaneous[group_for_row]
    other_mask.scatter_(1, event_rows[:, None], False)
    relevant = prefix_mask.any(0) | other_mask.any(0)
    causal_events = events.select(relevant)
    prefix_mask = prefix_mask[:, relevant]
    other_mask = other_mask[:, relevant]
    keys = causal_events.current_edge * encoder.n_edges + causal_events.next_edge
    turn_indices = torch.searchsorted(encoder.turn_keys, keys)
    if len(causal_events) and (len(encoder.turn_keys) == 0
            or bool((turn_indices >= len(encoder.turn_keys)).any())
            or not torch.equal(encoder.turn_keys[turn_indices], keys)):
        raise ValueError("causal action event contains an illegal turn")
    event_h = encoder.event_features(causal_events)
    event_buckets = F.one_hot(torch.floor(causal_events.phase * 3).long(), num_classes=3)

    bucketed_h = (event_buckets.to(event_h)[..., None] * event_h[:, None]).flatten(1)

    def pooled_raw(selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep exact integer counts without materializing an N x L x 3 int64
        # table when all decisions from a window are scored together.
        raw_count = torch.stack([(selected & event_buckets[:, b].bool()[None]).sum(1)
                                 for b in range(3)], -1)
        raw_sum = (selected.to(event_h) @ bucketed_h).reshape(-1, 3, 32)
        return raw_count, raw_sum

    other_count, other_sum = pooled_raw(other_mask)
    grouped_count, grouped_sum = pooled_raw(prefix_mask)
    prefix = encoder.compact_readout_counts(grouped_count, grouped_sum)[group_for_row].to(z)
    others = encoder.compact_readout_counts(other_count, other_sum).to(z)
    safe_candidates = candidates.masked_fill(~legal, 0)
    flat_current = currents[:, None].expand(-1, width).reshape(-1)
    flat_destination = destinations[:, None].expand(-1, width).reshape(-1)
    flat_phase = phases[:, None].expand(-1, width).reshape(-1)
    # Encode only legal candidates: an arbitrary padded edge need not be a
    # successor of the current edge, even though its resulting Q is masked.
    candidate_h = z.new_zeros((count * width, 32))
    flat_legal = legal.reshape(-1)
    legal_turn_keys = flat_current[flat_legal] * encoder.n_edges + safe_candidates.reshape(-1)[flat_legal]
    legal_turns = torch.searchsorted(encoder.turn_keys, legal_turn_keys)
    if legal_turn_keys.numel() and (len(encoder.turn_keys) == 0
            or bool((legal_turns >= len(encoder.turn_keys)).any())
            or not torch.equal(encoder.turn_keys[legal_turns], legal_turn_keys)):
        raise ValueError("legal candidate contains an illegal turn")
    candidate_h[flat_legal] = encoder.encode_edges(flat_current[flat_legal],
        safe_candidates.reshape(-1)[flat_legal], flat_destination[flat_legal],
        flat_phase[flat_legal]).to(z)
    candidate_h = candidate_h.reshape(count, width, 32)
    # The background has already subtracted the exact sampled event. Adding
    # one proposed event completes Psi - H(sampled) + H(candidate), while
    # preserving integer counts and feature sums until nonlinear readout.
    bins = torch.floor(phases * 3).long()
    bucket = F.one_hot(bins, num_classes=3).to(z)
    replaced_count = other_count[:, None, :].expand(-1, width, -1) + bucket[:, None, :].long()
    replaced_sum = (other_sum[:, None, :, :].expand(-1, width, -1, -1)
                    + bucket[:, None, :, None] * candidate_h[:, :, None, :])
    replacement = encoder.compact_readout_counts(replaced_count, replaced_sum).to(z)
    n = torch.arange(count, device=z.device)
    current_latent = z[n, currents]
    candidate_latent = z[n[:, None], safe_candidates]
    destination = encoder.edge_embedding(destinations).to(z)
    fixed = torch.cat((global_latent, current_latent, destination, phases[:, None].to(z),
                       context, prefix, others), dim=-1)
    features = torch.cat((fixed[:, None, :].expand(-1, width, -1), candidate_latent,
                          observations, replacement), dim=-1)
    return features.masked_fill(~legal[..., None], 0.).detach()


__all__ = ["EventActionCritic", "EVENT_Q_FEATURE_DIM", "build_event_q_inputs",
           "detached_counterfactual_baseline", "executed_action_q_loss"]
