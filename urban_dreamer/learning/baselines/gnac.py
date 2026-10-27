"""GNAC baseline: graph message-passing next-hop actor with a global traffic critic.

On-policy event actor-critic over the real SUMO environment, aligned with the
reference global-only setting:
the team reward is the causal cumulative mean-time difference -delta(C/max(B,1))/60,
a global critic values the flattened physical flow state, and per-window
lambda-return advantages are broadcast to each event row by its window index.
There is no vehicle-level critic, world model, or imagination.

The actor embeds every edge from its 8-field flow state, runs two rounds of
in-edge averaged message passing over the road graph, and scores each
candidate by the candidate/current edge embeddings plus the shared 10-dim
candidate features.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ...models.policy import (ActionSelection, CANDIDATE_FEATURE_DIM, HIDDEN_DIM,
                              _initialize, _make_mlp)
from . import METHODS
from .common import reward_context_dim, GlobalTrafficCritic, global_critic_update, policy_gradient_step

FLOW_FEATURE_DIM = 8
EDGE_EMBEDDING_DIM = 64
MESSAGE_PASSING_ROUNDS = 2


class GraphNextHopActor(nn.Module):
    """Message-passing scorer over the edge graph for destination-reachable candidates."""

    def __init__(self, adjacency: np.ndarray) -> None:
        super().__init__()
        adjacency = np.asarray(adjacency, dtype=np.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError('adjacency must be a square matrix')
        incoming = adjacency.T
        degree = incoming.sum(axis=1, keepdims=True)
        self.register_buffer('adjacency', torch.as_tensor(incoming / np.maximum(degree, 1.), dtype=torch.float32))
        self.edge_encoder = nn.Linear(FLOW_FEATURE_DIM, EDGE_EMBEDDING_DIM)
        self.message_passing = nn.ModuleList(
            nn.Linear(2 * EDGE_EMBEDDING_DIM, EDGE_EMBEDDING_DIM) for _ in range(MESSAGE_PASSING_ROUNDS))
        self.candidate_scorer = _make_mlp(2 * EDGE_EMBEDDING_DIM + CANDIDATE_FEATURE_DIM)
        self.policy_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.policy_head, 0.01)

    def _edge_embeddings(self, flow: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.edge_encoder(flow))
        for layer in self.message_passing:
            h = torch.tanh(layer(torch.cat((h, self.adjacency @ h), dim=-1)))
        return h

    def forward_logits(self, features: torch.Tensor, mask: torch.Tensor, edge_indices: torch.Tensor,
                       current_edges: torch.Tensor, flow: torch.Tensor, *,
                       flow_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Score events, encoding each frozen graph once within this forward pass.

        A shared (E, 8) flow serves every event. Batched (B, E, 8) inputs
        retain their per-event semantics. ``flow_indices`` additionally maps
        events to a smaller batch of distinct frozen-window graphs.
        """
        if features.ndim != 3 or features.shape[-1] != CANDIDATE_FEATURE_DIM:
            raise ValueError(f"expected candidate features (B, K, {CANDIDATE_FEATURE_DIM})")
        if mask.shape != features.shape[:2] or features.shape[1] == 0:
            raise ValueError("candidate mask must have shape (B, K) with K positive")
        mask = mask.to(device=features.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every decision must have at least one legal candidate")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("candidate features must be finite")
        batch, width = features.shape[:2]
        flow = flow.to(device=features.device, dtype=torch.float32)
        if flow.ndim == 2:
            flow = flow.unsqueeze(0)
            if flow_indices is None:
                flow_indices = torch.zeros(batch, dtype=torch.long, device=features.device)
        if flow.ndim != 3 or flow.shape[1] != self.adjacency.shape[0] \
                or flow.shape[2] != FLOW_FEATURE_DIM:
            raise ValueError(f"expected flow (E, {FLOW_FEATURE_DIM}) or (B, E, {FLOW_FEATURE_DIM})")
        if flow_indices is None:
            if flow.shape[0] != batch:
                raise ValueError("batched flow must have one graph per decision")
            flow_indices = torch.arange(batch, device=features.device)
        else:
            flow_indices = flow_indices.to(device=features.device, dtype=torch.long)
            if flow_indices.shape != (batch,) or bool((flow_indices < 0).any()) \
                    or bool((flow_indices >= flow.shape[0]).any()):
                raise ValueError("flow_indices must map each decision to an available graph")
        edge_indices = edge_indices.to(device=features.device, dtype=torch.long)
        if edge_indices.shape != (batch, width):
            raise ValueError("edge_indices must have shape (B, K)")
        current_edges = current_edges.to(device=features.device, dtype=torch.long).reshape(-1)
        if current_edges.shape != (batch,):
            raise ValueError("current_edges must have shape (B,)")
        embeddings = self._edge_embeddings(flow)
        batch_index = flow_indices.unsqueeze(1).expand(batch, width)
        candidate_embeddings = embeddings[batch_index, edge_indices.clamp_min(0)]
        candidate_embeddings = candidate_embeddings * mask.unsqueeze(-1).to(candidate_embeddings.dtype)
        current_embeddings = embeddings[batch_index, current_edges.unsqueeze(1)]
        scored = self.policy_head(self.candidate_scorer(
            torch.cat((candidate_embeddings, current_embeddings, features), dim=-1))).squeeze(-1)
        logits = scored.masked_fill(~mask, -torch.inf)
        return logits

    def sample_action(self, features: torch.Tensor, mask: torch.Tensor, edge_indices: torch.Tensor,
                      current_edges: torch.Tensor, flow: torch.Tensor, *, deterministic: bool = False,
                      temperature: float = 1.) -> ActionSelection:
        logits = self.forward_logits(features, mask, edge_indices, current_edges, flow)
        if isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("action temperature must be finite and positive")
        distribution = Categorical(logits=logits / temperature)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())


class GraphActorCriticMethod:
    """GNAC method wrapper: graph actor + global traffic critic, shared window reward only."""

    entropy_coef = .002

    def __init__(self, args, runner):
        self.args, self.runner = args, runner
        self.device = runner.device
        self.actor = GraphNextHopActor(runner.network.adjacency).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        # The added critic must not change actor initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 73861)
            self.global_critic = GlobalTrafficCritic(runner.network.edge_count, reward_context_dim(args)).to(self.device)
        self.global_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=args.critic_lr)

    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=True, temperature=1., sample_actions=None):
        with torch.no_grad():
            selection = self.actor.sample_action(
                torch.as_tensor(actor_obs.features, dtype=torch.float32, device=self.device),
                torch.as_tensor(actor_obs.mask, device=self.device),
                torch.as_tensor(actor_obs.edge_indices, device=self.device),
                torch.as_tensor([int(vehicle['current_edge']) for vehicle in vehicles],
                                dtype=torch.long, device=self.device),
                torch.as_tensor(np.asarray(frozen_flow), dtype=torch.float32, device=self.device),
                deterministic=not (explore if sample_actions is None else sample_actions),
                temperature=temperature)
        indices, entropies = selection.indices_and_entropy()
        return indices, [dict(entropy=value) for value in entropies]

    def _compute_logits(self, rows):
        features = torch.as_tensor(np.stack([r['features'] for r in rows]), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(np.stack([r['mask'] for r in rows]), device=self.device)
        edge_indices = torch.as_tensor(np.stack([r['edge_indices'] for r in rows]), device=self.device)
        current_edges = torch.as_tensor([r['current_edge'] for r in rows], dtype=torch.long, device=self.device)
        # Collection freezes flow for the entire window. Sharing the graph
        # computation sums all event gradients back into that one encoding.
        # Legacy callers without window indices keep independent flow rows.
        window_graphs, flows, flow_indices = {}, [], []
        for index, row in enumerate(rows):
            key = ('window', row['window']) if 'window' in row else ('row', index)
            if key not in window_graphs:
                window_graphs[key] = len(flows)
                flows.append(row['flow'])
            flow_indices.append(window_graphs[key])
        flow = torch.as_tensor(np.stack(flows), dtype=torch.float32, device=self.device)
        return self.actor.forward_logits(features, mask, edge_indices, current_edges, flow,
            flow_indices=torch.as_tensor(flow_indices, dtype=torch.long, device=self.device))

    def on_episode(self, rows, windows, *, episode):
        learning_started = time.monotonic()
        critic_started = time.monotonic()
        advantages, metrics = global_critic_update(global_critic=self.global_critic,
            global_optimizer=self.global_optimizer, windows=windows)
        metrics['critic_update_seconds'] = time.monotonic() - critic_started
        actor_started = time.monotonic()
        event_windows = torch.as_tensor([r['window'] for r in rows], dtype=torch.long,
                                        device=advantages.device)
        update = policy_gradient_step(compute_logits=self._compute_logits,
            actor_parameters=self.actor.parameters(), actor_optimizer=self.actor_optimizer,
            rows=rows, advantages=advantages[event_windows],
            entropy_coef=self.entropy_coef,
            normalize_global_advantages=True)
        metrics.update(actor_update_seconds=time.monotonic() - actor_started,
                       world_model_seconds=0., imagination_seconds=0.,
                       learning_seconds=time.monotonic() - learning_started)
        return dict(metrics, **update)

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), global_critic=self.global_critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            global_optimizer=self.global_optimizer.state_dict())

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.global_critic.load_state_dict(state['global_critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.global_optimizer.load_state_dict(state['global_optimizer'])


METHODS['gnac'] = GraphActorCriticMethod
