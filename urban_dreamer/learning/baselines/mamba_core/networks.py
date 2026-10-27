"""Discrete RSSM and attention networks adapted from jbr-ai-labs/mamba."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from .config import MambaConfig


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden: int,
    layers: int = 2,
    activation: type[nn.Module] = nn.ELU,
) -> nn.Sequential:
    if layers < 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    modules: list[nn.Module] = [nn.Linear(in_dim, hidden), activation()]
    for _ in range(layers - 1):
        modules.extend((nn.Linear(hidden, hidden), activation()))
    modules.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*modules)


def _mask3(mask: Tensor, reference: Tensor) -> Tensor:
    if mask.ndim == reference.ndim:
        mask = mask.squeeze(-1)
    if mask.shape != reference.shape[:-1]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match "
            f"tensor prefix {tuple(reference.shape[:-1])}"
        )
    return mask.to(device=reference.device, dtype=torch.bool)


class AgentAttentionEncoder(nn.Module):
    """Permutation-equivariant communication without positional encodings.

    PyTorch's encoder can produce NaNs when an entire row is padding.  A safe
    dummy token is temporarily exposed in such rows and then all padded output
    is zeroed, so empty communication chunks remain finite.
    """

    def __init__(
        self,
        dim: int,
        *,
        layers: int = 3,
        heads: int = 8,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=feedforward_dim or dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )

    def forward(self, inputs: Tensor, presence: Tensor | None = None) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("attention inputs must have shape [batch, agents, dim]")
        if presence is None:
            presence = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        presence = _mask3(presence, inputs)
        safe_presence = presence.clone()
        empty = ~safe_presence.any(dim=1)
        if empty.any():
            safe_presence[empty, 0] = True
        output = self.encoder(inputs, src_key_padding_mask=~safe_presence)
        return output * presence.unsqueeze(-1).to(output.dtype)


@dataclass
class RSSMState:
    logits: Tensor
    stoch: Tensor
    deter: Tensor

    def map(self, function: Callable[[Tensor], Tensor]) -> "RSSMState":
        return RSSMState(
            logits=function(self.logits),
            stoch=function(self.stoch),
            deter=function(self.deter),
        )

    def detach(self) -> "RSSMState":
        return self.map(Tensor.detach)

    @property
    def features(self) -> Tensor:
        return torch.cat((self.stoch, self.deter), dim=-1)

    @classmethod
    def stack(cls, states: Iterable["RSSMState"], dim: int = 0) -> "RSSMState":
        items = list(states)
        if not items:
            raise ValueError("cannot stack an empty state sequence")
        return cls(
            logits=torch.stack([state.logits for state in items], dim=dim),
            stoch=torch.stack([state.stoch for state in items], dim=dim),
            deter=torch.stack([state.deter for state in items], dim=dim),
        )


class DiscreteLatent(nn.Module):
    def __init__(self, in_dim: int, cfg: MambaConfig) -> None:
        super().__init__()
        self.n_categoricals = cfg.n_categoricals
        self.n_classes = cfg.n_classes
        self.net = _mlp(
            in_dim,
            cfg.stochastic_size,
            cfg.model_hidden_size,
            layers=1,
            activation=nn.ReLU,
        )

    def forward(self, inputs: Tensor, *, sample: bool = True) -> tuple[Tensor, Tensor]:
        shape = (*inputs.shape[:-1], self.n_categoricals, self.n_classes)
        # Keep categorical probabilities in FP32 even under CUDA BF16
        # autocast.  Sampling and KL are particularly sensitive to the coarse
        # epsilon and probability resolution of bfloat16.
        logits = self.net(inputs).float().reshape(shape)
        probabilities = logits.softmax(dim=-1)
        if sample:
            indices = Categorical(probs=probabilities).sample()
        else:
            indices = probabilities.argmax(dim=-1)
        hard = F.one_hot(indices, self.n_classes).to(probabilities.dtype)
        straight_through = hard + probabilities - probabilities.detach()
        return logits, straight_through.flatten(-2)


class RSSMTransition(nn.Module):
    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_model = nn.Sequential(
            nn.Linear(cfg.action_dim + cfg.stochastic_size, cfg.hidden_size),
            nn.ReLU(),
        )
        self.communication = AgentAttentionEncoder(
            cfg.hidden_size,
            layers=cfg.transformer_layers,
            heads=cfg.transformer_heads,
            feedforward_dim=cfg.hidden_size,
            dropout=cfg.transformer_dropout,
        )
        self.cell = nn.GRUCell(cfg.hidden_size, cfg.deterministic_size)
        self.prior = DiscreteLatent(cfg.deterministic_size, cfg)

    def forward(
        self,
        previous_action: Tensor,
        previous_state: RSSMState,
        presence: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        if previous_action.ndim != 3:
            raise ValueError("previous_action must have shape [batch, agents, actions]")
        presence = _mask3(presence, previous_action)
        inputs = self.input_model(torch.cat((previous_action, previous_state.stoch), dim=-1))
        communicated = self.communication(inputs, presence)
        flat_output = self.cell(
            communicated.flatten(0, 1),
            previous_state.deter.flatten(0, 1),
        ).reshape(*previous_action.shape[:2], self.cfg.deterministic_size)
        logits, stoch = self.prior(flat_output, sample=sample)
        multiplier = presence.unsqueeze(-1).to(flat_output.dtype)
        return RSSMState(
            logits=logits * multiplier.unsqueeze(-1),
            stoch=stoch * multiplier,
            deter=flat_output * multiplier,
        )


class RSSMRepresentation(nn.Module):
    def __init__(self, cfg: MambaConfig, transition: RSSMTransition) -> None:
        super().__init__()
        self.cfg = cfg
        self.transition = transition
        self.posterior = DiscreteLatent(cfg.deterministic_size + cfg.embedding_size, cfg)

    def initial_state(
        self,
        batch_size: int,
        n_agents: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> RSSMState:
        prefix = (batch_size, n_agents)
        return RSSMState(
            logits=torch.zeros(
                *prefix, self.cfg.n_categoricals, self.cfg.n_classes,
                device=device, dtype=dtype,
            ),
            stoch=torch.zeros(*prefix, self.cfg.stochastic_size, device=device, dtype=dtype),
            deter=torch.zeros(*prefix, self.cfg.deterministic_size, device=device, dtype=dtype),
        )

    def correct(
        self,
        prior: RSSMState,
        observation_embedding: Tensor,
        observation_valid: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        observation_valid = _mask3(observation_valid, observation_embedding)
        logits, stoch = self.posterior(
            torch.cat((prior.deter, observation_embedding), dim=-1),
            sample=sample,
        )
        mask = observation_valid.unsqueeze(-1)
        return RSSMState(
            logits=torch.where(mask.unsqueeze(-1), logits, prior.logits),
            stoch=torch.where(mask, stoch, prior.stoch),
            deter=prior.deter,
        )

    def forward(
        self,
        observation_embedding: Tensor,
        previous_action: Tensor,
        previous_state: RSSMState,
        presence: Tensor,
        observation_valid: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.transition(
            previous_action, previous_state, presence, sample=sample,
        )
        return prior, self.correct(
            prior, observation_embedding, observation_valid, sample=sample,
        )


class MambaWorldModel(nn.Module):
    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.observation_encoder = _mlp(
            cfg.obs_dim, cfg.embedding_size, cfg.hidden_size, cfg.mlp_layers, nn.ReLU,
        )
        self.transition = RSSMTransition(cfg)
        self.representation = RSSMRepresentation(cfg, self.transition)
        self.observation_decoder = _mlp(
            cfg.feature_size, cfg.obs_dim, cfg.hidden_size, cfg.mlp_layers, nn.ReLU,
        )
        self.reward_model = _mlp(
            cfg.feature_size, 1, cfg.hidden_size, cfg.mlp_layers, nn.ELU,
        )
        self.continuation_model = _mlp(
            cfg.feature_size, 1, cfg.hidden_size, cfg.mlp_layers, nn.ELU,
        )
        # Upstream MAMBA includes an auxiliary inverse-action objective so the
        # learned dynamics retain action-specific information instead of
        # explaining all constant-reward transitions with the same latent.
        self.action_information_model = _mlp(
            cfg.feature_size, cfg.action_dim, cfg.hidden_size, 1, nn.ReLU,
        )
        self.availability_model = _mlp(
            cfg.feature_size, cfg.action_dim, cfg.hidden_size, 1, nn.ReLU,
        )

    def initial_state(
        self, batch_size: int, n_agents: int, *, device: torch.device | str
    ) -> RSSMState:
        return self.representation.initial_state(batch_size, n_agents, device=device)

    def encode(self, observation: Tensor) -> Tensor:
        return self.observation_encoder(observation)

    def posterior_from_prior(
        self,
        prior: RSSMState,
        observation: Tensor,
        observation_valid: Tensor,
        *,
        sample: bool = True,
    ) -> RSSMState:
        return self.representation.correct(
            prior, self.encode(observation), observation_valid, sample=sample,
        )

    def observe(
        self,
        observation: Tensor,
        previous_action: Tensor,
        previous_state: RSSMState,
        presence: Tensor,
        observation_valid: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        return self.representation(
            self.encode(observation), previous_action, previous_state,
            presence, observation_valid, sample=sample,
        )


class MambaActor(nn.Module):
    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.action_dim = cfg.action_dim
        self.net = _mlp(
            cfg.feature_size, cfg.action_dim, cfg.hidden_size, cfg.mlp_layers, nn.ReLU,
        )

    def logits(self, features: Tensor, action_valid: Tensor | None = None) -> Tensor:
        logits = self.net(features)
        if action_valid is not None:
            valid = action_valid.to(device=logits.device, dtype=torch.bool)
            if valid.shape != logits.shape:
                raise ValueError("action_valid must have the same shape as actor logits")
            if (~valid.any(dim=-1)).any():
                raise ValueError("every sampled agent must have at least one valid action")
            logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        return logits

    def distribution(
        self, features: Tensor, action_valid: Tensor | None = None
    ) -> Categorical:
        return Categorical(logits=self.logits(features, action_valid))

    def forward(
        self,
        features: Tensor,
        action_valid: Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(features, action_valid)
        action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()


class MambaCritic(nn.Module):
    def __init__(self, cfg: MambaConfig) -> None:
        super().__init__()
        self.embedding = nn.Linear(cfg.feature_size, cfg.hidden_size)
        self.communication = AgentAttentionEncoder(
            cfg.hidden_size,
            layers=1,
            heads=cfg.transformer_heads,
            feedforward_dim=cfg.hidden_size,
            dropout=0.0,
        )
        self.value = _mlp(cfg.hidden_size, 1, cfg.hidden_size, 1, nn.ReLU)

    def forward(self, features: Tensor, presence: Tensor | None = None) -> Tensor:
        if features.ndim != 3:
            raise ValueError("critic features must have shape [batch, agents, features]")
        if presence is None:
            presence = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        embedded = F.relu(self.embedding(features))
        values = self.value(F.relu(self.communication(embedded, presence)))
        return values * presence.unsqueeze(-1).to(values.dtype)


def initialize_module(module: nn.Module, *, orthogonal: bool = False) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            if orthogonal:
                nn.init.orthogonal_(child.weight)
            else:
                nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
