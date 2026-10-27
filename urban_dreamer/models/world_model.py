from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from urban_dreamer.learning.action_events import SparseActionEvents


FLOW_FEATURE_DIM = 8
FLOW_DYNAMIC_DIM = 4
FLOW_LATENT_DIM = 64
WORLD_MODEL_DIM = 256
TRANSFORMER_DEPTH = 6
TRANSFORMER_HEADS = 4
REGISTER_TOKENS = 4
GRAPH_LAYERS = 2
ACTION_EDGE_DIM = 64
FLOW_SPEED_INDEX = 1
FLOW_VOLUME_INDEX = 2

DECODER_RECON_WEIGHT = 0.2
TRANSITION_DECODED_WEIGHT = 1.0
BPR_SUPERVISION_WEIGHT = 0.1
T_ALL_SUPERVISION_WEIGHT = 0.2

INITIAL_SPACE_CHUNK = 1
INITIAL_TIME_CHUNK = 256
INITIAL_MLP_CHUNK = 4096
ONLINE_SPACE_CHUNK = 16
ONLINE_TIME_CHUNK = 32768
ONLINE_MLP_CHUNK = 16384


@dataclass(frozen=True)
class DynamicsOutput:
    latent_velocity: torch.Tensor


@dataclass(frozen=True)
class ConditionEncoding:
    """A solve-local cache. Never reuse after an optimizer update."""

    edge_tokens: torch.Tensor
    reference_tokens: torch.Tensor | None = None
    current_state: torch.Tensor | None = None
    background: torch.Tensor | None = None


class LowFrequencyGraphPrior(nn.Module):
    """Directed message prediction with hard or soft symmetric spectral filtering."""

    def __init__(self, adjacency: torch.Tensor, *, rank: int = 16, noise_scale: float = 0.1,
                 prior_filter: str = "hard", prior_filter_strength: float = 10.0,
                 prior_action_conditioned: bool = True, prior_filter_input: bool = False) -> None:
        super().__init__()
        if rank < 1 or not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("rank must be positive and noise_scale finite and nonnegative")
        if prior_filter not in {"hard", "soft"}:
            raise ValueError("prior_filter must be 'hard' or 'soft'")
        if not math.isfinite(prior_filter_strength) or prior_filter_strength < 0:
            raise ValueError("prior_filter_strength must be finite and nonnegative")
        if not isinstance(prior_action_conditioned, bool):
            raise ValueError("prior_action_conditioned must be a boolean")
        if not isinstance(prior_filter_input, bool):
            raise ValueError("prior_filter_input must be a boolean")
        graph = adjacency.detach().float().cpu()
        if not bool(torch.isfinite(graph).all()) or bool((graph < 0).any()):
            raise ValueError("adjacency must be finite and nonnegative")
        edges = int(graph.shape[0])
        if edges < 1:
            raise ValueError("graph must contain at least one edge")
        # Use the supplied traffic adjacency weights (including its normalization),
        # symmetrized only for the spectral operator; message passing stays directed.
        symmetric = (graph + graph.T) / 2
        degree = symmetric.sum(-1)
        inv_sqrt = degree.clamp_min(1e-12).rsqrt()
        laplacian = torch.diag((degree > 0).float()) - inv_sqrt[:, None] * symmetric * inv_sqrt[None, :]
        selected_rank = min(int(rank), max(1, edges - 1))
        # Bound eigensolver work for large road graphs instead of forming M eigenvectors.
        if edges > 128 and selected_rank * 3 < edges:
            generator = torch.Generator().manual_seed(0)
            initial = torch.randn(edges, selected_rank, generator=generator)
            _, basis = torch.lobpcg(laplacian, X=initial, k=selected_rank, largest=False, niter=100, tol=1e-6)
        else:
            _, vectors = torch.linalg.eigh(laplacian)
            basis = vectors[:, :selected_rank]
        basis = torch.linalg.qr(basis, mode="reduced").Q
        # Public rank is the input noise dimension used by FM training/inference.
        # Keep the legacy E x selected_rank basis in checkpoints, even when the
        # soft filter samples all E road coordinates instead of truncating modes.
        self.rank = edges if prior_filter == "soft" else selected_rank
        self.noise_scale = float(noise_scale)
        self.prior_filter = prior_filter
        self.prior_filter_strength = float(prior_filter_strength)
        self.prior_action_conditioned = prior_action_conditioned
        self.filter_input = prior_filter_input
        # Preserve legacy hard-noise coordinates and checkpoint tensor shapes,
        # including signs/rotations within repeated-eigenvalue subspaces.
        self.register_buffer("basis", basis)
        if prior_filter == "soft":
            # Full-rank resolvent: every eigenmode has gain 1 / (1 + tau * lambda).
            # Build in double precision without changing the legacy noise basis.
            symmetric64 = (graph.double() + graph.double().T) / 2
            degree64 = symmetric64.sum(-1)
            inv_sqrt64 = degree64.clamp_min(1e-12).rsqrt()
            laplacian64 = (torch.diag((degree64 > 0).double())
                           - inv_sqrt64[:, None] * symmetric64 * inv_sqrt64[None, :])
            identity = torch.eye(edges, dtype=torch.float64)
            operator = torch.linalg.solve(identity + self.prior_filter_strength * laplacian64, identity)
            self.register_buffer("soft_filter_operator", operator.float())
            # Match the legacy rank-r source energy without truncating modes.
            # Derived buffers stay out of checkpoints and sampling avoids reductions.
            self.register_buffer("noise_normalization", (
                selected_rank / self.soft_filter_operator.double().square().sum()).sqrt().float(),
                persistent=False)
            self.register_load_state_dict_post_hook(self._restore_noise_normalization)
        self.register_buffer("adjacency", graph, persistent=False)
        self.state_proj = nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM)
        self.upstream_proj = nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM, bias=False)
        self.downstream_proj = nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM, bias=False)
        self.condition_proj = nn.Linear(WORLD_MODEL_DIM, FLOW_LATENT_DIM)
        if not prior_action_conditioned:
            self.condition_proj.requires_grad_(False)
        self.output = nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def projection_rank(self) -> int:
        return self.basis.shape[0] if self.prior_filter == "soft" else self.rank

    @property
    def original_noise_rank(self) -> int:
        return self.basis.shape[1]

    @property
    def expected_noise_energy(self) -> float:
        return self.noise_scale ** 2 * self.basis.shape[1] * FLOW_LATENT_DIM

    @property
    def source_std(self) -> float:
        return (self.noise_scale * self.noise_normalization.item()
                if self.prior_filter == "soft" else self.noise_scale)

    def _restore_noise_normalization(self, _module, _incompatible_keys):
        self.noise_normalization.copy_((
            self.basis.shape[1] / self.soft_filter_operator.double().square().sum()).sqrt())

    def project(self, values: torch.Tensor) -> torch.Tensor:
        if self.prior_filter == "soft":
            operator = self.soft_filter_operator.to(device=values.device, dtype=values.dtype)
            return torch.matmul(operator, values)
        basis = self.basis.to(device=values.device, dtype=values.dtype)
        return torch.matmul(basis, torch.matmul(basis.T, values))

    def forward(self, z: torch.Tensor, condition: ConditionEncoding) -> torch.Tensor:
        graph = self.adjacency.to(device=z.device, dtype=z.dtype)
        downstream = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
        upstream = graph.T / graph.T.sum(-1, keepdim=True).clamp_min(1e-12)
        filtered_z = self.project(z) if self.filter_input else z
        hidden = self.state_proj(filtered_z)
        hidden = hidden + torch.matmul(downstream, self.downstream_proj(filtered_z))
        hidden = hidden + torch.matmul(upstream, self.upstream_proj(filtered_z))
        if self.prior_action_conditioned:
            hidden = hidden + self.condition_proj(condition.edge_tokens)
        return self.project(self.output(F.silu(hidden)))

    def sample_noise(self, z: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        shape = (*z.shape[:-2], self.rank, FLOW_LATENT_DIM)
        if noise is None:
            noise = torch.randn(shape, device=z.device, dtype=z.dtype)
        if tuple(noise.shape) != shape:
            coordinates = "road" if self.prior_filter == "soft" else "low-frequency"
            raise ValueError(f"noise must have {coordinates} coefficient shape {shape}")
        if self.prior_filter == "soft":
            operator = self.soft_filter_operator.to(device=z.device, dtype=z.dtype)
            scale = self.noise_scale * self.noise_normalization.to(device=z.device, dtype=z.dtype)
            return scale * torch.matmul(operator, noise.to(device=z.device, dtype=z.dtype))
        basis = self.basis.to(device=z.device, dtype=z.dtype)
        return self.noise_scale * torch.matmul(basis, noise.to(device=z.device, dtype=z.dtype))


class EdgeGNNLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.message = nn.Linear(width, width, bias=False)
        self.self_proj = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.matmul(adjacency, self.message(hidden))
        update = F.gelu(self.self_proj(hidden) + message)
        return self.norm(hidden + update)


def decode_nonnegative_volume(raw_volume: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw_volume, beta=10.0)


def _positive_pred_speed_for_loss(raw_speed: torch.Tensor) -> torch.Tensor:
    raw = raw_speed.float()
    return torch.where(raw >= 0.0, raw, F.softplus(raw, beta=1.0))


def _positive_pred_volume_for_loss(raw_volume: torch.Tensor) -> torch.Tensor:
    raw = raw_volume.float()
    return torch.where(raw >= 0.0, raw, F.softplus(raw, beta=10.0))


def _transform_dynamic_targets(
    pred_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_decoded.float()
    target = target_decoded.float()

    pred_out = pred.clone()
    target_out = target.clone()

    pred_speed = _positive_pred_speed_for_loss(pred[..., FLOW_SPEED_INDEX]).clamp_min(0.01)
    target_speed = target[..., FLOW_SPEED_INDEX].clamp_min(0.1)
    pred_out[..., FLOW_SPEED_INDEX] = pred_speed.reciprocal() - 1.0
    target_out[..., FLOW_SPEED_INDEX] = target_speed.reciprocal() - 1.0

    pred_vol = _positive_pred_volume_for_loss(pred[..., FLOW_VOLUME_INDEX])
    target_vol = target[..., FLOW_VOLUME_INDEX].clamp_min(0.0)
    pred_out[..., FLOW_VOLUME_INDEX] = torch.log1p(pred_vol)
    target_out[..., FLOW_VOLUME_INDEX] = torch.log1p(target_vol)
    return pred_out, target_out


def _dynamic_weighted_mse(
    pred_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
    *,
    speed_target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_eval, target_eval = _transform_dynamic_targets(pred_decoded, target_decoded)
    err2 = (pred_eval - target_eval).pow(2)
    feature_weight = torch.ones((FLOW_DYNAMIC_DIM,), device=err2.device, dtype=err2.dtype)
    feature_weight[FLOW_SPEED_INDEX] = 4.0
    feature_weight[FLOW_VOLUME_INDEX] = 2.0
    weighted = err2 * feature_weight.view(1, *([1] * (err2.dim() - 2)), FLOW_DYNAMIC_DIM)
    if speed_target_mask is None:
        return weighted.mean()
    if speed_target_mask.shape != err2.shape[:-1] or speed_target_mask.dtype != torch.bool:
        raise ValueError("speed_target_mask must be boolean with the decoded speed shape")
    valid = torch.ones_like(err2)
    valid[..., FLOW_SPEED_INDEX] = speed_target_mask.to(device=err2.device, dtype=err2.dtype)
    return (weighted * valid).sum() / valid.sum().clamp_min(1.)


def _validate_transition_tensor(
    values: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(values.shape)}")
    return values


class FlowProjector(nn.Module):
    def __init__(self, *, adjacency: torch.Tensor, positive_speed: bool = False) -> None:
        super().__init__()
        if adjacency.dim() != 2 or int(adjacency.shape[0]) != int(adjacency.shape[1]):
            raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
        self.n_edges = int(adjacency.shape[0])
        self.positive_speed = bool(positive_speed)
        self.register_buffer("adjacency", adjacency.float(), persistent=False)

        self.input_proj = nn.Linear(FLOW_FEATURE_DIM, FLOW_LATENT_DIM)
        self.gnn_layers = nn.ModuleList([EdgeGNNLayer(FLOW_LATENT_DIM) for _ in range(GRAPH_LAYERS)])
        self.decoder_shared = nn.Sequential(
            nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM),
            nn.SiLU(),
        )
        self.decoder_speed = nn.Linear(FLOW_LATENT_DIM, 1)
        self.decoder_volume = nn.Linear(FLOW_LATENT_DIM, 1)
        self.decoder_other = nn.Linear(FLOW_LATENT_DIM, FLOW_DYNAMIC_DIM - 2)
        pooled_dim = 3 * FLOW_LATENT_DIM
        self.t_all_head = nn.Sequential(
            nn.Linear(pooled_dim, FLOW_LATENT_DIM),
            nn.GELU(),
            nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM // 2),
            nn.GELU(),
            nn.Linear(FLOW_LATENT_DIM // 2, 1),
        )

        nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.decoder_shared[0].weight, gain=1.0)
        nn.init.zeros_(self.decoder_shared[0].bias)
        nn.init.orthogonal_(self.decoder_speed.weight, gain=1.0)
        nn.init.zeros_(self.decoder_speed.bias)
        if self.positive_speed:
            # Inverse Softplus(beta=5) at free-flow ratio 1; only the new codec uses it.
            nn.init.constant_(self.decoder_speed.bias, math.log(math.expm1(5.)) / 5.)
        nn.init.orthogonal_(self.decoder_volume.weight, gain=1.0)
        nn.init.zeros_(self.decoder_volume.bias)
        nn.init.orthogonal_(self.decoder_other.weight, gain=1.0)
        nn.init.zeros_(self.decoder_other.bias)
        nn.init.zeros_(self.t_all_head[-1].weight)
        nn.init.constant_(self.t_all_head[-1].bias, 1.5)

    def forward(self, flow_features: torch.Tensor) -> torch.Tensor:
        if int(flow_features.shape[-1]) != FLOW_FEATURE_DIM:
            raise ValueError(
                f"flow_features last dim must be {FLOW_FEATURE_DIM}, got {int(flow_features.shape[-1])}"
            )
        hidden = self.input_proj(flow_features)
        adjacency = self.adjacency.to(device=hidden.device, dtype=hidden.dtype)
        for layer in self.gnn_layers:
            hidden = layer(hidden, adjacency)
        return hidden

    def decode(self, latent_edges: torch.Tensor) -> torch.Tensor:
        if int(latent_edges.shape[-1]) != FLOW_LATENT_DIM:
            raise ValueError(
                f"latent_edges last dim must be {FLOW_LATENT_DIM}, got {int(latent_edges.shape[-1])}"
            )
        hidden = self.decoder_shared(latent_edges)
        speed = self.decoder_speed(hidden)
        if self.positive_speed:
            speed = F.softplus(speed, beta=5.)
        volume = decode_nonnegative_volume(self.decoder_volume(hidden))
        other = self.decoder_other(hidden)
        return torch.cat([other[..., :1], speed, volume, other[..., 1:2]], dim=-1)

    def predict_t_all_from_z(self, latent_edges: torch.Tensor) -> torch.Tensor:
        decoded = self.decode(latent_edges)
        vehicle_volume = decoded[..., FLOW_VOLUME_INDEX].clamp_min(0.0)
        weights = vehicle_volume / vehicle_volume.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.cat(
            [
                latent_edges.mean(dim=-2),
                latent_edges.max(dim=-2).values,
                (latent_edges * weights.unsqueeze(-1)).sum(dim=-2),
            ],
            dim=-1,
        )
        return F.softplus(self.t_all_head(pooled).squeeze(-1))


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(variance + self.eps))


class MLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = 4 * width
        self.input = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)
        self.token_chunk_size = 0

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.input(x).chunk(2, dim=-1)
        return self.output(up * F.silu(gate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.token_chunk_size)
        if chunk_size <= 0:
            return self._forward_chunk(x)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if int(flat.shape[0]) <= chunk_size:
            return self._forward_chunk(x)
        chunks = [
            self._forward_chunk(flat[start : start + chunk_size])
            for start in range(0, int(flat.shape[0]), chunk_size)
        ]
        return torch.cat(chunks, dim=0).reshape(shape)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width={width} must be divisible by n_heads={n_heads}")
        self.width = int(width)
        self.n_heads = int(n_heads)
        self.head_dim = self.width // self.n_heads
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.output = nn.Linear(self.width, self.width)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)

    def forward_packed(self, x: torch.Tensor, *, batch_chunk_size: int = 0) -> torch.Tensor:
        """Project all tokens together; chunk only independent spatial attention.

        The batch axis remains separate throughout SDPA, so different trajectories
        never attend to each other. This opt-in path shares all existing weights.
        """
        batch, length, width = x.shape
        query, key, value = self.qkv(x.reshape(-1, width)).chunk(3, dim=-1)
        query = query.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        chunk_size = int(batch_chunk_size)
        if chunk_size <= 0 or batch <= chunk_size:
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        else:
            attended = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        query[start : start + chunk_size],
                        key[start : start + chunk_size],
                        value[start : start + chunk_size],
                        dropout_p=0.0,
                    )
                    for start in range(0, batch, chunk_size)
                ],
                dim=0,
            )
        packed = attended.transpose(1, 2).contiguous().reshape(-1, width)
        return self.output(packed).reshape(batch, length, width)


def sinusoid_table(size: int, width: int, *, device: torch.device) -> torch.Tensor:
    position = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    channel = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(0)
    exponent = torch.floor(channel / 2.0)
    scale = torch.exp(-(2.0 * exponent) / max(1.0, float(width)) * math.log(10000.0))
    angle = position * scale
    return torch.where((channel % 2) == 0, torch.sin(angle), torch.cos(angle))


def _build_sinusoidal_positions(time_steps: int, token_count: int, width: int,
                                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position_t = sinusoid_table(time_steps, width, device=device)
    position_s = sinusoid_table(token_count, width, device=device)
    position = (position_t[None, :, None, :] + position_s[None, None, :, :]) * (1.0 / math.sqrt(width))
    return position.to(dtype=dtype)


@lru_cache(maxsize=4)
def _sinusoidal_positions(time_steps: int, token_count: int, width: int,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # FM repeatedly evaluates identical positions. Keep only four execution
    # shapes and create ordinary constants even if inference warms the cache.
    with torch.inference_mode(False), torch.no_grad():
        return _build_sinusoidal_positions(time_steps, token_count, width, device, dtype)


def add_sinusoidal_positions(tokens: torch.Tensor) -> torch.Tensor:
    _, time_steps, token_count, width = tokens.shape
    # Tracing and CUDA graph capture must own their constants and allocations.
    uncached = torch.compiler.is_compiling() or (
        tokens.is_cuda and torch.cuda.is_current_stream_capturing())
    positions = _build_sinusoidal_positions if uncached else _sinusoidal_positions
    return tokens + positions(time_steps, token_count, width, tokens.device, tokens.dtype)


class SpaceSelfAttention(nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.attention_batch_chunk_size = 0
        self.packed_projections = False
        self.attention = MultiheadSelfAttention(WORLD_MODEL_DIM, TRANSFORMER_HEADS)
        mask = torch.ones((1, 1, token_count, token_count), dtype=torch.bool)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, token_count, width = x.shape
        flat = x.reshape(batch * time_steps, token_count, width)
        chunk_size = int(self.attention_batch_chunk_size)
        if self.packed_projections:
            output = self.attention.forward_packed(flat, batch_chunk_size=chunk_size)
        elif chunk_size <= 0 or int(flat.shape[0]) <= chunk_size:
            # All spatial tokens are visible. A redundant all-True mask can
            # prevent the fastest SDPA backend from being selected.
            output = self.attention(flat)
        else:
            output = torch.cat(
                [
                    self.attention(flat[start : start + chunk_size])
                    for start in range(0, int(flat.shape[0]), chunk_size)
                ],
                dim=0,
            )
        return output.reshape(batch, time_steps, token_count, width)


class TimeSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_batch_chunk_size = 0
        self.attention = MultiheadSelfAttention(WORLD_MODEL_DIM, TRANSFORMER_HEADS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, token_count, width = x.shape
        if time_steps == 1:
            # With one key, softmax is identically one and Q/K have zero
            # derivative. Avoid FlashAttention's padded short-sequence workspace.
            qkv = self.attention.qkv
            value = F.linear(x, qkv.weight[2 * width:], qkv.bias[2 * width:] if qkv.bias is not None else None)
            return self.attention.output(value)
        flat = x.permute(0, 2, 1, 3).contiguous().view(batch * token_count, time_steps, width)
        chunk_size = int(self.attention_batch_chunk_size)
        if chunk_size <= 0 or int(flat.shape[0]) <= chunk_size:
            output = self.attention(flat, is_causal=True)
        else:
            output = torch.cat(
                [
                    self.attention(flat[start : start + chunk_size], is_causal=True)
                    for start in range(0, int(flat.shape[0]), chunk_size)
                ],
                dim=0,
            )
        return output.view(batch, token_count, time_steps, width).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.norm_space = RMSNorm(WORLD_MODEL_DIM)
        self.space = SpaceSelfAttention(token_count)
        self.norm_time = RMSNorm(WORLD_MODEL_DIM)
        self.time = TimeSelfAttention()
        self.norm_mlp = RMSNorm(WORLD_MODEL_DIM)
        self.mlp = MLP(WORLD_MODEL_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.space(self.norm_space(x))
        x = x + self.time(self.norm_time(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BlockCausalLayer(token_count) for _ in range(TRANSFORMER_DEPTH)])
        self.activation_checkpointing = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if torch.is_grad_enabled() and self.activation_checkpointing:
                # These blocks use no dropout or other random operations.
                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = layer(x)
        return x


class FlowTimeEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 2 * WORLD_MODEL_DIM
        self.network = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, WORLD_MODEL_DIM),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        return self.network(flow_time.unsqueeze(-1)).unsqueeze(2)


class ActionGNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        static_features: torch.Tensor,
    ) -> None:
        super().__init__()
        if adjacency.dim() != 2 or int(adjacency.shape[0]) != int(adjacency.shape[1]):
            raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
        if static_features.dim() != 2 or int(static_features.shape[0]) != int(adjacency.shape[0]):
            raise ValueError(
                "static_features must have shape (E, S) with the same edge count as adjacency, "
                f"got {tuple(static_features.shape)}"
            )
        self.n_edges = int(adjacency.shape[0])
        self.n_static = int(static_features.shape[1])
        self.register_buffer("adjacency", adjacency.float(), persistent=False)
        self.register_buffer("static_features", static_features.float(), persistent=False)

        self.action_input = nn.Linear(1 + self.n_static, ACTION_EDGE_DIM)
        self.action_gnn = nn.ModuleList([EdgeGNNLayer(ACTION_EDGE_DIM) for _ in range(GRAPH_LAYERS)])
        self.edge_conditioning = nn.Linear(ACTION_EDGE_DIM, WORLD_MODEL_DIM)

    def _augment_with_static(self, values: torch.Tensor) -> torch.Tensor:
        batch, time_steps, edge_count = values.shape
        static = self.static_features.to(device=values.device, dtype=values.dtype)
        static = static.view(1, 1, edge_count, self.n_static).expand(batch, time_steps, edge_count, self.n_static)
        return torch.cat([values.unsqueeze(-1), static], dim=-1)

    def _run_stack(self, projection: nn.Linear, layers: nn.ModuleList, values: torch.Tensor) -> torch.Tensor:
        hidden = projection(self._augment_with_static(values))
        adjacency = self.adjacency.to(device=hidden.device, dtype=hidden.dtype)
        for layer in layers:
            hidden = layer(hidden, adjacency)
        return hidden

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() != 3:
            raise ValueError("action must have shape (B, T, E)")
        if int(action.shape[-1]) != self.n_edges:
            raise ValueError(f"expected {self.n_edges} edge-aligned actions, got {int(action.shape[-1])}")
        action_hidden = self._run_stack(self.action_input, self.action_gnn, action)
        return self.edge_conditioning(action_hidden)


@dataclass(frozen=True)
class RawEventAggregate:
    """Counts and differentiable sums before any nonlinear group readout."""

    count: torch.Tensor
    feature_sum: torch.Tensor


class EventActionEncoder(nn.Module):
    time_buckets = 3
    embedding_dim = 16
    event_dim = 32

    def __init__(self, *, adjacency: torch.Tensor, static_features: torch.Tensor) -> None:
        super().__init__()
        self.n_edges = int(adjacency.shape[0])
        turns = torch.nonzero(adjacency > 0, as_tuple=False)
        self.register_buffer('turns', turns.long(), persistent=False)
        self.register_buffer('turn_keys', turns[:, 0] * self.n_edges + turns[:, 1], persistent=False)
        self.register_buffer('static_features', static_features.float(), persistent=False)
        self.edge_embedding = nn.Embedding(self.n_edges, self.embedding_dim)
        self.event_mlp = nn.Sequential(nn.Linear(49, 64), nn.SiLU(), nn.Linear(64, 32))
        self.turn_mlp = nn.Sequential(nn.Linear(99 + 2 * static_features.shape[1], 64),
                                      nn.SiLU(), nn.Linear(64, 32))

    def encode_edges(self, current_edge: torch.Tensor, next_edge: torch.Tensor,
                     destination_edge: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        device = self.edge_embedding.weight.device
        edges = [value.to(device=device, dtype=torch.long) for value in
                 (current_edge, next_edge, destination_edge)]
        for value in edges:
            if bool(((value < 0) | (value >= self.n_edges)).any()):
                raise ValueError('action event edge is outside the road graph')
        return self.event_mlp(torch.cat([*[self.edge_embedding(value) for value in edges],
                                         phase.to(device=device, dtype=self.edge_embedding.weight.dtype).unsqueeze(-1)], -1))

    def event_features(self, events: SparseActionEvents) -> torch.Tensor:
        return self.encode_edges(events.current_edge, events.next_edge, events.destination_edge, events.phase)

    def aggregate(self, events: SparseActionEvents) -> RawEventAggregate:
        events = events.to(self.edge_embedding.weight.device)
        turn_count = len(self.turns)
        shape = (events.batch_size, events.time_steps, turn_count, self.time_buckets)
        keys = events.current_edge * self.n_edges + events.next_edge
        turn_index = torch.searchsorted(self.turn_keys, keys)
        if len(events) and (turn_count == 0 or bool((turn_index >= turn_count).any())
                            or not torch.equal(self.turn_keys[turn_index], keys)):
            raise ValueError('action event contains an illegal turn')
        bucket = torch.floor(events.phase * self.time_buckets).long()
        flat_index = (((events.batch_index * events.time_steps + events.time_index)
                       * turn_count + turn_index) * self.time_buckets + bucket)
        size = math.prod(shape)
        count = torch.zeros(size, device=keys.device, dtype=torch.long)
        count = count.index_add(0, flat_index, torch.ones_like(flat_index))
        feature_sum = self.edge_embedding.weight.new_zeros(size, self.event_dim)
        feature_sum = feature_sum.index_add(0, flat_index,
            self.event_features(events).to(dtype=feature_sum.dtype))
        return RawEventAggregate(count.reshape(shape), feature_sum.reshape(*shape, self.event_dim))

    @staticmethod
    def _group_features(count: torch.Tensor, feature_sum: torch.Tensor) -> torch.Tensor:
        mean = feature_sum / count.clamp_min(1).unsqueeze(-1)
        features = torch.cat([torch.log1p(count.to(feature_sum.dtype)).unsqueeze(-1), mean], -1)
        return features.flatten(-2)

    def compact_readout_counts(self, count: torch.Tensor, feature_sum: torch.Tensor) -> torch.Tensor:
        features = self._group_features(count, feature_sum)
        # Share the learned turn readout without fabricating an entry/exit for pooled traffic.
        hidden = F.silu(F.linear(features, self.turn_mlp[0].weight[:, :99], self.turn_mlp[0].bias))
        return hidden * (count.sum(-1) > 0).unsqueeze(-1)

    def compact_readout(self, raw: RawEventAggregate) -> torch.Tensor:
        return self.compact_readout_counts(raw.count.sum(-2), raw.feature_sum.sum(-3))

    def background(self, events: SparseActionEvents, event_row_index: int) -> tuple[RawEventAggregate, RawEventAggregate]:
        if not 0 <= event_row_index < len(events):
            raise IndexError('event row index is outside the sparse events')
        same_window = ((events.batch_index == events.batch_index[event_row_index])
                       & (events.time_index == events.time_index[event_row_index]))
        prefix = same_window & (events.phase < events.phase[event_row_index])
        simultaneous = same_window & (events.phase == events.phase[event_row_index])
        simultaneous[event_row_index] = False
        return self.aggregate(events.select(prefix)), self.aggregate(events.select(simultaneous))

    def replace_event(self, raw: RawEventAggregate, events: SparseActionEvents,
                      event_row_index: int, next_edge: int) -> RawEventAggregate:
        mask = torch.arange(len(events), device=events.phase.device) == event_row_index
        own = events.select(mask)
        if len(own) != 1:
            raise IndexError('event row index is outside the sparse events')
        from dataclasses import replace
        candidate = replace(own, next_edge=torch.full_like(own.next_edge, next_edge))
        old, new = self.aggregate(own), self.aggregate(candidate)
        return RawEventAggregate(raw.count - old.count + new.count,
                                 raw.feature_sum - old.feature_sum + new.feature_sum)

    def readout(self, raw: RawEventAggregate) -> torch.Tensor:
        batch, steps, turn_count, _ = raw.count.shape
        features = self._group_features(raw.count, raw.feature_sum)
        static = self.static_features.to(dtype=features.dtype)
        turn_static = torch.cat([static[self.turns[:, 0]], static[self.turns[:, 1]]], -1)
        turn_static = turn_static[None, None].expand(batch, steps, turn_count, -1)
        hidden = self.turn_mlp(torch.cat([features, turn_static], -1))
        hidden = hidden * (raw.count.sum(-1) > 0).unsqueeze(-1)
        incoming = hidden.new_zeros(batch, steps, self.n_edges, 32).index_add(2, self.turns[:, 1], hidden)
        outgoing = hidden.new_zeros(batch, steps, self.n_edges, 32).index_add(2, self.turns[:, 0], hidden)
        return torch.cat([outgoing, incoming], -1)

    def forward(self, events: SparseActionEvents) -> torch.Tensor:
        return self.readout(self.aggregate(events))


class BackgroundConditionResidual(nn.Module):
    """Direct conditional correction matching the evaluated B architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(WORLD_MODEL_DIM + 3 * FLOW_LATENT_DIM + 1, 128)
        self.output = nn.Linear(128, FLOW_LATENT_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.hidden(inputs)))


class Dynamics(nn.Module):
    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        static_features: torch.Tensor,
        low_frequency_rank: int = 16,
        prior_noise_scale: float = 0.1,
        prior_filter: str = "hard",
        prior_filter_strength: float = 10.0,
        prior_action_conditioned: bool = True,
        prior_filter_input: bool = False,
        event_protocol: bool = False,
        prior_usage: str = "source_reference",
    ) -> None:
        super().__init__()
        if prior_usage not in {"source_reference", "condition_only"}:
            raise ValueError("prior_usage must be 'source_reference' or 'condition_only'")
        if prior_usage == "condition_only" and prior_action_conditioned:
            raise ValueError("condition_only requires a state-only prior (prior_action_conditioned=False)")
        self.prior_usage = prior_usage
        self.n_edges = int(adjacency.shape[0])
        self.action_encoder = ActionGNNEncoder(adjacency=adjacency, static_features=static_features)
        self.event_protocol = bool(event_protocol)
        self.event_encoder = (EventActionEncoder(adjacency=adjacency, static_features=static_features)
                              if self.event_protocol else None)
        self.graph_prior = LowFrequencyGraphPrior(adjacency, rank=low_frequency_rank, noise_scale=prior_noise_scale,
            prior_filter=prior_filter, prior_filter_strength=prior_filter_strength,
            prior_action_conditioned=prior_action_conditioned, prior_filter_input=prior_filter_input)
        self.spatial_proj = nn.Linear(FLOW_LATENT_DIM, WORLD_MODEL_DIM)
        self.reference_proj = nn.Linear(FLOW_LATENT_DIM, WORLD_MODEL_DIM, bias=False)
        self.flow_time = FlowTimeEmbedder()
        self.register_tokens = nn.Parameter(torch.empty(REGISTER_TOKENS, WORLD_MODEL_DIM))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.edge_action_proj = nn.Linear(2, WORLD_MODEL_DIM)
        nn.init.zeros_(self.edge_action_proj.weight)
        nn.init.zeros_(self.edge_action_proj.bias)

        self.flow_time_slice = slice(0, 1)
        self.spatial_slice = slice(1, 1 + self.n_edges)
        self.register_slice = slice(1 + self.n_edges, 1 + self.n_edges + REGISTER_TOKENS)
        self.transformer = BlockCausalTransformer(1 + self.n_edges + REGISTER_TOKENS)

        self.flow_head = nn.Linear(WORLD_MODEL_DIM, FLOW_LATENT_DIM)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

        if prior_usage == "condition_only":
            self.condition_residual = BackgroundConditionResidual()
            self.register_buffer("condition_rms_z", torch.ones(()))
            self.register_buffer("condition_rms_background", torch.ones(()))
            self.register_buffer("condition_rms_relative", torch.ones(()))

        self.set_initial_chunk_profile()

    def _set_chunk_profile(self, *, space_chunk: int, time_chunk: int, mlp_chunk: int) -> None:
        for layer in self.transformer.layers:
            layer.space.attention_batch_chunk_size = int(space_chunk)
            layer.time.attention_batch_chunk_size = int(time_chunk)
            layer.mlp.token_chunk_size = int(mlp_chunk)

    def set_pretraining_execution_profile(self, *, space_chunk: int, time_chunk: int,
                                          mlp_chunk: int, activation_checkpointing: bool) -> None:
        for value in (space_chunk, time_chunk, mlp_chunk):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('pretraining chunk sizes must be positive integers')
        if not isinstance(activation_checkpointing, bool):
            raise ValueError('activation_checkpointing must be a boolean')
        self._set_chunk_profile(space_chunk=space_chunk, time_chunk=time_chunk, mlp_chunk=mlp_chunk)
        self.transformer.activation_checkpointing = activation_checkpointing

    def set_initial_chunk_profile(self) -> None:
        self.transformer.activation_checkpointing = True
        self._set_chunk_profile(
            space_chunk=INITIAL_SPACE_CHUNK,
            time_chunk=INITIAL_TIME_CHUNK,
            mlp_chunk=INITIAL_MLP_CHUNK,
        )

    def set_online_chunk_profile(self) -> None:
        self.transformer.activation_checkpointing = True
        self._set_chunk_profile(
            space_chunk=ONLINE_SPACE_CHUNK,
            time_chunk=ONLINE_TIME_CHUNK,
            mlp_chunk=ONLINE_MLP_CHUNK,
        )

    def _edge_action_features(self, edge_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_actions = edge_actions.float()
        mask_values = (edge_actions.abs() > 1e-6).to(edge_actions.dtype)
        mask = mask_values.unsqueeze(-1)
        active_edge_count = mask_values.sum(dim=-1, keepdim=True).clamp_min(1.0)
        features = torch.stack(
            [mask_values, edge_actions * active_edge_count],
            dim=-1,
        )
        return mask, features

    def _inject_actions(
        self,
        spatial_tokens: torch.Tensor,
        edge_actions: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        mask, features = self._edge_action_features(edge_actions.to(device=spatial_tokens.device, dtype=spatial_tokens.dtype))
        injected = projection(features.to(dtype=spatial_tokens.dtype))
        return spatial_tokens + injected * mask

    def forward(
        self,
        action: torch.Tensor,
        *,
        packed_latents: torch.Tensor,
        flow_time: torch.Tensor,
        condition: ConditionEncoding | None = None,
        action_events: SparseActionEvents | None = None,
    ) -> DynamicsOutput:
        if packed_latents.dim() != 4 or tuple(packed_latents.shape[-2:]) != (self.n_edges, FLOW_LATENT_DIM):
            raise ValueError(
                "packed_latents must have shape (B, T, E, 64), "
                f"got {tuple(packed_latents.shape)}"
            )
        batch, time_steps, _, _ = packed_latents.shape
        expected_edges = (batch, time_steps, self.n_edges)
        _validate_transition_tensor(action, name="action", expected_shape=expected_edges)
        if tuple(flow_time.shape) != (batch, time_steps):
            raise ValueError(f"flow_time must have shape {(batch, time_steps)}, got {tuple(flow_time.shape)}")

        spatial_tokens = self.spatial_proj(packed_latents)
        if condition is None:
            condition = self.encode_condition(action, action_events=action_events)
        spatial_tokens = spatial_tokens + condition.edge_tokens
        if condition.reference_tokens is not None:
            spatial_tokens = spatial_tokens + condition.reference_tokens

        register_tokens = self.register_tokens.view(1, 1, REGISTER_TOKENS, WORLD_MODEL_DIM)
        register_tokens = register_tokens.expand(batch, time_steps, -1, -1)
        time_token = self.flow_time(flow_time.to(dtype=spatial_tokens.dtype))
        tokens = torch.cat([time_token, spatial_tokens, register_tokens], dim=2)
        hidden = self.transformer(add_sinusoidal_positions(tokens))
        spatial_hidden = hidden[:, :, self.spatial_slice, :]
        velocity = self.flow_head(spatial_hidden)
        if self.prior_usage == "condition_only":
            if condition.current_state is None or condition.background is None:
                raise ValueError("condition_only requires current-state context from prior_start")
            current, background = condition.current_state, condition.background
            if current.shape != packed_latents.shape or background.shape != current.shape:
                raise ValueError("current-state and background context must match packed_latents")
            time_feature = flow_time[..., None, None].expand(*spatial_hidden.shape[:-1], 1)
            inputs = torch.cat((spatial_hidden, current / self.condition_rms_z,
                background / self.condition_rms_background,
                (packed_latents - current) / self.condition_rms_relative,
                time_feature.to(spatial_hidden.dtype)), dim=-1)
            velocity = velocity + self.condition_residual(inputs)
        return DynamicsOutput(latent_velocity=velocity)

    def encode_condition(self, action: torch.Tensor, *, action_events: SparseActionEvents | None = None) -> ConditionEncoding:
        if self.event_protocol:
            if action_events is None:
                raise ValueError('event protocol requires explicit action_events for every window')
            if (action_events.batch_size, action_events.time_steps) != tuple(action.shape[:2]):
                raise ValueError('action event windows must align with the action tensor')
            hidden = self.event_encoder(action_events)
            adjacency = self.action_encoder.adjacency.to(device=hidden.device, dtype=hidden.dtype)
            for layer in self.action_encoder.action_gnn:
                hidden = layer(hidden, adjacency)
            return ConditionEncoding(self.action_encoder.edge_conditioning(hidden))
        edge_tokens = self.action_encoder(action)
        edge_tokens = self._inject_actions(edge_tokens, action, self.edge_action_proj)
        return ConditionEncoding(edge_tokens)

    def prior_start(
        self, z: torch.Tensor, condition: ConditionEncoding, noise: torch.Tensor | None = None,
        *, detach_prior_increment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, ConditionEncoding]:
        """Build a noisy source and solve-local current-state context.

        Condition-only priors supply detached background features without moving
        the source mean or reference. Legacy state-only priors detach increments
        on source/reference paths; action-conditioned priors detach on request.
        The returned original increment remains live for prior supervision.
        Current-state, condition encoder and reference gradients remain live.
        """
        if not isinstance(detach_prior_increment, bool):
            raise ValueError('detach_prior_increment must be a boolean')
        mean, increment, cached = self._prior_condition(z, condition, detach_prior_increment)
        return mean + self.graph_prior.sample_noise(z, noise), increment, cached

    def _prior_condition(
        self, z: torch.Tensor, condition: ConditionEncoding, detach_prior_increment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, ConditionEncoding]:
        """Build context without drawing noise; disagreement reuses it across samples."""
        increment = self.graph_prior(z, condition)
        if self.prior_usage == "condition_only":
            return z, increment, ConditionEncoding(
                condition.edge_tokens, self.reference_proj(z), z, increment.detach())
        detach_increment = (detach_prior_increment
                            or not getattr(self.graph_prior, "prior_action_conditioned", True))
        coarse = z + (increment.detach() if detach_increment else increment)
        return coarse, increment, ConditionEncoding(condition.edge_tokens, self.reference_proj(coarse))

    def solve(
        self, start: torch.Tensor, action: torch.Tensor, condition: ConditionEncoding,
        *, steps: int = 2, solver: str = "euler", picard_iterations: int | None = None,
        picard_batch_chunk_size: int = 64,
        checkpoint_interval: int = 0,
    ) -> torch.Tensor:
        if steps < 1 or int(steps) != steps:
            raise ValueError("steps must be a positive integer")
        if solver not in {"euler", "midpoint", "picard"}:
            raise ValueError("solver must be 'euler', 'midpoint' or 'picard'")
        if (isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int)
                or checkpoint_interval < 0):
            raise ValueError("checkpoint_interval must be a nonnegative integer")
        batch, intervals, edges, width = start.shape
        if (checkpoint_interval and torch.is_grad_enabled() and solver in {"euler", "midpoint"}
                and steps > checkpoint_interval):
            # Save only integration-segment boundaries. Every node is recomputed
            # during backward, preserving the full solver and condition gradients.
            def integrate(state, action, edge_tokens, reference_tokens, current_state, background,
                          *, begin, end):
                segment_condition = ConditionEncoding(edge_tokens, reference_tokens, current_state, background)
                for node in range(begin, end):
                    time = torch.full((batch, intervals), node / steps, device=start.device, dtype=start.dtype)
                    velocity = self(action, packed_latents=state, flow_time=time,
                                    condition=segment_condition).latent_velocity
                    if solver == "midpoint":
                        midpoint = state + velocity / (2 * steps)
                        midpoint_time = torch.full((batch, intervals), (node + .5) / steps,
                                                   device=start.device, dtype=start.dtype)
                        velocity = self(action, packed_latents=midpoint, flow_time=midpoint_time,
                                        condition=segment_condition).latent_velocity
                    state = state + velocity / steps
                return state

            state = start
            for begin in range(0, steps, checkpoint_interval):
                segment = partial(integrate, begin=begin, end=min(begin + checkpoint_interval, steps))
                state = checkpoint(segment, state, action, condition.edge_tokens,
                    condition.reference_tokens, condition.current_state, condition.background,
                    use_reentrant=False, preserve_rng_state=False)
            return state
        if solver in {"euler", "midpoint"}:
            state = start
            for node in range(steps):
                time = torch.full((batch, intervals), node / steps, device=start.device, dtype=start.dtype)
                velocity = self(action, packed_latents=state, flow_time=time,
                                condition=condition).latent_velocity
                if solver == "midpoint":
                    # Two velocity evaluations per flow-time interval; reuse the
                    # physical-window condition and its single source-noise draw.
                    midpoint = state + velocity / (2 * steps)
                    midpoint_time = torch.full((batch, intervals), (node + .5) / steps,
                                               device=start.device, dtype=start.dtype)
                    velocity = self(action, packed_latents=midpoint, flow_time=midpoint_time,
                                    condition=condition).latent_velocity
                state = state + velocity / steps
            return state
        iterations = steps if picard_iterations is None else picard_iterations
        if iterations < 1 or int(iterations) != iterations:
            raise ValueError("picard_iterations must be a positive integer")
        if (isinstance(picard_batch_chunk_size, bool)
                or not isinstance(picard_batch_chunk_size, int) or picard_batch_chunk_size < 0):
            raise ValueError("picard_batch_chunk_size must be a nonnegative integer")
        # Bound complete field evaluations, including their condition tensors.
        # Nodes remain batch members; physical intervals remain causal tokens.
        size = picard_batch_chunk_size or steps * batch
        chunks = []
        for begin in range(0, steps * batch, size):
            flat_indices = torch.arange(begin, min(begin + size, steps * batch), device=start.device)
            node_indices = flat_indices.div(batch, rounding_mode='floor')
            sample_indices = flat_indices.remainder(batch)
            time = (node_indices.to(start.dtype) / steps)[:, None].expand(-1, intervals)
            chunks.append((node_indices, sample_indices, time))
        trajectory = start.unsqueeze(0).expand(steps + 1, *start.shape)
        for _ in range(iterations):
            velocities = []
            for nodes, samples, time in chunks:
                chunk_condition = ConditionEncoding(condition.edge_tokens[samples],
                    None if condition.reference_tokens is None else condition.reference_tokens[samples],
                    None if condition.current_state is None else condition.current_state[samples],
                    None if condition.background is None else condition.background[samples])
                velocity = self(action[samples], packed_latents=trajectory[nodes, samples],
                                flow_time=time, condition=chunk_condition).latent_velocity
                velocities.append(velocity)
            velocity = torch.cat(velocities, dim=0).reshape(steps, *start.shape)
            trajectory = torch.cat([start.unsqueeze(0), start.unsqueeze(0) + velocity.cumsum(0) / steps], dim=0)
        return trajectory[-1]

    def transition(
        self, z: torch.Tensor, action: torch.Tensor, *, noise: torch.Tensor | None = None,
        steps: int = 2, solver: str = "euler", picard_iterations: int | None = None,
        picard_batch_chunk_size: int = 64,
        action_events: SparseActionEvents | None = None,
    ) -> torch.Tensor:
        squeezed = z.dim() == 3
        if squeezed:
            z, action = z.unsqueeze(1), action.unsqueeze(1)
            if noise is not None:
                noise = noise.unsqueeze(1)
        if z.dim() != 4 or tuple(z.shape[-2:]) != (self.n_edges, FLOW_LATENT_DIM):
            raise ValueError("z must have shape (B, E, 64) or (B, T, E, 64)")
        _validate_transition_tensor(action, name="action", expected_shape=tuple(z.shape[:-1]))
        condition = self.encode_condition(action, action_events=action_events)
        start, _, condition = self.prior_start(z, condition, noise)
        result = self.solve(start, action, condition, steps=steps, solver=solver,
                            picard_iterations=picard_iterations, picard_batch_chunk_size=picard_batch_chunk_size)
        return result.squeeze(1) if squeezed else result

    def latent_disagreement(
        self, z: torch.Tensor, action: torch.Tensor, *, n_samples: int = 2,
        noise: torch.Tensor | None = None, steps: int = 2, solver: str = "euler",
        picard_iterations: int | None = None,
        picard_batch_chunk_size: int = 64,
        action_events: SparseActionEvents | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_samples < 2 or int(n_samples) != n_samples:
            raise ValueError("latent disagreement requires at least two samples")
        squeezed = z.dim() == 3
        if squeezed:
            z, action = z.unsqueeze(1), action.unsqueeze(1)
            if noise is not None:
                noise = noise.unsqueeze(2)
        if z.dim() != 4 or tuple(z.shape[-2:]) != (self.n_edges, FLOW_LATENT_DIM):
            raise ValueError("z must have shape (B, E, 64) or (B, T, E, 64)")
        _validate_transition_tensor(action, name="action", expected_shape=tuple(z.shape[:-1]))
        condition = self.encode_condition(action, action_events=action_events)
        coarse, _, condition = self._prior_condition(z, condition)

        def repeat_samples(value: torch.Tensor) -> torch.Tensor:
            return value.unsqueeze(0).expand(n_samples, *value.shape).reshape(n_samples * z.shape[0], *value.shape[1:])

        repeated_z = repeat_samples(z)
        if noise is not None:
            expected = (n_samples, *z.shape[:-2], self.graph_prior.rank, FLOW_LATENT_DIM)
            if tuple(noise.shape) != expected:
                raise ValueError(f"noise must have shape {expected}")
            noise = noise.reshape(n_samples * z.shape[0], *noise.shape[2:])
        starts = repeat_samples(coarse) + self.graph_prior.sample_noise(repeated_z, noise)
        repeated_condition = ConditionEncoding(
            repeat_samples(condition.edge_tokens), repeat_samples(condition.reference_tokens),
            None if condition.current_state is None else repeat_samples(condition.current_state),
            None if condition.background is None else repeat_samples(condition.background))
        samples = self.solve(starts, repeat_samples(action), repeated_condition,
                             steps=steps, solver=solver, picard_iterations=picard_iterations,
                             picard_batch_chunk_size=picard_batch_chunk_size)
        samples = samples.reshape(n_samples, *z.shape)
        if squeezed:
            samples = samples.squeeze(2)
        edge_variance = samples.float().var(dim=0, unbiased=True).mean(dim=-1)
        return samples, edge_variance.mean(dim=-1), edge_variance


class ActiveVehicleFractionReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fraction_eps = 1e-3
        hidden = FLOW_LATENT_DIM // 2
        pooled_dim = 4 * FLOW_LATENT_DIM
        self.latent_norm = nn.LayerNorm(pooled_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(pooled_dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("global_logit_bias", torch.zeros((), dtype=torch.float32))
        self.register_buffer("calibration_updates", torch.zeros((), dtype=torch.long))
        nn.init.orthogonal_(self.correction_head[0].weight)
        nn.init.zeros_(self.correction_head[0].bias)
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def corrected_logits(
        self,
        z_next: torch.Tensor,
        *,
        z_prev: torch.Tensor,
        raw_decoder_fraction: torch.Tensor | float,
        transition_seconds: torch.Tensor | float,
    ) -> torch.Tensor:
        if z_next.dim() < 2 or int(z_next.shape[-1]) != FLOW_LATENT_DIM:
            raise ValueError(
                "ActiveVehicleFractionReadout expects (..., E, 64), "
                f"got {tuple(z_next.shape)}"
            )
        if tuple(z_prev.shape) != tuple(z_next.shape):
            raise ValueError("z_prev must match z_next shape")

        leading_shape = z_next.shape[:-2]
        latent_features = self.latent_norm(
            torch.cat(
                [
                    z_prev.mean(dim=-2),
                    z_prev.max(dim=-2).values,
                    z_next.mean(dim=-2),
                    z_next.max(dim=-2).values,
                ],
                dim=-1,
            )
        )
        duration = torch.as_tensor(transition_seconds, device=z_next.device, dtype=z_next.dtype)
        while duration.dim() < len(leading_shape):
            duration = duration.unsqueeze(-1)
        duration = torch.broadcast_to(duration, leading_shape)
        if bool(torch.any(~torch.isfinite(duration)).item()) or bool(torch.any(duration <= 0.0).item()):
            raise ValueError("transition_seconds must be finite and positive")

        raw = torch.as_tensor(raw_decoder_fraction, device=z_next.device, dtype=z_next.dtype)
        while raw.dim() < len(leading_shape):
            raw = raw.unsqueeze(-1)
        raw = torch.broadcast_to(raw, leading_shape)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=0.0, max=1e6)
        base = raw.clamp(min=self.fraction_eps, max=1.0 - self.fraction_eps)
        raw_features = torch.stack([base, torch.log1p(raw)], dim=-1)
        correction = self.correction_head(
            torch.cat([latent_features, raw_features, torch.log1p(duration).unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        return torch.logit(base) + self.global_logit_bias.to(device=z_next.device, dtype=z_next.dtype) + correction

    def forward(
        self,
        z_next: torch.Tensor,
        *,
        z_prev: torch.Tensor,
        raw_decoder_fraction: torch.Tensor | float,
        transition_seconds: torch.Tensor | float,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.corrected_logits(
                z_next,
                z_prev=z_prev,
                raw_decoder_fraction=raw_decoder_fraction,
                transition_seconds=transition_seconds,
            )
        )


class UrbanDreamerWorldModel(nn.Module):
    def __init__(self, *, adjacency: torch.Tensor, static_features: torch.Tensor,
                 low_frequency_rank: int = 16, prior_noise_scale: float = 0.1,
                 prior_filter: str = "hard", prior_filter_strength: float = 10.0,
                 prior_action_conditioned: bool = True,
                 prior_filter_input: bool = False,
                 event_protocol: bool = False, codec_speed_positive: bool = False,
                 prior_usage: str = "source_reference") -> None:
        super().__init__()
        self.flow_projector = FlowProjector(adjacency=adjacency, positive_speed=codec_speed_positive)
        self.dynamics = Dynamics(adjacency=adjacency, static_features=static_features,
                                 low_frequency_rank=low_frequency_rank, prior_noise_scale=prior_noise_scale,
                                 prior_filter=prior_filter, prior_filter_strength=prior_filter_strength,
                                 prior_action_conditioned=prior_action_conditioned,
                                 prior_filter_input=prior_filter_input,
                                 event_protocol=event_protocol, prior_usage=prior_usage)
        self.active_vehicle_fraction_readout = ActiveVehicleFractionReadout()

    def encode(self, flow_features: torch.Tensor) -> torch.Tensor:
        return self.flow_projector(flow_features)

    def decode(self, latent_edges: torch.Tensor) -> torch.Tensor:
        return self.flow_projector.decode(latent_edges)

    def transition(self, z: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dynamics.transition(z, action, **kwargs)

    def latent_disagreement(self, z: torch.Tensor, action: torch.Tensor,
                            **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dynamics.latent_disagreement(z, action, **kwargs)

    def set_initial_chunk_profile(self) -> None:
        self.dynamics.set_initial_chunk_profile()

    def set_online_chunk_profile(self) -> None:
        self.dynamics.set_online_chunk_profile()


def decoder_reconstruction_loss(
    *,
    decoded_dyn: torch.Tensor,
    flow_targets: torch.Tensor,
    speed_target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_dyn = flow_targets[..., :FLOW_DYNAMIC_DIM].to(dtype=decoded_dyn.dtype)
    weighted_mse = _dynamic_weighted_mse(decoded_dyn, target_dyn, speed_target_mask=speed_target_mask)
    loss = DECODER_RECON_WEIGHT * weighted_mse
    return loss, {
        "decoder_recon_weighted_mse": weighted_mse.detach(),
        "decoder_recon_loss": loss.detach(),
    }


def transition_decoded_supervision_loss(
    *,
    z_pred_next: torch.Tensor,
    flow_projector: FlowProjector | nn.Module,
    flow_target_next: torch.Tensor,
    speed_target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_next_decoded = flow_projector.decode(z_pred_next)  # type: ignore[attr-defined]
    target_dyn = flow_target_next[..., :FLOW_DYNAMIC_DIM].to(dtype=pred_next_decoded.dtype)
    weighted_mse = _dynamic_weighted_mse(pred_next_decoded, target_dyn, speed_target_mask=speed_target_mask)
    loss = TRANSITION_DECODED_WEIGHT * weighted_mse
    return loss, {
        "pred_next_decoded_weighted_mse": weighted_mse.detach(),
        "pred_next_decoded_supervision_loss": loss.detach(),
    }


def decoder_bpr_supervision_loss(
    *,
    decoded_dyn: torch.Tensor,
    flow_targets: torch.Tensor,
    lengths: torch.Tensor,
    speed_limits: torch.Tensor,
    capacities: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_dyn = flow_targets[..., :FLOW_DYNAMIC_DIM].to(dtype=decoded_dyn.dtype)
    cap = capacities.to(device=decoded_dyn.device, dtype=torch.float32).clamp_min(1.0).view(1, *([1] * (decoded_dyn.dim() - 2)), -1)
    t0 = (
        lengths.to(device=decoded_dyn.device, dtype=torch.float32)
        / speed_limits.to(device=decoded_dyn.device, dtype=torch.float32).clamp_min(0.1)
    ).view(1, *([1] * (decoded_dyn.dim() - 2)), -1)
    pred_volume = _positive_pred_volume_for_loss(decoded_dyn[..., FLOW_VOLUME_INDEX]) * cap
    true_volume = target_dyn[..., FLOW_VOLUME_INDEX].float().clamp_min(0.0) * cap
    pred_time = t0 * (1.0 + 0.15 * (pred_volume / cap).pow(4.0))
    true_time = t0 * (1.0 + 0.15 * (true_volume / cap).pow(4.0))
    pred_eval = pred_time / t0.clamp_min(1e-6) - 1.0
    true_eval = true_time / t0.clamp_min(1e-6) - 1.0
    weighted_mse = (pred_eval - true_eval).pow(2).mean()
    loss = BPR_SUPERVISION_WEIGHT * weighted_mse
    return loss, {
        "bpr_weighted_mse": weighted_mse.detach(),
        "bpr_supervision_loss": loss.detach(),
    }


def episode_t_all_supervision_loss(
    *,
    flow_projector: FlowProjector,
    z_tokens: torch.Tensor,
    reward_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = z_tokens.device
    if reward_targets.dim() not in {1, 2}:
        raise ValueError(
            "reward_targets must have shape (B,) or (B, T), "
            f"got {tuple(reward_targets.shape)}"
        )
    reward_last = (
        reward_targets[:, -1].to(device=device, dtype=torch.float32)
        if reward_targets.dim() == 2
        else reward_targets.to(device=device, dtype=torch.float32)
    )
    target_t_all = -reward_last
    valid = torch.isfinite(target_t_all) & (target_t_all > 0.0) & (reward_last < 0.0)
    if not bool(valid.all().item()):
        raise ValueError(
            "reward_targets must contain finite negative total-travel-minute "
            "rewards"
        )

    pred_t_all = flow_projector.predict_t_all_from_z(z_tokens[:, -1])
    target_t_all = target_t_all.float()
    scale = target_t_all.mean().clamp_min(1e-6)
    mse = ((pred_t_all.float() / scale) - (target_t_all / scale)).pow(2).mean()
    loss = T_ALL_SUPERVISION_WEIGHT * mse
    return loss, {
        "t_all_supervision_loss": loss.detach(),
        "t_all_pred_mse": mse.detach(),
    }


def independent_transition_fm_loss(
    dynamics: Dynamics, *, z_inputs: torch.Tensor, z_targets: torch.Tensor,
    action: torch.Tensor, action_events: SparseActionEvents | None = None,
    steps: int = 2, solver: str = "euler", picard_iterations: int | None = None,
    prior_weight: float = 0.1, noise: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None, predict_endpoints: bool = True,
    detach_prior_increment: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Independent observed-state transitions with separate endpoint targets.

    Each window starts from its full observed state. Endpoint volume may omit
    births within that one window, but cannot become the next window's input.
    Sampled endpoints are optional. Posterior readout supervision reuses the
    already encoded real targets, eliminating unnecessary integration passes.
    """
    if not isinstance(predict_endpoints, bool):
        raise ValueError('predict_endpoints must be a boolean')
    if z_inputs.ndim != 4 or tuple(z_inputs.shape[-2:]) != (dynamics.n_edges, FLOW_LATENT_DIM):
        raise ValueError('z_inputs must have shape (B, H, E, 64)')
    if z_targets.shape != z_inputs.shape or z_inputs.shape[0] < 1 or z_inputs.shape[1] < 1:
        raise ValueError('z_targets must match nonempty z_inputs')
    batch, horizon, edges, _ = z_inputs.shape
    _validate_transition_tensor(action, name='action', expected_shape=(batch, horizon, edges))
    if not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError('prior_weight must be finite and nonnegative')
    if flow_time is None:
        flow_time = torch.rand(batch, horizon, device=z_inputs.device, dtype=z_inputs.dtype)
    _validate_transition_tensor(flow_time, name='flow_time', expected_shape=(batch, horizon))
    if bool(((flow_time < 0) | (flow_time > 1) | ~torch.isfinite(flow_time)).any()):
        raise ValueError('flow_time must be finite and lie in [0, 1]')
    if noise is not None:
        _validate_transition_tensor(noise, name='noise',
            expected_shape=(batch, horizon, dynamics.graph_prior.rank, FLOW_LATENT_DIM))
        noise = noise.reshape(batch * horizon, 1, dynamics.graph_prior.rank, FLOW_LATENT_DIM)
    if action_events is not None:
        if action_events.batch_size != batch or action_events.time_steps < horizon:
            raise ValueError('action event windows must cover independent supervision')
        selected = action_events.select(action_events.time_index < horizon)
        action_events = SparseActionEvents(
            batch_index=selected.batch_index * horizon + selected.time_index,
            time_index=torch.zeros_like(selected.time_index), phase=selected.phase,
            current_edge=selected.current_edge, next_edge=selected.next_edge,
            destination_edge=selected.destination_edge, event_index=selected.event_index,
            vehicle_ids=selected.vehicle_ids, batch_size=batch * horizon, time_steps=1)
    current = z_inputs.detach().reshape(batch * horizon, 1, edges, FLOW_LATENT_DIM)
    target = z_targets.detach().reshape_as(current)
    flattened_action = action.reshape(batch * horizon, 1, edges)
    condition = dynamics.encode_condition(flattened_action, action_events=action_events)
    start, increment, condition = dynamics.prior_start(current, condition, noise,
        detach_prior_increment=detach_prior_increment)
    tau = flow_time.reshape(batch * horizon, 1, 1, 1)
    interpolated = (1 - tau) * start + tau * target
    velocity = dynamics(flattened_action, packed_latents=interpolated,
                        flow_time=flow_time.reshape(batch * horizon, 1), condition=condition).latent_velocity
    target_velocity = (target - start).detach()
    flow_mse = (velocity.float() - target_velocity.float()).square().mean()
    prior_target = dynamics.graph_prior.project(target - current).detach()
    prior_mse = (increment.float() - prior_target.float()).square().mean()
    if predict_endpoints:
        with torch.no_grad():
            predicted = dynamics.solve(start.detach(), flattened_action, condition, steps=steps,
                                       solver=solver, picard_iterations=picard_iterations)
    else:
        predicted = target
    loss = (flow_mse if prior_weight == 0 and not getattr(dynamics.graph_prior, "prior_action_conditioned", True)
            else flow_mse + prior_weight * prior_mse)
    return loss, {'flow_mse': flow_mse.detach(), 'weighted_flow_mse': flow_mse.detach(),
                  'prior_mse': prior_mse.detach(), 'prior_loss': (prior_weight * prior_mse).detach()}, predicted.reshape_as(z_inputs)


def recursive_fm_loss(
    dynamics: Dynamics, *, z_sequence: torch.Tensor, action: torch.Tensor,
    steps: int = 2, solver: str = "euler", picard_iterations: int | None = None,
    prior_weight: float = 0.1, depth_weights: torch.Tensor | None = None,
    noise: torch.Tensor | None = None, flow_time: torch.Tensor | None = None,
    action_events: SparseActionEvents | None = None,
    predict_endpoints: bool = True,
    checkpoint_interval: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
    """Recorded-condition recursion; later losses retain gradients through earlier solves.

    Reference encodings, prior increment targets and target velocities are detached.
    Predicted stochastic endpoints have no endpoint reconstruction/MSE objective.
    When endpoints are not consumed, the final solve is unnecessary: every
    supervised velocity/prior loss and every earlier recursive solve remains.
    """
    if z_sequence.dim() != 4 or tuple(z_sequence.shape[-2:]) != (dynamics.n_edges, FLOW_LATENT_DIM):
        raise ValueError("z_sequence must have shape (B, H+1, E, 64)")
    batch, total_steps, edges, _ = z_sequence.shape
    horizon = total_steps - 1
    if horizon < 1:
        raise ValueError("recursive supervision requires at least one transition")
    _validate_transition_tensor(action, name="action", expected_shape=(batch, horizon, edges))
    if action_events is not None and (action_events.batch_size != batch or action_events.time_steps < horizon):
        raise ValueError('action event windows must cover the recursive supervision horizon')
    if not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("prior_weight must be finite and nonnegative")
    if not isinstance(predict_endpoints, bool):
        raise ValueError("predict_endpoints must be a boolean")
    if (isinstance(checkpoint_interval, bool) or not isinstance(checkpoint_interval, int)
            or checkpoint_interval < 0):
        raise ValueError("checkpoint_interval must be a nonnegative integer")
    if flow_time is None:
        flow_time = torch.rand(batch, horizon, device=z_sequence.device, dtype=z_sequence.dtype)
    _validate_transition_tensor(flow_time, name="flow_time", expected_shape=(batch, horizon))
    if bool(((flow_time < 0) | (flow_time > 1) | ~torch.isfinite(flow_time)).any()):
        raise ValueError("flow_time must be finite and lie in [0, 1]")
    if noise is not None:
        _validate_transition_tensor(noise, name="noise",
            expected_shape=(batch, horizon, dynamics.graph_prior.rank, FLOW_LATENT_DIM))
    if depth_weights is None:
        weights = torch.ones(horizon, device=z_sequence.device, dtype=torch.float32)
    else:
        weights = torch.as_tensor(depth_weights, device=z_sequence.device, dtype=torch.float32).detach()
    if tuple(weights.shape) != (horizon,) or bool((~torch.isfinite(weights) | (weights < 0)).any()) or weights.sum() <= 0:
        raise ValueError("depth_weights must be nonnegative finite horizon weights with positive sum")
    current = z_sequence[:, :1]
    references = z_sequence[:, 1:].detach()
    flow_losses, prior_losses, predictions = [], [], []
    for index in range(horizon):
        step_action = action[:, index:index+1]
        condition = dynamics.encode_condition(step_action, action_events=None if action_events is None else action_events.for_step(index))
        step_noise = None if noise is None else noise[:, index:index+1]
        start, increment, condition = dynamics.prior_start(current, condition, step_noise)
        target = references[:, index:index+1]
        tau = flow_time[:, index:index+1]
        interpolated = (1 - tau[..., None, None]) * start + tau[..., None, None] * target
        target_velocity = (target - start).detach()
        velocity = dynamics(step_action, packed_latents=interpolated,
                            flow_time=tau, condition=condition).latent_velocity
        flow_losses.append((velocity.float() - target_velocity.float()).square().mean())
        prior_target = dynamics.graph_prior.project(target - current).detach()
        prior_losses.append((increment.float() - prior_target.float()).square().mean())
        if predict_endpoints or index + 1 < horizon:
            solve_kwargs = ({'checkpoint_interval': checkpoint_interval} if checkpoint_interval else {})
            current = dynamics.solve(start, step_action, condition, steps=steps, solver=solver,
                                     picard_iterations=picard_iterations, **solve_kwargs)
            if predict_endpoints:
                predictions.append(current)
    flow_mse = (torch.stack(flow_losses) * weights).sum() / weights.sum()
    prior_mse = (torch.stack(prior_losses) * weights).sum() / weights.sum()
    loss = (flow_mse if prior_weight == 0 and not getattr(dynamics.graph_prior, "prior_action_conditioned", True)
            else flow_mse + prior_weight * prior_mse)
    return loss, {"flow_mse": flow_mse.detach(), "weighted_flow_mse": flow_mse.detach(),
                  "prior_mse": prior_mse.detach(), "prior_loss": (prior_weight * prior_mse).detach(),
                  "t_mean": flow_time.mean().detach(),
                  "supervision_horizon": torch.tensor(horizon, device=z_sequence.device)}, (
                      torch.cat(predictions, dim=1) if predict_endpoints else None)


def cfm_loss(
    dynamics: Dynamics | nn.Module,
    *,
    z_sequence: torch.Tensor,
    action: torch.Tensor,
    action_events: SparseActionEvents | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(dynamics, Dynamics):
        loss, metrics, _ = recursive_fm_loss(dynamics, z_sequence=z_sequence, action=action, action_events=action_events)
        return loss, metrics
    if z_sequence.dim() != 4 or int(z_sequence.shape[-1]) != FLOW_LATENT_DIM or int(z_sequence.shape[1]) < 2:
        raise ValueError(
            "z_sequence must have shape (B, T, E, 64) with T >= 2, "
            f"got {tuple(z_sequence.shape)}"
        )
    batch, total_steps, edge_count, _ = z_sequence.shape
    transition_steps = total_steps - 1
    edge_shape = (batch, transition_steps, edge_count)
    _validate_transition_tensor(action, name="action", expected_shape=edge_shape)
    source_z = z_sequence[:, :-1].contiguous()
    target_z = z_sequence[:, 1:].contiguous()
    flow_time = torch.rand(batch, transition_steps, device=z_sequence.device, dtype=torch.float32)
    x_t = (1.0 - flow_time[..., None, None]) * source_z + flow_time[..., None, None] * target_z
    target_velocity = target_z - source_z

    output = dynamics(
        action,
        packed_latents=x_t,
        flow_time=flow_time,
        **({} if action_events is None else {'action_events': action_events}),
    )
    error2 = (output.latent_velocity.float() - target_velocity.float()).pow(2)
    flow_mse = error2.mean()

    loss = flow_mse
    return loss, {
        "flow_mse": flow_mse.detach(),
        "weighted_flow_mse": flow_mse.detach(),
        "t_mean": flow_time.mean().detach(),
    }


def cfm_predict_one_timestep(
    dynamics: Dynamics | nn.Module,
    *,
    past_latents: torch.Tensor,
    action: torch.Tensor,
    action_events: SparseActionEvents | None = None,
) -> torch.Tensor:
    if past_latents.dim() != 4 or int(past_latents.shape[-1]) != FLOW_LATENT_DIM or int(past_latents.shape[1]) < 1:
        raise ValueError(
            "past_latents must have shape (B, T, E, 64) with T >= 1, "
            f"got {tuple(past_latents.shape)}"
        )
    current = past_latents[:, -1:].contiguous()
    batch, _, edge_count, _ = current.shape
    edge_shape = (batch, 1, edge_count)
    _validate_transition_tensor(action, name="action", expected_shape=edge_shape)
    if isinstance(dynamics, Dynamics):
        return dynamics.transition(current, action, action_events=action_events)

    velocity = dynamics(
        action,
        packed_latents=current,
        flow_time=torch.zeros((batch, 1), device=current.device, dtype=torch.float32),
        **({} if action_events is None else {'action_events': action_events}),
    ).latent_velocity
    return current + velocity


__all__ = [
    "UrbanDreamerWorldModel",
    "cfm_loss",
    "cfm_predict_one_timestep",
    "recursive_fm_loss",
    "independent_transition_fm_loss",
    "decoder_bpr_supervision_loss",
    "decoder_reconstruction_loss",
    "episode_t_all_supervision_loss",
    "transition_decoded_supervision_loss",
]
