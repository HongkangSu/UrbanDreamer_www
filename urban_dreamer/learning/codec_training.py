"""Same-frame codec supervision in vehicle units, separate from latent dynamics."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


CODEC_LEARNING_RATE = 1e-4
CODEC_GRADIENT_CLIP = 10.0


def _coefficient(method: Any, name: str, default: float) -> float:
    value = float(getattr(method, name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and non-negative')
    return value


def _projector(model):
    return getattr(model, 'flow_projector', model)


def _physical(network, flow):
    capacity = torch.as_tensor(network.capacities, device=flow.device, dtype=torch.float32).clamp_min(1.)
    speed_limit = torch.as_tensor(network.speed_limits_mps, device=flow.device, dtype=torch.float32).clamp_min(.1)
    if capacity.ndim != 1 or speed_limit.shape != capacity.shape or len(capacity) != flow.shape[-2]:
        raise ValueError('codec network capacity and speed arrays must match flow edges')
    if not bool(torch.isfinite(capacity).all() & torch.isfinite(speed_limit).all()):
        raise ValueError('codec physical arrays must be finite')
    return capacity, speed_limit


def _validate_flow(flow, next_flow):
    if flow.ndim < 2 or flow.shape[-1] != 8 or flow.shape[-2] < 1:
        raise ValueError('codec flow must have shape (..., E, 8) with E>0')
    if tuple(next_flow.shape) != tuple(flow.shape):
        raise ValueError('codec future speed frame must match flow shape')
    if not bool(torch.isfinite(flow).all() & torch.isfinite(next_flow).all()):
        raise ValueError('codec input and speed endpoint must be finite')
    if bool((flow[..., 2] < 0).any()):
        raise ValueError('codec observed vehicle density must be non-negative')


def _masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def _decode(model, flow):
    projector = _projector(model)
    return projector.decode(projector(flow))


def _speed_masks(flow, next_flow, capacity, speed_limit, codec):
    count = flow[..., 2].detach().float() * capacity
    speed = .5 * (flow[..., 1].detach().float() + next_flow[..., 1].detach().float()) * speed_limit
    return count > 0, ((count >= float(getattr(codec, 'slow_min_vehicles', 5.)) - 1e-5)
                       & (speed <= float(getattr(codec, 'slow_speed_mps', 2.))))


def codec_speed_masks(flow, next_flow, network, codec=None):
    """Masks use current-frame counts and the forward mean-speed target in m/s."""
    _validate_flow(flow, next_flow)
    capacity, speed_limit = _physical(network, flow)
    return _speed_masks(flow, next_flow, capacity, speed_limit, codec)


def codec_decoded_metrics(decoded, flow, next_flow, network, codec=None) -> dict[str, torch.Tensor]:
    """Raw held-out errors; volume labels always belong to the input frame."""
    _validate_flow(flow, next_flow)
    capacity, speed_limit = _physical(network, flow)
    predicted_count = decoded[..., 2].float().clamp_min(0) * capacity
    target_count = flow[..., 2].float() * capacity
    occupied, empty = target_count > 0, target_count == 0
    count_error = predicted_count - target_count
    total_target = target_count.sum(-1)
    total_error = count_error.sum(-1)
    speed_target = .5 * (flow[..., 1].float() + next_flow[..., 1].float())
    # Score the returned decoder speed itself, with no loss-only transform or clipping.
    speed_error = (decoded[..., 1].float() - speed_target) * speed_limit
    _, slow = _speed_masks(flow, next_flow, capacity, speed_limit, codec)
    return {name: value.detach() for name, value in dict(
        codec_count_mae_all=count_error.abs().mean(),
        codec_count_mae_occupied=_masked_mean(count_error.abs(), occupied),
        codec_count_mae_empty=_masked_mean(count_error.abs(), empty),
        codec_count_rmse_all=count_error.square().mean().sqrt(),
        codec_count_occupied_roads=occupied.sum(), codec_count_empty_roads=empty.sum(),
        codec_empty_predicted_total=(predicted_count * empty).sum(),
        codec_empty_false_positive_fraction=_masked_mean((predicted_count > .1).float(), empty),
        codec_total_count_mae=total_error.abs().mean(),
        codec_total_count_bias=total_error.mean(),
        codec_total_relative_error=(total_error.abs() / total_target.clamp_min(1.)).mean(),
        codec_predicted_count_total=predicted_count.sum(-1).mean(),
        codec_target_count_total=total_target.mean(),
        codec_speed_mae_mps=speed_error.abs().mean(),
        codec_speed_mae_occupied_mps=_masked_mean(speed_error.abs(), occupied),
        codec_slow_roads=slow.sum(),
        codec_speed_mae_slow_mps=_masked_mean(speed_error.abs(), slow),
    ).items()}


def codec_objective(model, flow, next_flow, network, method=None, *, codec=None, decoder_only=False):
    """Train codec only: same-frame counts and the existing forward speed-mean label.

    Default road weighting retains 80% all-road supervision and adds a modest
    20% occupied-road term. It does not give empty/occupied groups equal weight.
    The mass penalty normalizes by observed population before a bounded-tail
    asinh transform, including zero-population frames without singular gradients.
    """
    _validate_flow(flow, next_flow)
    capacity, speed_limit = _physical(network, flow)
    if decoder_only:
        with torch.no_grad():
            latent = _projector(model)(flow)
        decoded = _projector(model).decode(latent.detach()).float()
    else:
        decoded = _decode(model, flow).float()
    predicted_count = decoded[..., 2].clamp_min(0) * capacity
    target_count = flow[..., 2].float().detach() * capacity
    occupied = target_count > 0
    count_square = (torch.log1p(predicted_count) - torch.log1p(target_count)).square()
    foreground = _coefficient(method, 'codec_foreground_fraction', .2)
    if foreground > 1:
        raise ValueError('codec_foreground_fraction must be in [0, 1]')
    if not bool(occupied.any()):
        foreground = 0.
    count_loss = (1 - foreground) * count_square.mean() + foreground * _masked_mean(count_square, occupied)
    normalized_mass_error = ((predicted_count.sum(-1) - target_count.sum(-1))
                             / (target_count.sum(-1) + 1.))
    mass_loss = torch.asinh(normalized_mass_error).square().mean()
    # No reciprocal speed: its near-zero derivative formerly dominated the count channel.
    predicted_speed = decoded[..., 1]
    speed_target = (.5 * (flow[..., 1].float() + next_flow[..., 1].float())).detach().clamp_min(0)
    speed_cells = F.smooth_l1_loss(predicted_speed, speed_target, beta=.1, reduction='none')
    _, slow = _speed_masks(flow, next_flow, capacity, speed_limit, codec)
    occupied_speed_loss = _masked_mean(speed_cells, occupied)
    slow_speed_loss = _masked_mean(speed_cells, slow)
    speed_loss = (speed_cells.mean()
                  + _coefficient(codec, 'occupied_speed_weight', 0.) * occupied_speed_loss
                  + _coefficient(codec, 'slow_speed_weight', 0.) * slow_speed_loss)
    occupancy_loss = F.mse_loss(decoded[..., 0], flow[..., 0].float().detach())
    halt_loss = F.mse_loss(decoded[..., 3], flow[..., 3].float().detach())
    loss = (_coefficient(method, 'codec_count_weight', 1.) * count_loss
            + _coefficient(method, 'codec_mass_weight', .1) * mass_loss
            + _coefficient(method, 'codec_speed_weight', 1.) * speed_loss
            + _coefficient(method, 'codec_other_weight', .05) * (occupancy_loss + halt_loss))
    metrics = codec_decoded_metrics(decoded, flow, next_flow, network, codec)
    metrics.update({name: value.detach() for name, value in dict(
        codec_loss=loss, codec_count_loss=count_loss, codec_count_loss_all=count_square.mean(),
        codec_count_loss_occupied=_masked_mean(count_square, occupied), codec_mass_loss=mass_loss,
        codec_speed_loss=speed_loss, codec_occupancy_loss=occupancy_loss, codec_halt_loss=halt_loss,
        codec_speed_loss_occupied=occupied_speed_loss, codec_speed_loss_slow=slow_speed_loss,
    ).items()})
    return loss, metrics


@torch.no_grad()
def codec_metrics(model, flow, next_flow, network, codec=None):
    return codec_decoded_metrics(_decode(model, flow), flow, next_flow, network, codec)


def decoder_parameters(model):
    projector = _projector(model)
    return [parameter for name in ('decoder_shared', 'decoder_speed', 'decoder_volume', 'decoder_other')
            for parameter in getattr(projector, name).parameters()]


def make_decoder_optimizer(model, *, lr=CODEC_LEARNING_RATE):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('decoder learning rate must be finite and positive')
    return torch.optim.Adam(decoder_parameters(model), lr=lr)


def make_codec_optimizer(model, *, lr=CODEC_LEARNING_RATE):
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('codec learning rate must be finite and positive')
    parameters = [parameter for parameter in _projector(model).parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError('codec optimizer requires trainable codec parameters')
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)


def codec_train_step(model, flow, next_flow, network, method, optimizer, codec=None):
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = codec_objective(model, flow, next_flow, network, method, codec=codec)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite codec loss')
    loss.backward()
    parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
    norm = torch.nn.utils.clip_grad_norm_(parameters, CODEC_GRADIENT_CLIP)
    if not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('nonfinite codec gradient')
    optimizer.step()
    metrics['codec_gradient_norm'] = norm.detach()
    return metrics
