"""Static full-batch velocity inference without changing the FP32 FM integrator."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path

import torch

from ..models.world_model import DynamicsOutput, FLOW_LATENT_DIM


FULL_IMAGINATION_BATCH = 24  # Eight trajectories, three independent noise samples.


@dataclass(frozen=True)
class ImaginationInferenceProfile:
    space_chunk: int
    time_chunk: int
    mlp_chunk: int
    precision: str
    compile: bool
    full_batch: int = FULL_IMAGINATION_BATCH
    compile_mode: str = "default"


def _profile(resources):
    profile = ImaginationInferenceProfile(**{
        name: getattr(resources, 'world_model_imagination_' + name)
        for name in ('space_chunk', 'time_chunk', 'mlp_chunk', 'precision', 'compile', 'compile_mode')})
    for value in (profile.space_chunk, profile.time_chunk, profile.mlp_chunk):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('imagination chunk sizes must be nonnegative integers')
    if profile.precision not in {'fp32', 'fp16', 'bf16', 'tf32'}:
        raise ValueError('imagination precision must be fp32, fp16, bf16, or tf32')
    if profile.compile_mode not in {'default', 'reduce-overhead'}:
        raise ValueError('imagination compile mode must be default or reduce-overhead')
    if not isinstance(profile.compile, bool):
        raise ValueError('imagination compile must be boolean')
    return profile


def _supports_velocity_profile(packed_latents, condition):
    return (not torch.is_grad_enabled() and packed_latents.device.type == 'cuda'
            and packed_latents.dtype == torch.float32 and condition is not None)


@contextmanager
def _velocity_tf32():
    """Use one precision API, restoring the caller's policy even on failure."""
    matmul = torch.backends.cuda.matmul
    # PyTorch 2.9+ exposes fp32_precision. Do not mix that API with the
    # deprecated allow_tf32 setter; older builds use only the legacy branch.
    if hasattr(matmul, 'fp32_precision'):
        previous = matmul.fp32_precision
        try:
            matmul.fp32_precision = 'tf32'
            yield
        finally:
            matmul.fp32_precision = previous
    else:
        previous = matmul.allow_tf32
        try:
            matmul.allow_tf32 = True
            yield
        finally:
            matmul.allow_tf32 = previous


class _VelocityInference:
    """A plain callable so optimizer/module registries remain untouched."""
    def __init__(self, dynamics, profile):
        self.eager = dynamics.forward
        self.dynamics = dynamics
        self.n_edges = dynamics.n_edges
        self.profile = profile
        self.compiled = None
        if profile.compile:
            self.compiled = torch.compile(self.eager, backend='inductor', mode=profile.compile_mode,
                                          fullgraph=True, dynamic=False)

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.eager = copy.deepcopy(self.eager, memo)
        copied.dynamics = copy.deepcopy(self.dynamics, memo)
        copied.n_edges = self.n_edges
        copied.profile = self.profile
        copied.compiled = (torch.compile(copied.eager, backend='inductor', mode=copied.profile.compile_mode,
                                         fullgraph=True, dynamic=False)
                           if self.compiled is not None else None)
        return copied

    def __call__(self, action, *, packed_latents, flow_time, condition=None, action_events=None):
        supported = (not getattr(self.dynamics, 'training', False)
                     and _supports_velocity_profile(packed_latents, condition))
        forward = self.eager
        full_shape = (self.profile.full_batch, 1, self.n_edges, FLOW_LATENT_DIM)
        if (supported and tuple(packed_latents.shape) == full_shape and self.compiled is not None
                and action_events is None):
            forward = self.compiled
        # Autocast surrounds only the velocity call. The caller retains FP32
        # prior/noise/conditions and integrator state for either autocast dtype.
        amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(self.profile.precision)
        amp = (torch.autocast('cuda', dtype=amp_dtype)
               if supported and amp_dtype is not None else nullcontext())
        tf32 = _velocity_tf32() if supported and self.profile.precision == 'tf32' else nullcontext()
        with tf32, amp:
            result = forward(action, packed_latents=packed_latents, flow_time=flow_time,
                             condition=condition, action_events=action_events)
        if supported:
            return DynamicsOutput(latent_velocity=result.latent_velocity.float())
        return result


def configure_imagination_inference(trainer, *, shared_policy_search=False, full_batch=None):
    """Apply before an imagination pass, after training has reset chunk sizes.

    Calls are idempotent. CPU/gradient execution remains eager, and every tail
    batch remains eager. No padding, trajectory dropping or solver change occurs.
    """
    if full_batch is None:
        full_batch = (trainer.config.resources.imagination_rollouts_per_pass
                      * trainer.config.method.disagreement_samples)
    if isinstance(full_batch, bool) or not isinstance(full_batch, int) or full_batch < 1:
        raise ValueError('imagination compile batch must be a positive integer')
    profile = replace(_profile(trainer.config.resources), full_batch=full_batch)
    if shared_policy_search:
        # Full/tail MLP chunks may select different BF16 GEMM kernels for
        # identical rows. Search's duplicate controls require one kernel shape
        # across all counterfactuals, including both disagreement samples.
        profile = replace(profile, mlp_chunk=0)
    dynamics = trainer.world_model.dynamics
    device = next(dynamics.parameters()).device
    if device.type != 'cuda':
        profile = replace(profile, precision='fp32', compile=False)
    elif profile.precision == 'bf16':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                profile = replace(profile, precision='fp32')
    parameters = tuple(id(parameter) for parameter in dynamics.parameters())
    keys = tuple(dynamics.state_dict())
    # Existing chunk setters support zero as an unchunked operation.
    dynamics._set_chunk_profile(space_chunk=profile.space_chunk, time_chunk=profile.time_chunk,
                                mlp_chunk=profile.mlp_chunk)
    wrapper = getattr(dynamics, '_imagination_velocity_inference', None)
    if wrapper is None or wrapper.profile != profile:
        if wrapper is not None:
            dynamics.forward = wrapper.eager
        if profile.compile:
            os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(
                (Path(trainer.config.run.output_dir).parent / 'inductor_cache').resolve()))
        wrapper = _VelocityInference(dynamics, profile)
        dynamics.forward = wrapper
        dynamics._imagination_velocity_inference = wrapper
    if parameters != tuple(id(parameter) for parameter in dynamics.parameters()):
        raise RuntimeError('imagination inference changed optimizer parameter identity')
    if keys != tuple(dynamics.state_dict()):
        raise RuntimeError('imagination inference changed checkpoint state keys')
    return asdict(profile)
