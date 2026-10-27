from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .portable import PROJECT_ROOT, portable_data, resolve_resource


NONARRIVAL_PENALTY = -5000.0


class ConfigError(ValueError):
    """Raised when the mainline TOML contract is invalid."""


@dataclass(frozen=True)
class InputConfig:
    network_name: str
    network_dir: Path


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    device: str
    seed: int
    world_model_initialization_epochs: int
    online_training_episodes: int
    evaluation_episodes: int
    # Accepted only to read frozen collector/baseline configurations. The
    # mainline starts from archived data and never collects warmup episodes.
    warmup_episodes: int = 0
    codec_pretraining_epochs: int = 10


@dataclass(frozen=True)
class SimulationConfig:
    vehicle_max_speed_mps: float
    world_model_intervals_per_episode: int
    episode_wall_timeout_seconds: float
    training_completion_buffer_seconds: int
    evaluation_horizon_seconds: int
    incident_edge: str
    incident_start_seconds: float
    incident_end_seconds: float
    incident_speed_mps: float
    sumo_extra_args: tuple[str, ...]
    demand_duration_seconds: float = 1800.0
    session_protocol: str = 'event'
    observation_step_seconds: float = 1.0
    backend: str = 'libsumo'
    traffic_model: str = 'meso'
    meso_junction_control: bool = True
    meso_lane_queue: bool = False
    meso_multi_queue: bool = False
    meso_edge_length_m: float = 98.0
    meso_interpolate_positions: bool = True
    routing_mode: str = 'current_to_dest'
    # Deprecated standalone collector settings, retained in frozen snapshots.
    warmup_horizon_seconds: int = 1800
    warmup_completion_drain_seconds: int = 3000
    warmup_completion_patience_seconds: int = 0


@dataclass(frozen=True)
class DataConfig:
    archive_roots: tuple[Path, ...] = ()
    cache_episodes: int = 2
    pretrain_cache_manifest: Path | None = None
    validation_cache_manifest: Path | None = None
    validation_archive_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ResourceConfig:
    world_model_initialization_batch_size: int
    world_model_initialization_sequence_length: int
    world_model_gradient_accumulation_steps: int
    online_world_model_batch_size: int
    world_model_pretrain_space_chunk: int = 1
    world_model_pretrain_time_chunk: int = 256
    world_model_pretrain_mlp_chunk: int = 4096
    world_model_pretrain_activation_checkpointing: bool = True
    world_model_pretrain_precision: str = "fp32"
    world_model_pretrain_compile_blocks: bool = False
    world_model_imagination_space_chunk: int = 16
    world_model_imagination_time_chunk: int = 32768
    world_model_imagination_mlp_chunk: int = 16384
    world_model_imagination_precision: str = "fp32"
    world_model_imagination_compile: bool = False
    world_model_imagination_compile_mode: str = "default"
    imagination_rollouts_per_pass: int = 2
    imagination_passes_per_episode: int = 1
    imagination_time_budget_seconds: float = 0.


@dataclass(frozen=True)
class CodecConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_updates: int = 20000
    validation_every: int = 500
    occupied_vehicle_mae_max: float = .5
    vehicle_mae_max: float = .1
    empty_vehicle_mae_max: float = .05
    total_relative_mae_max: float = .1
    speed_mae_mps_max: float = 1.5
    speed_positive: bool = True
    decoder_pretrain_enabled: bool = False
    decoder_pretrain_learning_rate: float = 1e-4
    decoder_online_enabled: bool = False
    decoder_online_learning_rate: float = 1e-5
    decoder_allow_unready: bool = False
    occupied_speed_weight: float = 0.
    slow_speed_weight: float = 0.
    slow_speed_mps: float = 2.
    slow_min_vehicles: float = 5.
    occupied_speed_mae_mps_max: float = 1.5
    slow_speed_mae_mps_max: float = 1.5
    slow_validation_min_roads: int = 0


@dataclass(frozen=True)
class MethodConfig:
    low_frequency_rank: int = 16
    prior_noise_scale: float = 0.1
    prior_loss_weight: float = 0.1
    prior_filter: str = 'soft'
    prior_filter_strength: float = 0.1
    prior_action_conditioned: bool = False
    prior_filter_input: bool = False
    prior_usage: str = 'condition_only'
    fm_steps: int = 10
    fm_solver: str = 'euler'
    picard_iterations: int = 2
    supervision_horizon: int = 3
    supervision_increment_updates: int = 100
    disagreement_samples: int = 2
    disagreement_weight: float = 0.0
    critic_value_mode: str = 'legacy'
    critic_target_tau: float = .01
    critic_terminal_aux_ratio: float = .25
    sp_prior_weight: float = 0.0
    sp_prior_evaluation_weight: float = 0.0
    sp_prior_temperature: float = 30.0
    sp_prior_cost_mode: str = 'free_flow'
    sp_prior_dynamic_speed_floor_mps: float = .5
    sp_prior_hold_episodes: int = 200
    sp_prior_off_episode: int = 1000
    evaluation_demand_seed: int = 314159
    transition_volume_scope: str = 'previous_state_excluding_current_births'
    exposure_readout_supervision: str = 'observed_endpoint'
    # Omitted fields retain frozen configurations' historical entropy units.
    actor_entropy_mode: str = 'annealed'
    actor_objective: str = 'counterfactual_actor_critic_recoverq_v1'
    imagination_actor_epochs: int = 512
    imagination_value_epochs: int = 1
    imagination_actor_advantage: str = 'event_mc_q_v1'
    imagination_horizon_windows: int = 8
    imagination_seed_scope: str = 'full_episode_v1'
    imagination_real_policy_guard: str = 'none'
    training_temperature_profile: str = 'none'
    real_policy_update: str = 'none'
    real_policy_critic_epochs: int = 8
    real_policy_actor_epochs: int = 4
    real_policy_value_mode: str = 'shared'
    model_policy_search_schedule: str = 'round_robin'
    actor_entropy_initial_coefficient: float = .002
    actor_entropy_final_coefficient: float = .0001
    actor_entropy_hold_fraction: float = .1
    actor_entropy_end_fraction: float = .8
    # Offline initialization always uses the native independent one-step loss.
    fm_training_mode: str = 'recursive_3step'


@dataclass(frozen=True)
class MainlineConfig:
    inputs: InputConfig
    run: RunConfig
    simulation: SimulationConfig
    resources: ResourceConfig
    semantic_fingerprint: str
    source_path: Path
    method: MethodConfig = field(default_factory=MethodConfig)
    data: DataConfig = field(default_factory=DataConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)

    def resolved_snapshot(self, *, root: Path | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("semantic_fingerprint")
        payload.pop("source_path")
        return portable_data(payload, root=root or self.source_path.parent)


_SECTION_FIELDS: dict[str, frozenset[str]] = {
    "inputs": frozenset(InputConfig.__dataclass_fields__),
    "run": frozenset(RunConfig.__dataclass_fields__),
    "simulation": frozenset(SimulationConfig.__dataclass_fields__),
    "resources": frozenset(ResourceConfig.__dataclass_fields__),
    "method": frozenset(MethodConfig.__dataclass_fields__),
    "data": frozenset(DataConfig.__dataclass_fields__),
    "codec": frozenset(CodecConfig.__dataclass_fields__),
}


def load_config(path: str | Path) -> MainlineConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read TOML config resource {source_path.name}") from None

    _validate_exact_schema(raw)
    base_dir = source_path.parent
    inputs = _parse_inputs(raw["inputs"], base_dir)
    run = _parse_run(raw["run"], base_dir)
    simulation = _parse_simulation(raw["simulation"])
    resources = _parse_resources(raw["resources"])
    method = _parse_method(raw.get('method', {}))
    from ..learning.counterfactual_training import validate_counterfactual_method
    try:
        validate_counterfactual_method(method)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if (resources.imagination_rollouts_per_pass != 2
            or resources.imagination_passes_per_episode != 1
            or resources.imagination_time_budget_seconds != 0):
        raise ConfigError('native mainline requires two rollouts, one pass, and no partial-pass timeout')
    validate_fm_training_profile(method, resources)
    data = _parse_data(raw.get('data', {}), base_dir)
    codec = _parse_codec(raw.get('codec', {}))
    semantic_fingerprint = _semantic_fingerprint(
        inputs=inputs,
        run=run,
        simulation=simulation,
        resources=resources,
        method=method,
        data=data,
        codec=codec,
    )
    return MainlineConfig(
        inputs=inputs,
        run=run,
        simulation=simulation,
        resources=resources,
        semantic_fingerprint=semantic_fingerprint,
        source_path=source_path,
        method=method,
        data=data,
        codec=codec,
    )


def _validate_exact_schema(raw: Mapping[str, Any]) -> None:
    unknown_sections = set(raw) - set(_SECTION_FIELDS)
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ConfigError(f"unknown top-level section(s): {names}")
    missing_sections = set(_SECTION_FIELDS) - set(raw)
    missing_sections.discard('method')
    missing_sections.discard('data')
    missing_sections.discard('codec')
    if missing_sections:
        names = ", ".join(sorted(missing_sections))
        raise ConfigError(f"missing top-level section(s): {names}")
    for section_name, expected_fields in _SECTION_FIELDS.items():
        if section_name in {'method', 'data', 'codec'} and section_name not in raw:
            continue
        section = raw[section_name]
        if not isinstance(section, dict):
            raise ConfigError(f"[{section_name}] must be a TOML table")
        unknown_fields = set(section) - set(expected_fields)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ConfigError(f"unknown [{section_name}] field(s): {names}")
        missing_fields = set(expected_fields) - set(section)
        if section_name == 'run':
            missing_fields.discard('warmup_episodes')
            missing_fields.discard('codec_pretraining_epochs')
        if section_name == 'simulation':
            missing_fields.discard('routing_mode')
            missing_fields.discard('demand_duration_seconds')
            missing_fields.difference_update({'session_protocol', 'observation_step_seconds', 'backend'})
            missing_fields.difference_update({'traffic_model', 'meso_junction_control',
                'meso_lane_queue', 'meso_multi_queue', 'meso_edge_length_m',
                'meso_interpolate_positions'})
            missing_fields.difference_update({'warmup_horizon_seconds', 'warmup_completion_drain_seconds',
                                              'warmup_completion_patience_seconds'})
        if section_name == 'resources':
            missing_fields.difference_update({'imagination_rollouts_per_pass', 'imagination_passes_per_episode',
                                              'imagination_time_budget_seconds'})
            missing_fields.difference_update(name for name in expected_fields
                                             if name.startswith('world_model_imagination_'))
            missing_fields.difference_update({'world_model_pretrain_space_chunk',
                'world_model_pretrain_time_chunk', 'world_model_pretrain_mlp_chunk',
                'world_model_pretrain_activation_checkpointing', 'world_model_pretrain_precision',
                'world_model_pretrain_compile_blocks'})
        if section_name in {'method', 'data', 'codec'}:
            missing_fields.clear()
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            raise ConfigError(f"missing [{section_name}] field(s): {names}")


def _parse_inputs(raw: Mapping[str, Any], base_dir: Path) -> InputConfig:
    return InputConfig(
        network_name=_nonempty_string(raw, "network_name", "inputs"),
        network_dir=_resolved_path(raw, "network_dir", "inputs", base_dir),

    )


def _parse_run(raw: Mapping[str, Any], base_dir: Path) -> RunConfig:
    return RunConfig(
        output_dir=_resolved_path(raw, "output_dir", "run", base_dir),
        device=_nonempty_string(raw, "device", "run"),
        seed=_integer(raw, "seed", "run"),
        warmup_episodes=_nonnegative_integer(
            {'warmup_episodes': raw.get('warmup_episodes', 0)}, 'warmup_episodes', 'run'),
        world_model_initialization_epochs=_positive_integer(
            raw, "world_model_initialization_epochs", "run"
        ),
        online_training_episodes=_positive_integer(
            raw, "online_training_episodes", "run"
        ),
        evaluation_episodes=_positive_integer(
            raw, "evaluation_episodes", "run"
        ),
        codec_pretraining_epochs=_positive_integer(
            {'codec_pretraining_epochs': raw.get('codec_pretraining_epochs', 10)},
            'codec_pretraining_epochs', 'run'),
    )


def _parse_simulation(raw: Mapping[str, Any]) -> SimulationConfig:
    protocol = raw.get('session_protocol', 'event')
    step = raw.get('observation_step_seconds', 1.0)
    backend = raw.get('backend', 'libsumo')
    routing_mode = raw.get('routing_mode', 'current_to_dest')
    if not isinstance(routing_mode, str) or routing_mode not in {'next_hop', 'current_to_dest'}:
        raise ConfigError('[simulation].routing_mode must be next_hop or current_to_dest')
    traffic_model = raw.get('traffic_model', 'meso')
    if not isinstance(traffic_model, str) or traffic_model not in {'micro', 'meso'}:
        raise ConfigError('[simulation].traffic_model must be micro or meso')
    meso = {}
    for name, default in (('meso_junction_control', True), ('meso_lane_queue', False),
                          ('meso_multi_queue', False), ('meso_interpolate_positions', True)):
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise ConfigError(f'[simulation].{name} must be boolean')
        if traffic_model == 'meso' and value != default:
            raise ConfigError(f'[simulation].{name} must be {default} for the meso deployment contract')
        meso[name] = value
    meso['meso_edge_length_m'] = _positive_number(
        {'meso_edge_length_m': raw.get('meso_edge_length_m', 98.)},
        'meso_edge_length_m', 'simulation')
    if protocol != 'event':
        raise ConfigError('[simulation].session_protocol must be event for the mainline')
    if isinstance(step, bool) or not isinstance(step, (int, float)) or step != 1.0:
        raise ConfigError('[simulation].observation_step_seconds must be 1.0')
    if backend not in {'traci', 'libsumo'}:
        raise ConfigError('[simulation].backend must be traci or libsumo')
    incident_start_seconds = _nonnegative_number(
        raw, "incident_start_seconds", "simulation"
    )
    incident_end_seconds = _positive_number(
        raw, "incident_end_seconds", "simulation"
    )
    if incident_end_seconds <= incident_start_seconds:
        raise ConfigError(
            "[simulation].incident_end_seconds must be greater than "
            "incident_start_seconds"
        )
    sumo_extra_args = raw["sumo_extra_args"]
    if not isinstance(sumo_extra_args, list) or not all(
        isinstance(value, str) and value for value in sumo_extra_args
    ):
        raise ConfigError("[simulation].sumo_extra_args must be an array of strings")
    reserved = {
        "-c", "--configuration-file", "--configuration", "-n", "--net-file",
        "-r", "--route-files", "--routes", "-a", "--additional-files", "--additional",
        "--step-length", "--begin", "-b", "--end", "-e", "--seed", "--random",
        "--default.speedfactor", "--default.speeddev", "--speed-factor",
        "--default.action-step-length", "--default.departspeed", "--default.departpos",
        "--load-state", "--route-steps", "--scale", "--device.rerouting.probability",
        "--device.rerouting.explicit", "--remote-port", "--num-clients",
        "--no-internal-links",
    }
    for argument in sumo_extra_args:
        option = argument.split("=", 1)[0]
        if option in reserved or option == '--mesosim' or option.startswith('--meso'):
            raise ConfigError(
                f"[simulation].sumo_extra_args contains reserved SUMO option {option}"
            )
    vehicle_max_speed = _positive_number(raw, "vehicle_max_speed_mps", "simulation")
    incident_speed = _positive_number(raw, "incident_speed_mps", "simulation")
    if incident_speed > vehicle_max_speed:
        raise ConfigError(
            "[simulation].incident_speed_mps must not exceed vehicle_max_speed_mps"
        )
    return SimulationConfig(
        vehicle_max_speed_mps=vehicle_max_speed,
        world_model_intervals_per_episode=_positive_integer(
            raw, "world_model_intervals_per_episode", "simulation"
        ),
        warmup_horizon_seconds=_positive_integer(
            {'warmup_horizon_seconds': raw.get('warmup_horizon_seconds', 1800)},
            "warmup_horizon_seconds", "simulation"
        ),
        warmup_completion_drain_seconds=_nonnegative_integer(
            {'warmup_completion_drain_seconds': raw.get('warmup_completion_drain_seconds', 3000)},
            "warmup_completion_drain_seconds", "simulation"
        ),
        warmup_completion_patience_seconds=_nonnegative_integer(
            {'warmup_completion_patience_seconds': raw.get('warmup_completion_patience_seconds', 0)},
            "warmup_completion_patience_seconds", "simulation"
        ),
        episode_wall_timeout_seconds=_positive_number(
            raw, "episode_wall_timeout_seconds", "simulation"
        ),
        training_completion_buffer_seconds=_nonnegative_integer(
            raw, "training_completion_buffer_seconds", "simulation"
        ),
        evaluation_horizon_seconds=_positive_integer(
            raw, "evaluation_horizon_seconds", "simulation"
        ),
        incident_edge=_nonempty_string(raw, "incident_edge", "simulation"),
        incident_start_seconds=incident_start_seconds,
        incident_end_seconds=incident_end_seconds,
        incident_speed_mps=incident_speed,
        sumo_extra_args=tuple(sumo_extra_args),
        demand_duration_seconds=_positive_number(
            {"demand_duration_seconds": raw.get("demand_duration_seconds", 1800.)},
            "demand_duration_seconds", "simulation"),
        session_protocol=protocol, observation_step_seconds=float(step), backend=backend,
        traffic_model=traffic_model, **meso,
        routing_mode=routing_mode,
    )


def _parse_resources(raw: Mapping[str, Any]) -> ResourceConfig:
    section = "resources"
    sequence_length = _positive_integer(
        raw,
        "world_model_initialization_sequence_length",
        section,
    )
    defaults = dict(world_model_pretrain_space_chunk=1,
        world_model_pretrain_time_chunk=256, world_model_pretrain_mlp_chunk=4096,
        world_model_pretrain_activation_checkpointing=True)
    imagination_defaults = {name: spec.default for name, spec in ResourceConfig.__dataclass_fields__.items()
                            if name.startswith('world_model_imagination_')}
    values = {**defaults, **imagination_defaults, 'world_model_pretrain_precision': 'fp32',
              'world_model_pretrain_compile_blocks': False,
              'imagination_rollouts_per_pass': 8,
              'imagination_passes_per_episode': 16,
              'imagination_time_budget_seconds': 0., **raw}
    precision = _nonempty_string(values, 'world_model_pretrain_precision', section)
    if precision not in {'fp32', 'bf16'}:
        raise ConfigError('[resources].world_model_pretrain_precision must be fp32 or bf16')
    activation = values['world_model_pretrain_activation_checkpointing']
    if not isinstance(activation, bool):
        raise ConfigError('[resources].world_model_pretrain_activation_checkpointing must be boolean')
    compile_blocks = values['world_model_pretrain_compile_blocks']
    if not isinstance(compile_blocks, bool):
        raise ConfigError('[resources].world_model_pretrain_compile_blocks must be boolean')
    chunks = {name: _positive_integer(values, name, section)
              for name in ('world_model_pretrain_space_chunk', 'world_model_pretrain_time_chunk',
                           'world_model_pretrain_mlp_chunk')}
    imagination = {name: _nonnegative_integer(values, name, section)
                   for name in imagination_defaults if name.endswith('_chunk')}
    imagination_precision = _nonempty_string(values, 'world_model_imagination_precision', section)
    if imagination_precision not in {'fp32', 'fp16', 'bf16', 'tf32'}:
        raise ConfigError('[resources].world_model_imagination_precision must be fp32, fp16, bf16, or tf32')
    imagination_compile = values['world_model_imagination_compile']
    if not isinstance(imagination_compile, bool):
        raise ConfigError('[resources].world_model_imagination_compile must be boolean')
    compile_mode = _nonempty_string(values, 'world_model_imagination_compile_mode', section)
    if compile_mode not in {'default', 'reduce-overhead'}:
        raise ConfigError('[resources].world_model_imagination_compile_mode must be default or reduce-overhead')
    imagination.update(world_model_imagination_precision=imagination_precision,
                       world_model_imagination_compile=imagination_compile,
                       world_model_imagination_compile_mode=compile_mode)
    return ResourceConfig(
        **imagination,
        imagination_rollouts_per_pass=_positive_integer(
            values, 'imagination_rollouts_per_pass', section),
        imagination_passes_per_episode=_nonnegative_integer(
            values, 'imagination_passes_per_episode', section),
        imagination_time_budget_seconds=_nonnegative_number(
            values, 'imagination_time_budget_seconds', section),
        world_model_initialization_batch_size=_positive_integer(
            raw, "world_model_initialization_batch_size", section
        ),
        world_model_initialization_sequence_length=sequence_length,
        world_model_gradient_accumulation_steps=_positive_integer(
            raw, "world_model_gradient_accumulation_steps", section
        ),
        online_world_model_batch_size=_positive_integer(
            raw, "online_world_model_batch_size", section
        ),
        **chunks,
        world_model_pretrain_activation_checkpointing=activation,
        world_model_pretrain_precision=precision,
        world_model_pretrain_compile_blocks=compile_blocks,
    )


def _semantic_fingerprint(
    *,
    inputs: InputConfig,
    run: RunConfig,
    simulation: SimulationConfig,
    resources: ResourceConfig,
    method: MethodConfig,
    data: DataConfig,
    codec: CodecConfig,
    reward_contract: str = 'causal_cumulative_mean_global_v1',
) -> str:
    run_values = asdict(run)
    run_values.pop("output_dir")
    run_values.pop("device")
    run_values.pop('warmup_episodes')
    simulation_values = asdict(simulation)
    if simulation.routing_mode == 'next_hop':
        # Existing runs and their pretrained checkpoint identities stay exact.
        simulation_values.pop('routing_mode')
    if simulation.traffic_model == 'micro':
        # Keep historical micro checkpoint hashes byte-for-byte compatible.
        for name in tuple(simulation_values):
            if name == 'traffic_model' or name.startswith('meso_'):
                simulation_values.pop(name)
    for name in ('warmup_horizon_seconds', 'warmup_completion_drain_seconds',
                 'warmup_completion_patience_seconds'):
        simulation_values.pop(name)
    method_values = asdict(method)
    if method.real_policy_update == 'none':
        method_values.pop('real_policy_update', None)
    if method.real_policy_critic_epochs == 8:
        method_values.pop('real_policy_critic_epochs', None)
    if method.real_policy_actor_epochs == 4:
        method_values.pop('real_policy_actor_epochs', None)
    if method.real_policy_value_mode == 'shared':
        method_values.pop('real_policy_value_mode', None)
    if method.model_policy_search_schedule == 'round_robin':
        method_values.pop('model_policy_search_schedule', None)
    if method.actor_objective == 'centered_event_mean_v1':
        method_values.pop('actor_objective', None)
    if method.imagination_actor_epochs == 1:
        method_values.pop('imagination_actor_epochs', None)
    if method.imagination_value_epochs == 1:
        method_values.pop('imagination_value_epochs', None)
    if method.imagination_actor_advantage == 'event_mc_q_v1':
        method_values.pop('imagination_actor_advantage', None)
    if method.imagination_horizon_windows == 8:
        method_values.pop('imagination_horizon_windows', None)
    if method.imagination_seed_scope == 'full_episode_v1':
        method_values.pop('imagination_seed_scope', None)
    if method.imagination_real_policy_guard == 'none':
        method_values.pop('imagination_real_policy_guard', None)
    if method.training_temperature_profile == 'none':
        method_values.pop('training_temperature_profile', None)
    if method.prior_usage == 'source_reference':
        # Omitted usage in archived configs preserves the original checkpoint identity.
        method_values.pop('prior_usage', None)
    if method.prior_filter == 'hard' and method.prior_action_conditioned and not method.prior_filter_input:
        # Archived configs without these fields keep their original contract.
        for name in ('prior_filter', 'prior_filter_strength', 'prior_action_conditioned', 'prior_filter_input'):
            method_values.pop(name, None)
    if method.actor_entropy_mode != 'annealed':
        # Inactive schedule settings retain existing checkpoint identities.
        for name in ('actor_entropy_initial_coefficient', 'actor_entropy_final_coefficient',
                     'actor_entropy_hold_fraction', 'actor_entropy_end_fraction'):
            method_values.pop(name, None)
    if method.critic_value_mode == 'legacy':
        for name in ('critic_value_mode', 'critic_target_tau', 'critic_terminal_aux_ratio'):
            method_values.pop(name, None)
    elif method.critic_value_mode == 'ema_legacy_v1':
        method_values.pop('critic_terminal_aux_ratio', None)
    if method.fm_training_mode == 'legacy_independent':
        method_values.pop('fm_training_mode', None)
    if method.actor_entropy_mode == 'legacy':
        method_values.pop('actor_entropy_mode', None)
    if method.sp_prior_cost_mode == 'free_flow':
        method_values.pop('sp_prior_cost_mode', None)
        method_values.pop('sp_prior_dynamic_speed_floor_mps', None)
    if method.sp_prior_evaluation_weight == 0.:
        method_values.pop('sp_prior_evaluation_weight', None)
    if method.sp_prior_weight == 0. and method.sp_prior_evaluation_weight == 0.:
        # Disabled behavior settings must retain historical checkpoint identities.
        for name in tuple(method_values):
            if name.startswith("sp_prior_"):
                method_values.pop(name)
    resource_values = asdict(resources)
    for name, spec in ResourceConfig.__dataclass_fields__.items():
        # Keep checkpoint identities for explicitly configured legacy FP32 runs.
        # FP16 is the execution default, but must be represented in the fingerprint.
        historical_default = 'fp32' if name == 'world_model_imagination_precision' else spec.default
        if (name.startswith('world_model_imagination_') or name in {
                'imagination_rollouts_per_pass', 'imagination_passes_per_episode', 'imagination_time_budget_seconds'
                }) and getattr(resources, name) == historical_default:
            resource_values.pop(name, None)
    if not resources.world_model_pretrain_compile_blocks:
        # Disabled compilation retains historical checkpoint fingerprints.
        resource_values.pop('world_model_pretrain_compile_blocks', None)
    codec_values = asdict(codec)
    for name in ('decoder_pretrain_enabled', 'decoder_pretrain_learning_rate',
                 'decoder_online_enabled', 'decoder_online_learning_rate', 'decoder_allow_unready',
                 'occupied_speed_weight', 'slow_speed_weight', 'slow_speed_mps',
                 'slow_min_vehicles', 'occupied_speed_mae_mps_max', 'slow_speed_mae_mps_max',
                 'slow_validation_min_roads'):
        if getattr(codec, name) == CodecConfig.__dataclass_fields__[name].default:
            codec_values.pop(name, None)
    payload = {
        "training_lifecycle": "validated_codec_independent_full_prior_fm_precision_exposure_contract_v4",
        "codec_objective": ("car_unit_occupied_empty_count_total_relative_group_speed_v2"
            if codec.occupied_speed_weight or codec.slow_speed_weight
            else "car_unit_occupied_empty_count_total_relative_speed_masked_v1"),
        "codec_validation": ("heldout_raw_hybrid_occupied_slow_current_decoder_revision_v3"
            if codec.decoder_pretrain_enabled or codec.decoder_online_enabled
            else "heldout_raw_and_current_birth_subtracted_endpoints_strict_v2"),
        "transition_volume_target": "observed_endpoint_subtract_only_current_interval_newborns_v2",
        "transition_input": "full_observed_prior_at_each_window_teacher_forcing_v1",
        "codec_during_dynamics_training": ("frozen_encoder_supervised_decoder_continuous_v1"
            if codec.decoder_online_enabled else "frozen_encoder_supervised_decoder_pretrain_v1"
            if codec.decoder_pretrain_enabled else "frozen_encoder_decoder_v1"),
        "world_model_initialization": "finite_shuffled_sequence_epochs_v1",
        "world_model_pretraining_mode": "independent_single_step_v1",
        "decoder_speed_target": "forward_endpoint_mean_v1",
        "world_model_conditions": "raw_event_turn_destination_phase_3x10s_embedding16_mlp64_32_readout64_v1",
        "world_model_transition": "independent_teacher_forced_full_prior_fm_windows_v2",
        "imagination_horizon": "eight_windows_full_episode_sampling_v1",
        "nonarrival_penalty": NONARRIVAL_PENALTY,
        "reward_mode": "global_with_individual_timeout",
        "global_reward": "event_reward_to_go_exact_seed_cohort_minutes_endpoint_latent_risk_v2",
        "nonarrival_penalty_scope": "own_vehicle_terminal_mc_return_v1",
        "policy_execution": "train_stochastic_evaluation_argmax_v1",
        "advantage_normalization": "global_scale_only_std_floor_1_then_individual_timeout_clip_2_5",
        "actor_global_advantage": "event_return_minus_detached_preupdate_masked_policy_event_q_expectation_v1",
        "mainline_demand": "independent_od_poisson_counts_uniform_times_v1",
        "network_name": inputs.network_name,
        "run": run_values,
        "simulation": simulation_values,
        "resources": resource_values,
        "method": method_values,
        "codec": codec_values,
        "offline_data": {
            "enabled": bool(data.archive_roots or data.pretrain_cache_manifest),
            "streaming": "frozen_content_manifest_v1",
            "pretrain_compact_cache": data.pretrain_cache_manifest is not None,
            "validation_compact_cache": data.validation_cache_manifest is not None,
            "explicit_validation_archives": bool(data.validation_archive_roots),
            "validation_split": "disjoint_demand_groups_or_explicit_sources_v1",
        },
    }
    if reward_contract == 'causal_cumulative_mean_global_v1':
        payload.update(
            training_lifecycle='validated_codec_independent_full_prior_fm_causal_mean_contract_v5',
            reward_contract=reward_contract,
            nonarrival_penalty=0., reward_mode='global_only',
            global_reward='minus_delta_cumulative_vehicle_seconds_per_started_vehicle_minutes_v1',
            nonarrival_penalty_scope='none',
            advantage_normalization='global_scale_only_std_floor_1_clip_2_5',
            trip_start='actual_departure_strictly_before_physical_cutoff_v1')
    elif reward_contract != 'event_q_counterfactual_global_individual_timeout_v2':
        raise ValueError('unknown reward contract for configuration fingerprint')
    if method.fm_training_mode != 'legacy_independent':
        payload['world_model_transition'] = method.fm_training_mode + '_v1'
    if method.fm_training_mode == 'recursive_3step':
        payload['transition_volume_target'] = 'fixed_sequence_start_cohort_v1'
        payload['transition_input'] = 'generated_previous_latent_with_recorded_cohort_actions_v1'
    if simulation.traffic_model == 'meso':
        from ..traffic.simulation_settings import physics_identity
        payload['traffic_physics'] = physics_identity(simulation)
    encoded = json.dumps(
        _json_compatible(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def legacy_semantic_fingerprint(config: MainlineConfig) -> str:
    """Verify immutable old sources without treating their reward as current."""
    return _semantic_fingerprint(**{name: getattr(config, name) for name in (
        'inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')},
        reward_contract='event_q_counterfactual_global_individual_timeout_v2')


def validate_fm_training_profile(method: MethodConfig, resources: ResourceConfig) -> None:
    if method.fm_training_mode == 'recursive_3step' and (
            method.supervision_horizon < 3):
        raise ConfigError('online recursive_3step requires [method].supervision_horizon >= 3')


def _parse_method(raw: Mapping[str, Any]) -> MethodConfig:
    defaults = asdict(MethodConfig())
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ConfigError('[method] contains unsupported setting(s): ' + ', '.join(sorted(unknown)))
    for name, value in raw.items():
        expected = defaults[name]
        if isinstance(expected, bool):
            valid = type(value) is bool and value == expected
        elif isinstance(expected, int):
            valid = type(value) is int and value == expected
        elif isinstance(expected, float):
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and value == expected
        else:
            valid = type(value) is type(expected) and value == expected
        if not valid:
            raise ConfigError(f'[method].{name} must use the native mainline setting {expected!r}')
    return MethodConfig(**{**defaults, **raw})


def _parse_codec(raw: Mapping[str, Any]) -> CodecConfig:
    values = {**asdict(CodecConfig()), **raw}
    parsed = {name: _positive_integer(values, name, 'codec')
              for name in ('batch_size', 'max_updates', 'validation_every')}
    parsed['learning_rate'] = _positive_number(values, 'learning_rate', 'codec')
    for name in ('occupied_vehicle_mae_max', 'vehicle_mae_max', 'empty_vehicle_mae_max',
                 'total_relative_mae_max', 'speed_mae_mps_max'):
        parsed[name] = _nonnegative_number(values, name, 'codec')
    if not isinstance(values['speed_positive'], bool):
        raise ConfigError('[codec].speed_positive must be boolean')
    parsed['speed_positive'] = values['speed_positive']
    if not isinstance(values['decoder_pretrain_enabled'], bool):
        raise ConfigError('[codec].decoder_pretrain_enabled must be boolean')
    parsed['decoder_pretrain_enabled'] = values['decoder_pretrain_enabled']
    for name in ('decoder_online_enabled', 'decoder_allow_unready'):
        if not isinstance(values[name], bool):
            raise ConfigError(f'[codec].{name} must be boolean')
        parsed[name] = values[name]
    parsed['decoder_online_learning_rate'] = _positive_number(values, 'decoder_online_learning_rate', 'codec')
    parsed['decoder_pretrain_learning_rate'] = _positive_number(values, 'decoder_pretrain_learning_rate', 'codec')
    parsed['slow_validation_min_roads'] = _nonnegative_integer(values, 'slow_validation_min_roads', 'codec')
    for name in ('occupied_speed_weight', 'slow_speed_weight', 'slow_speed_mps',
                 'slow_min_vehicles', 'occupied_speed_mae_mps_max', 'slow_speed_mae_mps_max'):
        parsed[name] = _nonnegative_number(values, name, 'codec')
    return CodecConfig(**parsed)


def _parse_data(raw: Mapping[str, Any], base_dir: Path) -> DataConfig:
    roots = {}
    for name in ('archive_roots', 'validation_archive_roots'):
        values = raw.get(name, [])
        if not isinstance(values, list) or not all(isinstance(root, str) and root.strip() for root in values):
            raise ConfigError(f'[data].{name} must be an array of non-empty paths')
        roots[name] = tuple(_resolved_path({'root': root}, 'root', 'data', base_dir) for root in values)
    manifests = {}
    for name in ('pretrain_cache_manifest', 'validation_cache_manifest'):
        manifests[name] = None if raw.get(name) is None else _resolved_path(raw, name, 'data', base_dir)
    cache = _positive_integer({'cache_episodes': raw.get('cache_episodes', 2)}, 'cache_episodes', 'data')
    if cache > 2:
        raise ConfigError('[data].cache_episodes must be at most 2')
    return DataConfig(**roots, **manifests, cache_episodes=cache)


def _resolved_path(
    raw: Mapping[str, Any],
    key: str,
    section: str,
    base_dir: Path,
) -> Path:
    value = _nonempty_string(raw, key, section)
    try:
        return resolve_resource(value, root=base_dir)
    except ValueError as error:
        raise ConfigError(f'[{section}].{key}: {error}') from None


def _nonempty_string(raw: Mapping[str, Any], key: str, section: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"[{section}].{key} must be a non-empty string")
    return value.strip()


def _integer(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[{section}].{key} must be an integer")
    return value


def _positive_integer(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = _integer(raw, key, section)
    if value <= 0:
        raise ConfigError(f"[{section}].{key} must be positive")
    return value


def _nonnegative_integer(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = _integer(raw, key, section)
    if value < 0:
        raise ConfigError(f"[{section}].{key} must be non-negative")
    return value


def _number(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"[{section}].{key} must be a number")
    if not math.isfinite(float(value)):
        raise ConfigError(f"[{section}].{key} must be finite")
    return float(value)


def _positive_number(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = _number(raw, key, section)
    if value <= 0.0:
        raise ConfigError(f"[{section}].{key} must be positive")
    return value


def _nonnegative_number(raw: Mapping[str, Any], key: str, section: str) -> float:
    value = _number(raw, key, section)
    if value < 0.0:
        raise ConfigError(f"[{section}].{key} must be non-negative")
    return value


def _json_compatible(value: Any) -> Any:
    return portable_data(value, root=PROJECT_ROOT)
