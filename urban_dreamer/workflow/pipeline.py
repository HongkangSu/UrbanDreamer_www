from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol

from .checkpoint import (
    CheckpointPhase,
    OutputDirectoryDecision,
    decide_output_directory,
    save_checkpoint,
)
from .config import ConfigError, MainlineConfig
from ..traffic.routing import PreparedInputs, validate_prepared_inputs
from ..traffic.topology import TrafficTopology, load_traffic_topology


class MainlineRunError(RuntimeError):
    """Raised when the single supported run cannot safely start or resume."""


# Units between full boundary checkpoints per phase; None saves only at phase end.
_CHECKPOINT_SAVE_INTERVALS: dict[CheckpointPhase, int | None] = {
    CheckpointPhase.CODEC_PRETRAIN: 1,
    CheckpointPhase.PRETRAIN: 1,
    CheckpointPhase.ONLINE_TRAINING: 100,
    CheckpointPhase.FINAL_EVALUATION: 1,
}


class MainlineTrainerProtocol(Protocol):
    def validate_pretraining_data(self) -> None: ...

    def initialize_codec_epoch(self, epoch_index: int) -> dict[str, Any]: ...

    def codec_ready(self) -> bool: ...

    def validate_codec_ready(self) -> None: ...

    def initialize_world_model_epoch(self, epoch_index: int) -> dict[str, Any]: ...

    def train_online_episode(self, episode_index: int) -> dict[str, Any]: ...

    def evaluate_episode(self, episode_index: int) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def summary(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class MainlinePipeline:
    """Strict phase coordinator for the only supported training lifecycle."""

    def __init__(
        self,
        *,
        config: MainlineConfig,
        input_hash: str,
        trainer_factory: Callable[[], MainlineTrainerProtocol],
        pretrain_only: bool = False,
        initial_state: dict[str, Any] | None = None,
        evaluate_only: bool = False,
    ) -> None:
        self.config = config
        self.input_hash = str(input_hash)
        self._trainer_factory = trainer_factory
        self.pretrain_only = bool(pretrain_only)
        self.initial_state = initial_state
        self.evaluate_only = bool(evaluate_only)

    def run(self, status=None) -> dict[str, Any]:
        status = status or decide_output_directory(
            output_dir=self.config.run.output_dir,
            expected_config_fingerprint=self.config.semantic_fingerprint,
            expected_input_hash=self.input_hash,
        )
        if status.decision is OutputDirectoryDecision.NONEMPTY_INVALID:
            raise MainlineRunError(status.reason or "output directory cannot be resumed")
        if status.decision is OutputDirectoryDecision.COMPLETED:
            assert status.checkpoint is not None
            return _completed_summary(
                self.config.run.output_dir,
                status.checkpoint.training_state,
            )

        trainer = self._trainer_factory()
        try:
            if status.decision is OutputDirectoryDecision.VALID_RESUME:
                assert status.checkpoint is not None
                phase = status.checkpoint.phase
                next_unit = status.checkpoint.next_unit
                trainer.load_state_dict(status.checkpoint.training_state)
                self._trim_metrics(phase, next_unit)
            else:
                if self.initial_state is None:
                    trainer.validate_pretraining_data()
                    phase = CheckpointPhase.CODEC_PRETRAIN
                else:
                    trainer.load_state_dict(self.initial_state)
                    phase = (CheckpointPhase.FINAL_EVALUATION if self.evaluate_only
                             else CheckpointPhase.ONLINE_TRAINING)
                next_unit = 0
                self._save_boundary(trainer, phase=phase, next_unit=next_unit)
                _atomic_write_json(
                    self.config.run.output_dir / "resolved_config.json",
                    self.config.resolved_snapshot(root=self.config.run.output_dir),
                )

            phase, next_unit = self._run_remaining_phases(
                trainer,
                phase=phase,
                next_unit=next_unit,
            )
            if self.pretrain_only and phase is CheckpointPhase.ONLINE_TRAINING and next_unit == 0:
                summary = {"status": "pretrained", **trainer.summary()}
                _atomic_write_json(self.config.run.output_dir / 'pretrain_summary.json', summary)
                return summary
            if phase is not CheckpointPhase.COMPLETED or next_unit != 0:
                raise AssertionError("phase coordinator did not reach completed")

            summary = {"status": "completed", **trainer.summary()}
            self._save_boundary(
                trainer,
                phase=CheckpointPhase.COMPLETED,
                next_unit=0,
            )
            _atomic_write_json(
                self.config.run.output_dir / "summary.json",
                summary,
            )
            return summary
        finally:
            trainer.close()

    def _run_remaining_phases(
        self,
        trainer: MainlineTrainerProtocol,
        *,
        phase: CheckpointPhase,
        next_unit: int,
    ) -> tuple[CheckpointPhase, int]:
        stages: tuple[
            tuple[CheckpointPhase, int, Callable[[int], dict[str, Any]]],
            ...,
        ] = (
            (
                CheckpointPhase.CODEC_PRETRAIN,
                self.config.run.codec_pretraining_epochs,
                trainer.initialize_codec_epoch,
            ),
            (
                CheckpointPhase.PRETRAIN,
                self.config.run.world_model_initialization_epochs,
                trainer.initialize_world_model_epoch,
            ),
            (
                CheckpointPhase.ONLINE_TRAINING,
                self.config.run.online_training_episodes,
                trainer.train_online_episode,
            ),
            (
                CheckpointPhase.FINAL_EVALUATION,
                self.config.run.evaluation_episodes,
                trainer.evaluate_episode,
            ),
        )
        stage_index = _phase_index(phase)
        if stage_index == len(stages):
            return CheckpointPhase.COMPLETED, 0

        for index in range(stage_index, len(stages)):
            current_phase, target_units, execute_unit = stages[index]
            cursor = next_unit if index == stage_index else 0
            if current_phase is CheckpointPhase.PRETRAIN and hasattr(trainer, 'validate_codec_pretraining_ready'):
                trainer.validate_codec_pretraining_ready()
            elif current_phase is not CheckpointPhase.CODEC_PRETRAIN:
                trainer.validate_codec_ready()
            if self.pretrain_only and current_phase in {
                    CheckpointPhase.ONLINE_TRAINING, CheckpointPhase.FINAL_EVALUATION}:
                if current_phase is not CheckpointPhase.ONLINE_TRAINING or cursor != 0:
                    raise MainlineRunError('--pretrain-only cannot resume a run that has started online training')
                return CheckpointPhase.ONLINE_TRAINING, 0
            if cursor > target_units:
                raise MainlineRunError(
                    f"checkpoint cursor {cursor} exceeds {current_phase.value} "
                    f"target {target_units}"
                )
            save_interval = _CHECKPOINT_SAVE_INTERVALS[current_phase]
            for unit_index in range(cursor, target_units):
                if current_phase is CheckpointPhase.CODEC_PRETRAIN and trainer.codec_ready():
                    break
                result = execute_unit(unit_index)
                self._record_unit(current_phase, unit_index, result)
                # A full boundary checkpoint serializes the entire replay and
                # is saved after each initialization epoch. Online training
                # checkpoints sparsely; the final unit of every
                # phase is always saved (followed by the next-phase boundary).
                if (save_interval is not None and (unit_index + 1) % save_interval == 0) \
                        or unit_index + 1 == target_units:
                    self._save_boundary(
                        trainer,
                        phase=current_phase,
                        next_unit=unit_index + 1,
                    )
                if current_phase is CheckpointPhase.CODEC_PRETRAIN and trainer.codec_ready():
                    break
            if current_phase is CheckpointPhase.CODEC_PRETRAIN:
                # Exhausting a training budget never substitutes for held-out
                # reconstruction validation. Keep the codec-phase checkpoint
                # if validation fails, and never begin FM with an unready codec.
                trainer.validate_codec_ready()
            next_phase = (
                stages[index + 1][0]
                if index + 1 < len(stages)
                else CheckpointPhase.COMPLETED
            )
            self._save_boundary(
                trainer,
                phase=next_phase,
                next_unit=0,
            )
            next_unit = 0
        return CheckpointPhase.COMPLETED, 0

    def _trim_metrics(self, phase, next_unit):
        path = self.config.run.output_dir / 'metrics.jsonl'
        if not path.is_file():
            return
        boundary = _phase_index(phase)
        retained = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index = _phase_index(CheckpointPhase(row['phase']))
            if index < boundary or (index == boundary and row['unit_index'] < next_unit):
                retained.append(line)
        path.write_text(''.join(line + '\n' for line in retained))

    def _record_unit(self, phase, unit_index, result):
        from .portable import portable_data
        simulation = self.config.simulation
        formal = (self.config.run.online_training_episodes == 1000
            and self.config.run.evaluation_episodes == 3
            and simulation.world_model_intervals_per_episode == 127
            and simulation.training_completion_buffer_seconds == 0
            and simulation.demand_duration_seconds == 1800.)
        row = dict(result, phase=phase.value, unit_index=unit_index,
                   method='urban_dreamer', smoke=not formal)
        if phase is CheckpointPhase.ONLINE_TRAINING:
            row.update(episode_index=unit_index + 1,
                online_att_seconds=result['online_all_demand_censored_att_seconds'],
                online_decision_windows=127 * (unit_index + 1))
        elif phase is CheckpointPhase.FINAL_EVALUATION:
            row.update(evaluation_index=unit_index,
                ATT=result['evaluation_all_demand_censored_att_seconds'],
                P90=result['evaluation_all_demand_censored_p90_seconds'],
                P95=result['evaluation_all_demand_censored_p95_seconds'],
                completion_rate=result['evaluation_completion_rate'],
                demand_hash=result['evaluation_demand_hash'])
        with (self.config.run.output_dir / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(portable_data(row, root=self.config.run.output_dir),
                                   allow_nan=False, sort_keys=True) + '\n')

    def _save_boundary(
        self,
        trainer: MainlineTrainerProtocol,
        *,
        phase: CheckpointPhase,
        next_unit: int,
    ) -> None:
        save_checkpoint(
            output_dir=self.config.run.output_dir,
            config_fingerprint=self.config.semantic_fingerprint,
            input_hash=self.input_hash,
            phase=phase,
            next_unit=next_unit,
            training_state=trainer.state_dict(),
        )


def run_mainline(config: MainlineConfig, *, pretrain_only: bool = False,
                 pretrained: str | Path | None = None, evaluate_only: bool = False) -> dict[str, Any]:
    """Validate prepared inputs and execute or strictly resume the sole mainline."""
    if not isinstance(config, MainlineConfig):
        raise TypeError("run_mainline expects a MainlineConfig from load_config()")
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    input_hash = combined_input_hash(prepared)
    status = decide_output_directory(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=input_hash)
    initial_state = None
    source_state = status.checkpoint.training_state if status.checkpoint is not None else None
    if pretrained is not None:
        from .pretrained import load_pretrained
        native = load_pretrained(pretrained, config=config, prepared=prepared)
        if source_state is not None:
            if source_state['metrics'].get('pretrained_checkpoint_sha256') != native.checkpoint_sha256:
                raise MainlineRunError('existing run was initialized from a different pretrained checkpoint')
        else:
            initial_state = native.training_state
            initial_state['metrics'].update(initialization_kind='pretrained',
                pretrained_checkpoint_sha256=native.checkpoint_sha256, pretrained_city=native.city)
            source_state = initial_state
    elif evaluate_only:
        raise ConfigError('--evaluate-only requires --pretrained')
    if (source_state is None and not config.data.archive_roots
            and config.data.pretrain_cache_manifest is None):
        raise ConfigError('fresh training requires compatible offline archives or --pretrained')
    reference = (source_state if source_state is not None
                 and source_state['metrics'].get('initialization_kind') == 'pretrained' else None)
    return MainlinePipeline(
        config=config,
        input_hash=input_hash,
        trainer_factory=lambda: _build_trainer(config, prepared, topology, reference=reference),
        pretrain_only=pretrain_only,
        initial_state=initial_state,
        evaluate_only=evaluate_only,
    ).run(status)


def combined_input_hash(prepared: PreparedInputs) -> str:
    encoded = json.dumps(
        sorted(prepared.input_hashes.items()),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_trainer(
    config: MainlineConfig,
    prepared: PreparedInputs,
    topology: TrafficTopology,
    *, reference: dict[str, Any] | None = None,
) -> MainlineTrainerProtocol:
    from ..learning.training import MainlineTrainer

    datasets = {}
    if reference is not None:
        from .pretrained import StoredOfflineReference
        datasets['_offline_data'] = StoredOfflineReference(reference['replay']['offline'])
        saved_validation = reference['replay']['codec_validation']
        datasets['_codec_validation_data'] = StoredOfflineReference(saved_validation) if saved_validation else None
    return MainlineTrainer(
        config=config,
        prepared_inputs=prepared,
        topology=topology,
        **datasets,
    )


def _phase_index(phase: CheckpointPhase) -> int:
    ordered = (
        CheckpointPhase.CODEC_PRETRAIN,
        CheckpointPhase.PRETRAIN,
        CheckpointPhase.ONLINE_TRAINING,
        CheckpointPhase.FINAL_EVALUATION,
        CheckpointPhase.COMPLETED,
    )
    try:
        return ordered.index(phase)
    except ValueError as exc:  # pragma: no cover - enum makes this defensive
        raise MainlineRunError(f"unsupported phase {phase!r}") from exc


def _completed_summary(
    output_dir: Path,
    training_state: dict[str, Any],
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MainlineRunError(f"cannot read completed summary: {exc}") from exc
        if not isinstance(value, dict):
            raise MainlineRunError("completed summary must be a JSON object")
        return value
    metrics = training_state.get("metrics", {})
    if not isinstance(metrics, dict):
        raise MainlineRunError("completed checkpoint metrics must be an object")
    return {"status": "completed", **metrics}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp.",
        suffix=path.suffix,
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
