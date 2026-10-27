from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .portable import portable_data


CHECKPOINT_SCHEMA_VERSION = 10
LATEST_CHECKPOINT_FILENAME = "latest.pt"
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "config_fingerprint",
        "input_hash",
        "phase",
        "next_unit",
        "training_state",
    }
)
_TRAINING_STATE_KEYS = frozenset(
    {"reward_contract", "models", "optimizers", "replay", "rng", "metrics", "next_environment_episode"}
)


class CheckpointError(ValueError):
    """Raised when a checkpoint payload or resume decision is invalid."""


class CheckpointPhase(str, Enum):
    CODEC_PRETRAIN = 'codec_pretrain'
    PRETRAIN = "pretrain"
    ONLINE_TRAINING = "online_training"
    FINAL_EVALUATION = "final_evaluation"
    COMPLETED = "completed"


class OutputDirectoryDecision(str, Enum):
    FRESH = "fresh"
    VALID_RESUME = "valid-resume"
    NONEMPTY_INVALID = "nonempty-invalid"
    COMPLETED = "completed"


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    schema_version: int
    config_fingerprint: str
    input_hash: str
    phase: CheckpointPhase
    next_unit: int
    training_state: dict[str, Any]


@dataclass(frozen=True)
class OutputDirectoryStatus:
    decision: OutputDirectoryDecision
    latest_path: Path
    checkpoint: LoadedCheckpoint | None = None
    reason: str | None = None


def latest_checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir).expanduser().resolve() / "checkpoints" / LATEST_CHECKPOINT_FILENAME


def save_checkpoint(
    *,
    output_dir: str | Path,
    config_fingerprint: str,
    input_hash: str,
    phase: CheckpointPhase | str,
    next_unit: int,
    training_state: dict[str, Any],
) -> Path:
    latest_path = latest_checkpoint_path(output_dir)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(
        config_fingerprint=config_fingerprint,
        input_hash=input_hash,
        phase=phase,
        next_unit=next_unit,
        training_state=portable_data(training_state, root=Path(output_dir)),
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{latest_path.stem}.tmp.",
        suffix=latest_path.suffix,
        dir=str(latest_path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, latest_path)
        _fsync_directory(latest_path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return latest_path


def load_checkpoint(
    *,
    output_dir: str | Path,
    expected_config_fingerprint: str,
    expected_input_hash: str,
) -> LoadedCheckpoint:
    latest_path = latest_checkpoint_path(output_dir)
    if not latest_path.is_file():
        raise CheckpointError(f"checkpoint does not exist: {latest_path.name}")
    try:
        raw = torch.load(latest_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exercised by tests
        raise CheckpointError(f"cannot load checkpoint resource {latest_path.name}") from None
    return _validate_loaded_payload(
        payload=raw,
        path=latest_path,
        expected_config_fingerprint=expected_config_fingerprint,
        expected_input_hash=expected_input_hash,
    )


def decide_output_directory(
    *,
    output_dir: str | Path,
    expected_config_fingerprint: str,
    expected_input_hash: str,
) -> OutputDirectoryStatus:
    root = Path(output_dir).expanduser().resolve()
    latest_path = latest_checkpoint_path(root)
    if not root.exists():
        return OutputDirectoryStatus(
            decision=OutputDirectoryDecision.FRESH,
            latest_path=latest_path,
        )

    if latest_path.is_file():
        try:
            checkpoint = load_checkpoint(
                output_dir=root,
                expected_config_fingerprint=expected_config_fingerprint,
                expected_input_hash=expected_input_hash,
            )
        except CheckpointError as exc:
            return OutputDirectoryStatus(
                decision=OutputDirectoryDecision.NONEMPTY_INVALID,
                latest_path=latest_path,
                reason=str(exc),
            )
        decision = (
            OutputDirectoryDecision.COMPLETED
            if checkpoint.phase is CheckpointPhase.COMPLETED
            else OutputDirectoryDecision.VALID_RESUME
        )
        return OutputDirectoryStatus(
            decision=decision,
            latest_path=latest_path,
            checkpoint=checkpoint,
        )

    has_any_file = any(path.is_file() for path in root.rglob("*"))
    if not has_any_file:
        return OutputDirectoryStatus(
            decision=OutputDirectoryDecision.FRESH,
            latest_path=latest_path,
        )
    return OutputDirectoryStatus(
        decision=OutputDirectoryDecision.NONEMPTY_INVALID,
        latest_path=latest_path,
        reason=f"output directory is non-empty without a valid checkpoint: {root.name}",
    )


def _build_payload(
    *,
    config_fingerprint: str,
    input_hash: str,
    phase: CheckpointPhase | str,
    next_unit: int,
    training_state: dict[str, Any],
) -> dict[str, Any]:
    normalized_phase = _validate_phase(phase)
    normalized_next_unit = _validate_next_unit(next_unit)
    normalized_config_fingerprint = _validate_nonempty_string(
        config_fingerprint, "config_fingerprint"
    )
    normalized_input_hash = _validate_nonempty_string(input_hash, "input_hash")
    normalized_training_state = _normalize_training_state(training_state)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_fingerprint": normalized_config_fingerprint,
        "input_hash": normalized_input_hash,
        "phase": normalized_phase.value,
        "next_unit": normalized_next_unit,
        "training_state": normalized_training_state,
    }


def _validate_loaded_payload(
    *,
    payload: Any,
    path: Path,
    expected_config_fingerprint: str,
    expected_input_hash: str,
) -> LoadedCheckpoint:
    if not isinstance(payload, dict):
        raise CheckpointError(f"checkpoint payload is not an object: {path.name}")
    _validate_exact_keys(payload, _ROOT_KEYS, "checkpoint")

    schema_version = payload["schema_version"]
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema version {schema_version!r} in {path.name}"
        )
    config_fingerprint = _validate_nonempty_string(
        payload["config_fingerprint"], "config_fingerprint"
    )
    input_hash = _validate_nonempty_string(payload["input_hash"], "input_hash")
    if config_fingerprint != _validate_nonempty_string(
        expected_config_fingerprint, "expected_config_fingerprint"
    ):
        raise CheckpointError("checkpoint config fingerprint does not match")
    if input_hash != _validate_nonempty_string(expected_input_hash, "expected_input_hash"):
        raise CheckpointError("checkpoint input hash does not match")

    phase = _validate_phase(payload["phase"])
    next_unit = _validate_next_unit(payload["next_unit"])
    training_state_raw = payload["training_state"]
    if not isinstance(training_state_raw, dict):
        raise CheckpointError("checkpoint training_state is not an object")
    training_state = _normalize_training_state(training_state_raw)
    return LoadedCheckpoint(
        path=path,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        config_fingerprint=config_fingerprint,
        input_hash=input_hash,
        phase=phase,
        next_unit=next_unit,
        training_state=training_state,
    )


def _normalize_training_state(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointError("training_state must be an object")
    _validate_exact_keys(value, _TRAINING_STATE_KEYS, "training_state")
    # Structural readers must be able to inspect immutable pretraining sources.
    # MainlineTrainer requires the current contract before loading for learning;
    # legacy online state needs an explicit, recorded reward migration.
    if value['reward_contract'] not in {
            'event_q_counterfactual_global_individual_timeout_v2',
            'causal_cumulative_mean_global_v1'}:
        raise CheckpointError('unsupported global reward contract')
    episode = value["next_environment_episode"]
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise CheckpointError("next_environment_episode must be a non-negative integer")
    return {key: _normalize_plain_value(value[key], path=f"training_state.{key}") for key in value}


def _normalize_plain_value(value: Any, *, path: str) -> Any:
    # Replay snapshots contain millions of plain scalars. Avoid the tensor and
    # NumPy checks for those leaves; subclasses still use the checks below.
    if value is None or type(value) in (str, float, int, bool):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise CheckpointError(f"{path} contains an object-dtype NumPy array")
        return torch.as_tensor(value).detach().cpu()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    if isinstance(value, tuple):
        return [_normalize_plain_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, list):
        return [_normalize_plain_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, dict):
        normalized: dict[str | int, Any] = {}
        for key, item in value.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise CheckpointError(f"{path} has a key that is not a string or integer")
            normalized[key] = _normalize_plain_value(item, path=f"{path}.{key}")
        return normalized
    raise CheckpointError(
        f"{path} must contain only arrays/CPU tensors, dict/list/tuple containers, and scalar values"
    )


def _validate_exact_keys(mapping: dict[str, Any], expected: frozenset[str], context: str) -> None:
    missing = expected - set(mapping)
    if missing:
        names = ", ".join(sorted(missing))
        raise CheckpointError(f"{context} is missing key(s): {names}")
    extra = set(mapping) - expected
    if extra:
        names = ", ".join(sorted(extra))
        raise CheckpointError(f"{context} has unexpected key(s): {names}")


def _validate_phase(value: CheckpointPhase | str) -> CheckpointPhase:
    if isinstance(value, CheckpointPhase):
        return value
    if not isinstance(value, str):
        raise CheckpointError("phase must be a string")
    try:
        return CheckpointPhase(value)
    except ValueError as exc:
        raise CheckpointError(f"invalid checkpoint phase {value!r}") from exc


def _validate_next_unit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckpointError("next_unit must be a non-negative integer")
    return value


def _validate_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointError(f"{field_name} must be a non-empty string")
    return value.strip()


def _fsync_directory(directory: Path) -> None:
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
