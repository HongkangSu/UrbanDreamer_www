"""Relative resource identifiers and the baseline route observation format."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any

import numpy as np
import torch


FEATURE_CONTRACT = 'route_observation_44_v1'
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _external_id(path: str | PurePath) -> str:
    candidate = Path(path)
    digest = (content_hash(candidate) if not PureWindowsPath(str(path)).drive and candidate.is_file()
              else hashlib.sha256(str(path).encode()).hexdigest())
    return 'external/' + digest


def resource_id(path: str | Path, *, root: Path) -> str:
    """Identify a resource relative to its explicitly selected storage root."""
    if PureWindowsPath(str(path)).drive:
        return _external_id(path)
    actual, base = Path(path).resolve(), Path(root).resolve()
    if actual.is_relative_to(base):
        return actual.relative_to(base).as_posix()
    if actual.is_relative_to(PROJECT_ROOT):
        return 'project/' + actual.relative_to(PROJECT_ROOT).as_posix()
    return _external_id(actual)


def resolve_resource(identifier: str, *, root: Path) -> Path:
    path = Path(identifier)
    if path.is_absolute() or PureWindowsPath(identifier).drive or identifier.startswith('~'):
        raise ValueError('persisted resource identifiers must be relative')
    if identifier.startswith('external/'):
        raise ValueError('external resource requires an explicit dataset root binding')
    if identifier.startswith('project/'):
        resource = (PROJECT_ROOT / identifier.removeprefix('project/')).resolve()
        if not resource.is_relative_to(PROJECT_ROOT):
            raise ValueError('project resource escapes the release directory')
        return resource
    return (Path(root) / path).resolve()


def content_hash(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def portable_data(value: Any, *, root: Path) -> Any:
    """Serialize metadata without embedding a machine-specific filesystem root.

    Callers choose the root whose directory structure travels with the artifact.
    Network URLs remain unchanged; their slashes are not filesystem paths.
    Tensor/array payloads are passed through for checkpoint serialization.
    """
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, PurePath):
        return resource_id(value, root=root)
    if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).drive):
        return resource_id(value, root=root)
    if isinstance(value, dict):
        return {portable_data(key, root=root): portable_data(item, root=root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_data(item, root=root) for item in value]
    if hasattr(value, 'item') and getattr(value, 'ndim', 1) == 0:
        return value.item()
    return value
