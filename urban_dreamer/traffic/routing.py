from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..workflow.config import MainlineConfig


class InputValidationError(ValueError):
    """Raised when the network and demand input bundle is inconsistent."""


@dataclass(frozen=True)
class PreparedInputs:
    network_xml: Path
    network_agents_csv: Path
    od_file: Path
    edge_registry: Path
    vehicle_count: int
    input_hashes: dict[str, str]


def validate_prepared_inputs(config: MainlineConfig) -> PreparedInputs:
    name = config.inputs.network_name
    directory = config.inputs.network_dir
    paths = {
        'network_xml': directory / f'{name}.net.xml',
        'network_agents': directory / 'agents.csv',
        'od_file': directory / f'od_{name}.txt',
        'edge_registry': directory / 'edge_registry.json',
    }
    missing = [f'{key}={path}' for key, path in paths.items() if not path.is_file()]
    if missing:
        raise InputValidationError('prepared input bundle is incomplete: ' + ', '.join(missing))
    registry = _read_json_object(paths['edge_registry'])
    edge_ids = registry.get('edge_ids')
    if not isinstance(edge_ids, list) or not edge_ids or not all(isinstance(edge, str) and edge for edge in edge_ids):
        raise InputValidationError('edge_registry.json must contain non-empty string edge_ids')
    if len(edge_ids) != len(set(edge_ids)):
        raise InputValidationError('edge_registry.json contains duplicate edge_ids')
    if registry.get('network') != name:
        raise InputValidationError('edge_registry.json network does not match [inputs].network_name')
    try:
        od = ast.literal_eval(paths['od_file'].read_text(encoding='utf-8'))
    except (OSError, ValueError, SyntaxError) as exc:
        raise InputValidationError(f'cannot read OD definition: {exc}') from exc
    if not isinstance(od, dict) or set(od) != {'origins', 'destinations'}:
        raise InputValidationError('OD definition must contain origins and destinations')
    for key in ('origins', 'destinations'):
        if not isinstance(od[key], list) or not od[key] or not all(isinstance(edge, str) and edge in edge_ids for edge in od[key]):
            raise InputValidationError(f'OD {key} must contain registered edge IDs')
    vehicle_count = _validate_demand(paths['network_agents'], od)
    return PreparedInputs(
        network_xml=paths['network_xml'],
        network_agents_csv=paths['network_agents'],
        od_file=paths['od_file'],
        edge_registry=paths['edge_registry'],
        vehicle_count=vehicle_count,
        input_hashes={key: _sha256(path) for key, path in paths.items()},
    )


def _validate_demand(path: Path, od: dict[str, Any]) -> int:
    vehicle_ids: set[str] = set()
    with path.open('r', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'id', 'origin', 'destination', 'start_time'}
        if reader.fieldnames is None or set(reader.fieldnames) != required or len(reader.fieldnames) != 4:
            raise InputValidationError(f'{path} columns must be exactly id,origin,destination,start_time')
        for row in reader:
            vehicle_id = str(row.get('id') or '').strip()
            if not vehicle_id:
                raise InputValidationError(f'{path} must contain non-empty vehicle IDs')
            if vehicle_id in vehicle_ids:
                raise InputValidationError(f'{path} contains duplicate vehicle IDs')
            vehicle_ids.add(vehicle_id)
            for field, choices in [('origin', od['origins']), ('destination', od['destinations'])]:
                try:
                    index = int(row[field])
                except (ValueError, TypeError) as exc:
                    raise InputValidationError(f'{path} {field} must be an integer OD index') from exc
                if not 0 <= index < len(choices):
                    raise InputValidationError(f'{path} {field} is outside the OD definition')
            try:
                departure = float(row['start_time'])
            except (ValueError, TypeError) as exc:
                raise InputValidationError(f'{path} start_time must be finite and non-negative') from exc
            if not math.isfinite(departure) or departure < 0:
                raise InputValidationError(f'{path} start_time must be finite and non-negative')
    if not vehicle_ids:
        raise InputValidationError(f'{path} must contain non-empty vehicle demand')
    return len(vehicle_ids)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputValidationError(f'cannot read JSON object {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise InputValidationError(f'{path} must contain a JSON object')
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
