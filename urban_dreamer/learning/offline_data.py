"""Frozen, content-addressed real SUMO archives for finite initialization epochs.

Only two decoded episodes can be resident. Checkpoints retain manifest identity
and sampler progress; archives remain separate, read-only files.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator
import zipfile

import numpy as np

from .episode_archive import archive_physics_identity, archive_to_replay_episode, read_episode_archive
from .replay import ReplayBuffer, SequenceBatch
from ..workflow.portable import resolve_resource, resource_id


def archive_content_hash(path: Path, *, trajectory_only: bool = False) -> str:
    """Hash decompressed NPY members, independent of ZIP compression/ordering."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
        if not names or len(names) != len(set(names)) or any(not n.endswith('.npy') for n in names):
            raise ValueError(f'{path.name}: expected unique NPY members')
        for name in names:
            if trajectory_only and name == 'summary_json.npy':
                continue
            digest.update(name.encode() + b'\0')
            with archive.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b'\0')
    return digest.hexdigest()


def _identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _source_config(path: Path, metadata: dict) -> dict:
    snapshot = metadata.get('config') or metadata.get('resolved_config')
    if isinstance(snapshot, dict):
        return snapshot
    raise ValueError('archive must embed its physical configuration; external source snapshots are not read')


class OfflineEpisodeDataset:
    def __init__(self, config, prepared_inputs, topology, sequence_length: int, *,
                 archive_paths=None, manifest_path: Path | None = None,
                 excluded_trajectory_hashes=()):
        if isinstance(sequence_length, bool) or sequence_length < 1:
            raise ValueError('sequence_length must be positive')
        self.config, self.prepared, self.topology = config, prepared_inputs, topology
        self.sequence_length = int(sequence_length)
        self.cache_limit = config.data.cache_episodes
        self._cache = OrderedDict()
        self._epoch_order: list[int] = []
        self._start_order: list[int] = []
        self._episode_cursor = self._start_cursor = self.epochs_completed = 0
        self._explicit = archive_paths is not None
        self._resource_roots = tuple(Path(path).resolve() for path in config.data.archive_roots)
        roots = [f'archive/{index}' for index in range(len(self._resource_roots))]
        self.manifest_path = Path(manifest_path or config.run.output_dir / 'offline_manifest.json')
        self._contract = dict(city=config.inputs.network_name, input_hashes=prepared_inputs.input_hashes,
            sequence_length=self.sequence_length, roots=roots,
            target_deadline_seconds=self.target_deadline_seconds)
        excluded = set(excluded_trajectory_hashes)
        if excluded:
            self._contract['excluded_trajectory_hashes'] = sorted(excluded)
        self._contract['physical'] = {key: config.resolved_snapshot()['simulation'][key] for key in (
            'vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds', 'incident_end_seconds',
            'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')}
        from ..traffic.simulation_settings import physics_identity
        physics = physics_identity(config.simulation)
        if physics['traffic_model'] == 'meso':
            self._contract['physics_identity'] = physics
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get('contract') != self._contract:
                raise ValueError('offline manifest contract mismatch')
            self._verify_manifest_identity()
            self.verify_integrity()
        else:
            paths = sorted({Path(p).resolve() for p in archive_paths}) if self._explicit else self._discover()
            entries, files, duplicates, seen = [], [], 0, set(excluded)
            for path in paths:
                digest = archive_content_hash(path)
                files.append(dict(path=self._path_id(path), content_hash=digest))
                trajectory = archive_content_hash(path, trajectory_only=True)
                data = read_episode_archive(path, expected_session_protocol='event')
                self._validate(path, data)
                if trajectory in seen:
                    duplicates += 1
                    continue
                entries.append(dict(path=self._path_id(path), content_hash=digest, windows=len(data['action']),
                    trajectory_hash=trajectory,
                    events=len(data['decisions']), population=data['metadata']['total_vehicle_count'],
                    completed=bool(data['completed'])))
                seen.add(trajectory)
            self.manifest = dict(schema_version=1, contract=self._contract, entries=entries,
                                 files=files,
                                 discovered_files=len(paths), duplicate_files=duplicates)
            self.manifest['identity'] = _identity(self.manifest)
            if paths:
                self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.manifest_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.manifest, indent=2) + '\n')
                temporary.replace(self.manifest_path)

    def _path_id(self, path: Path) -> str:
        actual = Path(path).resolve()
        for index, root in enumerate(self._resource_roots):
            if actual.is_relative_to(root):
                return f'archive/{index}/' + resource_id(actual, root=root)
        raise ValueError('archive is outside the configured dataset roots')

    def resolve_path(self, identifier: str) -> Path:
        parts = Path(identifier).parts
        if len(parts) < 3 or parts[0] != 'archive' or not parts[1].isdigit():
            raise ValueError('offline archive resource identifier is invalid')
        index = int(parts[1])
        if index >= len(self._resource_roots):
            raise ValueError('offline archive root is not configured')
        root = self._resource_roots[index]
        actual = resolve_resource('/'.join(parts[2:]), root=root)
        if not actual.is_relative_to(root):
            raise ValueError('offline archive escapes its configured root')
        return actual

    @property
    def target_deadline_seconds(self) -> float:
        sim = self.config.simulation
        seconds = 30 * sim.world_model_intervals_per_episode + sim.training_completion_buffer_seconds
        return float(30 * math.ceil(seconds / 30))

    @property
    def has_samples(self) -> bool:
        return any(entry['windows'] for entry in self.manifest['entries'])

    @property
    def summary(self) -> dict:
        entries = self.manifest['entries']
        return dict(manifest_identity=self.manifest['identity'], imported_episodes=len(entries),
                    discovered_files=self.manifest['discovered_files'],
                    duplicate_files=self.manifest['duplicate_files'],
                    events=sum(e['events'] for e in entries), windows=sum(e['windows'] for e in entries),
                    completed_episodes=sum(e['completed'] for e in entries),
                    truncated_episodes=sum(not e['completed'] for e in entries),
                    cached_episodes=len(self._cache), cache_limit=self.cache_limit)

    def _discover(self) -> list[Path]:
        paths = set()
        for root in self.config.data.archive_roots:
            if not root.is_dir():
                raise ValueError('offline archive root does not exist')
            # A collection root may contain all cities. Select the target city
            # first; explicit method/city roots remain usable too.
            selected = root / self.config.inputs.network_name
            if not selected.is_dir():
                selected = root
            paths.update(p.resolve() for p in selected.rglob('*.npz') if 'train' in p.parts)
        return sorted(paths)

    def _validate(self, path: Path, data: dict) -> None:
        def fail(reason):
            raise ValueError(f'{path.name}: {reason}')
        meta = data['metadata']
        if meta.get('city') != self.config.inputs.network_name:
            fail('city mismatch')
        if meta.get('phase') not in {'train', 'warmup', 'online'} or ('train' not in path.parts and not self._explicit):
            fail('archive is not from a training partition')
        if data.get('source') != 'real_sumo':
            fail('archive must contain real SUMO transitions')
        if meta.get('input_hashes') != self.prepared.input_hashes:
            fail('prepared input hashes mismatch')
        if meta.get('edge_ids') != list(self.topology.network.edge_ids):
            fail('edge order mismatch or missing edge registry')
        step = meta.get('requested_step_seconds', meta.get('step_seconds', meta.get('observation_step_seconds')))
        if step != 1.:
            fail('event observation step must be one second')
        if meta.get('macro_step_seconds', 30.) != 30.:
            fail('world model window must be 30 seconds')
        source = _source_config(path, meta)
        from ..traffic.simulation_settings import physics_identity
        physics = archive_physics_identity(meta, simulation=source.get('simulation', {}))
        if physics != physics_identity(self.config.simulation):
            fail('traffic physics configuration mismatch')
        physical = ('vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds',
                    'incident_end_seconds', 'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')
        target = self.config.resolved_snapshot()['simulation']
        for key in physical:
            if source.get('simulation', {}).get(key) != target[key]:
                fail(f'physical/demand configuration mismatch: {key}')
        demand = meta.get('demand', {})
        law = meta.get('demand_law', demand.get('law'))
        if law not in {'independent_od_poisson', 'independent_poisson_per_od'}:
            fail('missing or incompatible demand law')
        duration = meta.get('demand_duration_seconds', demand.get('duration_seconds'))
        if duration != target['demand_duration_seconds']:
            fail('demand duration mismatch')
        from ..traffic.demand import PoissonDemand
        expected_demand = PoissonDemand.from_vehicles(self.topology.vehicles, duration_seconds=duration)
        rates = meta.get('od_rates', demand.get('od_rates'))
        if rates is None:
            fail('missing OD demand rates')
        try:
            actual_rates = sorted((row['origin_index'], row['destination_index'],
                row.get('rate_veh_per_second', row.get('rate_vehicles_per_second'))) for row in rates)
            if len(actual_rates) != len(expected_demand.od_rates) or any(
                    pair[:2] != wanted[:2] or not np.isclose(pair[2], wanted[2], rtol=1e-8, atol=1e-12)
                    for pair, wanted in zip(actual_rates, expected_demand.od_rates)):
                fail('OD demand rates mismatch')
        except (KeyError, TypeError):
            fail('invalid OD demand rates')
        times = np.asarray(data['time'])
        n, edges = len(data['action']), self.topology.network.edge_count
        if (n < 1 or times.shape != (n + 1,) or not np.isfinite(times).all()
                or not np.allclose(np.diff(times), 30., rtol=0, atol=1e-7)
                or data['flow'].shape != (n + 1, edges, 8) or not np.isfinite(data['flow']).all()):
            fail('invalid physical boundaries or flow')
        population = meta['total_vehicle_count']
        if isinstance(population, bool) or not isinstance(population, int) or population < 0:
            fail('invalid population')
        if len(data['states']) != n + 1:
            fail('snapshot boundary count mismatch')
        table = {v['vehicle_id']: v for v in data['states'][0]['vehicles']}
        if len(table) != population:
            fail('full initial vehicle table does not match population')
        realization = meta.get('demand_realization', meta.get('realized_demand'))
        if realization is None or len(realization) != population:
            fail('missing complete realized demand')
        if {row['vehicle_id'] for row in realization} != set(table):
            fail('realized demand vehicle association mismatch')
        for row in realization:
            vehicle = table[row['vehicle_id']]
            try:
                origin = self.topology.network.edge_to_index[self.topology.origins[row['origin_index']]]
                destination = self.topology.network.edge_to_index[self.topology.destinations[row['destination_index']]]
            except (KeyError, IndexError, TypeError):
                fail('realized demand OD indices are invalid')
            if (vehicle['destination_edge'] != destination or vehicle['current_edge'] != origin
                    or not np.isclose(vehicle['departure_time'], row['departure_second'], rtol=0, atol=1e-7)):
                fail('realized demand disagrees with full vehicle table')
        for index, state in enumerate(data['states']):
            if physics['traffic_model'] == 'meso' and any(
                    v.get('in_junction', False) or v.get('junction_connection') is not None
                    or v.get('junction_remaining_distance', 0.) != 0. for v in state['vehicles']):
                fail('meso snapshots must have zero internal junction state')
            if (len(state['vehicles']) != population or {v['vehicle_id'] for v in state['vehicles']} != set(table)
                    or state['time'] != times[index]):
                fail('boundary vehicle table or time mismatch')
            if any(v['destination_edge'] != table[v['vehicle_id']]['destination_edge'] for v in state['vehicles']):
                fail('vehicle destination changed inside episode')
        counts = np.zeros((n, edges), dtype=np.int64)
        previous_time = -1.
        for index, event in enumerate(data['decisions']):
            wi, current, nxt, dest = (event[k] for k in ('window_index', 'current_edge', 'next_edge', 'destination_edge'))
            vehicle = table.get(event['vehicle_id'])
            if (vehicle is None or event['event_index'] != index or not 0 <= wi < n
                    or not 0 <= current < edges or not 0 <= nxt < edges or not 0 <= dest < edges
                    or dest != vehicle['destination_edge'] or not times[wi] <= event['decision_time'] < times[wi+1]
                    or event['decision_time'] < previous_time):
                fail('invalid event identity, destination, time or half-open window')
            if nxt not in self.topology.network.adjacency_list[current]:
                fail('event turn is not legal')
            counts[wi, nxt] += 1
            previous_time = event['decision_time']
        if not np.array_equal(counts, data['action_counts']):
            fail('raw event reconstruction differs from archived action counts')
        denominator = max(population, 1)
        for name in ('action', 'departure'):
            raw = np.asarray(data[name + '_counts'])
            if (raw.shape != (n, edges) or not np.isfinite(raw).all() or (raw < 0).any()
                    or not np.equal(raw, np.floor(raw)).all()
                    or not np.allclose(data[name], raw / denominator, rtol=1e-6, atol=1e-8)):
                fail(f'invalid {name} counts or population normalization')
        exposure = np.asarray(data['active_vehicle_seconds'])
        if exposure.shape != (n,) or not np.isfinite(exposure).all() or (exposure < 0).any() or (exposure > population * 30 + 1e-5).any():
            fail('invalid population exposure')
        if bool(data['completed']) != bool(data['states'][-1]['done']):
            fail('completion disagrees with physical termination')
        if data['terminated'].shape != (n,) or data['truncated'].shape != (n,):
            fail('invalid completion/truncation flags')
        if (np.any(data['terminated'][:-1]) or bool(data['terminated'][-1]) != bool(data['completed'])
                or np.any(data['truncated'][:-1]) or bool(data['truncated'][-1]) == bool(data['completed'])):
            fail('completion/truncation flags do not describe episode')
        arrived = {v['vehicle_id'] for v in data['states'][-1]['vehicles'] if v['arrived']}
        travel = data['travel_times']
        if (set(travel) != arrived or (data['completed'] and arrived != set(table))
                or any(not np.isfinite(value) or value < 0 for value in travel.values())):
            fail('arrived vehicle travel-time supervision mismatch')

    def _verify_manifest_identity(self):
        payload = {k: v for k, v in self.manifest.items() if k != 'identity'}
        if self.manifest.get('schema_version') != 1 or self.manifest.get('identity') != _identity(payload):
            raise ValueError('invalid offline manifest identity')

    def verify_integrity(self, *, duplicates_only: bool = False) -> None:
        selected_paths = {entry['path'] for entry in self.manifest['entries']}
        for entry in self.manifest['files']:
            if duplicates_only and entry['path'] in selected_paths:
                continue
            if archive_content_hash(self.resolve_path(entry['path'])) != entry['content_hash']:
                raise ValueError(f"offline archive content changed: {entry['path']}")

    def _episode(self, index):
        entry = self.manifest['entries'][index]
        path = self.resolve_path(entry['path'])
        # Recheck even cached entries: replacing a file during a run cannot
        # silently mix old and new conditions in subsequent epochs.
        if archive_content_hash(path) != entry['content_hash']:
            raise ValueError(f'offline archive content changed: {self._path_id(path)}')
        if index not in self._cache:
            while len(self._cache) >= self.cache_limit:
                self._cache.popitem(last=False)
            data = read_episode_archive(path, expected_session_protocol='event')
            self._validate(path, data)
            episode = archive_to_replay_episode(data, target_deadline_seconds=self.target_deadline_seconds)
            self._cache[index] = episode
        self._cache.move_to_end(index)
        return self._cache[index]

    def iter_epoch(self, batch_size: int, rng: np.random.Generator) -> Iterator[SequenceBatch]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.verify_integrity(duplicates_only=True)
        if not self.has_samples:
            return
        if not self._epoch_order:
            self._epoch_order = rng.permutation(len(self.manifest['entries'])).tolist()
        while self._episode_cursor < len(self._epoch_order):
            index = self._epoch_order[self._episode_cursor]
            episode = self._episode(index)
            replay = ReplayBuffer(self.sequence_length, max_episodes=1)
            replay.append_episode(episode['transitions'], total_travel_minutes=episode['total_travel_minutes'],
                                  total_vehicle_count=episode['total_vehicle_count'],
                                  arrival_times=episode['arrival_times'])
            starts = max(1, len(episode['transitions']) - self.sequence_length + 1)
            if not self._start_order:
                self._start_order = rng.permutation(starts).tolist()
            while self._start_cursor < len(self._start_order):
                selected = self._start_order[self._start_cursor:self._start_cursor + batch_size]
                self._start_cursor += len(selected)
                yield replay._batch_from_picks([(0, start) for start in selected])
            self._episode_cursor += 1
            self._start_cursor = 0
            self._start_order = []
        self.epochs_completed += 1
        self._epoch_order = []
        self._episode_cursor = 0

    def state_dict(self) -> dict:
        return dict(schema_version=1, manifest_identity=self.manifest['identity'],
                    epoch_order=list(self._epoch_order), episode_cursor=self._episode_cursor,
                    start_order=list(self._start_order), start_cursor=self._start_cursor,
                    epochs_completed=self.epochs_completed)

    def release_cache(self) -> None:
        """Release decoded episodes when initialization switches data sources."""
        self._cache.clear()

    def load_state_dict(self, state: dict) -> None:
        expected = set(self.state_dict())
        if set(state) != expected or state['schema_version'] != 1 or state['manifest_identity'] != self.manifest['identity']:
            raise ValueError('offline dataset state/manifest mismatch')
        order = list(state['epoch_order'])
        if order and sorted(order) != list(range(len(self.manifest['entries']))):
            raise ValueError('invalid offline episode permutation')
        cursor = state['episode_cursor']
        starts = list(state['start_order'])
        if not isinstance(cursor, int) or not 0 <= cursor <= len(order):
            raise ValueError('invalid offline episode cursor')
        if starts:
            if cursor == len(order):
                raise ValueError('invalid offline sequence progress')
            count = max(1, self.manifest['entries'][order[cursor]]['windows'] - self.sequence_length + 1)
            if sorted(starts) != list(range(count)):
                raise ValueError('invalid offline start permutation')
        if not isinstance(state['start_cursor'], int) or not 0 <= state['start_cursor'] <= len(starts):
            raise ValueError('invalid offline start cursor')
        if not isinstance(state['epochs_completed'], int) or state['epochs_completed'] < 0:
            raise ValueError('invalid offline epoch count')
        self.verify_integrity()
        self._epoch_order, self._episode_cursor = order, cursor
        self._start_order, self._start_cursor = starts, state['start_cursor']
        self.epochs_completed = state['epochs_completed']
