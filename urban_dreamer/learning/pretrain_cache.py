"""Lossless compact real-episode cache with source identities and sparse vehicles.

The builder hashes each decompressed source member while reading it once. Cache
episodes retain raw observations/actions and physical membership/arrival data;
learned conditions and fixed-cohort targets are rebuilt when batching.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .action_events import SparseActionEvents
from .episode_archive import archive_physics_identity, _decode, _json_array, observed_arrival_times
from .offline_data import OfflineEpisodeDataset, _identity, archive_content_hash
from .replay import SequenceBatch


CACHE_SCHEMA_VERSION = 1
# Reuse content validation across datasets/splits only while the file identity is unchanged.
_VERIFIED_COMPACT_FILES: dict[tuple, list[int]] = {}
TRANSITION_TARGET_SCOPE = 'previous_state_excluding_current_births'


def file_stamp(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def member_signature(members, *, trajectory_only=False) -> str:
    rows = [(m.filename, m.file_size, m.CRC) for m in sorted(members, key=lambda m: m.filename)
            if not trajectory_only or m.filename != 'summary_json.npy']
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def read_hash_source(path: Path) -> tuple[dict, dict]:
    """Decode and compute both canonical SHA256 identities in one member pass."""
    content, trajectory = hashlib.sha256(), hashlib.sha256()
    arrays = {}
    stamp = file_stamp(path)
    with zipfile.ZipFile(path) as archive:
        members = sorted(archive.infolist(), key=lambda m: m.filename)
        names = [m.filename for m in members]
        if not names or len(names) != len(set(names)) or any(not n.endswith('.npy') for n in names):
            raise ValueError(f'{path}: expected unique NPY members')
        for member in members:
            digests = (content,) if member.filename == 'summary_json.npy' else (content, trajectory)
            for digest in digests:
                digest.update(member.filename.encode() + b'\0')
            buffer = io.BytesIO()
            with archive.open(member) as stream:
                while chunk := stream.read(1024 * 1024):
                    for digest in digests:
                        digest.update(chunk)
                    buffer.write(chunk)
            for digest in digests:
                digest.update(b'\0')
            buffer.seek(0)
            value = np.load(buffer, allow_pickle=False)
            key = member.filename.removesuffix('.npy')
            arrays[key] = _decode(value) if key.endswith('_json') else value
        signatures = dict(source_member_signature=member_signature(members),
                          census_trajectory_signature=member_signature(members, trajectory_only=True))
    identity = dict(content_hash=content.hexdigest(), trajectory_hash=trajectory.hexdigest(), **signatures)
    if file_stamp(path) != stamp:
        # Atomic lossless repacking may race a reader. Its decoded content must
        # still match; substantively different replacements are never accepted.
        if archive_content_hash(path) != identity['content_hash']:
            raise ValueError(f'{path}: source changed while building cache')
        stamp = file_stamp(path)
    identity['source_stamp'] = stamp
    summary = arrays.pop('summary_json')
    arrays.update(summary)
    arrays['states'] = arrays.pop('states_json')
    arrays['decisions'] = arrays.pop('decisions_json')
    if arrays['schema_version'] not in {1, 2}:
        raise ValueError(f'{path}: unsupported source archive schema')
    if arrays['metadata'].get('session_protocol') != 'event' or arrays['metadata'].get('action_semantics') != 'branch-only':
        raise ValueError(f'{path}: compact pretraining requires event branch-only archives')
    archive_physics_identity(arrays['metadata'])
    destinations = {v['vehicle_id']: v['destination_edge'] for v in arrays['states'][0]['vehicles']}
    for index, event in enumerate(arrays['decisions']):
        event.setdefault('event_index', index)
        event.setdefault('destination_edge', destinations.get(event['vehicle_id']))
    return arrays, identity


def compact_arrays(data: dict) -> dict[str, np.ndarray]:
    """Replace repeated JSON vehicle snapshots with a registry and boundary CSR."""
    table = data['states'][0]['vehicles']
    ids = [v['vehicle_id'] for v in table]
    index = {name: i for i, name in enumerate(ids)}
    n = data['metadata']['total_vehicle_count']
    denominator = max(n, 1)
    metadata = dict(schema_version=CACHE_SCHEMA_VERSION, city=data['metadata']['city'],
                    population=n, completed=bool(data['completed']),
                    total_travel_minutes=(sum(data['travel_times'].values()) if data['completed']
                                          else float(data['active_vehicle_seconds'].sum())) / (60 * denominator))
    metadata['physics_identity'] = archive_physics_identity(data['metadata'])
    arrays = {key: np.asarray(data[key]) for key in ('flow', 'time', 'action_counts', 'departure_counts',
              'active_vehicle_seconds', 'terminated', 'truncated')}
    arrays.update(metadata_json=_json_array(metadata), vehicle_ids=np.asarray(ids, dtype=np.str_),
        vehicle_destination=np.asarray([v['destination_edge'] for v in table], dtype=np.int32),
        vehicle_origin=np.asarray([v['current_edge'] for v in table], dtype=np.int32),
        vehicle_departure=np.asarray([v['departure_time'] for v in table], dtype=np.float64))
    offsets, active = [0], []
    for state in data['states']:
        active.extend(v for v in state['vehicles'] if v['departed'] and not v['arrived'])
        offsets.append(len(active))
    arrays['boundary_offsets'] = np.asarray(offsets, dtype=np.int64)
    arrays['active_vehicle'] = np.asarray([index[v['vehicle_id']] for v in active], dtype=np.int32)
    for field in ('current_edge', 'committed_next_edge'):
        arrays['active_' + field] = np.asarray([v.get(field) if v.get(field) is not None else -1 for v in active], dtype=np.int32)
    for field in ('remaining_distance', 'departure_time', 'actual_departure_time', 'observed_departure_time',
                  'junction_remaining_distance', 'junction_speed'):
        default = -1. if field in {'actual_departure_time', 'observed_departure_time'} else 0.
        arrays['active_' + field] = np.asarray([v.get(field) if v.get(field) is not None else default for v in active], dtype=np.float64)
    for field in ('in_junction', 'pending_decision'):
        arrays['active_' + field] = np.asarray([bool(v.get(field, False)) for v in active], dtype=np.bool_)
    arrays['active_external_edge'] = np.where(arrays['active_in_junction'], -1, arrays['active_current_edge']).astype(np.int32)
    for i, field in enumerate(('junction_source', 'junction_target')):
        arrays['active_' + field] = np.asarray([v['junction_connection'][i] if v.get('junction_connection') else -1 for v in active], dtype=np.int32)
    arrivals = observed_arrival_times(data)
    arrays['arrival_vehicle'] = np.asarray([index[name] for name in arrivals], dtype=np.int32)
    arrays['arrival_time'] = np.asarray(list(arrivals.values()), dtype=np.float64)
    events = data['decisions']
    arrays['event_vehicle'] = np.asarray([index[e['vehicle_id']] for e in events], dtype=np.int32)
    for field in ('window_index', 'event_index', 'current_edge', 'next_edge', 'destination_edge'):
        arrays['event_' + field] = np.asarray([e[field] for e in events], dtype=np.int64)
    arrays['event_decision_time'] = np.asarray([e['decision_time'] for e in events], dtype=np.float64)
    arrays['event_phase'] = (arrays['event_decision_time'] - arrays['time'][arrays['event_window_index']]) / 30.
    return arrays


def boundary_vehicles(arrays: dict, frame: int) -> tuple[dict, ...]:
    start, end = arrays['boundary_offsets'][frame:frame + 2]
    result = []
    for offset in range(int(start), int(end)):
        vehicle = int(arrays['active_vehicle'][offset])
        row = dict(vehicle_id=str(arrays['vehicle_ids'][vehicle]),
                   destination_edge=int(arrays['vehicle_destination'][vehicle]),
                   departed=True, arrived=False)
        for name in ('current_edge', 'committed_next_edge'):
            value = int(arrays['active_' + name][offset])
            row[name] = None if name == 'committed_next_edge' and value < 0 else value
        for name in ('remaining_distance', 'departure_time', 'junction_remaining_distance', 'junction_speed'):
            row[name] = float(arrays['active_' + name][offset])
        for name in ('actual_departure_time', 'observed_departure_time'):
            value = float(arrays['active_' + name][offset])
            row[name] = None if value < 0 else value
        for name in ('in_junction', 'pending_decision'):
            row[name] = bool(arrays['active_' + name][offset])
        connection = (int(arrays['active_junction_source'][offset]), int(arrays['active_junction_target'][offset]))
        row['junction_connection'] = None if connection[0] < 0 else connection
        result.append(row)
    return tuple(result)


class LazyBoundaries:
    """A sampled sequence view; snapshots are decoded only if explicitly read."""
    def __init__(self, arrays, start, length):
        self.arrays, self.start, self.length = arrays, start, length
        self.seed = None

    def __len__(self):
        return self.length + 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('compact boundary outside sampled sequence')
        if index==0:
            if self.seed is None:
                self.seed=boundary_vehicles(self.arrays,self.start)
            return self.seed
        return boundary_vehicles(self.arrays, self.start + index)


def capacity_identity(capacities) -> str:
    values = np.maximum(np.asarray(capacities, dtype=np.float32), 1.)
    return hashlib.sha256(values.tobytes()).hexdigest()


def compact_cohort_targets(arrays, starts, length, capacities):
    """Legacy fixed-sequence C0 targets, retained for historical diagnostics only."""
    edges = arrays['flow'].shape[1]
    population = arrays['metadata']['population']
    capacity = np.maximum(np.asarray(capacities, dtype=np.float32), 1.)
    arrival = np.full(population, np.inf)
    arrival[arrays['arrival_vehicle']] = arrays['arrival_time']
    transition = np.stack([arrays['flow'][start:start+length+1] for start in starts]).copy()
    transition[..., 2] = 0.
    actions = np.zeros((len(starts),length,edges),dtype=np.float32)
    fractions = np.zeros((len(starts),length),dtype=np.float32)
    counts, selections = [], []
    for batch,start in enumerate(starts):
        left,right = arrays['boundary_offsets'][start:start+2]
        cohort = arrays['active_vehicle'][left:right]
        membership = np.zeros(population,dtype=np.bool_)
        membership[cohort] = True
        counts.append(len(cohort))
        when = arrays['time'][start]
        if np.any(arrival[cohort] <= when):
            raise ValueError('compact active seed has stale observed arrival')
        for step in range(length+1):
            frame=start+step
            first,last=arrays['boundary_offsets'][frame:frame+2]
            active=arrays['active_vehicle'][first:last]
            present=membership[active]
            expected=cohort[arrival[cohort] > arrays['time'][frame]]
            if not np.array_equal(np.sort(active[present]),np.sort(expected)):
                raise ValueError('compact cohort boundary/observed arrival disagreement')
            external=present & ~arrays['active_in_junction'][first:last]
            road=arrays['active_current_edge'][first:last][external]
            if np.any((road < 0) | (road >= edges)):
                raise ValueError('compact cohort external edge outside registry')
            transition[batch,step,:,2]=np.bincount(road,minlength=edges) / capacity
            if step < length:
                timestamp=arrays['time'][frame]
                exposure=np.maximum(0.,np.minimum(timestamp+30.,arrival[cohort])-timestamp).sum()
                fractions[batch,step]=exposure/(30*max(population,1))
        positions=np.flatnonzero((arrays['event_window_index']>=start)&(arrays['event_window_index']<start+length))
        vehicles=arrays['event_vehicle'][positions]
        if np.any(arrays['event_decision_time'][positions] >= arrival[vehicles]):
            raise ValueError('compact event occurred after observed arrival')
        selected=positions[membership[vehicles]]
        flat=(arrays['event_window_index'][selected]-start)*edges+arrays['event_next_edge'][selected]
        actions[batch]=np.bincount(flat,minlength=length*edges).reshape(length,edges)/max(population,1)
        selections.append((positions,selected))
    return transition,actions,fractions,np.asarray(counts,dtype=np.int64),selections


def compact_transition_targets(arrays,starts,length,capacities):
    """Full raw inputs, with only each endpoint's current-window newborns removed.

    Membership is refreshed at every observed window start. A vehicle born in
    window0 therefore belongs to window1's input and next endpoint target.
    Actions, events, exposure, and all non-volume target channels are untouched.
    """
    edges=arrays['flow'].shape[1]
    population=arrays['metadata']['population']
    capacity=np.maximum(np.asarray(capacities,dtype=np.float32),1.)
    arrival=np.full(population,np.inf)
    arrival[arrays['arrival_vehicle']]=arrays['arrival_time']
    target=np.stack([arrays['flow'][start+1:start+length+1] for start in starts]).copy()
    target[...,2]=0.
    counts=np.zeros((len(starts),length),dtype=np.int64)
    for batch,start in enumerate(starts):
        for step in range(length):
            frame=start+step
            first,last=arrays['boundary_offsets'][frame:frame+2]
            cohort=arrays['active_vehicle'][first:last]
            counts[batch,step]=len(cohort)
            if np.any(arrival[cohort]<=arrays['time'][frame]):
                raise ValueError('compact prior-state member has stale observed arrival')
            membership=np.zeros(population,dtype=np.bool_)
            membership[cohort]=True
            next_first,next_last=arrays['boundary_offsets'][frame+1:frame+3]
            active=arrays['active_vehicle'][next_first:next_last]
            present=membership[active]
            expected=cohort[arrival[cohort]>arrays['time'][frame+1]]
            if not np.array_equal(np.sort(active[present]),np.sort(expected)):
                raise ValueError('compact transition boundary/observed arrival disagreement')
            external=present & ~arrays['active_in_junction'][next_first:next_last]
            road=arrays['active_current_edge'][next_first:next_last][external]
            if np.any((road<0)|(road>=edges)):
                raise ValueError('compact transition external edge outside registry')
            target[batch,step,:,2]=np.bincount(road,minlength=edges)/capacity
    return target,counts


class CompactPretrainDataset(OfflineEpisodeDataset):
    """Bounded compact-episode streaming, with ordinary sparse SequenceBatch API.

    Source files are validated by the cache builder. Unchanged file stamps and
    member identities avoid rereading their large JSON on every epoch; changed
    source stamps require full decompressed SHA256, permitting recompression.
    Set ``verify_sources='always'`` for unconditional source rehashing.
    Legacy compact cache bodies are SHA256 checked when loaded. Explicit
    ``compact_only_v1`` manifests retain raw provenance but validate canonical
    compact files instead: every cache is hashed on first verification, with
    process-local reuse only for an unchanged file stamp and expected hashes.
    """
    def __init__(self, config, prepared_inputs, topology, sequence_length: int, *,
                 cache_manifest_path: Path, split: str = 'train', verify_sources: str = 'changed'):
        if split not in {'train', 'validation'} or verify_sources not in {'changed', 'always'}:
            raise ValueError('invalid compact split or source verification policy')
        if isinstance(sequence_length, bool) or sequence_length < 1:
            raise ValueError('sequence_length must be positive')
        self.config, self.prepared, self.topology = config, prepared_inputs, topology
        self.sequence_length, self.cache_limit = int(sequence_length), config.data.cache_episodes
        self._cache, self._source_stamps = OrderedDict(), {}
        self._epoch_order, self._start_order = [], []
        self._episode_cursor = self._start_cursor = self.epochs_completed = 0
        self.verify_sources = verify_sources
        self.cache_manifest_path = Path(cache_manifest_path)
        source = json.loads(self.cache_manifest_path.read_text())
        if source.get('status') != 'complete' or source.get('schema_version') != CACHE_SCHEMA_VERSION:
            raise ValueError('compact cache is incomplete or has unsupported schema')
        if source.get('identity') != _identity({k: v for k, v in source.items() if k != 'identity'}):
            raise ValueError('invalid compact cache manifest identity')
        self.source_storage = source.get('source_storage', 'raw_required')
        if self.source_storage not in {'raw_required', 'compact_only_v1'}:
            raise ValueError('unsupported compact source_storage')
        self._canonical_entries = {}
        if self.source_storage == 'compact_only_v1':
            for entry in source['entries']:
                trajectory = entry.get('trajectory_hash')
                if trajectory in self._canonical_entries:
                    raise ValueError('duplicate canonical compact trajectory')
                self._canonical_entries[trajectory] = entry
            sources_by_path = {row['path']: row for row in source['sources']}
            if len(sources_by_path) != len(source['sources']):
                raise ValueError('duplicate compact source provenance path')
            for entry in source['entries']:
                row = sources_by_path.get(entry['path'])
                if row is None or any(row.get(key) != entry.get(key) for key in (
                        'content_hash', 'trajectory_hash', 'source_member_signature', 'source_stamp',
                        'cache_path', 'cache_content_hash', 'cache_member_signature')):
                    raise ValueError('canonical compact entry missing matching source provenance')
        contract = source['contract']
        if contract['input_hashes'] != prepared_inputs.input_hashes or contract['city'] != config.inputs.network_name:
            raise ValueError('compact cache prepared inputs/city mismatch')
        target = config.resolved_snapshot()['simulation']
        from ..traffic.simulation_settings import physics_identity
        self._physics_identity = physics_identity(config.simulation)
        if contract.get('physics_identity', {'traffic_model': 'micro'}) != self._physics_identity:
            raise ValueError('compact cache traffic physics contract mismatch')
        if any(target[key] != value for key, value in contract['physical'].items()):
            raise ValueError('compact cache physical/demand contract mismatch')
        entries = [entry for entry in source['entries'] if entry['split'] == split]
        self.manifest = dict(schema_version=1, contract=dict(source_manifest=source['identity'],
            split=split, sequence_length=self.sequence_length, target_deadline_seconds=self.target_deadline_seconds,
            transition_target_scope=TRANSITION_TARGET_SCOPE+'_v1'),
            entries=entries, files=source['sources'], discovered_files=len(source['sources']),
            duplicate_files=source['duplicate_files'])
        self.manifest['identity'] = _identity(self.manifest)
        self._all_sources = source['sources']
        self.verify_integrity()

    def _verify_compact_binding(self, source):
        canonical = self._canonical_entries.get(source.get('trajectory_hash'))
        if canonical is None:
            raise ValueError('compact source has no canonical trajectory')
        for key in ('content_hash', 'trajectory_hash', 'source_member_signature',
                    'cache_content_hash', 'cache_member_signature'):
            value = source.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'invalid compact source provenance: {key}')
        if not isinstance(source.get('path'), str) or not source['path']:
            raise ValueError('invalid compact source provenance path')
        stamp = source.get('source_stamp')
        if not isinstance(stamp, list) or len(stamp) != 4 or any(type(v) is not int or v < 0 for v in stamp):
            raise ValueError('invalid compact source provenance stamp')
        if any(source.get(key) != canonical.get(key) for key in (
                'cache_path', 'cache_content_hash', 'cache_member_signature')):
            raise ValueError('compact source/canonical cache binding mismatch')
        cache_path = source.get('cache_path')
        if not isinstance(cache_path, str) or not cache_path:
            raise ValueError('invalid compact canonical cache path')
        return Path(cache_path)

    def _verify_compact_file(self, source):
        path = self._verify_compact_binding(source)
        stamp = file_stamp(path)
        key = (str(path.resolve()), source['cache_content_hash'], source['cache_member_signature'])
        if self.verify_sources != 'always' and _VERIFIED_COMPACT_FILES.get(key) == stamp:
            return
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or member_signature(members) != source['cache_member_signature']:
                raise ValueError(f'compact episode member signature changed: {path}')
        if archive_content_hash(path) != source['cache_content_hash']:
            raise ValueError(f'compact episode content changed: {path}')
        if file_stamp(path) != stamp:
            raise ValueError(f'compact episode changed during verification: {path}')
        _VERIFIED_COMPACT_FILES[key] = stamp

    def _verify_source(self, source):
        if self.source_storage == 'compact_only_v1':
            self._verify_compact_file(source)
            return
        path = Path(source['path'])
        stamp = file_stamp(path)
        known = self._source_stamps.get(str(path), source['source_stamp'])
        with zipfile.ZipFile(path) as archive:
            signature = member_signature(archive.infolist())
        if signature != source['source_member_signature']:
            raise ValueError(f'compact cache source content changed: {path}')
        if self.verify_sources == 'always' or stamp != known:
            if archive_content_hash(path) != source['content_hash']:
                raise ValueError(f'compact cache source content changed: {path}')
        self._source_stamps[str(path)] = stamp

    def verify_integrity(self, *, duplicates_only=False):
        selected = {e['path'] for e in self.manifest['entries']}
        for source in self._all_sources:
            if not duplicates_only or source['path'] not in selected:
                self._verify_source(source)

    def _episode(self, index, *, flow_only=False):
        entry = self.manifest['entries'][index]
        self._verify_source(entry)
        path = Path(entry['cache_path'])
        stamp = file_stamp(path)
        changed = index in self._cache and self._cache[index]['_cache_stamp'] != stamp
        if self.source_storage != 'compact_only_v1' and (index not in self._cache or changed):
            if archive_content_hash(path) != entry['cache_content_hash']:
                raise ValueError(f'compact episode content changed: {path}')
        if changed:
            del self._cache[index]
        if index not in self._cache or (not flow_only and 'metadata' not in self._cache[index]):
            while len(self._cache) >= self.cache_limit:
                self._cache.popitem(last=False)
            with np.load(path, allow_pickle=False) as archive:
                metadata = _decode(archive['metadata_json'])
                if metadata.get('schema_version') != CACHE_SCHEMA_VERSION:
                    raise ValueError('unsupported compact episode schema')
                if metadata.get('physics_identity', {'traffic_model': 'micro'}) != self._physics_identity:
                    raise ValueError('compact episode traffic physics mismatch')
                arrays = {'flow': archive['flow']} if flow_only else {key: archive[key] for key in archive.files}
            if not flow_only:
                arrays['metadata'] = _decode(arrays.pop('metadata_json'))
            arrays['_cache_stamp'] = stamp
            if not flow_only and arrays['metadata']['schema_version'] != CACHE_SCHEMA_VERSION:
                raise ValueError('unsupported compact episode schema')
            self._cache[index] = arrays
        self._cache.move_to_end(index)
        self._cache[index]['_cache_stamp'] = stamp
        return self._cache[index]

    def batch_from_starts(self, episode_index: int, starts: list[int], *,
                          materialization_horizon: int | None = None, fm_only: bool = False) -> SequenceBatch:
        """Materialize an optional supervised prefix without changing legal starts.

        External callers retain full sequence_length by default. FM-only reads
        omit Python seed snapshots/arrival dictionaries; lazily accessible
        boundary metadata remains available for explicit diagnostic callers.
        """
        if not starts:
            raise ValueError('compact batching requires sequence starts')
        if materialization_horizon is not None and (isinstance(materialization_horizon,bool)
                or not isinstance(materialization_horizon,(int,np.integer)) or materialization_horizon<1):
            raise ValueError('materialization_horizon must be positive')
        arrays = self._episode(episode_index)
        windows = len(arrays['time']) - 1
        if any(isinstance(start, bool) or start < 0 or start >= windows for start in starts):
            raise ValueError('compact sequence start outside episode')
        length = min(self.sequence_length, min(windows - start for start in starts))
        if materialization_horizon is not None:
            length=min(length,int(materialization_horizon))
        population = arrays['metadata']['population']
        denominator = max(population, 1)
        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype).clone()
        def sliced(name, boundary=False):
            return tensor(np.stack([arrays[name][start:start + length + int(boundary)] for start in starts]))
        boundaries = tuple(LazyBoundaries(arrays,start,length) for start in starts)
        context = np.asarray([[[min(arrays['time'][start + offset] / self.target_deadline_seconds, 1.),
                              (arrays['boundary_offsets'][start+offset+1]-arrays['boundary_offsets'][start+offset]) / denominator] for offset in range(length)]
                              for b, start in enumerate(starts)], dtype=np.float32)
        # Traffic-only archives have no online critic budget context. Match the
        # zero padding applied to these same records by ReplayTransition.
        context = np.pad(context, ((0, 0), (0, 0), (0, 2)))
        transition,cohort_counts=compact_transition_targets(
            arrays,starts,length,self.topology.network.capacities)
        def pack():
            columns=[[] for _ in range(7)];ids=[]
            for batch,start in enumerate(starts):
                positions=np.flatnonzero((arrays['event_window_index']>=start)&(arrays['event_window_index']<start+length))
                values=(np.full(len(positions),batch,dtype=np.int64),arrays['event_window_index'][positions]-start,
                    arrays['event_phase'][positions],arrays['event_current_edge'][positions],arrays['event_next_edge'][positions],
                    arrays['event_destination_edge'][positions],arrays['event_event_index'][positions])
                for column,value in zip(columns,values):
                    column.append(value)
                ids.extend(arrays['vehicle_ids'][arrays['event_vehicle'][positions]].tolist())
            tensors=[torch.from_numpy(np.concatenate(values)).to(torch.float64 if i==2 else torch.long)
                     for i,values in enumerate(columns)]
            return SparseActionEvents(*tensors,tuple(ids),len(starts),length)
        events=pack()
        arrivals = None if fm_only else {str(arrays['vehicle_ids'][vehicle]): float(at) for vehicle, at in
                                        zip(arrays['arrival_vehicle'], arrays['arrival_time'])}
        decoder_following, decoder_valid = None, None
        if self.config.codec.decoder_pretrain_enabled:
            indices = np.asarray(starts)[:, None] + np.arange(length)[None, :] + 2
            valid = indices < len(arrays['flow'])
            values = np.zeros((len(starts), length, *arrays['flow'].shape[1:]), dtype=arrays['flow'].dtype)
            values[valid] = arrays['flow'][indices[valid]]
            decoder_following, decoder_valid = tensor(values), tensor(valid, torch.bool)
        return SequenceBatch(flow=sliced('flow', True), action=sliced('action_counts') / denominator,
            departure=sliced('departure_counts') / denominator, context=tensor(context),
            active_vehicle_fraction=sliced('active_vehicle_seconds') / (30 * denominator),
            vehicles=tuple(() for _ in starts) if fm_only else tuple(frames[0] for frames in boundaries),
            time=tensor(arrays['time'][starts], torch.float64),
            terminal=sliced('terminated').bool(), total_travel_minutes=torch.full((len(starts),), arrays['metadata']['total_travel_minutes']),
            episode_complete=torch.full((len(starts),), arrays['metadata']['completed'], dtype=torch.bool),
            episode=torch.full((len(starts),), self.manifest['entries'][episode_index]['episode'], dtype=torch.long),
            start_index=torch.tensor(starts), total_vehicle_count=torch.full((len(starts),), population, dtype=torch.long),
            action_events=events, boundary_vehicles=boundaries,
            arrival_times=None if fm_only else tuple(dict(arrivals) for _ in starts),
            transition_target_flow=tensor(transition),transition_cohort_count=tensor(cohort_counts,torch.long),
            transition_volume_scope=TRANSITION_TARGET_SCOPE,
            transition_capacity_identity=capacity_identity(self.topology.network.capacities),
            decoder_following_flow=decoder_following, decoder_following_valid=decoder_valid)

    def iter_epoch(self, batch_size, rng):
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.verify_integrity(duplicates_only=True)
        if not self.has_samples:
            return
        if not self._epoch_order:
            self._epoch_order = rng.permutation(len(self.manifest['entries'])).tolist()
        while self._episode_cursor < len(self._epoch_order):
            index = self._epoch_order[self._episode_cursor]
            count = max(1, self.manifest['entries'][index]['windows'] - self.sequence_length + 1)
            if not self._start_order:
                self._start_order = rng.permutation(count).tolist()
            while self._start_cursor < len(self._start_order):
                starts = self._start_order[self._start_cursor:self._start_cursor + batch_size]
                self._start_cursor += len(starts)
                yield self.batch_from_starts(index, starts,
                    materialization_horizon=min(self.sequence_length,self.config.method.supervision_horizon),
                    fm_only=self.config.method.fm_training_mode != 'recursive_3step')
            self._episode_cursor += 1
            self._start_cursor = 0
            self._start_order = []
        self.epochs_completed += 1
        self._epoch_order = []
        self._episode_cursor = 0

    def iter_codec_batches(self, batch_size, rng=None, *, shuffle=False):
        """Yield (current,next) raw frame tensors once, without vehicle/event decoding.

        There is no terminal frame padding: the final boundary supplies a speed
        endpoint only. Codec iteration does not advance the FM sequence sampler.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('codec frame batch_size must be positive')
        if shuffle and rng is None:
            raise ValueError('shuffled codec iteration requires an explicit RNG')
        self.verify_integrity(duplicates_only=True)
        order = rng.permutation(len(self.manifest['entries'])) if shuffle else range(len(self.manifest['entries']))
        current, following, pending = [], [], 0
        for index in order:
            flow = self._episode(int(index), flow_only=True)['flow']
            frames = rng.permutation(len(flow)-1) if shuffle else np.arange(len(flow)-1)
            cursor = 0
            while cursor < len(frames):
                take = min(batch_size-pending, len(frames)-cursor)
                positions = frames[cursor:cursor+take]
                current.append(torch.from_numpy(flow[positions]).clone())
                following.append(torch.from_numpy(flow[positions+1]).clone())
                pending += take;cursor += take
                if pending == batch_size:
                    yield torch.cat(current), torch.cat(following)
                    current, following, pending = [], [], 0
        if pending:
            yield torch.cat(current), torch.cat(following)

    def iter_cohort_codec_batches(self,batch_size,rng=None,*,shuffle=False):
        """Legacy fixed-sequence C0 codec stream for retained diagnostics only.

        Legal anchors are uniformly spaced, rounded and deduplicated. Four
        valid current-frame anchors cover0..T-1; following=anchor+1<=T always
        has an observed forward speed endpoint in the same seed cohort.
        This stream does not modify the ordinary FM epoch sampler.
        """
        if isinstance(batch_size,bool) or not isinstance(batch_size,(int,np.integer)) or batch_size<1:
            raise ValueError('cohort codec batch_size must be positive')
        if shuffle and rng is None:
            raise ValueError('shuffled cohort codec iteration requires explicit RNG')
        self.verify_integrity(duplicates_only=True)
        order=rng.permutation(len(self.manifest['entries'])) if shuffle else range(len(self.manifest['entries']))
        current,following,pending=[],[],0
        for index in order:
            arrays=self._episode(int(index))
            windows=len(arrays['time'])-1
            legal=max(1,windows-self.sequence_length+1)
            starts=np.unique(np.rint(np.linspace(0,legal-1,min(4,legal))).astype(int))
            if shuffle:
                rng.shuffle(starts)
            for start in starts:
                length=min(self.sequence_length,windows-int(start))
                positions=np.unique(np.rint(np.linspace(0,length-1,min(4,length))).astype(int))
                transition,*_=compact_cohort_targets(arrays,[int(start)],length,self.topology.network.capacities)
                flow=transition[0]
                future=positions+1
                cursor=0
                while cursor<len(positions):
                    take=min(batch_size-pending,len(positions)-cursor)
                    chosen=positions[cursor:cursor+take]
                    next_chosen=future[cursor:cursor+take]
                    current.append(torch.from_numpy(flow[chosen]).clone())
                    following.append(torch.from_numpy(flow[next_chosen]).clone())
                    cursor+=take;pending+=take
                    if pending==batch_size:
                        yield torch.cat(current),torch.cat(following)
                        current,following,pending=[],[],0
        if pending:
            yield torch.cat(current),torch.cat(following)

    def iter_transition_codec_batches(self,batch_size,rng=None,*,shuffle=False):
        """Representative corrected endpoint Y[t+1] codec inputs, next=raw[t+2].

        Four uniform legal sequence starts and four uniform window anchors are
        rounded/deduplicated. A final physical endpoint without a following
        speed observation is skipped; no artificial next=current pair exists.
        Raw prior-state codec inputs are separately provided by iter_codec_batches.
        """
        if isinstance(batch_size,bool) or not isinstance(batch_size,(int,np.integer)) or batch_size<1:
            raise ValueError('transition codec batch_size must be positive')
        if shuffle and rng is None:
            raise ValueError('shuffled transition codec iteration requires explicit RNG')
        self.verify_integrity(duplicates_only=True)
        order=rng.permutation(len(self.manifest['entries'])) if shuffle else range(len(self.manifest['entries']))
        current,following,pending=[],[],0
        for index in order:
            arrays=self._episode(int(index))
            windows=len(arrays['time'])-1
            legal=max(1,windows-self.sequence_length+1)
            starts=np.unique(np.rint(np.linspace(0,legal-1,min(4,legal))).astype(int))
            if shuffle:
                rng.shuffle(starts)
            for start in starts:
                length=min(self.sequence_length,windows-int(start))
                positions=np.unique(np.rint(np.linspace(0,length-1,min(4,length))).astype(int))
                positions=positions[int(start)+positions+2<=windows]
                if not len(positions):
                    continue
                target,_=compact_transition_targets(arrays,[int(start)],length,self.topology.network.capacities)
                flow=target[0]
                cursor=0
                while cursor<len(positions):
                    take=min(batch_size-pending,len(positions)-cursor)
                    chosen=positions[cursor:cursor+take]
                    current.append(torch.from_numpy(flow[chosen]).clone())
                    following.append(torch.from_numpy(arrays['flow'][int(start)+chosen+2]).clone())
                    cursor+=take;pending+=take
                    if pending==batch_size:
                        yield torch.cat(current),torch.cat(following)
                        current,following,pending=[],[],0
        if pending:
            yield torch.cat(current),torch.cat(following)
