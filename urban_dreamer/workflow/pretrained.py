"""Verify and load the original three-city native pretraining checkpoints."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import torch

from .portable import content_hash, portable_data


@dataclass(frozen=True)
class NativePretrained:
    training_state: dict[str, Any]
    checkpoint_sha256: str
    city: str
    history: dict[str, Any]


def _read_sources():
    path = Path(__file__).resolve().parents[2] / 'configs/pretrained_sources.json'
    return json.loads(path.read_text(encoding='utf-8'))


def xml_structure_hash(path):
    """Hash canonical XML without comments or formatting-only whitespace."""
    digest = hashlib.sha256()

    class Writer:
        def write(self, text):
            digest.update(text.encode('utf-8'))

    ET.canonicalize(from_file=path, out=Writer(), with_comments=False, strip_text=True)
    return digest.hexdigest()


def _native_metadata(value, *, root):
    # Adam's step counters are scalar tensors. Metadata serialization must never
    # turn them into Python scalars or copy/reshape any model/optimizer tensor.
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {_native_metadata(key, root=root): _native_metadata(item, root=root)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_native_metadata(item, root=root) for item in value]
    if isinstance(value, tuple):
        return tuple(_native_metadata(item, root=root) for item in value)
    if isinstance(value, (str, Path)):
        return portable_data(value, root=root)
    return value


def _validate_initial_state(raw, source):
    if not isinstance(raw, dict) or raw.get('schema_version') != 10:
        raise ValueError('pretrained checkpoint requires original native schema 10')
    if raw.get('phase') != 'online_training':
        raise ValueError('pretrained checkpoint must be at the online_training boundary')
    if type(raw.get('next_unit')) is not int or raw['next_unit'] != 0:
        raise ValueError('pretrained checkpoint next_unit must be zero')
    if raw.get('input_hash') != source['source']['input_hash']:
        raise ValueError('pretrained source input hash does not match its provenance')
    state = raw.get('training_state')
    fields = {'reward_contract', 'models', 'optimizers', 'replay', 'rng', 'metrics',
              'next_environment_episode'}
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError('pretrained training_state does not have its original native fields')
    if set(state['models']) != {'world_model', 'actor', 'latent_critic', 'event_critic'}:
        raise ValueError('pretrained models must contain exactly the four original models')
    if type(state['next_environment_episode']) is not int or state['next_environment_episode'] != 0:
        raise ValueError('pretrained environment episode cursor must be zero')
    if state['reward_contract'] != 'causal_cumulative_mean_global_v1':
        raise ValueError('pretrained native reward definition does not match')
    metrics = state['metrics']
    if not isinstance(metrics, dict) or metrics.get('codec_ready') != 1.:
        raise ValueError('pretrained checkpoint must retain its original codec_ready evidence')
    for name in ('online_episodes_completed', 'evaluation_episodes_completed', 'decoder_online_updates'):
        if name not in metrics or metrics[name] != 0:
            raise ValueError(f'pretrained {name} counter must be zero')
    for name, value in metrics.items():
        if name.startswith(('online_', 'evaluation_')) and isinstance(value, (int, float)) and value != 0:
            raise ValueError(f'pretrained {name} counter must be zero')
    replay = state['replay']
    if not isinstance(replay, dict) or set(replay) != {'online', 'offline', 'codec_validation'}:
        raise ValueError('pretrained replay must contain original online and offline states')
    online = replay['online']
    if (not isinstance(online, dict) or online.get('schema_version') != 8
            or online.get('episodes') != [] or online.get('sample_order') != []
            or online.get('sample_cursor') != 0 or online.get('next_storage_id') != 0):
        raise ValueError('pretrained online replay must be empty with zero cursors')
    for name, identity in (('offline', source['offline_manifest_identity']),
                           ('codec_validation', source['codec_validation_manifest_identity'])):
        saved = replay[name]
        if not isinstance(saved, dict) or saved.get('manifest_identity') != identity:
            raise ValueError(f'pretrained {name} manifest does not match its provenance')
        StoredOfflineReference(saved)
    return state


def _verify_assets(prepared, source):
    expected = source['source']['input_hashes']
    combined = hashlib.sha256(json.dumps(sorted(expected.items()),
        separators=(',', ':')).encode('utf-8')).hexdigest()
    if combined != source['source']['input_hash']:
        raise ValueError('pretrained source input hash manifest is inconsistent')
    for key, path in (('network_agents', prepared.network_agents_csv),
                      ('od_file', prepared.od_file), ('edge_registry', prepared.edge_registry)):
        if content_hash(path) != expected[key]:
            raise ValueError(f'prepared input content differs from native pretraining: {key}')
    if xml_structure_hash(prepared.network_xml) != source['network_xml_structure_sha256']:
        raise ValueError('prepared network XML structure differs from native pretraining')


def load_pretrained(path, *, config, prepared):
    """Load the native payload without changing parameters, optimizers or counters.

    Only the supplied checkpoint and this release's anonymous provenance and
    input assets are read. Original paths stored in metadata are never opened.
    """
    path = Path(path)
    if path.is_dir():
        path = path / 'checkpoints/latest.pt'
    if not path.is_file():
        raise FileNotFoundError('pretrained checkpoint file is missing')
    digest = content_hash(path)
    sources = _read_sources()['sources']
    matches = [(city, source) for city, source in sources.items()
               if source['checkpoint_sha256'] == digest]
    if len(matches) != 1:
        raise ValueError('pretrained checkpoint SHA256 does not match a supplied native artifact')
    city, source = matches[0]
    if city != config.inputs.network_name:
        raise ValueError('pretrained checkpoint city does not match the configured city')
    _verify_assets(prepared, source)
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = _validate_initial_state(raw, source)
    history = dict(city=city, checkpoint_sha256=digest,
        **{key: value for key, value in state['metrics'].items() if type(value) in (int, float)})
    history.update({'offline_' + key: value for key, value in source.get('offline_summary', {}).items()
                    if type(value) in (int, float)})
    history['offline_epochs_completed'] = state['replay']['offline']['epochs_completed']
    history['codec_validation_epochs_completed'] = state['replay']['codec_validation']['epochs_completed']
    return NativePretrained(_native_metadata(state, root=config.run.output_dir), digest, city, history)


class StoredOfflineReference:
    """Original sampler history whose archive bodies are unavailable locally."""

    available = False

    def __init__(self, state):
        self._state = deepcopy(state)
        self.manifest = {'identity': state['manifest_identity']}
        self.load_state_dict(state)

    def state_dict(self):
        return deepcopy(self._state)

    def load_state_dict(self, state):
        required = {'schema_version', 'manifest_identity', 'epoch_order', 'episode_cursor',
                    'start_order', 'start_cursor', 'epochs_completed'}
        if (not isinstance(state, dict) or set(state) != required or state['schema_version'] != 1
                or state['manifest_identity'] != self.manifest['identity']):
            raise ValueError('stored offline sampler schema or manifest differs from its checkpoint')
        for key in ('episode_cursor', 'start_cursor', 'epochs_completed'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError('stored offline sampler counters must remain nonnegative integers')
        for key in ('epoch_order', 'start_order'):
            if not isinstance(state[key], list) or any(type(item) is not int or item < 0 for item in state[key]):
                raise ValueError('stored offline sampler orders must remain integer lists')
        self._state = deepcopy(state)

    @property
    def has_samples(self):
        # The checkpoint records completed initialization on this manifest;
        # availability of the underlying archives is a separate fact.
        return True

    @property
    def epochs_completed(self):
        return self._state['epochs_completed']

    @property
    def summary(self):
        result = dict(manifest_identity=self.manifest['identity'],
            epochs_completed=self.epochs_completed, available=False)
        for source in _read_sources()['sources'].values():
            if source['offline_manifest_identity'] == self.manifest['identity']:
                result.update(source.get('offline_summary', {}))
                break
        return result

    def release_cache(self):
        pass

    def _missing(self, *args, **kwargs):
        raise FileNotFoundError('original offline archives are unavailable; configure a real offline dataset to read samples')

    __iter__ = _missing
    iter_epoch = _missing
    iter_codec_batches = _missing
    iter_cohort_codec_batches = _missing
    iter_transition_codec_batches = _missing
    read = _missing
    read_episode = _missing
    _read = _missing
    _episode = _missing
    verify_integrity = _missing
