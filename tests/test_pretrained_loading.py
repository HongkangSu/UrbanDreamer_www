"""Native initialization preserves checkpoint payloads and verifies provenance."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offline_state(identity='a' * 64):
    return dict(schema_version=1, manifest_identity=identity, epoch_order=[], episode_cursor=0,
                start_order=[], start_cursor=0, epochs_completed=1)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    from urban_dreamer.workflow import pretrained
    city = 'ingolstadt_custom'
    files = {key: tmp_path / name for key, name in {
        'network_xml': 'network.net.xml', 'network_agents': 'agents.csv',
        'od_file': 'od.txt', 'edge_registry': 'edge_registry.json'}.items()}
    files['network_xml'].write_text('<net><!-- original metadata --><edge id="a" speed="10"/></net>')
    files['network_agents'].write_text('id,origin,destination,start_time\nv0,0,0,0\n')
    files['od_file'].write_text("{'origins':['a'],'destinations':['a']}")
    files['edge_registry'].write_text('{"network":"ingolstadt_custom","edge_ids":["a"]}')
    source_hashes = {key: digest(path) for key, path in files.items()}
    source_hash = hashlib.sha256(json.dumps(sorted(source_hashes.items()),
        separators=(',', ':')).encode()).hexdigest()
    model_state = {name: {'weight': torch.tensor([1., 2.])}
                   for name in ('world_model', 'actor', 'latent_critic', 'event_critic')}
    raw = dict(schema_version=10, config_fingerprint='b' * 64, input_hash=source_hash,
        phase='online_training', next_unit=0,
        training_state=dict(reward_contract='causal_cumulative_mean_global_v1',
            models=model_state, optimizers={'world_model_initial': {'state': {0: {'step': torch.tensor(7.)}}}},
            replay=dict(online=dict(schema_version=8, episodes=[], sample_order=[],
                sample_cursor=0, next_storage_id=0, max_episodes=8, sequence_length=3, active_only=False),
                offline=offline_state(), codec_validation=offline_state('c' * 64)),
            rng={'torch_cpu': torch.get_rng_state()}, metrics=dict(codec_ready=1., codec_updates=5.,
                online_episodes_completed=0., evaluation_episodes_completed=0.,
                decoder_online_updates=0., world_model_initialization_epochs_completed=1.,
                world_model_initialization_samples_seen=100.,
                codec_warm_start_checkpoint=str(tmp_path / 'private' / 'old.pt')),
            next_environment_episode=0))
    checkpoint = tmp_path / 'checkpoints/latest.pt'
    checkpoint.parent.mkdir()
    source = dict(checkpoint_sha256='', source=dict(input_hash=source_hash, input_hashes=source_hashes),
        network_xml_structure_sha256=pretrained.xml_structure_hash(files['network_xml']),
        offline_manifest_identity='a' * 64, codec_validation_manifest_identity='c' * 64,
        offline_summary=dict(imported_episodes=20, windows=100, events=50))
    sources = {'sources': {city: source}}
    monkeypatch.setattr(pretrained, '_read_sources', lambda: sources)

    def save():
        torch.save(raw, checkpoint)
        source['checkpoint_sha256'] = digest(checkpoint)

    save()
    # Anonymous XML may have a different byte hash while its structure is identical.
    files['network_xml'].write_text('<net><edge speed="10" id="a" /></net>')
    prepared = SimpleNamespace(network_xml=files['network_xml'], network_agents_csv=files['network_agents'],
        od_file=files['od_file'], edge_registry=files['edge_registry'],
        input_hashes={key: digest(path) for key, path in files.items()})
    config = SimpleNamespace(inputs=SimpleNamespace(network_name=city),
                             run=SimpleNamespace(output_dir=tmp_path / 'output'))
    return SimpleNamespace(module=pretrained, raw=raw, config=config, prepared=prepared,
                           checkpoint=checkpoint, source=source, files=files, save=save)


def test_native_load_preserves_tensors_optimizer_scalars_and_history(artifact, monkeypatch):
    module = artifact.module
    original_load = module.torch.load
    observed = {}

    def load(*args, **kwargs):
        observed.update(kwargs)
        observed['raw'] = original_load(*args, **kwargs)
        return observed['raw']

    monkeypatch.setattr(module.torch, 'load', load)
    loaded = module.load_pretrained(artifact.checkpoint.parent.parent,
                                   config=artifact.config, prepared=artifact.prepared)
    assert observed['weights_only'] is True and observed['mmap'] is True
    for name in artifact.raw['training_state']['models']:
        assert loaded.training_state['models'][name]['weight'] is observed['raw']['training_state']['models'][name]['weight']
    step = loaded.training_state['optimizers']['world_model_initial']['state'][0]['step']
    assert isinstance(step, torch.Tensor) and step.ndim == 0 and step.item() == 7.
    assert step is observed['raw']['training_state']['optimizers']['world_model_initial']['state'][0]['step']
    assert loaded.training_state['metrics']['codec_updates'] == 5.
    assert loaded.city == 'ingolstadt_custom' and loaded.checkpoint_sha256 == digest(artifact.checkpoint)
    assert all(isinstance(v, (int, float)) for k, v in loaded.history.items()
               if k not in ('city', 'checkpoint_sha256'))
    assert str(artifact.checkpoint.parent.parent) not in json.dumps(loaded.history)
    assert str(artifact.checkpoint.parent.parent) not in loaded.training_state['metrics']['codec_warm_start_checkpoint']


@pytest.mark.parametrize('change, message', [
    (lambda raw: raw.update(schema_version=11), 'schema'),
    (lambda raw: raw.update(phase='pretrain'), 'online'),
    (lambda raw: raw.update(next_unit=1), 'next_unit'),
    (lambda raw: raw['training_state'].update(next_environment_episode=1), 'episode'),
    (lambda raw: raw['training_state']['metrics'].update(online_episodes_completed=1.), 'online'),
    (lambda raw: raw['training_state']['metrics'].update(evaluation_episodes_completed=1.), 'evaluation'),
    (lambda raw: raw['training_state']['replay']['online'].update(episodes=[{}]), 'replay'),
    (lambda raw: raw['training_state']['models'].update(extra={}), 'models'),
])
def test_rejects_noninitial_native_payloads(artifact, change, message):
    change(artifact.raw)
    artifact.save()
    with pytest.raises(ValueError, match=message):
        artifact.module.load_pretrained(artifact.checkpoint, config=artifact.config, prepared=artifact.prepared)


def test_wrong_city_and_modified_checkpoint_are_rejected(artifact):
    artifact.config.inputs.network_name = 'coulommiers'
    with pytest.raises(ValueError, match='city'):
        artifact.module.load_pretrained(artifact.checkpoint, config=artifact.config, prepared=artifact.prepared)
    artifact.config.inputs.network_name = 'ingolstadt_custom'
    with artifact.checkpoint.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        artifact.module.load_pretrained(artifact.checkpoint, config=artifact.config, prepared=artifact.prepared)


@pytest.mark.parametrize('field', ['network_agents', 'od_file', 'edge_registry', 'network_xml'])
def test_rejects_changed_traffic_assets(artifact, field):
    if field == 'network_xml':
        artifact.files[field].write_text('<net><edge speed="11" id="a"/></net>')
    else:
        with artifact.files[field].open('a') as stream:
            stream.write('changed')
    with pytest.raises(ValueError, match='input|structure'):
        artifact.module.load_pretrained(artifact.checkpoint, config=artifact.config, prepared=artifact.prepared)


def test_stored_offline_history_never_pretends_archives_are_available(artifact):
    module = artifact.module
    reference = module.StoredOfflineReference(offline_state())
    assert reference.manifest == {'identity': 'a' * 64}
    assert reference.has_samples is True and reference.available is False
    assert reference.summary['imported_episodes'] == 20
    assert reference.summary['epochs_completed'] == 1
    assert reference.state_dict() == offline_state()
    state = reference.state_dict()
    state['epoch_order'].append(9)
    assert reference.state_dict() == offline_state()
    for method in (reference.iter_epoch, reference.iter_codec_batches,
                   reference.iter_cohort_codec_batches, reference.iter_transition_codec_batches,
                   reference.read, reference.verify_integrity):
        with pytest.raises(FileNotFoundError, match='offline'):
            method()
    with pytest.raises(FileNotFoundError, match='offline'):
        iter(reference)
    with pytest.raises(ValueError, match='manifest'):
        reference.load_state_dict(offline_state('d' * 64))
