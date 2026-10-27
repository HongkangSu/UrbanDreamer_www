from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.workflow.config import ConfigError, _parse_method


def test_native_mainline_defaults_and_rejected_alternative_modes():
    method = _parse_method({})
    assert method.actor_objective == 'counterfactual_actor_critic_recoverq_v1'
    assert (method.fm_steps, method.disagreement_samples, method.disagreement_weight) == (10, 2, 0.)
    assert method.prior_filter == 'soft' and not method.prior_filter_input
    assert method.transition_volume_scope == 'previous_state_excluding_current_births'
    assert method.actor_entropy_mode == 'annealed'
    from urban_dreamer.workflow.config import load_config
    for city in ('ingolstadt_custom','montereau_fault_yonne','coulommiers'):
        config=load_config(Path(__file__).resolve().parents[1]/'configs'/f'{city}.toml')
        assert config.method.disagreement_samples==2
        assert config.method.prior_filter=='soft'
        assert config.resources.world_model_initialization_sequence_length==1
        assert config.method.supervision_horizon==3
    for override in ({'actor_objective': 'ppo_actor_critic_v1'}, {'disagreement_weight': .1},
                     {'prior_usage': 'source_reference'}, {'fm_training_mode': 'single_step'},
                     {'real_policy_update': 'ppo_anchor_v1'}):
        with pytest.raises(ConfigError):
            _parse_method(override)


def test_portable_resource_round_trip_and_relocation(tmp_path):
    from urban_dreamer.workflow.portable import portable_data, resource_id, resolve_resource
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    payload = {'file': first / 'data' / 'episode.npz', 'url': 'https://sumo.dlr.de/xsd/net_file.xsd'}
    saved = portable_data(payload, root=first)
    assert saved == {'file': 'data/episode.npz', 'url': payload['url']}
    assert resolve_resource(saved['file'], root=second) == second / 'data' / 'episode.npz'
    assert resource_id(first / 'data' / 'episode.npz', root=first) == saved['file']
    assert str(tmp_path) not in json.dumps(saved)


def test_absolute_paths_are_not_accepted_as_persisted_resource_ids(tmp_path):
    from urban_dreamer.workflow.portable import resolve_resource
    with pytest.raises(ValueError, match='relative'):
        resolve_resource(str(tmp_path / 'data'), root=tmp_path)


def test_cross_project_snapshot_and_windows_paths_do_not_expose_owner_directories(tmp_path):
    from pathlib import PureWindowsPath
    from urban_dreamer.workflow.config import InputConfig, load_config
    from urban_dreamer.workflow.portable import portable_data, resolve_resource
    token = 'private_owner_token'
    private = tmp_path / token / 'project'
    public = tmp_path / 'public' / 'run'
    config = load_config(Path(__file__).resolve().parents[1] / 'configs' / 'ingolstadt_custom.toml')
    config = replace(config, inputs=InputConfig('city', private / 'networks' / 'city'),
                     source_path=private / 'configs' / 'main.toml',
                     run=replace(config.run, output_dir=public))
    snapshot = config.resolved_snapshot(root=public)
    assert token not in json.dumps(snapshot)
    assert snapshot['inputs']['network_dir'].startswith('external/')
    windows_path = PureWindowsPath('Q:' + chr(92), token, 'dataset.npz')
    sanitized = portable_data({'path': windows_path, 'string': str(windows_path)}, root=public)
    assert token not in json.dumps(sanitized)
    with pytest.raises(ValueError, match='relative'):
        resolve_resource(str(windows_path), root=public)


def test_project_resources_resolve_without_original_output_location(tmp_path):
    from urban_dreamer.workflow.portable import PROJECT_ROOT, portable_data, resolve_resource
    resource = PROJECT_ROOT / 'configs' / 'ingolstadt_custom.toml'
    identifier = portable_data(resource, root=tmp_path / 'run')
    assert identifier == 'project/configs/ingolstadt_custom.toml'
    assert resolve_resource(identifier, root=tmp_path / 'different_run') == resource


def test_native_cache_configuration_remains_readable(tmp_path):
    from urban_dreamer.workflow.config import _parse_data
    data = _parse_data({'pretrain_cache_manifest': 'manifest.json'}, tmp_path)
    assert data.pretrain_cache_manifest == tmp_path / 'manifest.json'


@pytest.mark.parametrize('city', ['ingolstadt_custom', 'montereau_fault_yonne', 'coulommiers'])
def test_native_snapshot_toml_round_trip(tmp_path, city):
    from urban_dreamer.workflow.config import load_config
    original = load_config(Path(__file__).resolve().parents[1] / 'configs' / f'{city}.toml')
    snapshot = original.resolved_snapshot(root=tmp_path)
    assert 'feature_contract' not in snapshot
    assert snapshot['method']['actor_objective'] == 'counterfactual_actor_critic_recoverq_v1'
    lines = []
    for section, values in snapshot.items():
        lines.extend(['', f'[{section}]'])
        lines.extend(f'{key} = {json.dumps(value)}' for key, value in values.items() if value is not None)
    exported = tmp_path / 'run_config.toml'
    exported.write_text('\n'.join(lines) + '\n')
    restored = load_config(exported)
    assert restored.semantic_fingerprint == original.semantic_fingerprint
    assert restored.inputs.network_dir == original.inputs.network_dir
    assert restored.run.output_dir == original.run.output_dir


def test_native_schema10_checkpoint_relocation_and_strict_identity(tmp_path):
    import torch
    from urban_dreamer.workflow.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
    original = tmp_path / 'original'
    state = dict(reward_contract='causal_cumulative_mean_global_v1', models={'weight': torch.tensor([2.])},
                 optimizers={}, replay={}, rng={}, metrics={'resource': str(original / 'metrics.json')},
                 next_environment_episode=4)
    saved = save_checkpoint(output_dir=original, config_fingerprint='config', input_hash='inputs',
                            phase='online_training', next_unit=4, training_state=state)
    raw = torch.load(saved, weights_only=True)
    assert raw['schema_version'] == 10
    assert 'feature_contract' not in raw
    assert raw['training_state']['metrics']['resource'] == 'metrics.json'
    moved = tmp_path / 'moved'
    shutil.copytree(original, moved)
    restored = load_checkpoint(output_dir=moved, expected_config_fingerprint='config', expected_input_hash='inputs')
    assert restored.next_unit == 4
    assert restored.training_state['models']['weight'].item() == 2.
    with pytest.raises(CheckpointError, match='config fingerprint'):
        load_checkpoint(output_dir=moved, expected_config_fingerprint='changed', expected_input_hash='inputs')


def test_native_continuous_occupancy_is_preserved():
    from urban_dreamer.traffic.topology import EdgeNetwork
    network = SimpleNamespace(edge_count=2, edge_ids=('a', 'b'), speed_limits_mps=np.ones(2)*10,
        capacities=np.ones(2)*5, static_features=np.ones((2,4), dtype=np.float32))
    flow = EdgeNetwork.extract_flow(network, {
        'a': {'occupancy': 25., 'vehicle_number': 1, 'mean_speed': 4},
        'b': {'occupancy': .4, 'vehicle_number': 2, 'mean_speed': 10}})
    np.testing.assert_allclose(flow[:,0], [.25, .4])


def test_native_codec_keeps_forward_mean_speed_target():
    import torch
    from urban_dreamer.learning.codec_training import codec_decoded_metrics
    flow = torch.tensor([[[.25, .2, .5, 0., 1., 1., 1., 1.]]])
    following = flow.clone(); following[...,1] = .8
    decoded = flow[...,:4].clone(); decoded[...,1] = .5
    network = SimpleNamespace(capacities=np.array([2.]), speed_limits_mps=np.array([10.]))
    metrics = codec_decoded_metrics(decoded, flow, following, network)
    assert float(metrics['codec_speed_mae_mps']) == 0.


def test_dataset_manifest_identity_survives_directory_relocation(tmp_path):
    from urban_dreamer.learning.episode_archive import EpisodeArchive
    from urban_dreamer.learning.offline_data import OfflineEpisodeDataset
    from urban_dreamer.workflow.config import DataConfig, load_config
    config = load_config(Path(__file__).resolve().parents[1] / 'configs' / 'ingolstadt_custom.toml')
    original = tmp_path / 'original'
    archive_path = original / 'data' / 'train' / 'episode.npz'
    config = replace(config, data=DataConfig(archive_roots=(original / 'data',)),
                     run=replace(config.run, output_dir=original / 'run'))
    meta = dict(city=config.inputs.network_name, method='test', seed=42, episode=0, phase='train',
                input_hashes={'network': 'content'}, total_vehicle_count=0, session_protocol='event',
                routing_mode='current_to_dest', config=config.resolved_snapshot(),
                edge_ids=['e0', 'e1'], requested_step_seconds=1., demand_law='independent_od_poisson',
                demand_duration_seconds=1800., od_rates=[], demand_realization=[])
    episode = EpisodeArchive(meta)
    flow = np.zeros((2, 8), dtype=np.float32)
    flow[:,0] = [.25, .4]
    episode.start({'time': 0., 'done': False, 'vehicles': [], 'flow': flow})
    episode.append(SimpleNamespace(start_time=0., end_time=30.,
        next_state={'time': 30., 'done': True, 'vehicles': [], 'flow': flow},
        action_counts=np.zeros(2), departure_counts=np.zeros(2), active_vehicle_seconds=0., decisions=[]))
    episode.save(archive_path, metrics={}, travel_times={}, completed=True, deadline_seconds=3810.)
    from urban_dreamer.learning.episode_archive import read_episode_archive
    restored = read_episode_archive(archive_path)
    np.testing.assert_array_equal(restored['flow'][0,:,0], flow[:,0])
    assert restored['schema_version'] == 2
    assert 'feature_contract' not in restored['metadata']
    topology = SimpleNamespace(network=SimpleNamespace(edge_ids=('e0', 'e1'), edge_count=2,
        edge_to_index={'e0': 0, 'e1': 1}, adjacency_list=((), ())), vehicles=(), origins=(), destinations=())
    prepared = SimpleNamespace(input_hashes={'network': 'content'})
    first = OfflineEpisodeDataset(config, prepared, topology, 3)
    identity = first.manifest['identity']
    manifest = json.dumps(first.manifest)
    assert str(tmp_path) not in manifest
    assert first.manifest['entries'][0]['path'] == 'archive/0/train/episode.npz'
    second_root = tmp_path / 'relocated'
    shutil.copytree(original, second_root)
    moved_config = replace(config, data=DataConfig(archive_roots=(second_root / 'data',)),
                           run=replace(config.run, output_dir=second_root / 'run'))
    second = OfflineEpisodeDataset(moved_config, prepared, topology, 3)
    assert second.manifest['identity'] == identity
    second.verify_integrity()
