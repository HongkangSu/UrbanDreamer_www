"""The release baselines share legal actions, paired demands and real updates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def changed(module, before):
    return any(not torch.equal(value, before[key]) for key, value in module.state_dict().items())


def test_release_cli_has_exactly_nine_methods_and_paired_evaluation():
    from tools.run_baseline import METHODS, arguments, evaluation_identity
    assert set(METHODS) == {'greedy', 'ga', 'iql', 'ippo', 'mappo', 'gnac', 'dreamer', 'mamba', 'mbpo'}
    args = arguments(['--method', 'greedy', '--config', 'configs/ingolstadt_custom.toml',
                      '--output-dir', 'outputs/example'])
    assert args.episodes == 1000 and args.seed == 42
    assert [evaluation_identity(i) for i in range(3)] == [
        {'demand_seed': 314159, 'demand_episode_index': i, 'simulation_seed': 314159 + i}
        for i in range(3)]
    with pytest.raises(SystemExit):
        arguments(['--method', 'dta', '--config', 'example.toml', '--output-dir', 'outputs/example'])


def test_heuristics_use_only_shared_legal_candidate_indices():
    from urban_dreamer.learning.baselines.heuristics import GeneticCandidatePolicy, GreedyCandidatePolicy
    network = SimpleNamespace(lengths_m=np.array([10., 20., 30., 40., 50.]),
                              speed_limits_mps=np.ones(5) * 10.,
                              free_flow_travel_seconds=np.arange(1., 6.))
    vehicles = [{'vehicle_id': 'v', 'current_edge': 0, 'remaining_distance': 10.}]
    observation = SimpleNamespace(mask=np.array([[True, False, True, False]]),
        routes=(((0, 1, 4), (), (0, 2, 4), ()),))
    flow = np.zeros((5, 8), dtype=np.float32)
    flow[:, 1] = 1.
    flow[1, 1] = .01
    greedy = GreedyCandidatePolicy(network)
    genetic = GeneticCandidatePolicy(network, seed=42)
    assert greedy.select(vehicles, observation, flow, 0.)[0] == [0]
    # Current congestion makes the longer static route faster for GA.
    assert genetic.select(vehicles, observation, flow, 0.)[0] == [2]
    costs = np.array([[1., np.inf, 3., np.inf], [np.inf, 2., np.inf, 1.]])
    mask = np.isfinite(costs)
    population = genetic.initialize_population(mask)
    assert population.shape == (100, 2)
    assert all(mask[np.arange(2), chromosome].all() for chromosome in population)
    for _ in range(20):
        assert mask[np.arange(2), genetic.mutate(population[0], mask)].all()


@pytest.mark.parametrize('name', ['iql', 'ippo', 'mappo', 'gnac'])
def test_model_free_baselines_make_finite_parameter_updates(name):
    from urban_dreamer.learning.baselines import build_method
    torch.manual_seed(37)
    settings = SimpleNamespace(seed=42, actor_lr=2e-4, critic_lr=3e-4,
        reward_objective='observed_mean', global_reward_only=True,
        iql_batch_size=8, iql_replay_capacity=32, iql_sample_to_insert_ratio=1.,
        ppo_epochs=1, ppo_critic_epochs=1, ppo_critic_min_epochs=1,
        ppo_critic_patience=1)
    network = SimpleNamespace(edge_count=3, adjacency=np.ones((3, 3), np.float32) - np.eye(3))
    runner = SimpleNamespace(device=torch.device('cpu'), network=network, deadline=90.)
    model = build_method(name, settings, runner)
    rng = np.random.default_rng(23)
    features = rng.normal(size=(6, 4, 10)).astype(np.float32)
    masks = np.tile([True, True, False, False], (6, 1))
    edge_indices = np.tile([1, 2, 0, 0], (6, 1))
    observation = SimpleNamespace(features=features, mask=masks, edge_indices=edge_indices)
    flow = rng.random((3, 8), dtype=np.float32)
    indices, auxiliary = model.select([{'current_edge': 0}] * 6, observation, flow, 0.)
    rows = [dict(vehicle_id=f'v{i % 2}', window=i // 2, time=float((i // 2) * 30),
        current_edge=0, features=features[i], mask=masks[i], edge_indices=edge_indices[i],
        flow=flow.copy(), action=indices[i], **auxiliary[i]) for i in range(6)]
    windows = [dict(obs=rng.random(16, dtype=np.float32), next_obs=rng.random(16, dtype=np.float32),
        start_time=i * 30., end_time=(i + 1) * 30., reward=-float(i + 1),
        terminal=i == 2) for i in range(3)]
    actor_before = deepcopy(model.actor.state_dict())
    metrics = model.on_episode(rows, windows, episode=1)
    assert changed(model.actor, actor_before)
    assert all(np.isfinite(value) for value in metrics.values()
               if isinstance(value, (int, float, np.number)))
    if name != 'iql':
        assert metrics.get('actor_optimizer_steps', 1) > 0


@pytest.mark.parametrize('name', ['dreamer', 'mamba', 'mbpo'])
def test_native_models_update_dynamics_and_actor_without_environment(name):
    from tools.run_baseline import make_native_agent
    from urban_dreamer.learning.baselines.generic_config import GenericConfig
    config = GenericConfig(obs_dim=44, action_dim=4, seed=42, hidden_dim=32,
        batch_size=8, sequence_length=4, sequence_batch_size=2,
        mamba_sequence_batch_size=1, mamba_imagination_starts=2,
        imagination_horizon=3, imagination_starts=2, rollout_batch_size=8,
        max_agents_per_chunk=2, model_updates=1, policy_updates=1, replay_capacity=128)
    model = make_native_agent(name, config)
    rng = np.random.default_rng(23)
    count = 16
    observation = rng.normal(size=(count, 44)).astype(np.float32)
    masks = np.tile([True, True, False, True], (count, 1))
    actions = np.tile([0, 1], count // 2)
    terminals = np.arange(count) >= count - 2
    next_masks = masks.copy()
    next_masks[terminals] = False
    model.observe(dict(obs=observation, actions=actions,
        rewards=rng.normal(size=count).astype(np.float32), next_obs=observation + .1,
        terminals=terminals, masks=masks, next_masks=next_masks,
        agent_ids=np.tile([0, 1], count // 2), frame_indices=np.repeat(np.arange(8), 2)))
    dynamics = model.models if name == 'mbpo' else model.world_model
    before = deepcopy(dynamics.state_dict())
    actor_before = deepcopy(model.actor.state_dict())
    metrics = model.update()
    assert changed(dynamics, before) and changed(model.actor, actor_before)
    assert metrics['model_optimizer_steps'] > 0 and metrics['actor_optimizer_steps'] > 0
    assert all(np.isfinite(value) for value in metrics.values()
               if isinstance(value, (int, float, np.number)))
    assert not any(hasattr(model, key) for key in ('network', 'runner', 'deadline', 'speed_limits'))


@pytest.mark.parametrize('name', ['greedy', 'ga'])
def test_heuristics_complete_real_sumo_and_write_portable_paired_metrics(name, tmp_path):
    if importlib.util.find_spec('libsumo') is None:
        pytest.skip('real simulator smoke requires the documented libsumo dependency')
    from tools.run_baseline import main
    root = Path(__file__).resolve().parents[1]
    source = (root / 'configs/ingolstadt_custom.toml').read_text()
    fixture = tmp_path / 'inputs'
    shutil.copytree(root / 'tests/fixtures/toy_cross_bottleneck', fixture)
    source = source.replace('network_name = "ingolstadt_custom"', 'network_name = "toy_cross_bottleneck"')
    source = source.replace('network_dir = "../networks/ingolstadt_custom"',
        'network_dir = "' + Path(os.path.relpath(fixture, tmp_path)).as_posix() + '"')
    source = source.replace('incident_edge = "201201950#0"', 'incident_edge = "A_in"')
    config = tmp_path / 'toy.toml'
    config.write_text(source)
    result = main(['--method', name, '--config', str(config), '--output-dir', str(tmp_path / name), '--smoke'])
    assert result['smoke'] is True and len(result['evaluations']) == 3
    for index, evaluation in enumerate(result['evaluations']):
        assert evaluation['demand_seed'] == 314159 and evaluation['demand_index'] == index
        assert evaluation['simulation_seed'] == 314159 + index
        assert all(np.isfinite(evaluation[key]) for key in ('ATT', 'P90', 'P95', 'completion_rate'))
    exported = (tmp_path / name / 'result.json').read_text()
    assert str(root.resolve()) not in exported and str(tmp_path.resolve()) not in exported


def test_project_inputs_and_external_output_do_not_expose_machine_directories(tmp_path):
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.portable import portable_data
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs/ingolstadt_custom.toml')
    exported = json.dumps(portable_data(config.resolved_snapshot(root=tmp_path), root=root))
    assert str(root.resolve()) not in exported
    assert str(Path.home()) not in exported and str(tmp_path) not in exported


@pytest.mark.parametrize('name', ['iql', 'ippo', 'mappo', 'gnac', 'dreamer', 'mamba', 'mbpo'])
def test_learned_baseline_cli_updates_from_real_sumo_and_archives_data(name, tmp_path):
    if importlib.util.find_spec('libsumo') is None:
        pytest.skip('real simulator smoke requires libsumo')
    from tools.run_baseline import main
    from urban_dreamer.learning.episode_archive import read_episode_archive
    from urban_dreamer.workflow.portable import FEATURE_CONTRACT
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / 'tests/fixtures/toy_cross_bottleneck', tmp_path / 'inputs')
    source = (root / 'configs/ingolstadt_custom.toml').read_text()
    source = source.replace('network_name = "ingolstadt_custom"', 'network_name = "toy_cross_bottleneck"')
    source = source.replace('network_dir = "../networks/ingolstadt_custom"', 'network_dir = "inputs"')
    source = source.replace('incident_edge = "201201950#0"', 'incident_edge = "A_in"')
    config = tmp_path / 'toy.toml'
    config.write_text(source)
    output = tmp_path / name
    result = main(['--method', name, '--config', str(config), '--output-dir', str(output),
                   '--smoke', '--save-episodes'])
    assert (output / 'final.pt').is_file() and len(result['evaluations']) == 3
    rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    training = next(row for row in rows if row['phase'] == 'train')
    assert training['online_att_seconds'] == training['all_demand_censored_att_seconds']
    assert training['episode_index'] == 1 and training['smoke'] is True
    assert training.get('actor_optimizer_steps', training.get('q_updates', 1)) > 0
    archive = read_episode_archive(output / 'episodes/train/episode_000001.npz')
    assert archive['metadata']['feature_contract'] == FEATURE_CONTRACT
    restored = main(['--method', name, '--config', str(config),
        '--output-dir', str(tmp_path / (name + '_restored')), '--smoke',
        '--evaluate-only', '--checkpoint', str(output / 'final.pt')])
    assert [row['demand_hash'] for row in restored['evaluations']] == [
        row['demand_hash'] for row in result['evaluations']]
    assert [row['ATT'] for row in restored['evaluations']] == [
        row['ATT'] for row in result['evaluations']]
