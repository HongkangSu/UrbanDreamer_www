from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
import json
import shutil

import pytest
import torch

from smoke_support import make_project
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.heuristics import GreedyCandidatePolicy
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import run_mainline


def collect_archives(path, count=3):
    args = Namespace(config=path, output_dir=path.parent/'data/offline', seed=42,
        threads=1, device='cpu', method='greedy', vehicle_limit=0)
    runner = EpisodeRunner(args)
    policy = GreedyCandidatePolicy(runner.network, candidates=runner.candidates)
    for index in range(count):
        runner.run_episode(policy.select, seed=42+index, collect=True,
            demand_seed=42, demand_episode_index=index, simulation_seed=42+index,
            archive_path=path.parent/'data/offline/episodes/train'/f'episode_{index:03d}.npz',
            archive_metadata={'phase': 'train', 'episode': index})


def parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def same(before, module):
    return all(torch.equal(a, b) for a, b in zip(before, module.parameters()))


def test_real_offline_to_online_mainline_and_completed_resume(tmp_path):
    pytest.importorskip('libsumo')
    path = make_project(tmp_path/'bundle')
    collect_archives(path)
    config = load_config(path)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    try:
        trainer.initialize_codec_epoch(0)
        assert trainer.codec_ready()
        actor_before = parameters(trainer.actor)
        trainer.initialize_world_model_epoch(0)
        assert trainer.metrics['world_model_supervision_depth_last']==1.
        assert same(actor_before, trainer.actor)
        committed_state = deepcopy(trainer.state_dict())
        before = {name: parameters(getattr(trainer, name))
                  for name in ('actor', 'latent_critic', 'event_critic')}
        collected = trainer._run_episode(mode='online', episode_index=0)
        moved_output = config.run.output_dir.parent/'moved'
        shutil.copytree(config.run.output_dir,moved_output)
        moved_config = replace(config,run=replace(config.run,device='cpu:0',output_dir=moved_output))
        replayed = MainlineTrainer(config=moved_config,prepared_inputs=prepared,topology=topology)
        replayed.load_state_dict(committed_state)
        replayed._run_episode(mode='online',episode_index=0)
        assert (moved_output/'episodes/online/episode_000000.npz').is_file()
        replayed.close()
        trainer.online_replay.append_episode(collected.transitions,
            total_travel_minutes=collected.total_travel_minutes,
            total_vehicle_count=collected.total_vehicle_count,
            arrival_times=collected.arrival_times)
        trainer._update_world(trainer.online_replay, initial=False)
        assert all(same(values, getattr(trainer, name)) for name, values in before.items())
        world_before = parameters(trainer.world_model)
        imagined = trainer._train_episode_imagination()
        assert same(world_before, trainer.world_model)
        assert imagined['imagination_passes_completed'] >= 1
        assert imagined['counterfactual_model_rollouts_total'] > 0
        assert imagined['imagined_event_critic_optimizer_steps_total'] > 0
        saved = deepcopy(trainer.state_dict())
        restored = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
        restored.load_state_dict(saved)
        assert same(parameters(trainer.actor), restored.actor)
        restored.close()
    finally:
        trainer.close()
    config = replace(config, run=replace(config.run,
        output_dir=config.run.output_dir.parent/'pipeline'))
    result = run_mainline(config)
    assert result['status'] == 'completed'
    assert result['online_episodes_completed'] == 1
    assert result['evaluation_episodes_completed'] == 3
    assert run_mainline(config) == result
    rows = [json.loads(line) for line in (config.run.output_dir/'metrics.jsonl').read_text().splitlines()]
    assert sum(row['phase'] == 'online_training' for row in rows) == 1
    evaluation = [row for row in rows if row['phase'] == 'final_evaluation']
    assert len(evaluation) == 3
    assert all(row['smoke'] and row['ATT'] >= 0 and row['demand_hash'] for row in evaluation)


def test_native_pretraining_initialization_runs_online_without_offline_bodies(tmp_path, monkeypatch):
    from urban_dreamer.workflow.pretrained import NativePretrained
    from urban_dreamer.workflow import pretrained
    pytest.importorskip('libsumo')
    path = make_project(tmp_path/'native_bundle')
    collect_archives(path)
    config = load_config(path)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    try:
        trainer.initialize_codec_epoch(0)
        trainer.initialize_world_model_epoch(0)
        state = deepcopy(trainer.state_dict())
    finally:
        trainer.close()
    native = NativePretrained(state,'f'*64,'toy_cross_bottleneck',{})
    monkeypatch.setattr(pretrained,'load_pretrained',lambda *args,**kwargs:native)
    def forbidden(self):
        raise AssertionError('native pretrained startup must not discover offline archives')
    monkeypatch.setattr(MainlineTrainer,'_pretraining_sources',forbidden)
    shutil.move(path.parent/'data',path.parent/'unavailable_data')
    config = replace(config,run=replace(config.run,output_dir=path.parent/'native_run'))
    result = run_mainline(config,pretrained='supplied.pt',pretrain_only=True)
    assert result['status']=='pretrained' and result['online_episodes_completed']==0
    saved = torch.load(config.run.output_dir/'checkpoints/latest.pt',weights_only=True)
    assert saved['schema_version']==10 and saved['phase']=='online_training'
    for name,weights in state['models'].items():
        for key,value in weights.items():
            assert torch.equal(value,saved['training_state']['models'][name][key])
    def no_source(*args,**kwargs):
        raise AssertionError('native checkpoint resume must not reopen the source artifact')
    monkeypatch.setattr(pretrained,'load_pretrained',no_source)
    moved = path.parent/'moved_native_run'
    shutil.move(config.run.output_dir,moved)
    config = replace(config,run=replace(config.run,output_dir=moved))
    completed = run_mainline(config)
    assert completed['status']=='completed'
    assert completed['online_episodes_completed']==1
    assert completed['evaluation_episodes_completed']==3
    assert completed['pretrained_checkpoint_sha256']=='f'*64
