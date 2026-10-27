from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.run_baseline import CHECKPOINT_FORMAT, load_evaluation_checkpoint, physical_protocol
from urban_dreamer.learning.training import archive_configuration
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.portable import FEATURE_CONTRACT


def test_archive_provenance_is_independent_of_mount_device_and_output(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1]/'configs/coulommiers.toml')
    moved = replace(config, run=replace(config.run,device='cpu:0',output_dir=tmp_path/'run'),
        inputs=replace(config.inputs,network_dir=tmp_path/'network'),
        data=replace(config.data,archive_roots=(tmp_path/'data',)))
    assert archive_configuration(config) == archive_configuration(moved)
    provenance=archive_configuration(moved)
    assert 'device' not in provenance['run'] and 'output_dir' not in provenance['run']
    assert 'data' not in provenance and 'network_dir' not in provenance['inputs']


def checkpoint_case(tmp_path, *, smoke):
    runner = SimpleNamespace(prepared=SimpleNamespace(input_hashes={'network':'hash'}),
        simulation_settings={'traffic_model':'meso'}, deadline=180. if smoke else 3810.,
        demand=SimpleNamespace(duration_seconds=60. if smoke else 1800.))
    state = dict(format=CHECKPOINT_FORMAT,method='mappo',feature_contract=FEATURE_CONTRACT,
        input_hashes=runner.prepared.input_hashes,smoke=smoke,training_seed=43,
        completed_training_episodes=1 if smoke else 1000,
        physical_protocol=physical_protocol(runner),learner={})
    path=tmp_path/'final.pt';torch.save(state,path)
    return path,runner,state


def test_smoke_checkpoint_cannot_be_relabelled_as_formal(tmp_path):
    path,runner,_=checkpoint_case(tmp_path,smoke=True)
    with pytest.raises(ValueError,match='smoke'):
        load_evaluation_checkpoint(path,'mappo',None,runner,smoke=False)


def test_checkpoint_evaluation_preserves_source_training_provenance(tmp_path):
    path,runner,state=checkpoint_case(tmp_path,smoke=False)
    learner=SimpleNamespace(load_state_dict=lambda value:None)
    result=load_evaluation_checkpoint(path,'mappo',learner,runner)
    assert result==dict(training_seed=43,completed_training_episodes=1000)
    state['physical_protocol']['deadline_seconds']=180.
    torch.save(state,path)
    with pytest.raises(ValueError,match='physical protocol'):
        load_evaluation_checkpoint(path,'mappo',learner,runner)
