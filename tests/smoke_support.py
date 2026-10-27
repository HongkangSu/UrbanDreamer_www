"""Temporary small projects for real SUMO integration tests."""
from pathlib import Path
import json
import shutil
import tomllib

ROOT=Path(__file__).resolve().parents[1]


def write_toml(path, config):
    text=''
    for section, fields in config.items():
        text+='['+section+']\n'
        text+=''.join(key+' = '+json.dumps(value)+'\n' for key,value in fields.items())+'\n'
    path.write_text(text)
    return path


def make_project(root):
    root.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'tests/fixtures/toy_cross_bottleneck',root/'network')
    cfg=tomllib.loads((ROOT/'configs/coulommiers.toml').read_text())
    cfg['inputs']=dict(network_name='toy_cross_bottleneck',network_dir='network')
    cfg['run'].update(output_dir='outputs/mainline',device='cpu',world_model_initialization_epochs=1,
        online_training_episodes=1,evaluation_episodes=3)
    cfg['simulation'].update(demand_duration_seconds=60.,world_model_intervals_per_episode=6,
        training_completion_buffer_seconds=0,evaluation_horizon_seconds=180,incident_edge='A_in')
    cfg['resources'].update(world_model_gradient_accumulation_steps=1,online_world_model_batch_size=1,
        world_model_initialization_batch_size=1,world_model_pretrain_activation_checkpointing=False)
    cfg['data'].update(archive_roots=['data/offline'],validation_archive_roots=[])
    cfg['codec'].update(max_updates=32,validation_every=8,occupied_vehicle_mae_max=5.,vehicle_mae_max=5.,
        empty_vehicle_mae_max=5.,total_relative_mae_max=10.,speed_mae_mps_max=50.,
        occupied_speed_mae_mps_max=50.,slow_speed_mae_mps_max=50.,slow_validation_min_roads=0)
    return write_toml(root/'smoke.toml',cfg)
