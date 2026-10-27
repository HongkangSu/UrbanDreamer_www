"""Lock native input, codec and recursive-cohort semantics during cleanup."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from urban_dreamer.traffic.topology import EdgeNetwork
from urban_dreamer.learning.codec_training import codec_decoded_metrics
from urban_dreamer.learning import training
from urban_dreamer.learning.observations import build_actor_observations


def test_native_occupancy_retains_original_ratio_normalization():
    network=SimpleNamespace(edge_count=2,edge_ids=('a','b'),speed_limits_mps=np.ones(2)*10,
        capacities=np.ones(2)*5,static_features=np.ones((2,4),dtype=np.float32))
    flow=EdgeNetwork.extract_flow(network,{'a':{'occupancy':2.5,'vehicle_number':1,'mean_speed':4},
        'b':{'occupancy':.2,'vehicle_number':0,'mean_speed':10}})
    np.testing.assert_allclose(flow[:,0],[.025,.2])


def test_native_codec_preserves_forward_boundary_mean_speed():
    flow=torch.tensor([[[1.,.2,.5,0.,1.,1.,1.,1.]]])
    following=flow.clone();following[...,1]=.8
    network=SimpleNamespace(capacities=np.array([2.]),speed_limits_mps=np.array([10.]))
    metrics=codec_decoded_metrics(flow[...,:4],flow,following,network)
    assert float(metrics['codec_speed_mae_mps'])==pytest.approx(3.)


def test_original_actor_uses_the_original_global_context_features():
    network=SimpleNamespace(edge_count=4,lengths_m=np.ones(4)*10,free_flow_travel_seconds=np.ones(4))
    candidates=SimpleNamespace(network=network,routing_mode='current_to_dest',
        generate_paths=lambda *args:((0,1,2),(0,2),(0,1,2),(0,2)))
    vehicles=[dict(vehicle_id='v',current_edge=0,destination_edge=2,remaining_distance=5.)]
    flow=np.ones((4,2),dtype=np.float32)
    first=build_actor_observations(policy_edge_features=flow,vehicle_states=vehicles,candidates=candidates,
        episode_progress=.2,active_vehicle_fraction=.1)
    flow[3]=[999.,0.]
    second=build_actor_observations(policy_edge_features=flow,vehicle_states=vehicles,candidates=candidates,
        episode_progress=.2,active_vehicle_fraction=.9)
    assert not np.array_equal(first.features,second.features)
    assert first.features[0,0,9]==pytest.approx(.1)
    assert second.features[0,0,9]==pytest.approx(.9)


def test_recursive_training_retains_original_cohort_targets_and_controls(monkeypatch):
    trainer=training.MainlineTrainer.__new__(training.MainlineTrainer)
    trainer.device=torch.device('cpu');trainer.edge_count=2;trainer.metrics={'world_model_updates':0}
    trainer.validate_codec_ready=lambda:None
    trainer.network=SimpleNamespace(capacities=np.ones(2))
    trainer.config=SimpleNamespace(method=SimpleNamespace(fm_training_mode='recursive_3step',fm_steps=10,
        fm_solver='euler',picard_iterations=2,prior_loss_weight=.1,exposure_readout_supervision='observed_endpoint'),
        resources=SimpleNamespace())
    trainer.world_model=SimpleNamespace(encode=lambda f:torch.nn.functional.pad(f,(0,56)),
        decode=lambda z:z[...,:4],dynamics=SimpleNamespace(),
        active_vehicle_fraction_readout=lambda z,**kwargs:z[...,2].mean(-1))
    observed=torch.zeros(1,4,2,8);observed[:,1:,0,2]=torch.tensor([2.,2.,1.])
    cohort=observed.clone();cohort[:,1:,0,2]=torch.tensor([1.,1.,0.])
    event=SimpleNamespace();event.to=lambda device:event
    scoped=SimpleNamespace(transition_flow=cohort,transition_action=torch.zeros(1,3,2),
        transition_action_events=event,transition_active_vehicle_fraction=torch.ones(1,3))
    batch=SimpleNamespace(flow=observed,action=torch.ones(1,3,2))
    monkeypatch.setattr(training,'prepare_cohort_batch',lambda *args,**kwargs:scoped)
    def fm(dynamics,**kwargs):
        torch.testing.assert_close(kwargs['z_sequence'][...,:8],cohort)
        torch.testing.assert_close(kwargs['action'],scoped.transition_action)
        assert kwargs['action_events'] is event
        return torch.tensor(0.,requires_grad=True),{},kwargs['z_sequence'][:,1:]
    monkeypatch.setattr(training,'recursive_fm_loss',fm)
    trainer._world_loss(batch)


def test_offline_pretraining_uses_one_step_loss_only(monkeypatch):
    trainer=training.MainlineTrainer.__new__(training.MainlineTrainer)
    trainer.device=torch.device('cpu');trainer.edge_count=2;trainer.metrics={'world_model_updates':0}
    trainer.validate_codec_ready=lambda:None
    trainer.network=SimpleNamespace(capacities=np.ones(2))
    trainer.config=SimpleNamespace(method=SimpleNamespace(fm_training_mode='recursive_3step',fm_steps=10,
        fm_solver='euler',picard_iterations=2,prior_loss_weight=.1,exposure_readout_supervision='observed_endpoint'),
        resources=SimpleNamespace())
    trainer.world_model=SimpleNamespace(encode=lambda f:torch.nn.functional.pad(f,(0,56)),
        decode=lambda z:z[...,:4],dynamics=SimpleNamespace(),
        active_vehicle_fraction_readout=lambda z,**kwargs:z[...,2].mean(-1))
    flow=torch.randn(1,4,2,8)
    target=flow[:,1:].clone();target[...,2]=0.
    events=SimpleNamespace();events.to=lambda device:events
    batch=SimpleNamespace(flow=flow,action=torch.zeros(1,3,2),action_events=events,
        active_vehicle_fraction=torch.zeros(1,3))
    monkeypatch.setattr(training,'prepare_transition_targets',
        lambda *args,**kwargs:SimpleNamespace(transition_target_flow=target))
    def forbidden(*args,**kwargs):
        raise AssertionError('offline pretraining must not call recursive supervision')
    monkeypatch.setattr(training,'prepare_cohort_batch',forbidden)
    monkeypatch.setattr(training,'recursive_fm_loss',forbidden)
    calls=[]
    def independent(dynamics,**kwargs):
        calls.append(True)
        assert kwargs['z_inputs'].shape[1]==kwargs['z_targets'].shape[1]==1
        torch.testing.assert_close(kwargs['z_inputs'][...,:8],flow[:,:1])
        torch.testing.assert_close(kwargs['z_targets'][...,:8],target[:,:1])
        return torch.tensor(0.,requires_grad=True),{},kwargs['z_targets']
    monkeypatch.setattr(training,'independent_transition_fm_loss',independent)
    trainer._world_loss(batch,initial=True)
    assert calls==[True]
    assert trainer.metrics['world_model_supervision_depth_last']==1.
