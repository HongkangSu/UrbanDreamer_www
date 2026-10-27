"""Exercise native pretrained parameters using the ordinary model interfaces."""
from __future__ import annotations

import torch

from .pipeline import _build_trainer
from .pretrained import load_pretrained
from ..traffic.topology import load_traffic_topology
from ..learning.action_events import pack_action_events


def validate_pretrained_models(path, *, config, prepared):
    native = load_pretrained(path, config=config, prepared=prepared)
    topology = load_traffic_topology(prepared)
    trainer = _build_trainer(config, prepared, topology, reference=native.training_state)
    try:
        trainer.load_state_dict(native.training_state)
        matched = {}
        for name, expected in native.training_state['models'].items():
            model = getattr(trainer,name)
            actual = model.state_dict()
            if set(actual)!=set(expected):
                raise ValueError('native model tensor keys differ: '+name)
            if any(actual[key].dtype!=value.dtype or not torch.equal(actual[key].cpu(),value.cpu())
                   for key,value in expected.items()):
                raise ValueError('native model tensor values differ after strict loading: '+name)
            model.eval()
            matched[name]=dict(tensors=len(actual),strict=True,all_tensors_equal=True,
                               missing_keys=[],unexpected_keys=[])
        device=trainer.device
        edges=topology.network.edge_count
        flow=torch.zeros(1,edges,8,device=device)
        flow[...,0]=.25
        flow[...,1]=.75
        flow[...,2]=.05
        flow[...,4:]=torch.as_tensor(topology.network.static_features,device=device)
        with torch.no_grad():
            latent=trainer.world_model.encode(flow)
            decoded=trainer.world_model.decode(latent)
            actions=torch.zeros(1,edges,device=device)
            events=pack_action_events([[()]],window_starts=[[0.]]).to(device)
            generator=torch.Generator(device=device).manual_seed(2027)
            noise=torch.randn((1,trainer.world_model.dynamics.graph_prior.rank,64),
                              generator=generator,device=device)
            predicted=trainer.world_model.transition(latent,actions,noise=noise,
                steps=config.method.fm_steps,solver=config.method.fm_solver,
                picard_iterations=config.method.picard_iterations,action_events=events)
            features=torch.zeros(2,4,10,device=device)
            mask=torch.ones(2,4,dtype=torch.bool,device=device)
            policy=trainer.actor.forward_logits(features,mask)
            value=trainer.latent_critic(latent,torch.zeros(1,4,device=device))
            q=trainer.event_critic(torch.zeros(2,4,trainer.event_critic.input_dim,device=device),mask)
            if not all(bool(torch.isfinite(item).all()) for item in
                       (latent,decoded,predicted,policy,value,q)):
                raise ValueError('native pretrained forward pass produced nonfinite values')
        return dict(status='passed',city=native.city,checkpoint_schema=10,
            checkpoint_sha256=native.checkpoint_sha256,models=matched,
            forward=dict(encode=True,decode=True,transition=True,actor=True,value=True,q=True,
                         euler_steps=config.method.fm_steps,finite=True),
            codec_ready=trainer.codec_ready(),offline_archives_read=False,
            online_episodes_completed=trainer.metrics['online_episodes_completed'])
    finally:
        trainer.close()
