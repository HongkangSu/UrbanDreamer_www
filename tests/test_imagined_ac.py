"""Regression coverage for the native active imagined actor--critic path."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from urban_dreamer.learning import counterfactual_q
from urban_dreamer.learning.counterfactual_q import (
    advantage_batch, centered_action_values, collect_route_values, expected_action_loss,
    fit_action_advantage,
)
from urban_dreamer.learning.counterfactual_value import fit_counterfactual_value
from urban_dreamer.models.event_critic import EventActionCritic
from urban_dreamer.models.policy import CentralizedLatentValueCritic, NextHopActor
from urban_dreamer.models.world_model import EventActionEncoder
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.topology import EdgeNetwork


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def vehicle(vehicle_id="active", **updates):
    state = dict(vehicle_id=vehicle_id, current_edge=0, destination_edge=4,
        departure_time=0., departed=True, arrived=False, remaining_distance=10.,
        committed_next_edge=1, committed_route=(0, 1, 2, 4), pending_decision=False,
        in_junction=False, junction_remaining_distance=0., junction_speed=1.)
    state.update(updates)
    return state


class ToyWorld(nn.Module):
    def __init__(self, network, *, noise_scale=0., destination_speed=1.):
        super().__init__()
        self.noise_scale = noise_scale
        self.destination_speed = destination_speed
        self.encode_calls = 0
        self.noises = []
        self.outputs = []
        self.dynamics = SimpleNamespace(graph_prior=SimpleNamespace(rank=5),
            event_encoder=EventActionEncoder(adjacency=torch.as_tensor(network.adjacency),
                static_features=torch.as_tensor(network.static_features)))

    def encode(self, flow):
        self.encode_calls += 1
        return torch.zeros(len(flow), 5, 64)

    def decode(self, latent):
        features = torch.zeros(len(latent), 5, 4)
        features[..., 1] = 1.
        features[:, 4, 1] = self.destination_speed
        return features

    def transition(self, latent, action, *, noise, **kwargs):
        self.noises.append(noise.clone())
        successor = latent + self.noise_scale * noise
        self.outputs.append(successor.clone())
        return successor


class ConstantValue(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, latent, context):
        return latent.new_full((len(latent), 1), self.value)


def case(*, deadline=60., noise_scale=0., destination_speed=1., destination_length=400.):
    torch.manual_seed(312)
    lengths = np.array([10., 10., 10., 20., destination_length], dtype=np.float32)
    graph = np.zeros((5, 5), dtype=np.float32)
    adjacency = ((1,), (2, 3), (4,), (4,), ())
    for current, successors in enumerate(adjacency):
        graph[current, list(successors)] = 1.
    network = EdgeNetwork(tuple("abcde"), dict(zip("abcde", range(5))),
        np.ones((5, 4), dtype=np.float32), np.full(5, 10., dtype=np.float32),
        np.ones(5, dtype=np.float32), lengths, graph, adjacency,
        np.ones(5, dtype=bool), lengths / 10.)
    world = ToyWorld(network, noise_scale=noise_scale, destination_speed=destination_speed)
    states = [vehicle(), vehicle("background", current_edge=4,
        remaining_distance=destination_length, committed_route=(4,), committed_next_edge=None)]
    method = SimpleNamespace(actor_objective="counterfactual_actor_critic_recoverq_v1",
        real_policy_update="none", disagreement_samples=2, disagreement_weight=0.,
        imagination_seed_scope="full_episode_v1", imagination_horizon_windows=8,
        fm_steps=10, fm_solver="euler", picard_iterations=1)
    trainer = SimpleNamespace(world_model=world, actor=NextHopActor(),
        device=torch.device("cpu"), network=network,
        candidates=RouteCandidates(network, traffic_model="meso"),
        _episode_end_time=lambda: deadline,
        config=SimpleNamespace(method=method, simulation=SimpleNamespace(
            demand_duration_seconds=0., vehicle_max_speed_mps=20., traffic_model="meso")))
    batch = SimpleNamespace(flow=torch.zeros(1, 1, 5, 8), time=torch.tensor([0.]),
        started_count=torch.tensor([2]), total_vehicle_count=torch.tensor([2]),
        cumulative_vehicle_seconds=torch.tensor([0.]), vehicles=(states,))
    return trainer, batch



def test_native_route_returns_and_duplicate_randomness():
    trainer, batch = case()
    before = deepcopy(batch.vehicles)
    row = collect_route_values(trainer, batch, policy_seed=19, model_seed=29, max_windows=2)
    assert row['terminal_only'] and not row['bootstrap_used']
    assert row['duplicate_error'] == row['alias_error'] == 0.
    assert batch.vehicles == before
    for ordinal, action in enumerate(torch.where(row['mask'])[0]):
        assert float(row['returns'][action]) == pytest.approx(-row['arm_exposures'][ordinal] / 120.)
    assert all(torch.equal(noise, noise[:1].expand_as(noise)) for noise in trainer.world_model.noises)


def test_native_h8_bootstrap_and_value_clamp():
    trainer, batch = case(deadline=600., destination_speed=.001)
    row = collect_route_values(trainer, batch, policy_seed=19, model_seed=29,
        max_windows=8, bootstrap_value=ConstantValue(-2.))
    assert row['bootstrap_used'] and not row['terminal_only']
    assert row['bootstrap_tails'] == [-2.] * 5
    assert row['windows_per_arm'] == [8] * 5
    assert len(row['value_rows']) == 32
    for ordinal, action in enumerate(torch.where(row['mask'])[0]):
        assert float(row['returns'][action]) == pytest.approx(-row['arm_exposures'][ordinal] / 120. - 2.)
    positive = collect_route_values(trainer, batch, policy_seed=19, model_seed=29,
        max_windows=8, bootstrap_value=ConstantValue(25.))
    assert positive['bootstrap_tails'] == [0.] * 5


def test_native_q_target_population_scaling_is_retained():
    row = dict(q_features=torch.zeros(2, 3), mask=torch.tensor([True, True]),
        probabilities=torch.tensor([.25, .75]), returns=torch.tensor([-1., -3.]), population=2)
    first = advantage_batch([row], 'cpu')[-1]
    row['population'] = 2000
    second = advantage_batch([row], 'cpu')[-1]
    torch.testing.assert_close(second, first * 1000.)


def test_native_future_birth_compensation_and_fixed_return_denominator(monkeypatch):
    trainer, batch = case()
    batch.cumulative_vehicle_seconds[:] = 120.
    future = vehicle('future', current_edge=4, departure_time=15., departed=False,
        remaining_distance=400., committed_route=(4,), committed_next_edge=None)
    monkeypatch.setattr('urban_dreamer.learning.imagined_demand.sample_future_states',
        lambda *args, **kwargs: ([deepcopy(future)], 123))
    row = collect_route_values(trainer, batch, policy_seed=19, model_seed=29,
        max_windows=2, bootstrap_value=ConstantValue(0.))
    assert row['population'] == 3 and row['future_departure_count'] == 1
    assert trainer.world_model.encode_calls > 1
    for ordinal, action in enumerate(torch.where(row['mask'])[0]):
        assert float(row['returns'][action]) == pytest.approx(-row['arm_exposures'][ordinal] / 180.)


def test_actor_expected_gradient_and_critic_targets_are_detached():
    logits = torch.zeros(1, 3, requires_grad=True)
    values = torch.tensor([[1., -1., 1e6]], requires_grad=True)
    expected_action_loss(logits, values, torch.tensor([[True, True, False]])).backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0 and logits.grad[0, 2] == 0.
    assert values.grad is None
    trainer, batch = case(deadline=600., destination_speed=.001)
    row = collect_route_values(trainer, batch, policy_seed=19, model_seed=29,
        max_windows=8, bootstrap_value=ConstantValue(-2.))
    trainer.latent_critic = CentralizedLatentValueCritic()
    trainer.event_critic = EventActionCritic()
    trainer.optimizers = {'latent_critic': torch.optim.Adam(trainer.latent_critic.parameters(), lr=3e-4)}
    before = deepcopy(trainer.actor.state_dict())
    fit = fit_counterfactual_value(trainer, [row], epochs=8)
    assert fit['mse_after'] < fit['mse_before']
    assert all(torch.equal(v, trainer.actor.state_dict()[k]) for k, v in before.items())


def test_native_validation_keeps_physical_only_scores(monkeypatch):
    from urban_dreamer.learning import counterfactual_validation as validation
    from dataclasses import dataclass
    from urban_dreamer.workflow.config import MethodConfig
    @dataclass
    class Config:
        method: object
    trainer = SimpleNamespace(actor=nn.Linear(1, 1), config=Config(MethodConfig()))
    original = deepcopy(trainer.actor.state_dict())
    def collect(*args, **kwargs):
        assert kwargs['return_tracks'] and kwargs['reward_only']
        assert len(kwargs['policy_plan']['policies']) == 4
        def track(exposure):
            return dict(population=2, windows=[dict(start=60., terminal=False,
                next_value=torch.tensor(1e12), risk_cost=torch.tensor(1e12),
                output={'result': SimpleNamespace(active_vehicle_seconds=exposure)})])
        return [track(100.), track(98.), track(100.), track(98.)], 3810.
    monkeypatch.setattr(validation, 'collect_parallel_imagination', collect)
    result = validation.validate_actor_in_model(trainer, original)
    assert result['accepted'] and not result['bootstrap_used']
    assert result['pairs'][0]['delta'] == pytest.approx(2. / 120.)
    assert all(torch.equal(v, trainer.actor.state_dict()[k]) for k, v in original.items())
