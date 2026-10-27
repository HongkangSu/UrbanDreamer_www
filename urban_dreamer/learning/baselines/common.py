"""Shared episode collection and global-reward updates for routing baselines."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from ..global_reward import (REWARD_CONTRACT, causal_context, observed_mean_seconds, started_count_from_state)
from ..routing_evaluation_metrics import (DiversionCost, all_demand_censored_metrics,
    DIVERSION_COST_DEFINITION, ALL_DEMAND_CENSORED_DEFINITION)
from ..observations import build_actor_observations, selected_decisions
from ..sumo_sampling import SumoEpisodeSampler, resolve_session_factory
from ...models.policy import lambda_return
from ...traffic.demand import PoissonDemand
from ...traffic.routing import validate_prepared_inputs
from ...traffic.topology import load_traffic_topology
from ...traffic.simulation_settings import physics_identity
from ...workflow.config import load_config
from ...workflow.portable import FEATURE_CONTRACT


@contextmanager
def evaluation_rng(seed, method=None):
    """Fixed evaluation draws leave the training random streams untouched."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = getattr(method, 'rng', None)
    generator_state = copy.deepcopy(generator.bit_generator.state) if isinstance(generator, np.random.Generator) else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if generator_state is not None:
            generator.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if generator_state is not None:
            generator.bit_generator.state = generator_state


def reward_objective(args):
    return 'observed_mean'


def reward_context_dim(args):
    return 4


def accounting_context(state, deadline):
    """Only revealed trip starts and actually accumulated physical time."""
    started = state.get('started_vehicle_count')
    if started is None:
        started = started_count_from_state(state, deadline=deadline)
    return causal_context(timestamp=float(state['time']), deadline=deadline,
        active_count=int(state.get('active_vehicle_count', 0)), started_count=int(started),
        cumulative_seconds=float(state.get('cumulative_vehicle_seconds', 0.)))


def resolve_collection_settings(config, args):
    """All baselines use the paper's common physical routing interface."""
    simulation = config.simulation
    if (simulation.traffic_model != 'meso' or simulation.routing_mode != 'current_to_dest'
            or simulation.session_protocol != 'event' or simulation.backend != 'libsumo'
            or simulation.observation_step_seconds != 1.):
        raise ValueError('baselines require mesoscopic current-to-destination libsumo with one-second observations')
    args.session, args.backend, args.step_seconds = 'event', 'libsumo', 1.
    return dict(physics_identity(simulation), session_protocol='event',
        observation_step_seconds=1., backend='libsumo', routing_mode='current_to_dest',
        policy_decision_interval_seconds=30.)


def build_collection_candidates(network, simulation):
    from ...traffic.route_candidates import RouteCandidates
    return RouteCandidates(network, traffic_model=simulation.traffic_model)


class EpisodeRunner:
    """SUMO episode lifecycle, event accounting and metrics; no networks."""

    def __init__(self, args, session_factory=None):
        self.args = args
        args.reward_objective, args.global_reward_only = 'observed_mean', True
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device(args.device)
        config = load_config(args.config)
        self.config = replace(config, run=replace(config.run, device=args.device))
        self.simulation_settings = resolve_collection_settings(self.config, args)
        self.session_factory = session_factory or resolve_session_factory('event', 1., 'libsumo')
        self.prepared = validate_prepared_inputs(self.config)
        self.topology = load_traffic_topology(self.prepared)
        if getattr(args, 'vehicle_limit', 0):
            self.topology = replace(self.topology, vehicles=self.topology.vehicles[:args.vehicle_limit])
        self.network = self.topology.network
        self.demand = PoissonDemand.from_vehicles(self.topology.vehicles,
            duration_seconds=self.config.simulation.demand_duration_seconds)
        self.episode_topology = self.topology
        self.n = len(self.topology.vehicles)
        self.candidates = build_collection_candidates(self.network, self.config.simulation)
        simulation = self.config.simulation
        self.deadline = 30 * math.ceil((simulation.world_model_intervals_per_episode * 30
                                       + simulation.training_completion_buffer_seconds) / 30)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(self.topology.vehicles)}

    def global_observation(self, state):
        return np.concatenate((np.asarray(state['flow'][:, :4], dtype=np.float32).reshape(-1),
                               accounting_context(state, self.deadline)))


    def run_episode(self, select, *, seed, collect=False, row_flow=True, label=None,
                    on_demand=None, archive_path=None, archive_metadata=None,
                    demand_seed=None, demand_episode_index=None, simulation_seed=None):
        """Run one episode; ``select(vehicles, actor_obs, frozen_flow, now)`` picks candidates.

        ``select`` returns either a list of candidate indices (one per pending
        vehicle) or a ``(indices, aux)`` tuple whose per-vehicle ``aux`` dicts
        are merged into the collected rows; ``aux[i]['entropy']`` feeds the
        branch-entropy statistics. ``on_demand(topology, seed=seed)`` resets recurrent
        policy state after the Poisson draw and before session reset.
        ``archive_path`` saves every real macro transition independently of
        model-specific learning records, including zero-population episodes.
        Returns ``(rows, windows, metrics)``.

        Explicit demand and simulation seeds pair physical episodes while leaving
        ``seed`` as the policy/action RNG seed.
        """
        started = time.monotonic()
        for name, value in (('demand_seed', demand_seed), ('demand_episode_index', demand_episode_index),
                            ('simulation_seed', simulation_seed)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer')
        demand_seed = self.args.seed if demand_seed is None else demand_seed
        demand_index = int(seed) if demand_episode_index is None else demand_episode_index
        simulation_seed = int(seed) if simulation_seed is None else simulation_seed
        vehicles = self.demand.sample(seed=demand_seed, episode_index=demand_index)
        self.episode_topology = replace(self.topology, vehicles=vehicles)
        self.n = len(vehicles)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(vehicles)}
        denominator = max(self.n, 1)
        demand_payload = [{'vehicle_id': v.vehicle_id, 'origin_index': v.origin_index,
                           'destination_index': v.destination_index, 'departure_second': v.departure_second}
                          for v in vehicles]
        demand_sha256 = hashlib.sha256(json.dumps(demand_payload, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if on_demand is not None:
            on_demand(self.episode_topology, seed=seed)
        config = replace(self.config, run=replace(self.config.run, seed=simulation_seed))
        torch.manual_seed(seed)
        label = label or getattr(select, 'label', 'method')
        evaluation_metrics = (not collect and label != 'stochastic'
                              and (archive_metadata or {}).get('phase') != 'train')
        diversion_cost = DiversionCost(self.candidates) if evaluation_metrics else None
        rows = []
        choices, branch_choices = Counter(), Counter()
        branch_entropies = []
        exposure, event_count = 0., 0
        previous_mean = 0.
        windows = []
        session = self.session_factory(config, self.prepared, self.episode_topology)
        with SumoEpisodeSampler(session=session, candidates=self.candidates,
                deadline_seconds=self.deadline, total_vehicle_count=self.n,
                context_mode='window', sort_pending=False, pad_empty=False,
                advance_empty=True,
                observation_builder=build_actor_observations,
                reward_objective=reward_objective(self.args)) as sampler:
            session = sampler.session
            state = sampler.state
            destinations = {str(v['vehicle_id']): int(v['destination_edge']) for v in state['vehicles']} if diversion_cost is not None else {}
            archive = None
            if archive_path is not None:
                from ..episode_archive import EpisodeArchive
                metadata = dict(archive_metadata or {})
                metadata.update(city=self.config.inputs.network_name,
                    method=getattr(self.args, 'method', label), policy=label,
                    seed=int(seed), demand_seed=int(demand_seed), demand_index=demand_index,
                    simulation_seed=simulation_seed,
                    total_vehicle_count=self.n, realized_demand=demand_payload,
                    demand_sha256=demand_sha256,
                    demand=dict(law='independent_poisson_per_od',
                        duration_seconds=self.demand.duration_seconds,
                        expected_vehicle_count=self.demand.expected_vehicle_count,
                        od_rates=[dict(origin_index=o, destination_index=d,
                                       rate_vehicles_per_second=rate)
                                  for o, d, rate in self.demand.od_rates]),
                    edge_ids=list(self.network.edge_ids),
                    config=config.resolved_snapshot(root=Path(archive_path).parent),
                    config_fingerprint=self.config.semantic_fingerprint,
                    input_hashes=self.prepared.input_hashes,
                    session=getattr(self.args, 'session', 'microstep'),
                    session_protocol=getattr(self.args, 'session', 'microstep'),
                    feature_contract=FEATURE_CONTRACT,
                    reward_objective=reward_objective(self.args),
                    reward_contract=REWARD_CONTRACT,
                    reward_mode='global_only',
                    backend=getattr(self.args, 'backend', 'traci'),
                    step_seconds=float(session.microstep_seconds))
                metadata.update(self.simulation_settings)
                archive = EpisodeArchive(metadata)
                archive.start(state)
            global_obs = row_window_flow = None
            window_time = None

            def before_window(index, start_state, context):
                nonlocal global_obs, row_window_flow
                nonlocal window_time
                global_obs = self.global_observation(start_state)
                row_window_flow = None
                window_time = float(start_state['time'])

            def decide(vehicles, actor_obs, frozen, now, window_index, context):
                nonlocal event_count, row_window_flow
                # Rows share a read-only macro snapshot, copied before session advancement.
                if collect and row_flow and vehicles and row_window_flow is None:
                    row_window_flow = np.asarray(frozen, dtype=np.float32).copy()
                selected = select(vehicles, actor_obs, frozen, now)
                if isinstance(selected, tuple):
                    indices, aux = selected
                    aux = [dict(entry) for entry in aux]
                else:
                    indices, aux = list(selected), [{} for _ in vehicles]
                indices = [int(index) for index in indices]
                if len(indices) != len(vehicles) or len(aux) != len(vehicles):
                    raise ValueError('select must return one candidate index per pending vehicle')
                for i, vehicle in enumerate(vehicles):
                    vehicle_id = vehicle['vehicle_id']
                    if not 0 <= indices[i] < actor_obs.mask.shape[1] or not actor_obs.mask[i, indices[i]]:
                        raise ValueError(f'select returned illegal candidate index {indices[i]} for {vehicle_id}')
                    edge = int(actor_obs.edge_indices[i, indices[i]])
                    pair = f"{self.network.edge_ids[vehicle['current_edge']]}->{self.network.edge_ids[edge]}"
                    choices[pair] += 1
                    if actor_obs.mask[i].sum() > 1:
                        branch_choices[pair] += 1
                        branch_entropies.append(float(aux[i].get('entropy', 0.)))
                    if collect:
                        row = dict(vehicle_id=vehicle_id, time=float(now),
                            features=actor_obs.features[i].copy(), mask=actor_obs.mask[i].copy(), action=indices[i],
                            window=window_index, current_edge=int(vehicle['current_edge']),
                            edge_indices=actor_obs.edge_indices[i].copy())
                        if getattr(actor_obs, 'routes', None) is not None:
                            row['candidate_routes'] = actor_obs.routes[i]
                        if row_flow:
                            row['flow'] = row_window_flow
                        for key, value in aux[i].items():
                            if key in row:
                                raise ValueError(f'aux field {key!r} collides with a reserved row field')
                            row[key] = value
                        rows.append(row)
                decisions = selected_decisions(vehicles, actor_obs, indices, now)
                event_count += len(decisions)
                return decisions

            for sample in sampler.windows(decide, before_window=before_window):
                result = sample.result
                if diversion_cost is not None:
                    realized = getattr(result, 'conditioning_decisions', None)
                    diversion_cost.add(result.decisions if realized is None else realized, destinations)
                if archive is not None:
                    archive.append(result)
                exposure += result.active_vehicle_seconds
                state = result.next_state
                state.setdefault('cumulative_vehicle_seconds', exposure)
                state.setdefault('started_vehicle_count', started_count_from_state(state, deadline=self.deadline))
                current_mean = observed_mean_seconds(exposure, state['started_vehicle_count'])
                reward = -(current_mean - previous_mean) / 60.
                previous_mean = current_mean
                window = dict(obs=global_obs, next_obs=self.global_observation(state),
                    reward=reward,
                    start_time=window_time, end_time=float(state['time']),
                    terminal=bool(session.done or state['time'] >= self.deadline))
                windows.append(window)
            durations = np.asarray(list(session.travel_times.values()), dtype=np.float64)
            unfinished = self.n - len(durations)
            global_reward = sum(window['reward'] for window in windows)
            metrics = dict(policy=label, seed=int(seed), vehicle_count=self.n,
                demand_seed=int(demand_seed), demand_index=demand_index, simulation_seed=simulation_seed,
                demand_sha256=demand_sha256, demand_duration_seconds=self.demand.duration_seconds,
                completion_rate=len(durations) / self.n if self.n else 1., unfinished=unfinished,
                never_departed=sum(not vehicle["departed"] for vehicle in state["vehicles"]),
                mean_travel_seconds=float(durations.mean()) if durations.size else None,
                p90_travel_seconds=float(np.percentile(durations, 90)) if durations.size else None,
                exposure_seconds=exposure, mean_exposure_seconds=previous_mean,
                observed_mean_seconds=previous_mean, cumulative_vehicle_seconds=exposure,
                reward_contract=REWARD_CONTRACT,
                started_vehicle_count=int(state.get('started_vehicle_count', 0)),
                reward_objective=reward_objective(self.args),
                simulation_end_seconds=float(state['time']), microstep_seconds=session.microstep_seconds,
                microsteps=int(session.metrics['microsteps']), event_count=event_count,
                branch_event_count=len(branch_entropies), branch_entropy=float(np.mean(branch_entropies)) if branch_entropies else 0.,
                branch_choices=dict(branch_choices), choices=dict(choices), wall_seconds=time.monotonic() - started,
                window_count=len(windows), global_reward_sum=global_reward,
                selection_score=global_reward)
            metrics.update(collection_seconds=metrics['wall_seconds'], archive_seconds=0.)
            if getattr(self.config.simulation, 'routing_mode', 'next_hop') == 'current_to_dest':
                metrics.update(routing_mode='current_to_dest', policy_decision_interval_seconds=30.,
                    p95_travel_seconds=float(np.percentile(durations, 95)) if durations.size else None)
            if diversion_cost is not None:
                metrics.update(diversion_cost.metrics(int(state['started_vehicle_count'])))
                metrics.update(diversion_cost_definition=DIVERSION_COST_DEFINITION,
                    all_demand_censored_definition=ALL_DEMAND_CENSORED_DEFINITION)
            metrics.update(all_demand_censored_metrics(vehicles, state['vehicles'],
                min(float(state['time']), self.deadline)))
            if archive is not None:
                archive_started = time.monotonic()
                archive.save(archive_path, metrics=metrics, travel_times=dict(session.travel_times),
                    completed=bool(session.done), deadline_seconds=self.deadline)
                metrics['archive_seconds'] = time.monotonic() - archive_started
        metrics['collection_seconds'] = time.monotonic() - started
        return rows, windows, metrics


class GlobalTrafficCritic(torch.nn.Module):
    """Value of the real full-network traffic state for the shared window reward."""
    def __init__(self, edge_count, context_dim=4, *, normalize_hidden=False, center_free_flow=False):
        super().__init__()
        if center_free_flow:
            # Speed / speed_limit is one on free-flow edges. Subtracting its
            # fixed baseline preserves congestion information while preventing
            # thousands of static ones from amplifying coherent Adam updates.
            offset = torch.zeros(edge_count * 4 + context_dim)
            offset[1:edge_count * 4:4] = 1.
            self.register_buffer('input_offset', offset)
        layers = []
        for input_dim in (edge_count * 4 + context_dim, 256):
            layers.append(torch.nn.Linear(input_dim, 256))
            if normalize_hidden:
                # Wide, almost constant free-flow inputs can move all hidden
                # preactivations far into Tanh saturation after an Adam step.
                layers.append(torch.nn.LayerNorm(256))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(256, 1))
        self.network = torch.nn.Sequential(*layers)
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.))
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(self.network[-1].weight, gain=1.)

    def forward(self, obs):
        if hasattr(self, 'input_offset'):
            obs = obs - self.input_offset
        return self.network(obs)


def global_reward_integral(windows, *, deadline):
    """Return cumulative team reward and episode end for asynchronous events.

    Exact window integrals are allocated uniformly inside each window because
    archives do not contain a per-second reward trace. Prefer explicit physical
    bounds from the physical collector.
    """
    if not windows:
        raise ValueError('global reward integration needs collected windows')
    starts, ends, rewards = [], [], []
    for index, window in enumerate(windows):
        start, end = window['start_time'], window['end_time']
        starts.append(float(start))
        ends.append(float(end))
        rewards.append(float(window['reward']))
    starts, ends, rewards = (np.asarray(values, dtype=np.float64)
                            for values in (starts, ends, rewards))
    if (not all(np.isfinite(values).all() for values in (starts, ends, rewards))
            or (ends <= starts).any()
            or not np.allclose(starts[1:], ends[:-1], atol=1e-3, rtol=0)):
        raise ValueError('global reward windows must be finite, positive and contiguous')
    cumulative = np.concatenate(([0.], np.cumsum(rewards)))

    def integrated(timestamp):
        timestamp = np.asarray(timestamp, dtype=np.float64)
        if (not np.isfinite(timestamp).all() or (timestamp < starts[0] - 1e-3).any()
                or (timestamp > ends[-1] + 1e-3).any()):
            raise ValueError('decision time outside collected reward windows')
        timestamp = np.clip(timestamp, starts[0], ends[-1])
        index = np.minimum(np.searchsorted(starts, timestamp, side='right') - 1, len(starts) - 1)
        fraction = (timestamp - starts[index]) / (ends[index] - starts[index])
        result = cumulative[index] + rewards[index] * fraction
        return float(result) if result.ndim == 0 else result

    return integrated, float(ends[-1])


def global_critic_update(*, global_critic, global_optimizer=None, windows, grad_clip=10.0, update=True):
    """Global-critic evaluation (and optional fit) on shared-reward windows.

    Mirrors the reference global branch: advantages are per-window lambda-return
    targets minus the pre-update value estimate. With ``update=False`` no
    optimizer step is taken (used to score imagined windows). Returns
    ``(advantages, metrics)`` with advantages detached, one per window.
    """
    if not windows:
        raise ValueError('global critic needs collected macro windows')
    device = next(global_critic.parameters()).device
    obs = torch.as_tensor(np.stack([w['obs'] for w in windows]), dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.stack([w['next_obs'] for w in windows]), dtype=torch.float32, device=device)
    rewards = torch.as_tensor([w['reward'] for w in windows], dtype=torch.float32, device=device)
    terminal = torch.tensor([w['terminal'] for w in windows], dtype=torch.bool, device=device)
    if update:
        values = global_critic(obs).squeeze(-1)
    else:
        with torch.no_grad():
            values = global_critic(obs).squeeze(-1)
    with torch.no_grad():
        next_values = global_critic(next_obs).squeeze(-1)
        targets = lambda_return(rewards, next_values, terminal=terminal)
    advantages = (targets - values.detach()).detach()
    metrics = dict(global_advantage_std=float(advantages.std(unbiased=False)))
    if update:
        if global_optimizer is None:
            raise ValueError('global critic update needs an optimizer')
        loss = F.mse_loss(values, targets)
        global_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(global_critic.parameters(), grad_clip)
        if not torch.isfinite(gradient):
            raise RuntimeError('non-finite global critic gradient')
        global_optimizer.step()
        with torch.no_grad():
            fitted = global_critic(obs).squeeze(-1)
            explained = 1 - torch.var(targets - fitted, unbiased=False) / torch.var(targets, unbiased=False).clamp_min(1e-8)
        metrics.update(global_critic_loss=float(loss.detach()), global_critic_gradient_norm=float(gradient),
                       global_explained_variance=float(explained))
    return advantages, metrics


def policy_gradient_step(*, compute_logits, actor_parameters, actor_optimizer, rows, advantages,
                         entropy_coef=0.002, grad_clip=10.0,
                         normalize_global_advantages=False, minibatch_size=None,
                         behavior_log_prob_key=None, policy_clip=0.2):
    """Actor-only update with externally supplied per-row advantages.

    The explicit pure-global mode centers/scales meaningful choices before
    optimization.
    Chunked backward passes accumulate one sample-weighted full-batch update.
    """
    if not rows:
        return dict(actor_loss=0., actor_gradient_norm=0., advantage_std=0.)
    if behavior_log_prob_key is not None and (not math.isfinite(policy_clip) or not 0. < policy_clip < 1.):
        raise ValueError('policy_clip must be finite and between zero and one')
    if minibatch_size is not None and (isinstance(minibatch_size, bool)
            or not isinstance(minibatch_size, (int, np.integer)) or minibatch_size < 1):
        raise ValueError('actor minibatch_size must be a positive integer')
    actor_parameters = list(actor_parameters)
    device = actor_parameters[0].device
    if isinstance(advantages, torch.Tensor):
        # GNAC already supplies device-resident advantages; avoid one host read per event.
        advantage = advantages.detach().to(device=device, dtype=torch.float32)
    else:
        advantage = torch.as_tensor(np.asarray([float(a) for a in advantages], dtype=np.float32), device=device)
    if advantage.shape != (len(rows),) or not bool(torch.isfinite(advantage).all()):
        raise ValueError('advantages must be finite with one value per row')
    # Forced moves provide critic data but have no policy choice. Including them
    # dilutes decision gradients; a zero-gradient Adam step can still move its
    # parameters through momentum retained from the previous episode.
    choices = [index for index, row in enumerate(rows)
               if 'mask' not in row or np.count_nonzero(row['mask']) > 1]
    if not choices:
        actor_optimizer.zero_grad(set_to_none=True)
        return dict(actor_loss=0., actor_gradient_norm=0., advantage_std=0.)
    if len(choices) != len(rows):
        rows = [rows[index] for index in choices]
        advantage = advantage[choices]
    raw_mean, raw_std = float(advantage.mean()), float(advantage.std(unbiased=False))
    if normalize_global_advantages:
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
        else:
            # A singleton cannot estimate a batch baseline. Retain its critic
            # signal, scaling only unusually large magnitudes.
            advantage = advantage / advantage.abs().clamp_min(1.)
    if entropy_coef == 0. and not bool(advantage.any()):
        actor_optimizer.zero_grad(set_to_none=True)
        return dict(actor_loss=0., actor_gradient_norm=0., advantage_std=0.)
    actor_optimizer.zero_grad(set_to_none=True)
    count = len(rows)
    chunk_size = count if minibatch_size is None else int(minibatch_size)
    actor_loss = 0.
    for start in range(0, count, chunk_size):
        batch = rows[start:start + chunk_size]
        actions = torch.tensor([row['action'] for row in batch], device=device)
        distribution = Categorical(logits=compute_logits(batch))
        log_prob = distribution.log_prob(actions)
        batch_advantage = advantage[start:start + len(batch)]
        if behavior_log_prob_key is None:
            surrogate = log_prob * batch_advantage
        else:
            old = torch.tensor([row[behavior_log_prob_key] for row in batch], device=device)
            if not bool(torch.isfinite(old).all()):
                raise ValueError('behavior log probabilities must be finite')
            log_ratio = log_prob - old
            # Select the clipped branch before exponentiation to keep extreme
            # stale probabilities from overflowing an unused surrogate branch.
            log_ratio = torch.where(batch_advantage >= 0.,
                log_ratio.clamp_max(math.log1p(policy_clip)),
                log_ratio.clamp_min(math.log1p(-policy_clip)))
            # Extremely stale behavior can have finite log probabilities but
            # an overflowing ratio. Bound only this numerical tail; ordinary
            # PPO ratios and their gradients are unchanged.
            log_ratio = log_ratio.clamp_max(20.)
            surrogate = log_ratio.exp() * batch_advantage
        loss = -(surrogate + entropy_coef * distribution.entropy()).sum() / count
        loss.backward()
        actor_loss += float(loss.detach())
    actor_gradient = torch.nn.utils.clip_grad_norm_(actor_parameters, grad_clip)
    if not torch.isfinite(actor_gradient):
        raise RuntimeError('non-finite actor gradient')
    actor_optimizer.step()
    return dict(actor_loss=actor_loss, actor_gradient_norm=float(actor_gradient),
                advantage_std=float(advantage.std(unbiased=False)),
                raw_global_advantage_mean=raw_mean, raw_global_advantage_std=raw_std,
                actor_decision_count=count)
