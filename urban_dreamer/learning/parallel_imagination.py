"""Persistent CPU event windows with centralized, batched differentiable learning.

Workers never own CUDA objects. Each task has a private rollout RNG state, so
scheduling cannot change sampling; all policy/critic parameters stay fixed for
an entire collection pass. The parent alone creates backward graphs.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np
import torch
from .action_events import pack_action_events
from .imagination import advance_imagined_window, clip_road_speed_fractions
from .observations import build_actor_observations, prepare_actor_window, selected_decisions
from ..models.event_critic import EventActionCritic
from ..models.policy import CANDIDATE_FEATURE_DIM, NextHopActor, set_shortest_path_prior
from ..models.world_model import EventActionEncoder
from .global_reward import causal_context
from .policy_temperature import NONE, profile_state, reset_actor_temperature, require_actor_temperature
_WORKER = None

def _numpy_state(module):
    return {name: value.detach().cpu().numpy().copy() for name, value in module.state_dict().items()}

def _load_numpy(module, state):
    module.load_state_dict({name: torch.from_numpy(value) for name, value in state.items()}, strict=True)

def _feature_biased_logits(logits, features, mask, feature_index, strength):
    """Apply one shared candidate-feature direction, excluding padding first."""
    if not isinstance(feature_index, int) or isinstance(feature_index, bool) or (not 0 <= feature_index < CANDIDATE_FEATURE_DIM):
        raise ValueError('policy feature_index is outside the candidate features')
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or (not math.isfinite(strength)):
        raise ValueError('policy strength must be a finite number')
    costs = features[..., feature_index].masked_fill(~mask, 0.0)
    if not bool(torch.isfinite(costs).all()):
        raise ValueError('legal policy features must be finite')
    center = costs.sum(-1, keepdim=True) / mask.sum(-1, keepdim=True)
    return (logits + strength * (costs - center)).masked_fill(~mask, -torch.inf)

def _policy_uniform(seed, vehicle_id, timestamp, current_edge, destination_edge, occurrence):
    """Couple a vehicle's fork visit even if motion shifts its observation time.

    ``timestamp`` remains accepted for diagnostic callers, but never enters the
    key: one decoded-speed ULP can move a visit across a one-second boundary.
    """
    key = json.dumps([int(seed), str(vehicle_id), int(current_edge), int(destination_edge), int(occurrence)], separators=(',', ':'))
    bits = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') >> 11
    return bits / 2 ** 53

def _validated_policies(plan):
    policies = plan.get('policies') if isinstance(plan, dict) else None
    if not isinstance(policies, (list, tuple)) or len(policies) != 4:
        raise ValueError('shared policy plan requires exactly four policies')
    labels, result = (set(), [])
    for policy in policies:
        if not isinstance(policy, dict) or not isinstance(policy.get('label'), str) or (not policy['label']) or (policy['label'] in labels):
            raise ValueError('shared policy labels must be unique nonempty strings')
        axis, strength = (policy.get('feature_index'), policy.get('strength'))
        if not isinstance(axis, int) or isinstance(axis, bool) or (not 0 <= axis < CANDIDATE_FEATURE_DIM):
            raise ValueError('policy feature_index is outside the candidate features')
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or (not math.isfinite(strength)):
            raise ValueError('policy strength must be a finite number')
        state = policy.get('actor_state')
        if not isinstance(state, dict) or not all((isinstance(value, np.ndarray) for value in state.values())):
            raise ValueError('policy actor_state must be a NumPy parameter snapshot')
        labels.add(policy['label'])
        result.append(dict(label=policy['label'], feature_index=axis, strength=float(strength), actor_state={name: value.copy() for name, value in state.items()}))
    return result

def _seed_identity(start, population, states, cumulative_seconds):

    def scalar(value):
        return value.tolist() if hasattr(value, 'tolist') else value.item()
    payload = dict(time=start, population=population, states=states, cumulative_vehicle_seconds=cumulative_seconds)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=scalar).encode()).hexdigest()

def _initialize_worker(candidates, adjacency, static):
    global _WORKER
    torch.set_num_threads(1)
    actor = NextHopActor().eval().requires_grad_(False)
    critic = EventActionCritic().eval().requires_grad_(False)
    encoder = EventActionEncoder(adjacency=torch.from_numpy(adjacency), static_features=torch.from_numpy(static)).eval().requires_grad_(False)
    _WORKER = SimpleNamespace(candidates=candidates, actor=actor, critic=critic, world=SimpleNamespace(dynamics=SimpleNamespace(event_encoder=encoder)), version=None)

def _context(states, timestamp, deadline, population, cumulative_seconds):
    active = sum((bool(state['departed']) and (not bool(state['arrived'])) for state in states))
    return causal_context(timestamp, deadline, active, population, cumulative_seconds)

@torch.no_grad()
def _collect_window(task):
    worker = _WORKER
    weights = task['weights']
    reward_only = task.get('reward_only', False)
    temperature_state = weights.get('training_temperature_state')
    version = task['version']
    if 'training_temperature_episode' in task or 'training_temperature_profile' in task:
        raise ValueError('worker training temperature task requires explicit policy metadata')
    version = (version, reward_only)
    if worker.version != version:
        _load_numpy(worker.actor, weights['actor'])
        if 'score_gauge' in weights:
            worker.actor.score_gauge = weights['score_gauge']
        set_shortest_path_prior(worker.actor, *weights['prior'])
        reset_actor_temperature(worker.actor)
        worker.version = version
    states = task['states']
    generator = torch.Generator(device='cpu')
    generator.set_state(torch.from_numpy(task['rng']))
    decoded, population, deadline = (task['decoded'], task['population'], task['deadline'])
    cumulative_seconds = task['cumulative_vehicle_seconds']
    network = worker.candidates.network
    window = prepare_actor_window(np.asarray(decoded[:, [2, 1]], dtype=np.float32), network.edge_count)
    groups, observations, masks, edges, contexts, probabilities, selections = ([], [], [], [], [], [], [])
    behavior_logits, behavior_log_probs = ([], [])
    event_count = task['event_count']
    prior_cost_rows = []
    policy_occurrences = dict(task.get('policy_occurrences', {}))

    def decide(pending, frozen_flow, now):
        nonlocal event_count
        event_cumulative = cumulative_seconds + sum((max(0.0, min(float(state.get('arrival_time', now)), now) - task['start']) for state in states))
        context = _context(states, now, deadline, population, event_cumulative)
        prior_kwargs = dict(prior_cost_mode='dynamic', prior_speed_floor_mps=task['prior_speed_floor_mps'], prior_speed_cap_mps=task['max_speed']) if task.get('prior_cost_mode') == 'dynamic' else {}
        observation = build_actor_observations(policy_edge_features=window, vehicle_states=pending, candidates=worker.candidates, episode_progress=float(context[0]), active_vehicle_fraction=float(context[1]), **prior_kwargs)
        features, mask = (torch.from_numpy(observation.features), torch.from_numpy(observation.mask))
        prior_costs = getattr(observation, 'prior_costs_seconds', None)
        actor_kwargs = {'prior_costs_seconds': torch.as_tensor(prior_costs, dtype=torch.float32)} if prior_costs is not None else {}
        logits = worker.actor.forward_logits(features, mask, **actor_kwargs)
        if 'policy_seed' in task:
            logits = _feature_biased_logits(logits, features, mask, task['feature_index'], task['strength'])
        if not bool(torch.isfinite(logits[mask]).all()):
            raise ValueError('legal actor logits must be finite')
        probs = logits.softmax(-1)
        if 'policy_seed' in task:
            choices = []
            for row, state in enumerate(pending):
                key = json.dumps([str(state['vehicle_id']), int(state['current_edge']), int(state['destination_edge'])], separators=(',', ':'))
                occurrence = policy_occurrences.get(key, 0)
                policy_occurrences[key] = occurrence + 1
                uniform = _policy_uniform(task['policy_seed'], state['vehicle_id'], now, state['current_edge'], state['destination_edge'], occurrence)
                cdf = np.cumsum(probs[row].numpy(), dtype=np.float64)
                choice = min(int(np.searchsorted(cdf, uniform, side='right')), int(torch.nonzero(mask[row], as_tuple=True)[0][-1]))
                choices.append(choice)
            selected = torch.tensor(choices, dtype=torch.long)
        else:
            selected = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        choices = selected.tolist()
        decisions = selected_decisions(pending, observation, choices, now)
        event_count += len(pending)
        return decisions
    speeds = np.minimum(clip_road_speed_fractions(decoded[:, 1], task.get('traffic_model', 'micro')) * network.speed_limits_mps, task['max_speed'])
    result = advance_imagined_window(states, candidates=worker.candidates, edge_speeds_mps=speeds, frozen_flow=decoded, start_time=task['start'], interval_seconds=min(30.0, deadline - task['start']), total_vehicle_count=task['total_vehicle_count'], decide=decide, observation_step_seconds=1.0, session_protocol='event', traffic_model=task.get('traffic_model', 'micro'), event_index_start=task['event_count'], vehicle_max_speed_mps=task['max_speed'], stop_when_empty=True)
    output = dict(states=states, result=result, groups=groups, event_count=event_count, rng=generator.get_state().numpy().copy(), context=_context(states, result.end_time, deadline, population, cumulative_seconds + result.active_vehicle_seconds))
    if result.action_events:
        output['event_count'] = max(event_count, result.action_events[-1].event_index + 1)
    if 'policy_seed' in task:
        output['policy_occurrences'] = policy_occurrences
    return output

def close_parallel_imagination(trainer):
    """Release workers at trainer shutdown or after a failed collection pass."""
    executor = getattr(trainer, '_parallel_imagination_executor', None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
        del trainer._parallel_imagination_executor

def _executor(trainer, workers):
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('parallel imagination workers must be a positive integer')
    if getattr(trainer, '_parallel_imagination_executor', None) is not None and trainer._parallel_imagination_workers != workers:
        close_parallel_imagination(trainer)
    if getattr(trainer, '_parallel_imagination_executor', None) is None:
        encoder = trainer.world_model.dynamics.event_encoder
        adjacency = np.zeros((encoder.n_edges, encoder.n_edges), dtype=np.float32)
        turns = encoder.turns.detach().cpu().numpy()
        adjacency[turns[:, 0], turns[:, 1]] = 1.0
        trainer._parallel_imagination_executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_initialize_worker, initargs=(trainer.candidates, adjacency, encoder.static_features.detach().cpu().numpy()))
        trainer._parallel_imagination_workers = workers
    return trainer._parallel_imagination_executor

def collect_parallel_imagination(trainer, workers=4, *, policy_plan=None, return_tracks=False, reward_only=False):
    """Return event groups and per-rollout V losses; never take optimizer steps.

    Parent-generated rollout seeds are part of the existing checkpointed CPU
    RNG stream. Independent generators make CPU tasks scheduling-independent.
    Explicit per-rollout WM noise also makes active-batch packing irrelevant.
    Worker parameter payloads are NumPy-only and cached by pass version; no
    tensors or autograd graphs cross a process boundary.

    ``reward_only`` returns physical tracks for validation while omitting
    unused Critic features and value predictions. Model solves and RNG streams
    are identical to full collection; these tracks cannot train the old loss.
    """
    if policy_plan is None or not return_tracks or (not reward_only):
        raise ValueError('parallel collection is reserved for paired model validation')
    if not isinstance(reward_only, bool) or (reward_only and (not return_tracks)):
        raise ValueError('reward-only imagination requires boolean mode and return_tracks=True')
    from .training import resolved_imagination_horizon, imagination_seed_kwargs
    horizon = resolved_imagination_horizon(trainer.config.method)
    require_actor_temperature(trainer.actor, profile_state(NONE, 0))
    policies = _validated_policies(policy_plan) if policy_plan is not None else None
    rollout_count = len(policies) if policies is not None else trainer.config.resources.imagination_rollouts_per_pass
    paired = policies is None and False
    executor = _executor(trainer, 1 if policies is not None and workers == 0 else workers)
    trainer._parallel_imagination_version = getattr(trainer, '_parallel_imagination_version', 0) + 1
    version = trainer._parallel_imagination_version
    weights = dict(actor=_numpy_state(trainer.actor), prior=(trainer.actor.prior_weight, trainer.actor.prior_temperature_seconds))
    batch = trainer.online_replay.sample_active_seeds(batch_size=1 if policies is not None else rollout_count, rng=trainer.rng, initial_only=True, **imagination_seed_kwargs(trainer.config))
    if paired or policies is not None:
        repeats = rollout_count if policies is not None else 2
        batch = SimpleNamespace(vehicles=tuple((states for states in batch.vehicles for _ in range(repeats))), **{name: None if getattr(batch, name) is None else getattr(batch, name).repeat_interleave(repeats, dim=0) for name in ('flow', 'time', 'started_count', 'cumulative_vehicle_seconds', 'total_vehicle_count')})
    trainer._last_sampled_imagination_seeds = [(float(batch.time[i]), int(batch.started_count[i])) for i in range(rollout_count)]
    seeds = torch.randint(0, 2 ** 63 - 1, (1, 2), device='cpu').repeat(rollout_count, 1).tolist() if policies is not None else torch.randint(0, 2 ** 63 - 1, (rollout_count, 2), device='cpu').tolist()
    deadline = trainer._episode_end_time()
    tracks = []
    for i, (actor_seed, noise_seed) in enumerate(seeds):
        states = [state for state in batch.vehicles[i] if state['departed'] and (not state['arrived'])]
        if batch.started_count is None or batch.cumulative_vehicle_seconds is None:
            raise ValueError('imagination requires causal cumulative reward accounting')
        population, start = (int(batch.started_count[i]), float(batch.time[i]))
        cumulative_seconds = float(batch.cumulative_vehicle_seconds[i])
        if population <= 0 or not states or start >= deadline:
            raise ValueError('imagination seed needs positive population and active vehicles before deadline')
        tracks.append(dict(states=copy.deepcopy(states), population=population, start=start, total_vehicle_count=int(batch.total_vehicle_count[i]), cumulative_vehicle_seconds=cumulative_seconds, rng=torch.Generator().manual_seed(actor_seed).get_state().numpy().copy(), noise_rng=torch.Generator(device=trainer.device).manual_seed(noise_seed), event_count=0, windows=[], terminal=False))
        if policies is not None:
            identity = _seed_identity(start, population, states, cumulative_seconds)
            collection = hashlib.sha256(f'{identity}:{actor_seed}:{noise_seed}:{version}'.encode()).hexdigest()
            tracks[-1].update(initial_start=start, horizon_end=min(start + horizon * 30.0, deadline), policy_label=policies[i]['label'], feature_index=policies[i]['feature_index'], strength=policies[i]['strength'], policy_seed=actor_seed, noise_seed=noise_seed, seed_identity=identity, collection_identity=collection, policy_occurrences={})
    with torch.no_grad():
        latent = trainer.world_model.encode(batch.flow[:, 0].to(trainer.device))
    try:
        for window_index in range(horizon):
            active = [i for i, track in enumerate(tracks) if not track['terminal']]
            if not active:
                break
            with torch.no_grad():
                decoded = trainer.world_model.decode(latent[active]).cpu().numpy()
                latent_host = None
            tasks = []
            for ordinal, i in enumerate(active):
                track = tracks[i]
                policy_weights = weights
                policy_version = version
                if policies is not None:
                    policy_weights = dict(weights, actor=policies[i]['actor_state'], score_gauge=getattr(trainer.actor, 'score_gauge', 'legacy'))
                    policy_version = (version, policies[i]['label'])
                tasks.append(dict(version=policy_version, weights=policy_weights, states=track['states'], rng=track['rng'], decoded=decoded[ordinal], reward_only=reward_only, population=track['population'], deadline=deadline, start=track['start'], total_vehicle_count=track['total_vehicle_count'], cumulative_vehicle_seconds=track['cumulative_vehicle_seconds'], max_speed=trainer.config.simulation.vehicle_max_speed_mps, traffic_model=getattr(trainer.config.simulation, 'traffic_model', 'micro'), event_count=track['event_count'], capture_actor_behavior=False, prior_cost_mode=getattr(trainer.config.method, 'sp_prior_cost_mode', 'free_flow'), prior_speed_floor_mps=getattr(trainer.config.method, 'sp_prior_dynamic_speed_floor_mps', 0.5)))
                if policies is not None:
                    tasks[-1].update(policy_seed=track['policy_seed'], feature_index=track['feature_index'], strength=track['strength'], policy_occurrences=track['policy_occurrences'])
            outputs = list(executor.map(_collect_window, tasks, chunksize=1))
            full = [ordinal for ordinal, output in enumerate(outputs) if abs(output['result'].end_time - tracks[active[ordinal]]['start'] - 30.0) <= 1e-07]
            next_latent = latent[active].clone()
            risk = next_latent.new_zeros(len(active))
            with torch.no_grad():
                if full:
                    method = trainer.config.method
                    rank = trainer.world_model.dynamics.graph_prior.rank
                    noise = torch.stack([torch.randn((method.disagreement_samples, rank, 64), generator=tracks[active[ordinal]]['noise_rng'], device=trainer.device, dtype=latent.dtype) for ordinal in full], dim=1)
                    sparse = pack_action_events([[outputs[ordinal]['result'].action_events] for ordinal in full], window_starts=[[tracks[active[ordinal]]['start']] for ordinal in full])
                    actions = torch.as_tensor(np.stack([outputs[ordinal]['result'].action for ordinal in full]), device=trainer.device, dtype=latent.dtype)
                    samples, variance, _ = trainer.world_model.latent_disagreement(latent[active][full], actions, action_events=sparse.to(trainer.device), noise=noise, n_samples=method.disagreement_samples, steps=method.fm_steps, solver=method.fm_solver, picard_iterations=method.picard_iterations)
                    next_latent[full] = samples[0]
                    risk[full] = method.disagreement_weight * variance.reshape(-1)
            for ordinal, i in enumerate(active):
                track, output = (tracks[i], outputs[ordinal])
                result = output['result']
                terminal = result.end_time >= deadline - 1e-09 or all((state['arrived'] for state in output['states']))
                track['windows'].append(dict(output=output, start=track['start'], terminal=terminal, cumulative_vehicle_seconds=track['cumulative_vehicle_seconds'], global_reward=latent.new_tensor(-result.active_vehicle_seconds / (track['population'] * 60.0)), risk_cost=risk[ordinal], reward=latent.new_tensor(-result.active_vehicle_seconds / (track['population'] * 60.0)) - risk[ordinal]))
                track.update(states=output['states'], start=result.end_time, rng=output['rng'], event_count=output['event_count'], terminal=terminal, cumulative_vehicle_seconds=track['cumulative_vehicle_seconds'] + result.active_vehicle_seconds)
                if policies is not None:
                    track['policy_occurrences'] = output['policy_occurrences']
            latent[active] = next_latent
    except BaseException:
        close_parallel_imagination(trainer)
        raise
    windows = [window for track in tracks for window in track['windows']]
    trainer._last_parallel_work = dict(rollouts=len(tracks), windows=len(windows), seconds=sum((window['output']['result'].end_time - window['start'] for window in windows)), active_vehicle_seconds=sum((window['output']['result'].active_vehicle_seconds for window in windows)))
    return tracks, deadline
