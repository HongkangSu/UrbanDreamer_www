"""Model-only focal-route returns and action-advantage regression.

Every candidate is a Q training example. No maximizing-action labels or policy
search are produced. Full terminal rollouts avoid borrowing real value targets.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from .action_events import ActionEvent, pack_action_events
from .global_reward import causal_context
from .imagination import advance_imagined_window, clip_road_speed_fractions
from .observations import build_actor_observations, prepare_actor_window, selected_decisions
from .parallel_imagination import _policy_uniform
from ..models.event_critic import build_event_q_inputs


class NoEffectiveFocal(ValueError):
    """No distinguishable route can take effect within this control window."""


def focal_effectiveness(trainer, state, routes, decoded, remaining_seconds):
    unique = list(dict.fromkeys(tuple(route) for route in routes))
    if len(unique) < 2:
        return None
    prefix = []
    for edges in zip(*unique):
        if len(set(edges)) != 1:
            break
        prefix.append(edges[0])
    speeds = np.minimum(clip_road_speed_fractions(decoded[:, 1], 'meso') * trainer.network.speed_limits_mps,
        trainer.config.simulation.vehicle_max_speed_mps)
    distances = [max(0., float(state['remaining_distance']))] + [float(trainer.network.lengths_m[e]) for e in prefix[1:]]
    if any(speeds[e] <= 0 for e in prefix):
        return None
    divergence_seconds = sum(distance / float(speeds[e]) for distance, e in zip(distances, prefix))
    # Eligibility is decided before any returns: no hindsight route selection.
    minimum_route_distance = min(float(state['remaining_distance'])
        + sum(float(trainer.network.lengths_m[e]) for e in route[1:]) for route in unique)
    if divergence_seconds >= 29. or minimum_route_distance / trainer.config.simulation.vehicle_max_speed_mps > remaining_seconds:
        return None
    return dict(shared_prefix_edges=len(prefix), predicted_seconds_to_divergence=divergence_seconds,
        optimistic_seconds_to_destination=minimum_route_distance / trainer.config.simulation.vehicle_max_speed_mps)


def _observe(trainer, states, decoded, now, population, cumulative):
    context = causal_context(now, trainer._episode_end_time(),
        sum(s['departed'] and not s['arrived'] for s in states), population, cumulative)
    pending = sorted([s for s in states if s['departed'] and not s['arrived']
        and int(s['current_edge']) != int(s['destination_edge'])], key=lambda s: str(s['vehicle_id']))
    observation = build_actor_observations(
        policy_edge_features=prepare_actor_window(decoded[:, [2, 1]], trainer.network.edge_count),
        vehicle_states=pending, candidates=trainer.candidates,
        episode_progress=float(context[0]), active_vehicle_fraction=float(context[1]))
    return pending, observation, context


def _choices(actor, pending, observation, now, policy_seed, occurrences):
    with torch.no_grad():
        logits = actor.forward_logits(torch.from_numpy(observation.features), torch.from_numpy(observation.mask))
        probabilities = logits.softmax(-1).cpu().numpy()
    chosen = []
    for index, state in enumerate(pending):
        key = (str(state['vehicle_id']), int(state['current_edge']), int(state['destination_edge']))
        count = occurrences.get(key, 0)
        occurrences[key] = count + 1
        u = _policy_uniform(policy_seed, key[0], now, key[1], key[2], count)
        legal = np.flatnonzero(observation.mask[index])
        cdf = np.cumsum(probabilities[index], dtype=np.float64)
        chosen.append(min(int(np.searchsorted(cdf, u, side='right')), int(legal[-1])))
    return chosen, probabilities


@torch.no_grad()
def collect_route_values(trainer, batch, *, batch_index=0, policy_seed, model_seed,
                         focal_vehicle=None, max_windows=64, focal_selection='first', bootstrap_value=None):
    """Enumerate one focal vehicle's actions under a fixed continuation policy.

    Other initial actions are identical; future policy draws share visit keys
    while allowing causal state/policy responses. Each route ends at the real
    episode deadline or all-arrived termination, never a fictitious H cutoff.
    Only the replay seed is read, not its subsequent actions/rewards/states.
    """
    if focal_selection not in {'first', 'uniform'}:
        raise ValueError('unsupported causal focal selection')
    method = trainer.config.method
    if (method.real_policy_update != 'none' or trainer.actor.prior_weight != 0
            or method.disagreement_weight != 0):
        raise ValueError('counterfactual collection requires imagined-only, zero prior and zero risk')
    if getattr(trainer.candidates, 'routing_mode', '') != 'current_to_dest' or trainer.candidates.traffic_model != 'meso':
        raise ValueError('counterfactual route collection currently requires current_to_dest meso')
    if hasattr(trainer.world_model.dynamics, '_set_chunk_profile'):
        # Full/tail token chunks can use different GEMM shapes for duplicate
        # rows. Common noise alone does not remove those numerical differences.
        resources = trainer.config.resources
        trainer.world_model.dynamics._set_chunk_profile(
            space_chunk=resources.world_model_imagination_space_chunk,
            time_chunk=resources.world_model_imagination_time_chunk, mlp_chunk=0)
    start = float(batch.time[batch_index])
    deadline = float(trainer._episode_end_time())
    started_count = int(batch.started_count[batch_index])
    population = started_count
    full_scope = getattr(method, 'imagination_seed_scope', 'after_demand_v1') == 'full_episode_v1'
    initial_cumulative = float(batch.cumulative_vehicle_seconds[batch_index])
    if (population <= 0 or not 0 <= start < deadline
            or (bootstrap_value is None and not 0 < deadline - start <= 30 * max_windows)
            or (not full_scope and (population != int(batch.total_vehicle_count[batch_index])
                or start < trainer.config.simulation.demand_duration_seconds))):
        raise ValueError('counterfactual terminal rollout needs an eligible bounded cohort seed')
    initial = sorted([copy.deepcopy(s) for s in batch.vehicles[batch_index]
        if s['departed'] and not s['arrived']], key=lambda s: str(s['vehicle_id']))
    if not initial:
        raise ValueError('counterfactual seed must have active vehicles')
    past_arrived_count = started_count - len(initial)
    if past_arrived_count < 0:
        raise ValueError('active seed population exceeds its observed started count')
    future_states, future_seed = [], None
    if full_scope:
        from .imagined_demand import sample_future_states
        future_states, future_seed = sample_future_states(trainer,
            start_time=start, deadline=deadline, policy_seed=policy_seed)
        if {s['vehicle_id'] for s in initial} & {s['vehicle_id'] for s in future_states}:
            raise ValueError('imagined future vehicle identity collides with a causal seed')
        population += len(future_states)
        initial.extend(future_states)
    actor = copy.deepcopy(trainer.actor).cpu().eval().requires_grad_(False)
    latent = trainer.world_model.encode(batch.flow[batch_index:batch_index+1, 0].to(trainer.device))
    decoded = trainer.world_model.decode(latent)[0].cpu().numpy()
    pending, observation, context = _observe(trainer, initial, decoded, start, started_count, initial_cumulative)
    chosen, probabilities = _choices(actor, pending, observation, start, policy_seed, {})
    effective = {i:focal_effectiveness(trainer, s, observation.routes[i], decoded, deadline-start)
        for i,s in enumerate(pending) if focal_vehicle is None or str(s['vehicle_id']) == str(focal_vehicle)}
    eligible = [i for i,value in effective.items() if value is not None]
    if not eligible:
        raise NoEffectiveFocal('no causal focal route difference within the current control window')
    focal_index = (int(np.random.default_rng(policy_seed).choice(eligible))
        if focal_vehicle is None and focal_selection == 'uniform' else eligible[0])
    focal_id = str(pending[focal_index]['vehicle_id'])
    mask = np.asarray(observation.mask[focal_index], dtype=bool)
    routes = [tuple(r) for r in observation.routes[focal_index]]
    legal = np.flatnonzero(mask).tolist()
    factual = selected_decisions(pending, observation, chosen, start)
    events = tuple(ActionEvent(str(item.vehicle_id), i, float(start), int(item.current_edge),
        int(item.next_edge), int(pending[i]['destination_edge'])) for i, item in enumerate(factual))
    sparse = pack_action_events([[events]], window_starts=[[start]]).to(trainer.device)
    q_features = build_event_q_inputs(trainer.world_model, z=latent,
        observations=torch.as_tensor(observation.features, device=trainer.device),
        candidate_edges=torch.as_tensor(observation.edge_indices, device=trainer.device),
        mask=torch.as_tensor(observation.mask, device=trainer.device),
        current_edges=torch.tensor([e.current_edge for e in events], device=trainer.device),
        destination_edges=torch.tensor([e.destination_edge for e in events], device=trainer.device),
        decision_times=torch.tensor([start] * len(events), device=trainer.device),
        event_indices=torch.arange(len(events), device=trainer.device),
        context=torch.as_tensor(context, device=trainer.device), events=sparse, window_start=start)[focal_index].cpu()
    # The extra first-action arm is an exact duplicate control, not a new action.
    arm_actions = legal + [legal[0]]
    tracks = [dict(states=copy.deepcopy(initial), now=start, exposure=0., terminal=False,
        occurrences={}, intervened=False, initial_others=None, windows=[], value_rows=[]) for _ in arm_actions]
    latent = latent.expand(len(tracks), -1, -1).clone()
    noise_rng = torch.Generator(device=trainer.device).manual_seed(model_seed)
    max_error = 0.
    for window_index in range(min(max_windows, math.ceil((deadline - start) / 30.))):
        active = [i for i, t in enumerate(tracks) if not t['terminal']]
        if not active:
            break
        flows = trainer.world_model.decode(latent[active]).cpu().numpy()
        outputs, birth_volumes = [], []
        for ordinal, arm in enumerate(active):
            track = tracks[arm]
            now = track['now']
            cumulative = initial_cumulative + track['exposure']
            if bootstrap_value is not None:
                ctx = causal_context(now, deadline, sum(s['departed'] and not s['arrived'] for s in track['states']),
                    past_arrived_count + sum(s['departed'] for s in track['states']), cumulative)
                track['value_rows'].append(dict(pooled=latent[arm].mean(0).detach().cpu(),
                    context=torch.as_tensor(ctx).clone(), reward=0., next_value=0.))
            waiting_births = {str(s['vehicle_id']) for s in track['states'] if not s['departed']} if full_scope else set()

            def decide(vehicle_states, frozen, timestamp):
                event_cumulative = cumulative + sum(max(0., min(float(s.get('arrival_time', timestamp)), timestamp)
                    - max(now, float(s.get('actual_departure_time', s['departure_time']))))
                    for s in track['states'] if s['departed'])
                observed_started = past_arrived_count + sum(s['departed'] for s in track['states'])
                observed_pending, obs, _ = _observe(trainer, track['states'], flows[ordinal], timestamp,
                    observed_started, event_cumulative)
                # The physical executor supplies exactly the same sorted pending set.
                assert [str(s['vehicle_id']) for s in vehicle_states] == [str(s['vehicle_id']) for s in observed_pending]
                choices, _ = _choices(actor, vehicle_states, obs, timestamp, policy_seed, track['occurrences'])
                if window_index == 0 and not track['intervened']:
                    index = [str(s['vehicle_id']) for s in vehicle_states].index(focal_id)
                    assert [tuple(r) for r in obs.routes[index]] == routes
                    track['initial_others'] = [(str(s['vehicle_id']), tuple(obs.routes[i][a]))
                        for i, (s, a) in enumerate(zip(vehicle_states, choices)) if str(s['vehicle_id']) != focal_id]
                    choices[index] = arm_actions[arm]
                    track['intervened'] = True
                return selected_decisions(vehicle_states, obs, choices, timestamp)

            speeds = clip_road_speed_fractions(flows[ordinal, :, 1], 'meso') * trainer.network.speed_limits_mps
            result = advance_imagined_window(track['states'], candidates=trainer.candidates,
                edge_speeds_mps=speeds, frozen_flow=flows[ordinal], start_time=now,
                interval_seconds=min(30., deadline - now), total_vehicle_count=population, decide=decide,
                observation_step_seconds=1., session_protocol='event', traffic_model='meso',
                vehicle_max_speed_mps=trainer.config.simulation.vehicle_max_speed_mps, stop_when_empty=True)
            outputs.append(result)
            new_volume = None
            if waiting_births:
                new_volume = np.zeros(trainer.network.edge_count, dtype=np.float32)
                for state in track['states']:
                    if (str(state['vehicle_id']) in waiting_births and state['departed']
                            and not state['arrived'] and not state.get('in_junction', False)):
                        new_volume[int(state['current_edge'])] += 1.
            birth_volumes.append(new_volume)
            track['windows'].append(dict(start=now, exposure=float(result.active_vehicle_seconds)))
            track['exposure'] += result.active_vehicle_seconds
            track['now'] = result.end_time
            track['terminal'] = result.end_time >= deadline - 1e-9 or all(s['arrived'] for s in track['states'])
            if bootstrap_value is not None:
                track['value_rows'][-1]['reward'] = -float(result.active_vehicle_seconds)/(population*60.)
        full = [j for j, output in enumerate(outputs) if abs(output.end_time - (start + (window_index + 1) * 30.)) < 1e-7]
        # No successor is consumed once every arm has reached its real terminal
        # condition. Keep the original full batch (including terminal arms)
        # while any arm continues, preserving GEMM shapes and future noise draws.
        if full and not all(t['terminal'] for t in tracks):
            indexes = [active[j] for j in full]
            actions = torch.as_tensor(np.stack([outputs[j].action for j in full]), device=trainer.device, dtype=latent.dtype)
            events = pack_action_events([[outputs[j].action_events] for j in full],
                window_starts=[[start + window_index * 30.] for _ in full]).to(trainer.device)
            noise = torch.randn((1, trainer.world_model.dynamics.graph_prior.rank, 64), device=trainer.device, generator=noise_rng,
                dtype=latent.dtype).expand(len(full), -1, -1).clone()
            successor = trainer.world_model.transition(latent[indexes], actions, noise=noise,
                steps=method.fm_steps, solver=method.fm_solver, picard_iterations=method.picard_iterations,
                action_events=events)
            if full_scope and any(birth_volumes[j] is not None for j in full):
                # This WM predicts endpoint count for the previous-window
                # cohort; its other dynamic channels already target full
                # traffic. Reconstruct full count before the next transition.
                volume = torch.as_tensor(np.stack([birth_volumes[j] if birth_volumes[j] is not None
                    else np.zeros(trainer.network.edge_count, dtype=np.float32) for j in full]),
                    device=trainer.device, dtype=successor.dtype)
                affected = volume.sum(-1) > 0
                if bool(affected.any()):
                    decoded_successor = trainer.world_model.decode(successor[affected]).clone()
                    capacity = torch.as_tensor(trainer.network.capacities,
                        device=trainer.device, dtype=decoded_successor.dtype).clamp_min(1.)
                    decoded_successor[..., 2] += volume[affected] / capacity
                    static = batch.flow[batch_index, 0, :, 4:].to(decoded_successor)
                    corrected_flow = torch.cat((decoded_successor,
                        static.unsqueeze(0).expand(len(decoded_successor), -1, -1)), dim=-1)
                    successor[affected] = trainer.world_model.encode(corrected_flow)
            latent[indexes] = successor
        if not bool(torch.isfinite(latent).all()):
            raise ValueError('nonfinite counterfactual model successor')
        if bootstrap_value is not None:
            contexts = [causal_context(tracks[i]['now'], deadline,
                sum(s['departed'] and not s['arrived'] for s in tracks[i]['states']),
                past_arrived_count + sum(s['departed'] for s in tracks[i]['states']),
                initial_cumulative + tracks[i]['exposure']) for i in active]
            predicted = bootstrap_value(latent[active].mean(1, keepdim=True),
                torch.as_tensor(np.stack(contexts), device=trainer.device)).reshape(-1)
            if not bool(torch.isfinite(predicted).all()):
                raise ValueError('nonfinite imagined value bootstrap')
            budgets = predicted.new_tensor([(deadline-tracks[i]['now'])/60. for i in active])
            predicted = torch.maximum(predicted.clamp_max(0.), -budgets).cpu().tolist()
            for i, value in zip(active, predicted):
                tracks[i]['value_rows'][-1]['next_value'] = 0. if tracks[i]['terminal'] else value
    if not all(t['intervened'] and (t['terminal'] or bootstrap_value is not None) for t in tracks):
        raise ValueError('counterfactual return must end at true termination')
    assert all(t['initial_others'] == tracks[0]['initial_others'] for t in tracks)
    tails = [t['value_rows'][-1]['next_value'] if bootstrap_value is not None else 0. for t in tracks]
    returns = torch.tensor([-t['exposure'] / (population * 60.) + tail
                           for t, tail in zip(tracks, tails)], dtype=torch.float64)
    value_rows = []
    for track in tracks[:-1]:
        future = track['value_rows'][-1]['next_value'] if track['value_rows'] else 0.
        for row in reversed(track['value_rows']):
            future = row['reward'] + .05*row['next_value'] + .95*future
            value_rows.append(dict(pooled=row['pooled'], context=row['context'], target=future))
    duplicate_error = float(abs(returns[0] - returns[-1]))
    if duplicate_error > 1e-7:
        raise ValueError(f'duplicate counterfactual control mismatch: {duplicate_error}')
    for i, action in enumerate(legal):
        for j, other in enumerate(legal):
            if routes[action] == routes[other]:
                max_error = max(max_error, float(abs(returns[i] - returns[j])))
    if max_error > 1e-7:
        raise ValueError('identical route aliases disagree under common randomness')
    all_returns = torch.zeros(len(mask), dtype=torch.float64)
    all_returns[legal] = returns[:-1]
    return dict(actor_features=torch.from_numpy(observation.features[focal_index]).clone(),
        guard_actor_features=torch.from_numpy(observation.features).clone(),
        guard_mask=torch.from_numpy(observation.mask).clone(),
        guard_probabilities=torch.from_numpy(probabilities).clone(),
        mask=torch.from_numpy(mask).clone(), probabilities=torch.from_numpy(probabilities[focal_index]).clone(),
        q_features=q_features.clone(), returns=all_returns, population=population,
        focal_vehicle=focal_id, seed_time=start, policy_seed=policy_seed, model_seed=model_seed,
        initial_started_count=started_count, future_departure_count=len(future_states),
        future_demand_seed=future_seed,
        future_demand_source='independent_conditional_Poisson' if full_scope else 'none_closed_cohort',
        focal_effectiveness=effective[focal_index],
        route_actions=[list(r) for r in routes], duplicate_error=duplicate_error, alias_error=max_error,
        arm_exposures=[t['exposure'] for t in tracks], windows_per_arm=[len(t['windows']) for t in tracks],
        terminal_only=all(t['terminal'] for t in tracks), bootstrap_used=any(not t['terminal'] for t in tracks),
        value_rows=value_rows, bootstrap_tails=tails,
        initial_other_actions_identical=True)


def centered_action_values(values, probabilities, mask):
    """Remove a state-only offset, preserving action differences and gradients."""
    if values.shape != probabilities.shape or values.shape != mask.shape:
        raise ValueError('action values, probabilities and masks must share shape')
    if not bool(mask.any(-1).all()) or bool((probabilities[mask] < 0).any()):
        raise ValueError('invalid legal action probabilities')
    probs = probabilities.masked_fill(~mask, 0.)
    if not torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-6, rtol=0.):
        raise ValueError('legal probabilities must sum to one')
    return (values - (values.masked_fill(~mask, 0.) * probs).sum(-1, keepdim=True)).masked_fill(~mask, 0.)


def expected_action_loss(logits, values, mask, *, entropy_coefficient=0.):
    """Actor policy gradient from detached learned values, not argmax labels."""
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    values = values.detach().masked_fill(~mask, 0.)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    return -((probabilities * values).sum(-1) + entropy_coefficient * entropy).mean()


def advantage_batch(examples, device):
    """Population scaling resolves tiny per-vehicle contrasts in global returns.

    This critic estimates Q up to an action-independent state offset. Its
    centered output is explicitly not an absolute global value prediction.
    """
    if not examples:
        raise ValueError('counterfactual critic requires model-generated examples')
    width = max(len(row['mask']) for row in examples)
    features, masks, probabilities, targets = [], [], [], []
    for row in examples:
        mask = row['mask'].detach().to(device=device, dtype=torch.bool)
        probs = row['probabilities'].detach().to(device=device, dtype=torch.float64)
        returns = row['returns'].detach().to(device=device, dtype=torch.float64)
        target = centered_action_values(returns[None], probs[None], mask[None])[0] * row['population']
        extra = width - len(mask)
        features.append(torch.nn.functional.pad(row['q_features'].detach().to(device), (0, 0, 0, extra)))
        masks.append(torch.nn.functional.pad(mask, (0, extra)))
        probabilities.append(torch.nn.functional.pad(probs.float(), (0, extra)))
        targets.append(torch.nn.functional.pad(target.float(), (0, extra)))
    return torch.stack(features), torch.stack(masks), torch.stack(probabilities), torch.stack(targets)


def action_advantage_loss(critic, batch):
    features, mask, probabilities, targets = batch
    prediction = centered_action_values(critic(features, mask), probabilities, mask)
    return (((prediction - targets).square() * mask).sum(-1) / mask.sum(-1)).mean()


def fit_action_advantage(critic, optimizer, examples, *, device, epochs=256):
    """Fit every legal action to terminal imagined return contrasts only."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('positive critic fitting epoch count required')
    batch = advantage_batch(examples, device)
    from .imagined_value import _gradients, _restore_gradients
    with torch.no_grad():
        before = float(action_advantage_loss(critic, batch))
    if not math.isfinite(before):
        raise ValueError('nonfinite initial counterfactual critic loss')
    best_loss, best_epoch, attempted = before, 0, 0
    best = (copy.deepcopy(critic.state_dict()), copy.deepcopy(optimizer.state_dict()), _gradients(critic))
    try:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = action_advantage_loss(critic, batch)
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite counterfactual critic loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            attempted += 1
            with torch.no_grad():
                after = float(action_advantage_loss(critic, batch))
            if not math.isfinite(after):
                raise ValueError('nonfinite counterfactual critic post-step loss')
            if after < best_loss:
                best_loss, best_epoch = after, epoch
                best = (copy.deepcopy(critic.state_dict()), copy.deepcopy(optimizer.state_dict()), _gradients(critic))
    finally:
        critic.load_state_dict(best[0])
        optimizer.load_state_dict(best[1])
        _restore_gradients(critic, best[2])
    return dict(mse_before=before, mse_after=best_loss, actual_steps=attempted,
        retained_steps=best_epoch, candidate_targets=sum(int(row['mask'].sum()) for row in examples),
        target='population-scaled policy-centered terminal model returns; no maximizing-action labels')
