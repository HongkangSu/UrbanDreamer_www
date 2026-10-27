"""Pure imagined action-advantage Actor-Critic; no policy search or teacher."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import os
import time
import torch
from .counterfactual_q import collect_route_values, advantage_batch, centered_action_values, expected_action_loss, fit_action_advantage, NoEffectiveFocal
from .entropy_schedule import entropy_schedule_profile
from .counterfactual_value import uses_value_bootstrap, fit_counterfactual_value
RECOVER_MODE = 'counterfactual_actor_critic_recoverq_v1'
Q_EPOCHS = 256

def validate_counterfactual_method(method):
    if method.actor_objective != RECOVER_MODE or method.real_policy_update != 'none' or method.imagination_real_policy_guard != 'none' or (method.critic_value_mode != 'legacy') or (method.sp_prior_weight != 0) or (method.sp_prior_evaluation_weight != 0) or (method.disagreement_weight != 0) or (method.actor_entropy_mode == 'population') or (method.imagination_seed_scope != 'full_episode_v1') or (method.imagination_horizon_windows != 8) or (getattr(method, 'training_temperature_profile', 'none') != 'none') or (getattr(method, 'imagination_value_epochs', 1) != 1):
        raise ValueError('active recovered-Q AC requires original imagined-only H8 settings')

def update_expected_actor(trainer, examples, *, epochs, entropy_coefficient, select_best=False):
    """Optimize detached learned action advantages under the original policy KL."""
    device = trainer.device
    features = torch.stack([row['actor_features'] for row in examples]).to(device)
    batch = advantage_batch(examples, device)
    mask, old_probabilities = (batch[1], batch[2])
    guard_features = torch.cat([row.get('guard_actor_features', row['actor_features'][None]) for row in examples]).to(device)
    guard_mask = torch.cat([row.get('guard_mask', row['mask'][None]) for row in examples]).to(device)
    guard_probabilities = torch.cat([row.get('guard_probabilities', row['probabilities'][None]) for row in examples]).to(device)
    actor, optimizer = (trainer.actor, trainer.optimizers['actor'])
    with torch.no_grad():
        current = actor.forward_logits(features, mask).softmax(-1)
        if not torch.allclose(current, old_probabilities, atol=0.0001, rtol=0.0):
            raise ValueError('counterfactual Actor collection policy changed before update')
        values = centered_action_values(trainer.event_critic(batch[0], mask), old_probabilities, mask).detach()
        legal = values[mask]
        scale = max(1.0, float(torch.quantile(legal, 0.95) - torch.quantile(legal, 0.05)))
        values = values / scale
        old = current.detach()
        guard_old = actor.forward_logits(guard_features, guard_mask).softmax(-1).detach()
        if not torch.allclose(guard_old, guard_probabilities, atol=0.0001, rtol=0.0):
            raise ValueError('imagined guard observations do not match their collected policy')
        before = float(expected_action_loss(actor.forward_logits(features, mask), values, mask, entropy_coefficient=entropy_coefficient))
    accepted, attempted = (0, 0)
    from .imagined_value import _gradients, _restore_gradients
    best_loss, best_epoch = (before, 0)
    best = (deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict()), _gradients(actor)) if select_best else None
    for _ in range(epochs):
        parameters, adam = (deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict()))
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in actor.parameters()]
        optimizer.zero_grad(set_to_none=True)
        loss = expected_action_loss(actor.forward_logits(features, mask), values, mask, entropy_coefficient=entropy_coefficient)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        attempted += 1
        with torch.no_grad():
            guard_now = actor.forward_logits(guard_features, guard_mask).softmax(-1)
            kl = (guard_old * (guard_old.clamp_min(1e-30).log() - guard_now.clamp_min(1e-30).log())).sum(-1)
        if not bool(torch.isfinite(kl).all()) or float(kl.mean()) > 0.01 or float(kl.max()) > 0.1:
            actor.load_state_dict(parameters)
            optimizer.load_state_dict(adam)
            for parameter, gradient in zip(actor.parameters(), gradients):
                parameter.grad = gradient
            break
        accepted += 1
        if select_best:
            with torch.no_grad():
                candidate_loss = float(expected_action_loss(actor.forward_logits(features, mask), values, mask, entropy_coefficient=entropy_coefficient))
            if candidate_loss < best_loss:
                best_loss, best_epoch = (candidate_loss, accepted)
                best = (deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict()), _gradients(actor))
    if select_best:
        actor.load_state_dict(best[0])
        optimizer.load_state_dict(best[1])
        _restore_gradients(actor, best[2])
        accepted = best_epoch
    with torch.no_grad():
        guard_now = actor.forward_logits(guard_features, guard_mask).softmax(-1)
        kl = (guard_old * (guard_old.clamp_min(1e-30).log() - guard_now.clamp_min(1e-30).log())).sum(-1)
        after = float(expected_action_loss(actor.forward_logits(features, mask), values, mask, entropy_coefficient=entropy_coefficient))
    return dict(actor_loss=after, actor_optimizer_steps=accepted, actor_attempted_steps=attempted, actor_policy_kl=float(kl.mean()), actor_policy_kl_max=float(kl.max()), counterfactual_actor_loss_before=before, counterfactual_actor_loss_after=after, counterfactual_advantage_scale=scale, counterfactual_guard_observations=len(guard_features), counterfactual_actor_best_epoch=best_epoch if select_best else accepted)

def train_counterfactual_episode(trainer):
    method = trainer.config.method
    validate_counterfactual_method(method)
    validated = True
    minimum_signal = 0.1
    bootstrapped = uses_value_bootstrap(method)
    contract = 'population_scaled_policy_centered_model_Q_bootstrap_V_h8_v1'
    if trainer.metrics.get('online_episodes_completed', 0) > 0 and trainer.metrics.get('counterfactual_q_contract') != contract:
        raise ValueError('counterfactual Q requires fresh initialization or its own declared checkpoint contract')
    trainer.metrics['counterfactual_q_contract'] = contract
    if trainer.config.simulation.traffic_model != 'meso':
        raise ValueError('counterfactual route prototype currently requires meso')
    count = trainer.config.resources.imagination_rollouts_per_pass
    passes = trainer.config.resources.imagination_passes_per_episode
    deadline = trainer._episode_end_time()
    random_full_episode = True
    threshold = None
    eligible = int(trainer.online_replay.active_seed_summary(after_demand_seconds=threshold)['imagination_eligible_seed_count'])
    result = dict(imagination_eligible_seed_count=eligible, imagination_passes_requested=passes, imagination_passes_completed=0, imagination_no_eligible_seeds=float(not eligible), latent_critic_loss=0.0, imagined_latent_critic_optimizer_steps_total=0, imagined_actor_optimizer_steps_total=0, imagined_actor_attempted_steps_total=0, imagined_event_critic_optimizer_steps_total=0, imagined_events_total=0, counterfactual_contexts_total=0, counterfactual_candidate_targets_total=0, counterfactual_model_rollouts_total=0, counterfactual_macro_windows_total=0, counterfactual_informative_contexts_total=0, counterfactual_target_range_max=0.0, counterfactual_actor_min_signal=minimum_signal, counterfactual_model_validation_accepted_total=0, counterfactual_model_validation_rejected_total=0, counterfactual_actor_quality_skip_total=0, counterfactual_validation_rollouts_total=0, counterfactual_Q_additional_fit_passes=0, counterfactual_Q_additional_steps_total=0, counterfactual_Q_recovery_passes=0, counterfactual_Q_recovery_steps_total=0, counterfactual_validation_macro_windows_total=0, counterfactual_validation_duplicate_error_max=0.0, counterfactual_duplicate_error_max=0.0, counterfactual_terminal_targets_only=float(not bootstrapped), counterfactual_value_bootstrap=float(bootstrapped), counterfactual_fixed_horizon_windows=8, counterfactual_random_full_episode_seeds=float(random_full_episode), counterfactual_seed_candidates_sampled=0, counterfactual_sampled_seed_time_min=0.0, counterfactual_sampled_seed_time_max=0.0, counterfactual_seed_time_min=0.0, counterfactual_seed_time_max=0.0, counterfactual_terminal_horizon_windows_max=0, actor_loss=0.0, event_critic_loss=0.0)
    if not eligible:
        return result
    started = time.monotonic()
    trainer.world_model.eval()
    from .fm_inference import configure_imagination_inference
    entropy = entropy_schedule_profile(replace(method, actor_entropy_mode=os.environ.get('URB_ACTOR_ENTROPY_MODE', method.actor_entropy_mode)), completed_episodes=int(trainer.metrics.get('online_episodes_completed', 0)), total_episodes=int(trainer.config.run.online_training_episodes))['coefficient']
    for pass_index in range(passes):
        frozen_value = deepcopy(trainer.latent_critic).eval().requires_grad_(False)
        trainer.metrics['world_model_imagination_execution_profile'] = configure_imagination_inference(trainer, shared_policy_search=True, full_batch=getattr(getattr(trainer, 'candidates', None), 'max_candidates', 4) + 1)
        examples, windows, trajectories = ([], 0, 0)
        for _ in range(count * 32):
            if not validated and len(examples) >= count or (validated and (len(examples) >= count * 2 or sum((float((r['returns'][r['mask']].max() - r['returns'][r['mask']].min()) * r['population']) > minimum_signal for r in examples)) >= count)):
                break
            batch = trainer.online_replay.sample_active_seeds(1, trainer.rng, initial_only=True, after_demand_seconds=threshold, **{'causal_seed_only': True})
            seed_time = float(batch.time[0])
            horizon = method.imagination_horizon_windows
            sampled = result['counterfactual_seed_candidates_sampled']
            result['counterfactual_sampled_seed_time_min'] = min(result['counterfactual_sampled_seed_time_min'], seed_time) if sampled else seed_time
            result['counterfactual_sampled_seed_time_max'] = max(result['counterfactual_sampled_seed_time_max'], seed_time)
            result['counterfactual_seed_candidates_sampled'] += 1
            result['counterfactual_terminal_horizon_windows_max'] = max(result['counterfactual_terminal_horizon_windows_max'], horizon)
            policy_seed, noise_seed = torch.randint(0, 2 ** 31 - 1, (2,), device='cpu').tolist()
            try:
                first = collect_route_values(trainer, batch, policy_seed=policy_seed, model_seed=noise_seed, max_windows=horizon, **{'bootstrap_value': frozen_value}, focal_selection='uniform')
            except NoEffectiveFocal:
                continue
            replicas = [first] + [collect_route_values(trainer, batch, policy_seed=policy_seed, model_seed=noise_seed + replica, max_windows=horizon, **{'bootstrap_value': frozen_value}, focal_vehicle=first['focal_vehicle']) for replica in range(1, method.disagreement_samples)]
            row = dict(replicas[0])
            for other in replicas[1:]:
                if row['population'] != other['population']:
                    raise ValueError('paired model-noise replicas must share the same imagined population')
                torch.testing.assert_close(row['q_features'], other['q_features'], rtol=1e-05, atol=1e-06)
                torch.testing.assert_close(row['probabilities'], other['probabilities'], rtol=0, atol=0)
            row['returns'] = torch.stack([r['returns'] for r in replicas]).mean(0)
            row['value_rows'] = [item for replica in replicas for item in replica['value_rows']]
            row['model_noise_returns'] = torch.stack([r['returns'] for r in replicas])
            examples.append(row)
            previous_contexts = result['counterfactual_contexts_total'] + len(examples) - 1
            result['counterfactual_seed_time_min'] = min(result['counterfactual_seed_time_min'], seed_time) if previous_contexts else seed_time
            result['counterfactual_seed_time_max'] = max(result['counterfactual_seed_time_max'], seed_time)
            windows += sum((sum(r['windows_per_arm']) for r in replicas))
            trajectories += sum((len(r['windows_per_arm']) for r in replicas))
            result['counterfactual_duplicate_error_max'] = max(result['counterfactual_duplicate_error_max'], max((r['duplicate_error'] for r in replicas)))
        if not examples:
            result['counterfactual_no_effective_actions'] = 1.0
            continue
        if True and hasattr(trainer, 'log'):
            trainer.log(dict(phase='counterfactual_model_examples', episode_index=trainer.pilot_episode, pass_index=pass_index, targets_source='Learned model terminal returns only', examples=[{name: value.tolist() if isinstance(value, torch.Tensor) else value for name, value in row.items() if name in {'q_features', 'actor_features', 'mask', 'probabilities', 'returns', 'population', 'seed_time', 'focal_vehicle', 'policy_seed', 'model_seed', 'model_noise_returns', 'route_actions'}} for row in examples]))
        value_fit = fit_counterfactual_value(trainer, examples)
        result['imagined_latent_critic_optimizer_steps_total'] += value_fit['actual_steps']
        result['counterfactual_value_retained_steps_total'] = result.get('counterfactual_value_retained_steps_total', 0) + value_fit['retained_steps']
        result['counterfactual_value_mse_before'] = value_fit['mse_before']
        result['counterfactual_value_mse_after'] = value_fit['mse_after']
        result['counterfactual_value_normalized_mse_after'] = value_fit.get('normalized_mse_after', 0.0)
        result['counterfactual_value_target_energy'] = value_fit.get('target_energy', 0.0)
        result['latent_critic_loss'] = value_fit['mse_after']
        fit = fit_action_advantage(trainer.event_critic, trainer.optimizers['event_critic'], examples, device=trainer.device, epochs=Q_EPOCHS)
        spans = [float((row['returns'][row['mask']].max() - row['returns'][row['mask']].min()) * row['population']) for row in examples]
        informative = [row for row, span in zip(examples, spans) if span > minimum_signal]
        result['counterfactual_target_range_max'] = max(result['counterfactual_target_range_max'], max(spans))
        result['counterfactual_informative_contexts_total'] += len(informative)
        quality_ok = True
        if validated and informative:
            from .counterfactual_q import action_advantage_loss
            cache = advantage_batch(informative, trainer.device)
            energy = float((cache[3].square() * cache[1]).sum() / cache[1].sum())
            with torch.no_grad():
                relative_error = float(action_advantage_loss(trainer.event_critic, cache)) / max(energy, 1e-12)
            if True and relative_error > 0.25:
                extra = fit_action_advantage(trainer.event_critic, trainer.optimizers['event_critic'], examples, device=trainer.device, epochs=4096 - Q_EPOCHS)
                fit = dict(fit, mse_after=extra['mse_after'], actual_steps=fit['actual_steps'] + extra['actual_steps'], retained_steps=fit['retained_steps'] + extra['retained_steps'], additional_fit=extra)
                result['counterfactual_Q_additional_fit_passes'] += 1
                result['counterfactual_Q_additional_steps_total'] += extra['actual_steps']
                with torch.no_grad():
                    relative_error = float(action_advantage_loss(trainer.event_critic, cache)) / max(energy, 1e-12)
            if True and relative_error > 0.25:
                trainer.optimizers['event_critic'].state.clear()
                recovery = fit_action_advantage(trainer.event_critic, trainer.optimizers['event_critic'], examples, device=trainer.device, epochs=4096)
                fit = dict(fit, mse_after=recovery['mse_after'], actual_steps=fit['actual_steps'] + recovery['actual_steps'], retained_steps=fit['retained_steps'] + recovery['retained_steps'], recovery_fit=recovery, optimizer_history_resets=1)
                result['counterfactual_Q_recovery_passes'] += 1
                result['counterfactual_Q_recovery_steps_total'] += recovery['actual_steps']
                with torch.no_grad():
                    relative_error = float(action_advantage_loss(trainer.event_critic, cache)) / max(energy, 1e-12)
            quality_ok = relative_error <= 0.25
            result['counterfactual_relative_Q_error_last'] = relative_error
        from .imagined_value import _gradients, _restore_gradients
        original = (deepcopy(trainer.actor.state_dict()), deepcopy(trainer.optimizers['actor'].state_dict()), _gradients(trainer.actor))
        actor = update_expected_actor(trainer, informative, epochs=method.imagination_actor_epochs, entropy_coefficient=entropy, select_best=validated) if informative and quality_ok else dict(actor_loss=0.0, actor_optimizer_steps=0, actor_attempted_steps=0)
        validation = None
        if informative and (not quality_ok):
            result['counterfactual_actor_quality_skip_total'] += 1
        elif informative and actor['actor_optimizer_steps'] == 0:
            result['counterfactual_actor_quality_skip_total'] += 1
        if actor['actor_optimizer_steps'] > 0:
            from .counterfactual_validation import validate_actor_in_model
            try:
                validation = validate_actor_in_model(trainer, original[0])
            except BaseException:
                trainer.actor.load_state_dict(original[0])
                trainer.optimizers['actor'].load_state_dict(original[1])
                _restore_gradients(trainer.actor, original[2])
                raise
            for field in ('model_rollouts', 'macro_windows'):
                result['counterfactual_validation_' + ('rollouts_total' if field == 'model_rollouts' else 'macro_windows_total')] += validation[field]
            result['counterfactual_validation_duplicate_error_max'] = max(result['counterfactual_validation_duplicate_error_max'], validation['duplicate_error_max'])
            result['counterfactual_model_validation_' + ('accepted_total' if validation['accepted'] else 'rejected_total')] += 1
            if not validation['accepted']:
                trainer.actor.load_state_dict(original[0])
                trainer.optimizers['actor'].load_state_dict(original[1])
                _restore_gradients(trainer.actor, original[2])
                actor['actor_optimizer_steps'] = 0
                actor['actor_loss'] = actor['counterfactual_actor_loss_before']
                actor['counterfactual_actor_loss_after'] = actor['counterfactual_actor_loss_before']
                actor['actor_policy_kl'] = actor['actor_policy_kl_max'] = 0.0
        result['imagination_passes_completed'] += 1
        result['imagined_actor_optimizer_steps_total'] += actor['actor_optimizer_steps']
        result['imagined_actor_attempted_steps_total'] += actor['actor_attempted_steps']
        result['imagined_event_critic_optimizer_steps_total'] += fit['actual_steps']
        result['imagined_events_total'] += len(examples)
        result['counterfactual_contexts_total'] += len(examples)
        result['counterfactual_candidate_targets_total'] += fit['candidate_targets']
        result['counterfactual_model_rollouts_total'] += trajectories
        result['counterfactual_macro_windows_total'] += windows
        result['actor_loss'], result['event_critic_loss'] = (actor['actor_loss'], fit['mse_after'])
        result['counterfactual_actor_mean_kl_last'] = actor.get('actor_policy_kl', 0.0)
        result['counterfactual_actor_max_kl_last'] = actor.get('actor_policy_kl_max', 0.0)
        result['counterfactual_Q_mse_before'], result['counterfactual_Q_mse_after'] = (fit['mse_before'], fit['mse_after'])
        with torch.no_grad():
            guard_features = torch.cat([row.get('guard_actor_features', row['actor_features'][None]) for row in examples]).to(trainer.device)
            guard_mask = torch.cat([row.get('guard_mask', row['mask'][None]) for row in examples]).to(trainer.device)
            old_probabilities = torch.cat([row.get('guard_probabilities', row['probabilities'][None]) for row in examples]).to(trainer.device)
            probabilities = trainer.actor.forward_logits(guard_features, guard_mask).softmax(-1)
            result['actor_entropy_before'] = float(-(old_probabilities * old_probabilities.clamp_min(1e-30).log()).sum(-1).mean())
            result['actor_entropy'] = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean())
            actor['actor_entropy_before'] = result['actor_entropy_before']
            actor['actor_entropy_after'] = result['actor_entropy']
        if hasattr(trainer, 'log'):
            trainer.log(dict(phase='counterfactual_actor_critic_pass_finished', pass_index=pass_index, episode_index=trainer.pilot_episode, Q_fit=fit, Actor_fit=actor, model_rollouts=trajectories, macro_windows=windows, candidate_returns=[row['returns'].tolist() for row in examples], populations=[row['population'] for row in examples], population_scaled_spans=spans, model_validation=validation, minimum_action_contrast=minimum_signal, true_terminal_targets=not bootstrapped, value_bootstrap=bootstrapped, fixed_horizon_windows=8, real_policy_updates=0))
        if validated and actor['actor_optimizer_steps'] > 0:
            break
    result['imagination_seconds'] = time.monotonic() - started
    result['imagined_events'] = result['imagined_events_total']
    result['actor_optimizer_steps'] = result['imagined_actor_optimizer_steps_total']
    result['actor_attempted_steps'] = result['imagined_actor_attempted_steps_total']
    return result
