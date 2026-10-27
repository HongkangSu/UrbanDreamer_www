"""Shared next-hop MAPPO/IPPO with global traffic returns.

MAPPO values complete network windows. IPPO values local decision observations,
but its last decision target includes the remaining team episode rewards even
when that decision vehicle arrives earlier. Both critics retain global targets.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from ...models.policy import (
    CANDIDATE_FEATURE_DIM, DEFAULT_GAMMA, DEFAULT_LAMBDA, NextHopActor,
    lambda_return)
from . import METHODS
from .common import reward_context_dim, GlobalTrafficCritic, global_reward_integral



def _local_lambda_targets(rows, values, windows, *, deadline, trace_seconds=30.):
    """Compute detached asynchronous returns without per-event CUDA synchronizations.

    Only global window rewards contribute. A vehicle's final decision includes
    the team episode tail, including costs after that vehicle has arrived.
    Trace decay follows elapsed physical time, in 30-second macro-window units,
    rather than how many routing/forced events happen before a future reward.
    """
    if not rows:
        return torch.empty_like(values)
    if not math.isfinite(trace_seconds) or trace_seconds <= 0:
        raise ValueError('IPPO trace_seconds must be finite and positive')
    integrated, episode_end = global_reward_integral(windows, deadline=deadline)
    times = np.asarray([float(row['time']) for row in rows], dtype=np.float64)
    cumulative = integrated(times)
    tail = integrated(episode_end)
    trajectories = defaultdict(list)
    for index, row in enumerate(rows):
        trajectories[row['vehicle_id']].append(index)
    host_values = values.detach().cpu().numpy()
    targets = np.empty_like(host_values)
    scalar = host_values.dtype.type
    gamma = scalar(DEFAULT_GAMMA)
    for indices in trajectories.values():
        indices.sort(key=times.__getitem__)
        rewards = np.diff(np.append(cumulative[indices], tail)).astype(host_values.dtype)
        running = rewards[-1]
        targets[indices[-1]] = running
        for offset in range(len(indices) - 2, -1, -1):
            elapsed = times[indices[offset + 1]] - times[indices[offset]]
            lam = scalar(DEFAULT_LAMBDA ** (elapsed / trace_seconds))
            running = rewards[offset] + gamma * (
                (scalar(1.) - lam) * host_values[indices[offset + 1]] + lam * running)
            targets[indices[offset]] = running
    return torch.as_tensor(targets, dtype=values.dtype, device=values.device)


def clipped_policy_loss(log_prob, old_log_prob, advantage, *, clip=0.2):
    """PPO's pessimistic clipped surrogate, with the behavior policy frozen."""
    advantage = advantage.detach()
    log_ratio = log_prob - old_log_prob.detach()
    # Clip before exp: a large ratio on a positive/zero advantage has a finite
    # objective, but exp followed by minimum gives an inf * 0 gradient (NaN).
    pessimistic_log_ratio = torch.where(advantage >= 0.,
        log_ratio.clamp_max(math.log1p(clip)), log_ratio.clamp_min(math.log1p(-clip)))
    return -(pessimistic_log_ratio.exp() * advantage).mean()


def normalize_advantages(advantages, *, std_floor):
    """Remove the shared offset; do not amplify a numerically constant batch."""
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError('PPO advantage std floor must be finite and positive')
    if advantages.ndim != 1 or not bool(torch.isfinite(advantages).all()):
        raise ValueError('PPO advantages must be a finite vector')
    if not advantages.numel():
        return advantages.detach().clone()
    centered = advantages.detach() - advantages.detach().mean()
    scale = advantages.detach().std(unbiased=False)
    if float(scale) <= std_floor:
        return torch.zeros_like(advantages)
    return centered / scale


class LocalEventCritic(nn.Module):
    """Permutation-invariant value of a variable-width local candidate set."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(CANDIDATE_FEATURE_DIM, 64), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(64, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(self, features, mask):
        legal = mask.to(dtype=features.dtype).unsqueeze(-1)
        embeddings = self.encoder(features) * legal
        pooled = embeddings.sum(dim=1) / legal.sum(dim=1).clamp_min(1.)
        return self.value(pooled).squeeze(-1)


class RoutingPpoMethod:
    """On-policy clipped PPO over actual dynamic vehicle decision events."""

    needs_row_flow = False
    centralized = True
    ppo_epochs = 4
    minibatch_size = 512
    clip = 0.2
    entropy_coef = 0.002
    grad_clip = 10.

    def __init__(self, args, runner):
        self.args, self.runner, self.device = args, runner, runner.device
        self.ppo_epochs = getattr(args, 'ppo_epochs', 4)
        self.minibatch_size = getattr(args, 'ppo_minibatch_size', 512)
        self.target_kl = getattr(args, 'ppo_target_kl', .01)
        self.max_policy_kl = getattr(args, 'ppo_max_policy_kl', None)
        if self.max_policy_kl is None:
            self.max_policy_kl = 10. * self.target_kl
        self.max_backtracks = getattr(args, 'ppo_max_backtracks', 3)
        self.advantage_std_floor = getattr(args, 'ppo_advantage_std_floor', 1e-6)
        self.critic_epochs = getattr(args, 'ppo_critic_epochs', None)
        if self.critic_epochs is None:
            self.critic_epochs = 8 if self.centralized else 4
        self.critic_min_epochs = getattr(args, 'ppo_critic_min_epochs', None)
        if self.critic_min_epochs is None:
            self.critic_min_epochs = min(4, self.critic_epochs)
        self.critic_patience = getattr(args, 'ppo_critic_patience', 2)
        for name in ('ppo_epochs', 'minibatch_size', 'critic_epochs',
                     'critic_min_epochs', 'critic_patience'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'PPO {name} must be a positive integer')
        if self.critic_min_epochs > self.critic_epochs:
            raise ValueError('PPO critic_min_epochs must not exceed critic_epochs')
        if not math.isfinite(self.target_kl) or self.target_kl < 0:
            raise ValueError('PPO target_kl must be finite and nonnegative (0 disables stopping)')
        if not math.isfinite(self.max_policy_kl) or self.max_policy_kl < 0:
            raise ValueError('PPO max_policy_kl must be finite and nonnegative')
        if isinstance(self.max_backtracks, bool) or not isinstance(self.max_backtracks, int) or self.max_backtracks < 0:
            raise ValueError('PPO max_backtracks must be a nonnegative integer')
        if not math.isfinite(self.advantage_std_floor) or self.advantage_std_floor <= 0:
            raise ValueError('PPO advantage std floor must be finite and positive')
        self.actor = NextHopActor().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 73861)
            self.critic = (GlobalTrafficCritic(runner.network.edge_count, reward_context_dim(args),
                              normalize_hidden=True, center_free_flow=True) if self.centralized
                           else LocalEventCritic()).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        # Keep the centralized critic discoverable through the existing baseline interface.
        if self.centralized:
            self.global_critic, self.global_optimizer = self.critic, self.critic_optimizer
    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=True, temperature=1., sample_actions=None):
        with torch.no_grad():
            selection = self.actor.sample_action(
                torch.as_tensor(actor_obs.features, dtype=torch.float32, device=self.device),
                torch.as_tensor(actor_obs.mask, device=self.device), deterministic=not (explore if sample_actions is None else sample_actions),
                temperature=temperature)
            packed = torch.stack((selection.candidate_index.to(torch.float64),
                                  selection.entropy.to(torch.float64),
                                  selection.log_probability.to(torch.float64)), dim=-1).cpu().tolist()
        auxiliary = [dict(entropy=entry[1], old_log_prob=entry[2]) for entry in packed]
        return [int(entry[0]) for entry in packed], auxiliary

    def _row_tensors(self, rows):
        features = torch.as_tensor(np.stack([row['features'] for row in rows]),
                                   dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(np.stack([row['mask'] for row in rows]),
                               dtype=torch.bool, device=self.device)
        return features, mask

    def _targets(self, rows, windows):
        """Freeze values and lambda targets before any PPO optimization epoch."""
        if self.centralized:
            obs = torch.as_tensor(np.stack([window['obs'] for window in windows]),
                                  dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(np.stack([window['next_obs'] for window in windows]),
                                       dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor([window['reward'] for window in windows],
                                      dtype=torch.float32, device=self.device)
            terminal = torch.as_tensor([window['terminal'] for window in windows],
                                       dtype=torch.bool, device=self.device)
            with torch.no_grad():
                values = self.critic(obs).squeeze(-1)
                targets = lambda_return(rewards, self.critic(next_obs).squeeze(-1), terminal=terminal)
            event_windows = torch.as_tensor([row['window'] for row in rows],
                                            dtype=torch.long, device=self.device)
            return targets, (targets - values)[event_windows], (obs,)
        features, mask = self._row_tensors(rows)
        with torch.no_grad():
            values = self.critic(features, mask)
        targets = _local_lambda_targets(rows, values, windows, deadline=self.runner.deadline)
        return targets, targets - values, (features, mask)

    def learning_protocol(self):
        """Resolved clipped-PPO update semantics."""
        protocol = dict(version=2, centralized=self.centralized, ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size, target_kl=self.target_kl,
            actor_lr=float(self.args.actor_lr), critic_lr=float(self.args.critic_lr),
            advantage_normalization='centered_constant_zero',
            advantage_std_floor=self.advantage_std_floor, clip=self.clip,
            # The runtime coefficient is derived from the outer config and
            # episode counter, rather than being optimizer/checkpoint state.
            entropy_coef=type(self).entropy_coef, grad_clip=self.grad_clip,
            critic_epochs=self.critic_epochs, critic_min_epochs=self.critic_min_epochs,
            critic_patience=self.critic_patience, critic_selection='best_frozen_target_mse',
            gamma=DEFAULT_GAMMA, lambda_=DEFAULT_LAMBDA,
            reward_scope='global_team_episode_tail')
        protocol.update(version=7, actor_rows='multiple_legal_candidates',
                        kl_check='post_epoch_full_rollout_accept_or_restore_actor_and_adam',
                        max_policy_kl=self.max_policy_kl, max_backtracks=self.max_backtracks,
                        backtrack_factor=.5, actor_zero_gradient='skip_adam',
                        critic_conditioning='freeflow_centered_layernorm' if self.centralized else 'local_candidate_pool',
                        trace_clock='macro_windows' if self.centralized else 'physical_elapsed_seconds',
                        trace_seconds=30.,
                        actor_zero_objective='skip_each_minibatch',
                        policy_surrogate='sign_clipped_log_ratio')
        return protocol

    def _actor_advantages(self, advantages, rows):
        return normalize_advantages(advantages, std_floor=self.advantage_std_floor)

    @torch.no_grad()
    def _all_logits(self, features, mask):
        return torch.cat([self.actor.forward_logits(features[start:start + self.minibatch_size],
            mask[start:start + self.minibatch_size]) for start in range(0, len(features), self.minibatch_size)])

    @torch.no_grad()
    def _behavior_distribution(self, rows, features, mask, actions, old_log_prob):
        distribution = Categorical(logits=self._all_logits(features, mask))
        if not torch.allclose(distribution.log_prob(actions), old_log_prob, atol=1e-4, rtol=1e-5):
            raise ValueError('PPO collected behavior probabilities do not match the frozen actor')
        return distribution

    @torch.no_grad()
    def _critic_prediction(self, inputs):
        return torch.cat([self.critic(*(value[start:start + self.minibatch_size] for value in inputs))
            .reshape(-1) for start in range(0, len(inputs[0]), self.minibatch_size)])

    @staticmethod
    def _critic_statistics(prediction, targets):
        error = prediction - targets
        variance = targets.var(unbiased=False)
        explained = 1. - error.var(unbiased=False) / variance if float(variance) > 1e-8 else variance.new_zeros(())
        return dict(mse=float(error.square().mean()), explained_variance=float(explained),
                    bias=float(error.mean()))

    def _fit_critic(self, targets, inputs):
        """Bounded fit; restore both weights and Adam state at the best full-data MSE."""
        before = self._critic_statistics(self._critic_prediction(inputs), targets)
        best_loss, best_epoch, stale = before['mse'], 0, 0
        best = (deepcopy(self.critic.state_dict()), deepcopy(self.critic_optimizer.state_dict()))
        losses, gradients, steps = [], [], 0
        for epoch in range(1, self.critic_epochs + 1):
            for indices in torch.randperm(len(targets), device=self.device).split(self.minibatch_size):
                prediction = self.critic(*(tensor[indices] for tensor in inputs)).reshape(-1)
                loss = F.mse_loss(prediction, targets[indices])
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                if not bool(torch.isfinite(gradient)):
                    raise RuntimeError('non-finite PPO critic gradient')
                self.critic_optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
                gradients.append(float(gradient))
            fit_loss = self._critic_statistics(self._critic_prediction(inputs), targets)['mse']
            if fit_loss < best_loss:
                best_loss, best_epoch, stale = fit_loss, epoch, 0
                best = (deepcopy(self.critic.state_dict()), deepcopy(self.critic_optimizer.state_dict()))
            else:
                stale += 1
            if epoch >= self.critic_min_epochs and stale >= self.critic_patience:
                break
        self.critic.load_state_dict(best[0])
        self.critic_optimizer.load_state_dict(best[1])
        after = self._critic_statistics(self._critic_prediction(inputs), targets)
        return dict(critic_loss=float(np.mean(losses)), critic_gradient_norm=max(gradients),
            critic_optimizer_steps=steps, critic_epochs_completed=epoch, critic_best_epoch=best_epoch,
            critic_mse_before=before['mse'], critic_mse_after=after['mse'],
            critic_explained_variance_before=before['explained_variance'],
            critic_explained_variance_after=after['explained_variance'],
            critic_bias_before=before['bias'], critic_bias_after=after['bias'])

    @torch.no_grad()
    def _policy_statistics(self, old_distribution, features, mask, actions, old_log_prob):
        current = Categorical(logits=self._all_logits(features, mask))
        # Work in log space: softmax may underflow a tiny legal probability to
        # zero even though its finite logit gives a finite categorical KL.
        log_ratio = torch.where(mask, old_distribution.logits - current.logits, 0.)
        kl = (old_distribution.probs * log_ratio).sum(dim=-1).clamp_min(0.)
        ratio = (current.log_prob(actions) - old_log_prob).exp()
        return dict(policy_kl=float(kl.mean()), policy_kl_max=float(kl.max()),
            ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
            ratio_p01=float(torch.quantile(ratio, .01)), ratio_p99=float(torch.quantile(ratio, .99)),
            clip_fraction=float(((ratio < 1. - self.clip) | (ratio > 1. + self.clip)).float().mean()),
            policy_entropy=float(current.entropy().mean()))

    def _within_policy_budget(self, statistics):
        return (all(math.isfinite(value) for value in statistics.values())
            and (self.target_kl == 0 or statistics['policy_kl'] <= self.target_kl)
            and (self.max_policy_kl == 0 or statistics['policy_kl_max'] <= self.max_policy_kl))

    def _fit_actor(self, features, mask, actions, old_log_prob, old_distribution, advantages,
                   *, loss_function=None):
        """Accept bounded epochs; rejected proposals restore weights and Adam.

        Full-rollout checks occur once per attempted epoch, not per minibatch.
        Each retry reuses its permutation and halves only that proposal's LR.
        """
        count = len(actions)
        metrics = dict(actor_optimizer_steps=0, actor_attempted_steps=0,
            actor_epochs_completed=0, actor_rejected_updates=0, actor_backtracks=0,
            actor_lr_scale=0., actor_zero_gradient_steps=0, kl_early_stopped=False)
        metrics.update(self._policy_statistics(old_distribution, features, mask, actions, old_log_prob))
        losses, gradients = [], []
        if loss_function is None and self.entropy_coef == 0. and not bool(advantages.any()):
            return dict(metrics, actor_loss=0., actor_gradient_norm=0.)
        base_lrs = [group['lr'] for group in self.actor_optimizer.param_groups]
        for epoch in range(1, self.ppo_epochs + 1):
            actor_before = deepcopy(self.actor.state_dict())
            optimizer_before = deepcopy(self.actor_optimizer.state_dict())
            batches = torch.randperm(count, device=self.device).split(self.minibatch_size)
            accepted = False
            try:
                for retry in range(self.max_backtracks + 1):
                    for group, rate in zip(self.actor_optimizer.param_groups, base_lrs):
                        group['lr'] = rate * .5 ** retry
                    trial_losses, trial_gradients = [], []
                    minibatch_stopped = False
                    for indices in batches:
                        distribution = Categorical(logits=self.actor.forward_logits(features[indices], mask[indices]))
                        with torch.no_grad():
                            delta = torch.where(mask[indices],
                                old_distribution.logits[indices] - distribution.logits, 0.)
                            kl = (old_distribution.probs[indices] * delta).sum(-1).clamp_min(0.)
                        if ((self.target_kl > 0 and float(kl.mean()) > self.target_kl)
                                or (self.max_policy_kl > 0 and float(kl.max()) > self.max_policy_kl)):
                            minibatch_stopped = True
                            break
                        if loss_function is None and self.entropy_coef == 0. and not bool(advantages[indices].any()):
                            continue
                        loss = (clipped_policy_loss(distribution.log_prob(actions[indices]), old_log_prob[indices],
                                                   advantages[indices], clip=self.clip) if loss_function is None
                                else loss_function(distribution, indices))
                        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                            raise RuntimeError('non-finite or nonscalar actor objective')
                        loss = loss - self.entropy_coef * distribution.entropy().mean()
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                        if not bool(torch.isfinite(gradient)):
                            raise RuntimeError('non-finite PPO actor gradient')
                        if float(gradient) == 0.:
                            metrics['actor_zero_gradient_steps'] += 1
                            continue
                        self.actor_optimizer.step()
                        metrics['actor_attempted_steps'] += 1
                        trial_losses.append(float(loss.detach()))
                        trial_gradients.append(float(gradient))
                    statistics = self._policy_statistics(old_distribution, features, mask, actions, old_log_prob)
                    if self._within_policy_budget(statistics):
                        accepted = True
                        metrics.update(statistics)
                        metrics['actor_optimizer_steps'] += len(trial_losses)
                        metrics['actor_epochs_completed'] = epoch
                        metrics['actor_lr_scale'] = .5 ** retry
                        losses.extend(trial_losses)
                        gradients.extend(trial_gradients)
                        break
                    self.actor.load_state_dict(actor_before)
                    # Adam may alias input state tensors on the same device;
                    # each retry needs its own copy of the frozen snapshot.
                    self.actor_optimizer.load_state_dict(deepcopy(optimizer_before))
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    metrics['actor_rejected_updates'] += 1
                    if retry < self.max_backtracks:
                        metrics['actor_backtracks'] += 1
            except Exception:
                self.actor.load_state_dict(actor_before)
                self.actor_optimizer.load_state_dict(deepcopy(optimizer_before))
                self.actor_optimizer.zero_grad(set_to_none=True)
                raise
            finally:
                for group, rate in zip(self.actor_optimizer.param_groups, base_lrs):
                    group['lr'] = rate
            if not accepted or minibatch_stopped:
                metrics['kl_early_stopped'] = True
                break
        return dict(metrics, actor_loss=float(np.mean(losses)) if losses else 0.,
                    actor_gradient_norm=max(gradients, default=0.))

    def on_episode(self, rows, windows, *, episode):
        started = time.monotonic()
        zero = dict(actor_loss=0., actor_gradient_norm=0., critic_loss=0., critic_gradient_norm=0.,
                    advantage_std=0., actor_update_seconds=0., critic_update_seconds=0.,
                    world_model_seconds=0., imagination_seconds=0., learning_seconds=0.,
                    raw_advantage_mean=0., raw_advantage_std=0., raw_advantage_min=0.,
                    raw_advantage_max=0., raw_advantage_positive_fraction=0.,
                    policy_kl=0., policy_kl_max=0., ratio_min=1., ratio_max=1.,
                    ratio_p01=1., ratio_p99=1., clip_fraction=0., policy_entropy=0.,
                    actor_optimizer_steps=0, actor_epochs_completed=0, kl_early_stopped=False,
                    actor_attempted_steps=0, actor_rejected_updates=0, actor_backtracks=0,
                    actor_lr_scale=0., actor_zero_gradient_steps=0,
                    critic_optimizer_steps=0, critic_epochs_completed=0, critic_best_epoch=0,
                    critic_mse_before=0., critic_mse_after=0.,
                    critic_explained_variance_before=0., critic_explained_variance_after=0.,
                    critic_bias_before=0., critic_bias_after=0., ppo_epochs=self.ppo_epochs,
                    actor_decision_count=0, forced_decision_count=0)
        if not windows or (not self.centralized and not rows):
            return zero
        targets, advantages, critic_inputs = self._targets(rows, windows)
        if not bool(torch.isfinite(targets).all()):
            raise ValueError('PPO critic targets must be finite')
        metrics = dict(zero)
        actor_count = 0
        if rows:
            if any('old_log_prob' not in row for row in rows):
                raise ValueError('PPO needs collected behavior old_log_prob for every decision')
            features, mask = self._row_tensors(rows)
            actions = torch.as_tensor([row['action'] for row in rows], device=self.device)
            old_log_prob = torch.as_tensor([row['old_log_prob'] for row in rows],
                                           dtype=torch.float32, device=self.device)
            if not bool(torch.isfinite(old_log_prob).all()):
                raise ValueError('PPO behavior log probabilities must be finite')
            # The actor is unchanged during collection. Freeze its COMPLETE legal
            # distribution before either update, not just the sampled probabilities.
            old_distribution = self._behavior_distribution(rows, features, mask, actions, old_log_prob)
            # Forced routes have zero policy gradient, but including their returns
            # shifts/scales every actual choice's normalized advantage. They also
            # must not trigger Adam steps using stale moments from an earlier episode.
            choices = (mask.sum(dim=-1) > 1).nonzero(as_tuple=True)[0]
            actor_count = len(choices)
            metrics.update(actor_decision_count=actor_count,
                           forced_decision_count=len(rows) - actor_count)
            features, mask = features[choices], mask[choices]
            actions, old_log_prob = actions[choices], old_log_prob[choices]
            if actor_count:
                old_distribution = Categorical(logits=old_distribution.logits[choices])
                actor_rows = [rows[index] for index in choices.cpu().tolist()]
                advantages = advantages[choices]
                normalized = self._actor_advantages(advantages, actor_rows)
            else:
                advantages = advantages[:0]
        if actor_count:
            metrics.update(raw_advantage_mean=float(advantages.mean()),
                raw_advantage_std=float(advantages.std(unbiased=False)),
                raw_advantage_min=float(advantages.min()), raw_advantage_max=float(advantages.max()),
                raw_advantage_positive_fraction=float((advantages > 0).float().mean()),
                advantage_std=float(normalized.std(unbiased=False)))
            advantages = normalized
        critic_started = time.monotonic()
        metrics.update(self._fit_critic(targets, critic_inputs))
        metrics['critic_update_seconds'] = time.monotonic() - critic_started
        actor_started = time.monotonic()
        if actor_count:
            metrics.update(self._fit_actor(features, mask, actions, old_log_prob,
                                          old_distribution, advantages))
        metrics['actor_update_seconds'] = time.monotonic() - actor_started
        metrics['learning_seconds'] = time.monotonic() - started
        if self.centralized:
            metrics['global_critic_loss'] = metrics['critic_loss']
        return metrics

    def state_dict(self):
        return dict(learning_protocol=self.learning_protocol(), actor=self.actor.state_dict(),
            critic=self.critic.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict())

    def load_state_dict(self, state):
        if state.get('learning_protocol') != self.learning_protocol():
            raise ValueError('incompatible PPO learning protocol')
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])


class MappoMethod(RoutingPpoMethod):
    centralized = True


class IppoMethod(RoutingPpoMethod):
    centralized = False


METHODS['mappo'] = MappoMethod
METHODS['ippo'] = IppoMethod
