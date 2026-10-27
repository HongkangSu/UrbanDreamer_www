"""Generic discrete-action DreamerV2-style latent imagination actor--critic.

The learner consumes vector observations, action masks and transitions only.
No simulator, map, traffic model, physical state or deadline enters imagination.
This is a compact PyTorch vector-observation implementation, not DreamerV3.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import bisect

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def _mlp(dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def _available(logits):
    """Learned masks always retain their most likely action if all are false."""
    mask = logits > 0.
    fallback = F.one_hot(logits.argmax(-1), logits.shape[-1]).bool()
    return torch.where(mask.any(-1, keepdim=True), mask, fallback)


def _distribution(logits, mask):
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


class CategoricalRSSM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, categories=32, classes=32):
        super().__init__()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.hidden_dim, self.categories, self.classes = hidden_dim, categories, classes
        self.stoch_dim = categories * classes
        self.feature_dim = hidden_dim + self.stoch_dim
        self.encoder = _mlp([obs_dim, hidden_dim, hidden_dim])
        self.input = _mlp([self.stoch_dim + action_dim, hidden_dim])
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        self.prior = _mlp([hidden_dim, hidden_dim, self.stoch_dim])
        self.posterior = _mlp([2 * hidden_dim, hidden_dim, self.stoch_dim])
        self.decoder = _mlp([self.feature_dim, hidden_dim, hidden_dim, obs_dim])
        self.reward = _mlp([self.feature_dim, hidden_dim, hidden_dim, 1])
        self.continuation = _mlp([self.feature_dim, hidden_dim, hidden_dim, 1])
        self.availability = _mlp([self.feature_dim, hidden_dim, action_dim])

    def initial(self, batch, device):
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.stoch_dim, device=device))

    def _sample(self, logits, sample):
        logits = logits.reshape(*logits.shape[:-1], self.categories, self.classes)
        probs = logits.softmax(-1)
        indices = Categorical(logits=logits).sample() if sample else logits.argmax(-1)
        hard = F.one_hot(indices, self.classes).to(probs.dtype)
        return (hard + probs - probs.detach()).flatten(-2), logits

    def correct(self, h, observation, *, sample=True):
        raw = self.posterior(torch.cat((h, self.encoder(observation)), -1))
        z, logits = self._sample(raw, sample)
        return z, logits

    def transition(self, h, z, action, *, sample=True):
        onehot = F.one_hot(action.long(), self.action_dim).to(z.dtype)
        h = self.recurrent(self.input(torch.cat((z, onehot), -1)), h)
        z, logits = self._sample(self.prior(h), sample)
        return h, z, logits

    @staticmethod
    def features(h, z):
        return torch.cat((h, z), -1)


class GenericDreamer:
    """Categorical RSSM with a latent-only actor and learned latent rollouts."""
    schema = 'generic_vector_dreamer_v2_style_v1'

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.obs_dim, self.action_dim = int(config.obs_dim), int(config.action_dim)
        self.hidden_dim = int(getattr(config, 'hidden_dim', 256))
        self.gamma = float(getattr(config, 'gamma', .99))
        self.lambda_ = float(getattr(config, 'gae_lambda', getattr(config, 'lambda_', .95)))
        self.sequence_length = int(getattr(config, 'sequence_length', 16))
        self.batch_size = int(getattr(config, 'sequence_batch_size', min(16, getattr(config, 'batch_size', 256))))
        self.horizon = int(getattr(config, 'imagination_horizon', 15))
        self.capacity = int(getattr(config, 'replay_capacity', 200000))
        self.free_nats = float(getattr(config, 'free_nats', 1.))
        self.kl_scale = float(getattr(config, 'kl_scale', 1.))
        self.entropy_coef = float(getattr(config, 'entropy_coef', .001))
        self.target_tau = float(getattr(config, 'target_tau', .02))
        self.grad_clip = float(getattr(config, 'grad_clip', 100.))
        if min(self.obs_dim, self.action_dim, self.sequence_length, self.batch_size, self.horizon, self.capacity) < 1:
            raise ValueError('GenericDreamer dimensions and budgets must be positive')
        if not 0. <= self.gamma <= 1. or not 0. <= self.lambda_ <= 1.:
            raise ValueError('discount and lambda must be in [0,1]')
        torch.manual_seed(int(config.seed))
        self.rng = np.random.default_rng(int(config.seed) + 39239)
        self.world_model = CategoricalRSSM(self.obs_dim, self.action_dim, self.hidden_dim,
            int(getattr(config, 'stoch_categories', 32)), int(getattr(config, 'stoch_classes', 32))).to(self.device)
        dim = self.world_model.feature_dim
        self.actor = _mlp([dim, self.hidden_dim, self.hidden_dim, self.action_dim]).to(self.device)
        self.critic = _mlp([dim, self.hidden_dim, self.hidden_dim, 1]).to(self.device)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=float(getattr(config, 'model_lr', 3e-4)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, 'actor_lr', 3e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(getattr(config, 'critic_lr', 3e-4)))
        self.replay = deque()
        self.replay_size = 0
        self.observed_transitions = 0
        self.model_steps = self.actor_steps = self.critic_steps = 0
        self.reset()

    def reset(self):
        self._controller = {}

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, obs, mask, *, agent_ids=None, deterministic=False):
        obs = np.asarray(obs, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or mask.shape != (len(obs), self.action_dim):
            raise ValueError('observations/masks have incompatible shapes')
        if not np.isfinite(obs).all() or (len(mask) and not mask.any(-1).all()):
            raise ValueError('observations must be finite and each real action mask nonempty')
        if not len(obs):
            return np.empty(0, dtype=np.int64), []
        ids = list(range(len(obs))) if agent_ids is None else list(agent_ids)
        if len(ids) != len(obs) or len(set(ids)) != len(ids):
            raise ValueError('act requires one distinct identifier per observation')
        h, z = self.world_model.initial(len(obs), self.device)
        previous_action = torch.zeros(len(obs), dtype=torch.long, device=self.device)
        seen = torch.zeros(len(obs), dtype=torch.bool, device=self.device)
        for i, key in enumerate(ids):
            if key in self._controller:
                h[i], z[i], previous_action[i] = self._controller[key]
                seen[i] = True
        successor_h, _, _ = self.world_model.transition(h, z, previous_action, sample=not deterministic)
        h = torch.where(seen[:, None], successor_h, h)
        z, _ = self.world_model.correct(h, self._tensor(obs), sample=not deterministic)
        dist = _distribution(self.actor(self.world_model.features(h, z)), self._tensor(mask, torch.bool))
        action = dist.logits.argmax(-1) if deterministic else dist.sample()
        packed = torch.stack((action.float(), dist.entropy(), dist.log_prob(action)), -1).cpu().numpy()
        for i, key in enumerate(ids):
            self._controller[key] = (h[i].clone(), z[i].clone(), action[i].clone())
        return packed[:, 0].astype(np.int64), [dict(entropy=float(v[1]), old_log_prob=float(v[2])) for v in packed]

    def observe(self, episode):
        required = ('obs', 'actions', 'rewards', 'next_obs', 'terminals', 'masks', 'next_masks')
        arrays = {key: np.asarray(episode[key]) for key in required}
        count = len(arrays['actions'])
        if any(len(value) != count for value in arrays.values()):
            raise ValueError('all transition arrays must have the same length')
        if not count:
            return
        if arrays['obs'].shape != (count, self.obs_dim) or arrays['next_obs'].shape != (count, self.obs_dim):
            raise ValueError('observation dimension changed')
        if arrays['masks'].shape != (count, self.action_dim) or arrays['next_masks'].shape != (count, self.action_dim):
            raise ValueError('action dimension changed')
        for key in ('obs', 'next_obs', 'rewards'):
            if not np.isfinite(arrays[key]).all():
                raise ValueError('nonfinite transition '+key)
        if not arrays['masks'].any(-1).all():
            raise ValueError('real action masks must be nonempty')
        if bool((~arrays['terminals'].astype(bool) & ~arrays['next_masks'].any(-1)).any()):
            raise ValueError('nonterminal successors must have a legal action')
        actions = arrays['actions'].astype(np.int64)
        if ((actions < 0) | (actions >= self.action_dim)).any() or not arrays['masks'][np.arange(count), actions].all():
            raise ValueError('real actions must be legal under recorded masks')
        ids = np.asarray(episode.get('agent_ids', np.zeros(count, dtype=np.int64)))
        frames = np.asarray(episode.get('frame_indices', np.arange(count)))
        if ids.shape != (count,) or frames.shape != (count,):
            raise ValueError('agent identifiers and frame indices must have one value per transition')
        order = np.lexsort((frames, ids))
        ordered_ids = ids[order]
        boundaries = np.r_[0, np.flatnonzero(ordered_ids[1:] != ordered_ids[:-1]) + 1, count]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            selected = order[left:right]
            begin = 0
            for end in range(len(selected)):
                if bool(arrays['terminals'][selected[end]]) or end == len(selected) - 1:
                    chosen = selected[begin:end + 1]
                    if len(chosen) > self.capacity:
                        chosen = chosen[-self.capacity:]
                    data = {name: np.array(value[chosen], copy=True) for name, value in arrays.items()}
                    self.replay.append(data)
                    self.replay_size += len(chosen)
                    begin = end + 1
        while self.replay_size > self.capacity:
            self.replay_size -= len(self.replay.popleft()['actions'])
        self.observed_transitions += count

    def _sample(self):
        trajectories = list(self.replay)
        boundaries = np.cumsum([len(row['actions']) for row in trajectories]).tolist()
        picks = self.rng.integers(boundaries[-1], size=self.batch_size)
        shape = (self.batch_size, self.sequence_length)
        batch = dict(initial_obs=np.zeros((self.batch_size, self.obs_dim), np.float32),
            next_obs=np.zeros((*shape, self.obs_dim), np.float32), actions=np.zeros(shape, np.int64),
            rewards=np.zeros(shape, np.float32), terminals=np.ones(shape, bool),
            next_masks=np.zeros((*shape, self.action_dim), bool), valid=np.zeros(shape, bool))
        for b, pick in enumerate(picks):
            index = bisect.bisect_right(boundaries, int(pick))
            start = int(pick) - (boundaries[index - 1] if index else 0)
            row = trajectories[index]
            size = min(self.sequence_length, len(row['actions']) - start)
            batch['initial_obs'][b] = row['obs'][start]
            for key in ('next_obs', 'actions', 'rewards', 'terminals', 'next_masks'):
                batch[key][b, :size] = row[key][start:start + size]
            batch['valid'][b, :size] = True
        return {key: self._tensor(value, torch.long if key == 'actions' else torch.bool if key in ('terminals', 'next_masks', 'valid') else torch.float32)
                for key, value in batch.items()}

    def _world_loss(self, batch):
        model = self.world_model
        h, z = model.initial(len(batch['initial_obs']), self.device)
        z, _ = model.correct(h, batch['initial_obs'])
        valid = batch['valid'].float()
        denominator = valid.sum().clamp_min(1.)
        initial_features = model.features(h, z)
        losses = dict(reconstruction=F.mse_loss(model.decoder(initial_features), batch['initial_obs'],
            reduction='none').mean(-1).sum(), reward=0., continuation=0., availability=0., kl=0.)
        # Every recorded observation preceded a real legal action. Terminal
        # successors are reconstruction targets, never alive imagination seeds.
        seeds = [initial_features.detach()]
        for t in range(batch['actions'].shape[1]):
            h, _, prior = model.transition(h, z, batch['actions'][:, t])
            z, posterior = model.correct(h, batch['next_obs'][:, t])
            feature = model.features(h, z)
            target = batch['next_obs'][:, t]
            mask = valid[:, t]
            losses['reconstruction'] += (F.mse_loss(model.decoder(feature), target, reduction='none').mean(-1) * mask).sum()
            losses['reward'] += (F.smooth_l1_loss(model.reward(feature).squeeze(-1), batch['rewards'][:, t], reduction='none') * mask).sum()
            losses['continuation'] += (F.binary_cross_entropy_with_logits(model.continuation(feature).squeeze(-1), (~batch['terminals'][:, t]).float(), reduction='none') * mask).sum()
            losses['availability'] += (F.binary_cross_entropy_with_logits(model.availability(feature), batch['next_masks'][:, t].float(), reduction='none').mean(-1) * mask).sum()
            dynamic = torch.distributions.kl_divergence(Categorical(logits=posterior.detach()), Categorical(logits=prior)).sum(-1)
            representation = torch.distributions.kl_divergence(Categorical(logits=posterior), Categorical(logits=prior.detach())).sum(-1)
            losses['kl'] += ((.8 * dynamic.clamp_min(self.free_nats) + .2 * representation.clamp_min(self.free_nats)) * mask).sum()
            alive = batch['valid'][:, t] & ~batch['terminals'][:, t]
            if alive.any():
                seeds.append(feature[alive].detach())
        losses = {key: value / (denominator + len(batch['initial_obs']) if key == 'reconstruction'
                                else denominator) for key, value in losses.items()}
        total = sum(value * (self.kl_scale if key == 'kl' else 1.) for key, value in losses.items())
        return total, losses, torch.cat(seeds)

    def _step(self, optimizer, loss, parameters):
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite generic Dreamer loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('nonfinite generic Dreamer gradient')
        optimizer.step()

    @torch.no_grad()
    def _imagine(self, seed_features):
        model = self.world_model
        limit = int(getattr(self.config, 'batch_size', 256))
        if len(seed_features) > limit:
            selected = self.rng.choice(len(seed_features), size=limit, replace=False)
            seed_features = seed_features[selected]
        h, z = seed_features[:, :self.hidden_dim], seed_features[:, self.hidden_dim:]
        features, actions, masks, rewards, discounts = [], [], [], [], []
        for _ in range(self.horizon):
            feature = model.features(h, z)
            mask = _available(model.availability(feature))
            action = _distribution(self.actor(feature), mask).sample()
            features.append(feature)
            actions.append(action)
            masks.append(mask)
            h, z, _ = model.transition(h, z, action)
            successor = model.features(h, z)
            rewards.append(model.reward(successor).squeeze(-1))
            discounts.append(self.gamma * model.continuation(successor).sigmoid().squeeze(-1))
        bootstrap = self.target_critic(model.features(h, z)).squeeze(-1)
        values = self.target_critic(torch.stack(features)).squeeze(-1)
        running = bootstrap
        returns = []
        for t in reversed(range(self.horizon)):
            next_value = bootstrap if t == self.horizon - 1 else values[t + 1]
            running = rewards[t] + discounts[t] * ((1. - self.lambda_) * next_value + self.lambda_ * running)
            returns.append(running)
        discounts = torch.stack(discounts)
        weights = torch.cat((torch.ones_like(discounts[:1]), discounts[:-1]), 0).cumprod(0)
        return dict(features=torch.stack(features), actions=torch.stack(actions), masks=torch.stack(masks),
                    returns=torch.stack(returns[::-1]), weights=weights, rewards=torch.stack(rewards), discounts=discounts)

    def _policy_update(self, seed_features):
        imagined = self._imagine(seed_features)
        features, targets, weights = (imagined[key].detach() for key in ('features', 'returns', 'weights'))
        predictions = self.critic(features).squeeze(-1)
        advantages = (targets - predictions.detach())
        advantages = advantages / advantages.std(unbiased=False).clamp_min(1.)
        distribution = _distribution(self.actor(features), imagined['masks'])
        denominator = weights.sum().clamp_min(1.)
        actor_loss = -(weights * (distribution.log_prob(imagined['actions']) * advantages.detach()
                                  + self.entropy_coef * distribution.entropy())).sum() / denominator
        critic_loss = (weights * F.smooth_l1_loss(predictions, targets, reduction='none')).sum() / denominator
        self._step(self.actor_optimizer, actor_loss, list(self.actor.parameters()))
        self._step(self.critic_optimizer, critic_loss, list(self.critic.parameters()))
        with torch.no_grad():
            for source, target in zip(self.critic.parameters(), self.target_critic.parameters()):
                target.lerp_(source, self.target_tau)
        self.actor_steps += 1
        self.critic_steps += 1
        return dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                    imagined_reward=float(imagined['rewards'].mean()), imagined_continuation=float((imagined['discounts'] / max(self.gamma, 1e-8)).mean()))

    def update(self, updates=None):
        model_updates = int(getattr(self.config, 'model_updates', 4) if updates is None else updates)
        policy_updates = int(getattr(self.config, 'policy_updates', 4) if updates is None else updates)
        if min(model_updates, policy_updates) < 0:
            raise ValueError('update counts must be nonnegative')
        if not self.replay_size:
            return dict(model_ready=0., model_loss=0., actor_loss=0., critic_loss=0., model_optimizer_steps=0, actor_optimizer_steps=0, critic_optimizer_steps=0)
        losses = []
        seeds = None
        for _ in range(model_updates):
            loss, parts, seeds = self._world_loss(self._sample())
            self._step(self.model_optimizer, loss, list(self.world_model.parameters()))
            self.model_steps += 1
            losses.append({key+'_loss': float(value.detach()) for key, value in parts.items()} | {'model_loss': float(loss.detach())})
        if seeds is None and policy_updates:
            with torch.no_grad():
                _, _, seeds = self._world_loss(self._sample())
        policy = [self._policy_update(seeds) for _ in range(policy_updates)]
        metrics = {key: float(np.mean([entry[key] for entry in losses])) for key in losses[0]} if losses else {}
        if policy:
            metrics.update({key: float(np.mean([entry[key] for entry in policy])) for key in policy[0]})
        return dict(model_ready=1., replay_size=self.replay_size,
            model_optimizer_steps=model_updates, actor_optimizer_steps=policy_updates, critic_optimizer_steps=policy_updates,
            total_model_optimizer_steps=self.model_steps, total_actor_optimizer_steps=self.actor_steps,
            total_critic_optimizer_steps=self.critic_steps, **metrics)

    def learning_protocol(self):
        return dict(schema=self.schema, obs_dim=self.obs_dim, action_dim=self.action_dim,
            hidden_dim=self.hidden_dim, categories=self.world_model.categories, classes=self.world_model.classes,
            actor_input='rssm_latent_only', imagination='learned_latent_transition_reward_continuation_and_availability',
            real_actor_updates=False, gamma=self.gamma, lambda_=self.lambda_, free_nats=self.free_nats,
            kl_scale=self.kl_scale, target_tau=self.target_tau, sequence_length=self.sequence_length,
            imagination_horizon=self.horizon, sequence_batch_size=self.batch_size,
            imagination_batch_size=int(getattr(self.config, 'batch_size', 256)),
            entropy_coef=self.entropy_coef, gradient_clip=self.grad_clip, replay_capacity=self.capacity,
            model_updates=int(getattr(self.config, 'model_updates', 4)),
            policy_updates=int(getattr(self.config, 'policy_updates', 4)),
            model_lr=float(self.model_optimizer.param_groups[0]['lr']),
            actor_lr=float(self.actor_optimizer.param_groups[0]['lr']),
            critic_lr=float(self.critic_optimizer.param_groups[0]['lr']))

    def inference_state(self):
        return dict(protocol=self.learning_protocol(), world_model={key: value.detach().cpu().clone() for key, value in self.world_model.state_dict().items()},
                    actor={key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()})

    def load_inference_state(self, state):
        if state['protocol'] != self.learning_protocol():
            raise ValueError('generic Dreamer inference protocol differs')
        self.world_model.load_state_dict(state['world_model'])
        self.actor.load_state_dict(state['actor'])
        self.reset()

    def state_dict(self, include_replay=False):
        result = dict(protocol=self.learning_protocol(), world_model=self.world_model.state_dict(), actor=self.actor.state_dict(),
            critic=self.critic.state_dict(), target_critic=self.target_critic.state_dict(),
            model_optimizer=self.model_optimizer.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(), model_steps=self.model_steps, actor_steps=self.actor_steps,
            critic_steps=self.critic_steps, observed_transitions=self.observed_transitions,
            replay=deepcopy(list(self.replay)) if include_replay else [], replay_included=bool(include_replay),
            rng=deepcopy(self.rng.bit_generator.state), torch_rng=torch.get_rng_state())
        if self.device.type == 'cuda':
            result['cuda_rng'] = torch.cuda.get_rng_state(self.device)
        return result

    def load_state_dict(self, state):
        if state['protocol'] != self.learning_protocol():
            raise ValueError('generic Dreamer checkpoint protocol differs')
        for name in ('world_model', 'actor', 'critic', 'target_critic', 'model_optimizer', 'actor_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.model_steps, self.actor_steps, self.critic_steps = (int(state[key]) for key in ('model_steps', 'actor_steps', 'critic_steps'))
        self.observed_transitions = int(state['observed_transitions'])
        self.replay = deque(deepcopy(state.get('replay', [])))
        self.replay_size = sum(len(row['actions']) for row in self.replay)
        self.rng.bit_generator.state = deepcopy(state['rng'])
        torch.set_rng_state(state['torch_rng'])
        if self.device.type == 'cuda' and 'cuda_rng' in state:
            torch.cuda.set_rng_state(state['cuda_rng'], self.device)
        self.reset()
