"""Generic discrete MAMBA with learned latent imagination.

The learner accepts only observations, action masks, rewards, terminals and
agent identities. It has no environment, graph or physics access. Agent
communication is bounded to fixed-size chunks for memory, with all chunks
processed as one tensor batch. Parameters are shared across dynamic agents.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict
import math

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from .mamba_core.config import MambaConfig
from .mamba_core.networks import MambaActor, MambaCritic, MambaWorldModel, RSSMState

SCHEMA = 'generic_mamba_latent_v1'
_KEYS = ('obs', 'actions', 'rewards', 'next_obs', 'terminals', 'masks', 'next_masks',
         'agent_ids', 'frame_indices')


def learned_action_mask(logits: torch.Tensor) -> torch.Tensor:
    """Use learned availability; an empty prediction falls back to its argmax."""
    mask = logits >= 0.
    fallback = F.one_hot(logits.argmax(-1), logits.shape[-1]).bool()
    return torch.where(mask.any(-1, keepdim=True), mask, fallback)


def lambda_returns(rewards, values, continuation, bootstrap, gamma, lambda_):
    result = torch.empty_like(rewards)
    running = bootstrap
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        running = rewards[t] + gamma * continuation[t] * ((1. - lambda_) * next_value + lambda_ * running)
        result[t] = running
    return result


def _where_state(mask, yes, no):
    return RSSMState(torch.where(mask[..., None, None], yes.logits, no.logits),
                     torch.where(mask[..., None], yes.stoch, no.stoch),
                     torch.where(mask[..., None], yes.deter, no.deter))


def _mean(value, valid):
    return (value * valid.to(value)).sum() / valid.sum().clamp_min(1)


class GenericMAMBA:
    schema = SCHEMA

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.obs_dim, self.action_dim = int(config.obs_dim), int(config.action_dim)
        hidden = int(getattr(config, 'hidden_dim', 256))
        self.cfg = MambaConfig(obs_dim=self.obs_dim, action_dim=self.action_dim,
            device=str(self.device), seed=int(config.seed), hidden_size=hidden,
            model_hidden_size=hidden, embedding_size=hidden, deterministic_size=hidden,
            n_categoricals=int(getattr(config, 'latent_variables', 32)),
            n_classes=int(getattr(config, 'latent_classes', 32)),
            transformer_layers=int(getattr(config, 'transformer_layers', 3)),
            transformer_heads=8 if hidden % 8 == 0 else 1, transformer_dropout=0.)
        self.chunk = int(getattr(config, 'max_agents_per_chunk', getattr(config, 'agent_chunk_size', 32)))
        self.sequence_length = int(getattr(config, 'sequence_length', 16))
        self.sequence_batch_size = int(getattr(config, 'mamba_sequence_batch_size', getattr(config, 'sequence_batch_size', 2)))
        self.max_starts = int(getattr(config, 'mamba_imagination_starts', getattr(config, 'imagination_starts', getattr(config, 'max_imagination_starts', 8))))
        self.horizon = int(getattr(config, 'imagination_horizon', 15))
        self.gamma = float(getattr(config, 'gamma', .99))
        self.lambda_ = float(getattr(config, 'gae_lambda', getattr(config, 'lambda_', .95)))
        if min(self.chunk, self.sequence_batch_size, self.max_starts, self.horizon) < 1 or self.sequence_length < 2:
            raise ValueError('positive batch, chunk, horizon and sequence_length >=2 required')
        self.rng = np.random.default_rng(int(config.seed) + 9371)
        torch.manual_seed(int(config.seed))
        self.world_model = MambaWorldModel(self.cfg).to(self.device)
        self.actor = MambaActor(self.cfg).to(self.device)
        self.critic = MambaCritic(self.cfg).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=float(getattr(config, 'model_lr', 3e-4)))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(getattr(config, 'actor_lr', 3e-4)))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(getattr(config, 'critic_lr', 3e-4)))
        self.entropy_coef = float(getattr(config, 'entropy_coef', 1e-3))
        self.clip = float(getattr(config, 'ppo_clip', .075))
        self.tau = float(getattr(config, 'target_tau', .05))
        self.replay = deque()
        self.replay_size = 0
        self.updates = 0
        self.reset()

    def reset(self):
        self._ids = {}
        self._capacity = 0
        self._controller = self.world_model.initial_state(1, 0, device=self.device)
        self._previous = torch.zeros(1, 0, self.action_dim, device=self.device)

    def _capacity_for(self, size):
        if size <= self._capacity:
            return
        capacity = max(32, 1 << (size - 1).bit_length())
        state = self.world_model.initial_state(1, capacity, device=self.device)
        for name in ('logits', 'stoch', 'deter'):
            getattr(state, name)[:, :self._capacity] = getattr(self._controller, name)
        previous = torch.zeros(1, capacity, self.action_dim, device=self.device)
        previous[:, :self._capacity] = self._previous
        self._controller, self._previous, self._capacity = state, previous, capacity

    @torch.no_grad()
    def act(self, obs, mask, *, agent_ids=None, deterministic=False):
        obs = np.asarray(obs, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        count = len(obs)
        if obs.shape != (count, self.obs_dim) or mask.shape != (count, self.action_dim):
            raise ValueError('observation or action mask shape mismatch')
        if not count:
            return np.empty(0, dtype=np.int64), []
        if not np.isfinite(obs).all() or not mask.any(-1).all():
            raise ValueError('finite observations and a legal real action are required')
        ids = [str(value) for value in (range(count) if agent_ids is None else agent_ids)]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError('one distinct identity per simultaneous agent is required')
        for identity in ids:
            if identity not in self._ids:
                self._ids[identity] = len(self._ids)
        self._capacity_for(len(self._ids))
        # Identity-sorted grouping avoids dependence on input permutation.
        order = np.asarray(sorted(range(count), key=lambda i: ids[i]), dtype=np.int64)
        slots = torch.tensor([self._ids[ids[i]] for i in order], device=self.device)
        groups = math.ceil(count / self.chunk)
        size = groups * self.chunk
        present = torch.arange(size, device=self.device).reshape(groups, self.chunk) < count
        observations = torch.zeros(size, self.obs_dim, device=self.device)
        observations[:count] = torch.as_tensor(obs[order], device=self.device)
        availability = torch.ones(size, self.action_dim, dtype=torch.bool, device=self.device)
        availability[:count] = torch.as_tensor(mask[order], device=self.device)
        state = self.world_model.initial_state(groups, self.chunk, device=self.device)
        for name in ('logits', 'stoch', 'deter'):
            tensor = getattr(state, name)
            tensor.reshape(size, *tensor.shape[2:])[:count] = getattr(self._controller, name)[0, slots]
        previous = torch.zeros(size, self.action_dim, device=self.device)
        previous[:count] = self._previous[0, slots]
        self.world_model.eval()
        self.actor.eval()
        _, posterior = self.world_model.observe(observations.reshape(groups, self.chunk, -1),
            previous.reshape(groups, self.chunk, -1), state, present, present, sample=not deterministic)
        distribution = self.actor.distribution(posterior.features, availability.reshape(groups, self.chunk, -1))
        action = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        selected = action.reshape(-1)[:count]
        packed = torch.stack((selected.float(), distribution.log_prob(action).reshape(-1)[:count],
                              distribution.entropy().reshape(-1)[:count]), -1).cpu().numpy()
        for name in ('logits', 'stoch', 'deter'):
            tensor = getattr(posterior, name)
            getattr(self._controller, name)[0, slots] = tensor.reshape(size, *tensor.shape[2:])[:count]
        self._previous[0, slots] = F.one_hot(selected, self.action_dim).float()
        restored = np.empty_like(packed)
        restored[order] = packed
        return restored[:, 0].astype(np.int64), [dict(old_log_prob=float(row[1]), entropy=float(row[2])) for row in restored]

    def observe(self, episode):
        data = {}
        for name in _KEYS:
            if name not in episode:
                if name == 'agent_ids':
                    value = np.zeros(len(episode['actions']), dtype=np.int64)
                elif name == 'frame_indices':
                    value = np.arange(len(episode['actions']), dtype=np.int64)
                else:
                    raise ValueError('missing generic episode field: ' + name)
            else:
                value = episode[name]
            data[name] = np.asarray(value).copy()
        count = len(data['actions'])
        if not count:
            return
        if any(len(value) != count for value in data.values()):
            raise ValueError('generic transition fields must have equal lengths')
        if data['obs'].shape != (count, self.obs_dim) or data['next_obs'].shape != (count, self.obs_dim):
            raise ValueError('observation width mismatch')
        if data['masks'].shape != (count, self.action_dim) or data['next_masks'].shape != (count, self.action_dim):
            raise ValueError('mask width mismatch')
        if not all(np.isfinite(data[key]).all() for key in ('obs', 'next_obs', 'rewards')):
            raise ValueError('generic transitions must be finite')
        actions = data['actions']
        if ((actions < 0) | (actions >= self.action_dim)).any():
            raise ValueError('recorded actions must be in range')
        if not data['masks'][np.arange(count), actions.astype(np.int64)].all():
            raise ValueError('recorded actions must be legal under their masks')
        if not data['next_masks'][~data['terminals'].astype(bool)].any(-1).all():
            raise ValueError('nonterminal successors require a legal action')
        data['agent_ids'] = data['agent_ids'].astype(str)
        frames = []
        for time_index in np.unique(data['frame_indices']):
            indices = np.flatnonzero(data['frame_indices'] == time_index)
            # Repeated identities are successive events, never overwritten.
            current, seen = [], set()
            for index in indices:
                identity = data['agent_ids'][index]
                if identity in seen:
                    frames.append(np.asarray(current, dtype=np.int64))
                    current, seen = [], set()
                current.append(index)
                seen.add(identity)
            if current:
                frames.append(np.asarray(current, dtype=np.int64))
        data['_frames'] = frames
        self.replay.append(data)
        self.replay_size += count
        capacity = int(getattr(self.config, 'replay_capacity', 200000))
        while self.replay_size > capacity and len(self.replay) > 1:
            self.replay_size -= len(self.replay.popleft()['actions'])

    def _sample(self):
        eligible = [data for data in self.replay if len(data['_frames']) >= 2]
        if not eligible:
            return None
        b, t, a = self.sequence_batch_size, self.sequence_length, self.chunk
        arrays = dict(obs=np.zeros((t, b, a, self.obs_dim), np.float32),
            next_obs=np.zeros((t, b, a, self.obs_dim), np.float32),
            actions=np.zeros((t, b, a), np.int64), rewards=np.zeros((t, b, a), np.float32),
            terminals=np.ones((t, b, a), bool), masks=np.ones((t, b, a, self.action_dim), bool),
            next_masks=np.ones((t, b, a, self.action_dim), bool), presence=np.zeros((t, b, a), bool))
        for batch in range(b):
            data = eligible[int(self.rng.integers(len(eligible)))]
            start = int(self.rng.integers(max(1, len(data['_frames']) - 1)))
            frames = data['_frames'][start:start + t]
            identities = np.unique(np.concatenate([data['agent_ids'][frame] for frame in frames]))
            if len(identities) > a:
                identities = self.rng.choice(identities, a, replace=False)
            locations = {identity: i for i, identity in enumerate(identities)}
            for step, indices in enumerate(frames):
                for index in indices:
                    slot = locations.get(data['agent_ids'][index])
                    if slot is None:
                        continue
                    for key in ('obs', 'next_obs', 'actions', 'rewards', 'terminals', 'masks', 'next_masks'):
                        arrays[key][step, batch, slot] = data[key][index]
                    arrays['presence'][step, batch, slot] = True
        return {key: torch.as_tensor(value, device=self.device) for key, value in arrays.items()}

    def _world_loss(self, batch, *, seed_only=False):
        t, b, a = batch['presence'].shape
        state = self.world_model.initial_state(b, a, device=self.device)
        previous = torch.zeros(b, a, self.action_dim, device=self.device)
        losses, seeds = [], []
        for step in range(t):
            present = batch['presence'][step]
            prior, post = self.world_model.observe(batch['obs'][step], previous, state, present, present)
            post = _where_state(present, post, state)
            seeds.append(post)
            action = F.one_hot(batch['actions'][step], self.action_dim).float()
            if not seed_only:
                consequence = self.world_model.transition(action, post, present)
                features = torch.cat((post.stoch, consequence.deter), -1)
                reconstruction = F.smooth_l1_loss(self.world_model.observation_decoder(post.features),
                    batch['obs'][step], reduction='none').sum(-1)
                reward = F.smooth_l1_loss(self.world_model.reward_model(features).squeeze(-1), batch['rewards'][step], reduction='none')
                continuation = F.binary_cross_entropy_with_logits(self.world_model.continuation_model(features).squeeze(-1),
                    (~batch['terminals'][step]).float(), reduction='none')
                availability = F.binary_cross_entropy_with_logits(self.world_model.availability_model(post.features), batch['masks'][step].float(), reduction='none').mean(-1)
                information = F.cross_entropy(self.world_model.action_information_model(features).reshape(-1, self.action_dim), batch['actions'][step].reshape(-1), reduction='none').reshape(b, a)
                p, q = post.logits.softmax(-1), prior.logits.softmax(-1)
                log_p, log_q = post.logits.log_softmax(-1), prior.logits.log_softmax(-1)
                kl = .8 * (p.detach() * (log_p.detach() - log_q)).sum(-1).mean(-1)
                kl = kl + .2 * (p * (log_p - log_q.detach())).sum(-1).mean(-1)
                losses.append(_mean(reconstruction + reward + continuation + availability + information
                    + self.cfg.kl_coef * kl, present))
            previous = torch.where(present[..., None], action, previous)
            state = post
        logits = torch.stack([s.logits for s in seeds]).reshape(t * b, a, self.cfg.n_categoricals, self.cfg.n_classes)
        stoch = torch.stack([s.stoch for s in seeds]).reshape(t * b, a, self.cfg.stochastic_size)
        deter = torch.stack([s.deter for s in seeds]).reshape(t * b, a, self.cfg.deterministic_size)
        presence = batch['presence'].reshape(t * b, a)
        indices = torch.nonzero(presence.any(-1), as_tuple=False).flatten().cpu().numpy()
        if len(indices) > self.max_starts:
            indices = self.rng.choice(indices, self.max_starts, replace=False)
        indices = torch.as_tensor(indices, device=self.device)
        starts = RSSMState(logits[indices].detach(), stoch[indices].detach(), deter[indices].detach())
        seed = (starts, presence[indices], batch['masks'].reshape(t * b, a, self.action_dim)[indices])
        return (torch.stack(losses).mean() if losses else None), seed

    @torch.no_grad()
    def _imagine(self, seed):
        state, presence, availability = seed
        availability = torch.where(presence[..., None], availability, torch.ones_like(availability))
        features, actions, old_log_probs, rewards, continuations, values, weights, masks = [], [], [], [], [], [], [], []
        weight = presence.float()
        for _ in range(self.horizon):
            features.append(state.features)
            values.append(self.target_critic(state.features, presence).squeeze(-1))
            weights.append(weight)
            masks.append(availability)
            distribution = self.actor.distribution(state.features, availability)
            action = distribution.sample()
            actions.append(action)
            old_log_probs.append(distribution.log_prob(action))
            next_state = self.world_model.transition(F.one_hot(action, self.action_dim).float(), state, presence)
            consequence = torch.cat((state.stoch, next_state.deter), -1)
            individual_reward = self.world_model.reward_model(consequence).squeeze(-1)
            team_reward = (individual_reward * presence).sum(-1, keepdim=True) / presence.sum(-1, keepdim=True).clamp_min(1)
            rewards.append(team_reward.expand_as(individual_reward))
            continuation = self.world_model.continuation_model(consequence).sigmoid().squeeze(-1) * presence
            continuations.append(continuation)
            weight = weight * self.gamma * continuation
            state = next_state
            availability = learned_action_mask(self.world_model.availability_model(state.features))
            availability = torch.where(presence[..., None], availability, torch.ones_like(availability))
        bootstrap = self.target_critic(state.features, presence).squeeze(-1)
        returns = lambda_returns(torch.stack(rewards), torch.stack(values), torch.stack(continuations), bootstrap, self.gamma, self.lambda_)
        return dict(features=torch.stack(features), actions=torch.stack(actions), old_log_probs=torch.stack(old_log_probs),
            returns=returns, values=torch.stack(values), weights=torch.stack(weights), presence=presence,
            masks=torch.stack(masks))

    @staticmethod
    def _step(optimizer, loss, parameters):
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite generic MAMBA loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError('nonfinite generic MAMBA gradient')
        optimizer.step()

    def update(self, updates=None):
        model_steps = int(getattr(self.config, 'model_updates', 4))
        policy_steps = int(getattr(self.config, 'policy_updates', 4) if updates is None else updates)
        metrics = dict(world_model_loss=0., actor_loss=0., critic_loss=0., imagined_events=0,
                       model_optimizer_steps=0, actor_optimizer_steps=0, critic_optimizer_steps=0)
        if not any(len(data['_frames']) >= 2 for data in self.replay):
            return dict(metrics, skipped=1.)
        self.world_model.train()
        world_losses = []
        for _ in range(model_steps):
            batch = self._sample()
            loss, _ = self._world_loss(batch)
            self._step(self.model_optimizer, loss, list(self.world_model.parameters()))
            world_losses.append(float(loss.detach()))
        self.world_model.eval()
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            _, seed = self._world_loss(batch, seed_only=True)
        rollout = self._imagine(seed)
        # Keep availability and behavior probabilities frozen through PPO.
        features, actions = rollout['features'], rollout['actions']
        presence = rollout['presence'].expand(self.horizon, -1, -1)
        weights = rollout['weights']
        advantages = (rollout['returns'] - rollout['values']).detach()
        selected = advantages[presence]
        if selected.numel() > 1 and float(selected.std(unbiased=False)) >= 1e-6:
            advantages = (advantages - selected.mean()) / selected.std(unbiased=False)
        else:
            advantages = advantages / selected.abs().amax().clamp_min(1.)
        actor_losses, critic_losses = [], []
        for _ in range(policy_steps):
            distributions = Categorical(logits=self.actor.logits(features).masked_fill(~rollout['masks'], torch.finfo(features.dtype).min))
            log_ratio = distributions.log_prob(actions) - rollout['old_log_probs']
            clipped = torch.where(advantages >= 0., log_ratio.clamp_max(math.log1p(self.clip)), log_ratio.clamp_min(math.log1p(-self.clip))).clamp_max(20.)
            actor_loss = -_mean((clipped.exp() * advantages + self.entropy_coef * distributions.entropy()) * weights, presence)
            self._step(self.actor_optimizer, actor_loss, list(self.actor.parameters()))
            predicted = self.critic(features.flatten(0, 1), presence.flatten(0, 1)).reshape_as(advantages)
            critic_loss = _mean(F.smooth_l1_loss(predicted, rollout['returns'], reduction='none') * weights, presence)
            self._step(self.critic_optimizer, critic_loss, list(self.critic.parameters()))
            with torch.no_grad():
                for target, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target.lerp_(current, self.tau)
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
        self.updates += 1
        metrics.update(world_model_loss=float(np.mean(world_losses)), actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.,
            critic_loss=float(np.mean(critic_losses)) if critic_losses else 0., imagined_events=int(presence.sum()),
            model_optimizer_steps=model_steps, actor_optimizer_steps=policy_steps, critic_optimizer_steps=policy_steps,
            skipped=0., replay_transitions=self.replay_size, communication_chunk=self.chunk)
        return metrics

    def inference_state(self):
        return dict(schema=SCHEMA, world_model={k: v.detach().cpu().clone() for k, v in self.world_model.state_dict().items()},
                    actor={k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()})

    def load_inference_state(self, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('generic MAMBA inference schema mismatch')
        self.world_model.load_state_dict(state['world_model'])
        self.actor.load_state_dict(state['actor'])
        self.reset()

    def state_dict(self, include_replay=False):
        return dict(schema=SCHEMA, cfg=asdict(self.cfg), protocol=self.learning_protocol(),
            **{key: getattr(self, key).state_dict() for key in ('world_model', 'actor', 'critic', 'target_critic',
                'model_optimizer', 'actor_optimizer', 'critic_optimizer')},
            updates=self.updates, replay=copy.deepcopy(list(self.replay)) if include_replay else [],
            replay_reset_on_restore=not include_replay, rng=copy.deepcopy(self.rng.bit_generator.state),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None)

    def learning_protocol(self):
        return dict(schema=SCHEMA, chunk=self.chunk, sequence_length=self.sequence_length,
            sequence_batch_size=self.sequence_batch_size, max_starts=self.max_starts,
            horizon=self.horizon, gamma=self.gamma, lambda_=self.lambda_,
            entropy_coef=self.entropy_coef, clip=self.clip, target_tau=self.tau,
            model_updates=int(getattr(self.config, 'model_updates', 4)),
            policy_updates=int(getattr(self.config, 'policy_updates', 4)),
            replay_capacity=int(getattr(self.config, 'replay_capacity', 200000)),
            model_lr=float(getattr(self.config, 'model_lr', 3e-4)),
            actor_lr=float(getattr(self.config, 'actor_lr', 3e-4)),
            critic_lr=float(getattr(self.config, 'critic_lr', 3e-4)),
            kl_coef=self.cfg.kl_coef, reconstruction='smooth_l1_sum_observation_dimensions',
            imagination='learned_latent_transition_reward_continuation_and_availability',
            policy_input='posterior_latent_only', reward_scope='learned_team_mean',
            communication='batched_fixed_size_identity_sorted_chunks')

    def load_state_dict(self, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('generic MAMBA checkpoint schema mismatch')
        current, saved = asdict(self.cfg), dict(state['cfg'])
        current.pop('device'); saved.pop('device')
        if current != saved:
            raise ValueError('generic MAMBA architecture mismatch')
        if state.get('protocol') != self.learning_protocol():
            raise ValueError('generic MAMBA learning protocol mismatch')
        for key in ('world_model', 'actor', 'critic', 'target_critic', 'model_optimizer', 'actor_optimizer', 'critic_optimizer'):
            getattr(self, key).load_state_dict(state[key])
        self.updates = int(state['updates'])
        self.replay = deque(copy.deepcopy(state.get('replay', [])))
        self.replay_size = sum(len(data['actions']) for data in self.replay)
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if self.device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), self.device)
        self.reset()
