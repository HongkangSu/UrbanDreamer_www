"""Generic discrete-action MBPO with learned transitions and discrete SAC.

This is a PyTorch categorical-action variant of MBPO, not a byte-for-byte port
of the original continuous-action TensorFlow implementation.  The learner sees
only observation/action/reward/terminal/mask arrays.  Synthetic transitions come
entirely from a learned probabilistic ensemble, including termination and action
availability; no environment or task-specific transition rules are consulted.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FIELDS = ('obs', 'actions', 'rewards', 'next_obs', 'terminals', 'masks', 'next_masks')
SCHEMA = 'generic_discrete_mbpo_sac_v1'


def _mlp(input_dim, output_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim))


def _policy(logits, mask):
    """Finite log probabilities, including zero-probability padded actions."""
    legal = mask.bool()
    if not bool(legal.any(-1).all()):
        raise ValueError('each policy observation needs at least one legal action')
    log_probability = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=-1)
    probability = log_probability.exp()
    entropy = -(probability * log_probability).sum(-1)
    return probability, log_probability, entropy


def learned_action_mask(logits):
    """Use learned availability; an empty prediction falls back to its own argmax."""
    legal = logits >= 0.
    fallback = F.one_hot(logits.argmax(-1), logits.shape[-1]).bool()
    return torch.where(legal.any(-1, keepdim=True), legal, fallback)


class ArrayReplay:
    """Bounded numpy ring; only generic transition arrays are retained."""
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity, self.size, self.cursor = int(capacity), 0, 0
        if self.capacity < 1:
            raise ValueError('replay capacity must be positive')
        self.arrays = dict(obs=np.empty((capacity, obs_dim), np.float32),
            actions=np.empty(capacity, np.int64), rewards=np.empty(capacity, np.float32),
            next_obs=np.empty((capacity, obs_dim), np.float32), terminals=np.empty(capacity, bool),
            masks=np.empty((capacity, action_dim), bool), next_masks=np.empty((capacity, action_dim), bool))

    def add(self, data):
        count = len(data['actions'])
        if not count:
            return
        offset = max(0, count - self.capacity)
        values = {key: np.asarray(data[key], dtype=self.arrays[key].dtype)[offset:] for key in FIELDS}
        count = len(values['actions'])
        for key, value in values.items():
            if value.shape != (count, *self.arrays[key].shape[1:]):
                raise ValueError(f'invalid transition shape for {key}: {value.shape}')
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f'nonfinite transition array {key}')
        if ((values['actions'] < 0) | (values['actions'] >= self.arrays['masks'].shape[1])).any():
            raise ValueError('transition action is outside the action space')
        if not values['masks'][np.arange(count), values['actions']].all():
            raise ValueError('transition action is masked')
        if not (values['next_masks'].any(-1) | values['terminals']).all():
            raise ValueError('nonterminal next observation has no legal action')
        indices = (self.cursor + np.arange(count)) % self.capacity
        for key, value in values.items():
            self.arrays[key][indices] = value
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count, rng):
        if not self.size:
            raise ValueError('cannot sample empty replay')
        indices = rng.integers(self.size, size=count)
        return {key: array[indices] for key, array in self.arrays.items()}

    def state_dict(self):
        return dict(size=self.size, cursor=self.cursor,
                    arrays={key: value[:self.size].copy() for key, value in self.arrays.items()})

    def load_state_dict(self, state):
        if not 0 <= state['size'] <= self.capacity or not 0 <= state['cursor'] < self.capacity:
            raise ValueError('replay state exceeds configured capacity')
        self.size, self.cursor = int(state['size']), int(state['cursor'])
        for key, value in state['arrays'].items():
            self.arrays[key][:self.size] = value


class EnsembleMember(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.obs_dim = obs_dim
        self.trunk = _mlp(obs_dim + action_dim, hidden, hidden)
        self.gaussian = nn.Linear(hidden, 2 * (obs_dim + 1))
        self.terminal = nn.Linear(hidden, 1)
        self.availability = nn.Linear(hidden, action_dim)
        self.max_logvar = nn.Parameter(torch.full((obs_dim + 1,), .5))
        self.min_logvar = nn.Parameter(torch.full((obs_dim + 1,), -10.))

    def forward(self, observation, onehot_action):
        hidden = self.trunk(torch.cat((observation, onehot_action), -1))
        mean, raw_logvar = self.gaussian(hidden).chunk(2, -1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar, self.terminal(hidden).squeeze(-1), self.availability(hidden)


class GenericMBPO:
    """Probabilistic short model rollouts and entropy-regularized discrete SAC."""
    def __init__(self, config):
        self.config = config
        self.obs_dim, self.action_dim = int(config.obs_dim), int(config.action_dim)
        self.device = torch.device(config.device)
        self.seed = int(config.seed)
        self.rng = np.random.default_rng(self.seed + 7919)
        hidden = int(getattr(config, 'hidden_dim', 256))
        self.gamma = float(getattr(config, 'gamma', .99))
        self.batch_size = int(getattr(config, 'batch_size', 256))
        self.model_updates = int(getattr(config, 'model_updates', 4))
        self.policy_updates = int(getattr(config, 'policy_updates', 4))
        self.policy_sample_to_insert_ratio = float(getattr(config, 'policy_sample_to_insert_ratio', 0.))
        self.model_sample_to_insert_ratio = float(getattr(config, 'model_sample_to_insert_ratio', 0.))
        self.horizon = int(getattr(config, 'model_rollout_length', 1))
        self.ensemble_size = int(getattr(config, 'ensemble_size', 5))
        self.elite_count = min(int(getattr(config, 'elite_count', 3)), self.ensemble_size)
        self.rollout_batch_size = int(getattr(config, 'rollout_batch_size', min(1024, 4 * self.batch_size)))
        self.real_ratio = float(getattr(config, 'real_ratio', .1))
        self.tau = float(getattr(config, 'target_tau', .005))
        self.target_entropy_scale = float(getattr(config, 'target_entropy_scale', .5))
        self.alpha_min = float(getattr(config, 'alpha_min', 1e-4))
        self.alpha_max = float(getattr(config, 'alpha_max', 1.))
        if min(self.obs_dim, self.action_dim, self.batch_size, self.model_updates,
               self.policy_updates, self.horizon, self.ensemble_size, self.elite_count) < 1:
            raise ValueError('generic MBPO dimensions and budgets must be positive')
        if not 0 < self.real_ratio <= 1 or not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError('invalid SAC mixing, target or discount setting')
        if not 0 < self.target_entropy_scale <= 1:
            raise ValueError('target entropy scale must be in (0, 1]')
        if not 0 < self.alpha_min < self.alpha_max:
            raise ValueError('alpha bounds must satisfy 0 < alpha_min < alpha_max')
        if any(not math.isfinite(ratio) or ratio < 0. for ratio in
               (self.policy_sample_to_insert_ratio, self.model_sample_to_insert_ratio)):
            raise ValueError('sample-to-insert ratios must be finite and nonnegative')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.actor = _mlp(self.obs_dim, self.action_dim, hidden).to(self.device)
            self.q1 = _mlp(self.obs_dim, self.action_dim, hidden).to(self.device)
            self.q2 = _mlp(self.obs_dim, self.action_dim, hidden).to(self.device)
            self.models = nn.ModuleList(EnsembleMember(self.obs_dim, self.action_dim, hidden)
                                       for _ in range(self.ensemble_size)).to(self.device)
        self.target_q1, self.target_q2 = deepcopy(self.q1), deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.log_alpha_min = math.log(self.alpha_min)
        self.log_alpha_max = math.log(self.alpha_max)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(.2), device=self.device).clamp(
            self.log_alpha_min, self.log_alpha_max))
        actor_lr = float(getattr(config, 'actor_lr', 3e-4))
        critic_lr = float(getattr(config, 'critic_lr', 3e-4))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)
        self.model_optimizer = torch.optim.Adam(self.models.parameters(), lr=float(getattr(config, 'model_lr', 3e-4)))
        capacity = int(getattr(config, 'replay_capacity', 200000))
        self.real_replay = ArrayReplay(capacity, self.obs_dim, self.action_dim)
        self.model_replay = ArrayReplay(min(capacity, int(getattr(config, 'model_replay_capacity', 50000))),
                                        self.obs_dim, self.action_dim)
        self.elites = np.arange(self.elite_count, dtype=np.int64)
        self.observed_transitions = self.update_count = self.model_update_count = 0
        self.pending_insertions = 0

    @property
    def alpha(self):
        return self.log_alpha.clamp(self.log_alpha_min, self.log_alpha_max).exp()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs, mask, *, agent_ids=None, deterministic=False):
        observation = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        probability, log_probability, entropy = _policy(self.actor(observation), legal)
        action = probability.argmax(-1) if deterministic else torch.multinomial(probability, 1).squeeze(-1)
        selected = log_probability.gather(-1, action[:, None]).squeeze(-1)
        actions = action.cpu().numpy()
        aux = [dict(entropy=float(e), old_log_prob=float(p))
               for e, p in zip(entropy.cpu().numpy(), selected.cpu().numpy())]
        return actions, aux

    def observe(self, episode):
        self.real_replay.add(episode)
        self.observed_transitions += len(episode['actions'])
        self.pending_insertions += len(episode['actions'])

    def _tensor_batch(self, data):
        dtypes = dict(obs=torch.float32, next_obs=torch.float32, rewards=torch.float32,
                      actions=torch.long, terminals=torch.bool, masks=torch.bool, next_masks=torch.bool)
        return {key: torch.as_tensor(value, device=self.device, dtype=dtypes[key])
                for key, value in data.items()}

    def _model_loss(self, member, batch):
        onehot = F.one_hot(batch['actions'].long(), self.action_dim).float()
        mean, logvar, terminal, availability = member(batch['obs'], onehot)
        target = torch.cat((batch['next_obs'] - batch['obs'], batch['rewards'][:, None]), -1)
        nll = ((mean - target).square() * (-logvar).exp() + logvar).mean()
        done_loss = F.binary_cross_entropy_with_logits(terminal, batch['terminals'].float())
        mask_loss = F.binary_cross_entropy_with_logits(availability, batch['next_masks'].float())
        return nll + done_loss + mask_loss + .01 * (member.max_logvar.sum() - member.min_logvar.sum())

    def _train_models(self, steps=None):
        losses = []
        for _ in range(self.model_updates if steps is None else steps):
            self.model_optimizer.zero_grad(set_to_none=True)
            # Independent bootstrap samples preserve probabilistic ensemble diversity.
            members = [self._model_loss(member, self._tensor_batch(self.real_replay.sample(self.batch_size, self.rng)))
                       for member in self.models]
            loss = torch.stack(members).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models.parameters(), 100.)
            self.model_optimizer.step()
            losses.append(float(loss.detach()) / self.ensemble_size)
            self.model_update_count += 1
        with torch.no_grad():
            validation = self._tensor_batch(self.real_replay.sample(self.batch_size, self.rng))
            scores = [float(self._model_loss(member, validation)) for member in self.models]
            self.elites = np.argsort(scores)[:self.elite_count]
        return float(np.mean(losses))

    @torch.no_grad()
    def _generate_model_rollouts(self):
        observation = self._tensor_batch(self.real_replay.sample(self.rollout_batch_size, self.rng))
        obs, mask = observation['obs'], observation['masks']
        generated = 0
        for _ in range(self.horizon):
            if not len(obs):
                break
            probability, _, _ = _policy(self.actor(obs), mask)
            actions = torch.multinomial(probability, 1).squeeze(-1)
            onehot = F.one_hot(actions, self.action_dim).float()
            selected_members = self.rng.choice(self.elites, size=len(obs))
            next_obs = torch.empty_like(obs)
            rewards = obs.new_empty(len(obs))
            terminal_probability = obs.new_empty(len(obs))
            next_mask = torch.empty((len(obs), self.action_dim), dtype=torch.bool, device=self.device)
            for index, member in enumerate(self.models):
                indices = np.flatnonzero(selected_members == index)
                if not len(indices):
                    continue
                ids = torch.as_tensor(indices, device=self.device)
                mean, logvar, done, availability = member(obs[ids], onehot[ids])
                sample = mean + torch.randn_like(mean) * (.5 * logvar).exp()
                next_obs[ids] = obs[ids] + sample[:, :self.obs_dim]
                rewards[ids] = sample[:, self.obs_dim]
                terminal_probability[ids] = done.sigmoid()
                next_mask[ids] = learned_action_mask(availability)
            terminal = torch.rand_like(terminal_probability) < terminal_probability
            data = dict(obs=obs.cpu().numpy(), actions=actions.cpu().numpy(), rewards=rewards.cpu().numpy(),
                        next_obs=next_obs.cpu().numpy(), terminals=terminal.cpu().numpy(),
                        masks=mask.cpu().numpy(), next_masks=next_mask.cpu().numpy())
            self.model_replay.add(data)
            generated += len(obs)
            obs, mask = next_obs[~terminal], next_mask[~terminal]
        return generated

    def _mixed_batch(self):
        real_count = self.batch_size if not self.model_replay.size else max(1, round(self.batch_size * self.real_ratio))
        real = self.real_replay.sample(real_count, self.rng)
        model_count = self.batch_size - real_count
        if not model_count:
            return self._tensor_batch(real)
        model = self.model_replay.sample(model_count, self.rng)
        return self._tensor_batch({key: np.concatenate((real[key], model[key])) for key in FIELDS})

    @torch.no_grad()
    def _sac_target(self, batch):
        mask = batch['next_masks'].bool()
        # Terminal observations do not need a feasible next action.  Exposing
        # all actions only makes the zero-discount expectation well-defined.
        mask = torch.where(mask.any(-1, keepdim=True), mask, torch.ones_like(mask))
        probability, log_probability, _ = _policy(self.actor(batch['next_obs']), mask)
        minimum = torch.minimum(self.target_q1(batch['next_obs']), self.target_q2(batch['next_obs']))
        next_value = (probability * (minimum - self.alpha * log_probability)).sum(-1)
        return batch['rewards'] + self.gamma * (~batch['terminals'].bool()).float() * next_value

    def _sac_step(self, batch):
        target = self._sac_target(batch)
        actions = batch['actions'].long()[:, None]
        q1 = self.q1(batch['obs']).gather(-1, actions).squeeze(-1)
        q2 = self.q2(batch['obs']).gather(-1, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 100.)
        self.q_optimizer.step()
        probability, log_probability, entropy = _policy(self.actor(batch['obs']), batch['masks'])
        with torch.no_grad():
            minimum = torch.minimum(self.q1(batch['obs']), self.q2(batch['obs']))
        actor_loss = (probability * (self.alpha.detach() * log_probability - minimum)).sum(-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        target_entropy = self.target_entropy_scale * batch['masks'].sum(-1).float().clamp_min(1.).log()
        alpha_loss = (self.log_alpha * (entropy.detach() - target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], 10.)
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(self.log_alpha_min, self.log_alpha_max)
        with torch.no_grad():
            for target_net, online in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
                for destination, source in zip(target_net.parameters(), online.parameters()):
                    destination.lerp_(source, self.tau)
        self.update_count += 1
        return dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                    alpha_loss=float(alpha_loss.detach()), alpha=float(self.alpha.detach()),
                    entropy=float(entropy.detach().mean()),
                    target_entropy=float(target_entropy.detach().mean()))

    def update(self, updates=None):
        if not self.real_replay.size:
            return dict(skipped=1., model_loss=0., actor_loss=0., critic_loss=0., model_transitions=0)
        insertions = self.pending_insertions
        steps = (max(self.policy_updates, math.ceil(insertions * self.policy_sample_to_insert_ratio / self.batch_size))
                 if updates is None else int(updates))
        model_steps = max(self.model_updates, math.ceil(insertions * self.model_sample_to_insert_ratio / self.batch_size))
        if steps < 1:
            raise ValueError('SAC update count must be positive')
        model_loss = self._train_models(model_steps)
        generated = self._generate_model_rollouts()
        results = [self._sac_step(self._mixed_batch()) for _ in range(steps)]
        self.pending_insertions = 0
        return dict(skipped=0., model_loss=model_loss, model_transitions=generated,
                    real_replay_size=self.real_replay.size, model_replay_size=self.model_replay.size,
                    inserted_transitions=insertions, policy_optimization_draws=steps * self.batch_size,
                    model_optimization_draws_per_member=model_steps * self.batch_size,
                    model_optimization_draws_total=model_steps * self.batch_size * self.ensemble_size,
                    policy_draws_per_insertion=steps * self.batch_size / insertions if insertions else 0.,
                    model_draws_per_insertion_per_member=model_steps * self.batch_size / insertions if insertions else 0.,
                    model_optimizer_steps=model_steps, actor_optimizer_steps=steps,
                    critic_optimizer_steps=steps, temperature_optimizer_steps=steps,
                    **{key: float(np.mean([r[key] for r in results])) for key in results[0]})

    def inference_state(self):
        return dict(schema=SCHEMA, obs_dim=self.obs_dim, action_dim=self.action_dim,
                    actor={key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()})

    def load_inference_state(self, state):
        if state.get('schema') != SCHEMA or (state['obs_dim'], state['action_dim']) != (self.obs_dim, self.action_dim):
            raise ValueError('incompatible generic MBPO inference state')
        self.actor.load_state_dict(state['actor'])

    def state_dict(self, include_replay=False):
        state = dict(schema=SCHEMA, obs_dim=self.obs_dim, action_dim=self.action_dim,
                     gamma=self.gamma, real_ratio=self.real_ratio, horizon=self.horizon,
                     policy_sample_to_insert_ratio=self.policy_sample_to_insert_ratio,
                     model_sample_to_insert_ratio=self.model_sample_to_insert_ratio,
                     pending_insertions=self.pending_insertions,
                     elites=self.elites.copy(), observed_transitions=self.observed_transitions,
                     update_count=self.update_count, model_update_count=self.model_update_count,
                     rng=deepcopy(self.rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                     log_alpha=self.log_alpha.detach().cpu().clone(), replay_included=bool(include_replay))
        for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2', 'models', 'actor_optimizer',
                     'q_optimizer', 'alpha_optimizer', 'model_optimizer'):
            state[name] = deepcopy(getattr(self, name).state_dict())
        if self.device.type == 'cuda':
            state['cuda_rng'] = torch.cuda.get_rng_state(self.device)
        if include_replay:
            state['real_replay'] = self.real_replay.state_dict()
            state['model_replay'] = self.model_replay.state_dict()
        return state

    def load_state_dict(self, state):
        if state.get('schema') != SCHEMA or (state['obs_dim'], state['action_dim']) != (self.obs_dim, self.action_dim):
            raise ValueError('incompatible generic MBPO checkpoint')
        if (state['gamma'], state['real_ratio'], state['horizon']) != (self.gamma, self.real_ratio, self.horizon):
            raise ValueError('generic MBPO learning settings changed')
        if (state.get('policy_sample_to_insert_ratio', 0.), state.get('model_sample_to_insert_ratio', 0.)) != (
                self.policy_sample_to_insert_ratio, self.model_sample_to_insert_ratio):
            raise ValueError('generic MBPO insertion budgets changed')
        for name in ('actor', 'q1', 'q2', 'target_q1', 'target_q2', 'models', 'actor_optimizer',
                     'q_optimizer', 'alpha_optimizer', 'model_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'].to(self.device).clamp(
                self.log_alpha_min, self.log_alpha_max))
        self.elites = state['elites'].copy()
        self.observed_transitions = int(state['observed_transitions'])
        self.pending_insertions = int(state.get('pending_insertions', 0))
        self.update_count, self.model_update_count = int(state['update_count']), int(state['model_update_count'])
        self.rng.bit_generator.state = deepcopy(state['rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if 'cuda_rng' in state and self.device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], self.device)
        if state.get('replay_included'):
            self.real_replay.load_state_dict(state['real_replay'])
            self.model_replay.load_state_dict(state['model_replay'])
        else:
            self.real_replay.size = self.real_replay.cursor = 0
            self.model_replay.size = self.model_replay.cursor = 0
