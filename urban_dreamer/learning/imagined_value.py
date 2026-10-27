"""Snapshot helpers and the native real-phase policy immutability audit."""
from copy import deepcopy
from numbers import Integral

import torch


def imagination_value_epochs(method):
    epochs = getattr(method, 'imagination_value_epochs', 1)
    if isinstance(epochs, bool) or not isinstance(epochs, Integral) or epochs != 1:
        raise ValueError('the active recovered-Q path uses one imagined-value pass')
    return int(epochs)


def require_parallel_value_fit(trainer):
    return imagination_value_epochs(getattr(trainer.config, 'method', None))


def _gradients(module):
    return [None if p.grad is None else p.grad.detach().clone() for p in module.parameters()]


def _restore_gradients(module, gradients):
    for p, gradient in zip(module.parameters(), gradients):
        p.grad = None if gradient is None else gradient.clone()


def _same_tree(before, after):
    if isinstance(before, torch.Tensor):
        return isinstance(after, torch.Tensor) and torch.equal(before, after)
    if isinstance(before, dict):
        return isinstance(after, dict) and before.keys() == after.keys() and all(_same_tree(v, after[k]) for k, v in before.items())
    if isinstance(before, (tuple, list)):
        return type(before) is type(after) and len(before) == len(after) and all(_same_tree(a, b) for a, b in zip(before, after))
    return before == after


class RealPhaseImmutability:
    """Audit that collection and model fitting never train policy/value heads."""
    def __init__(self, trainer):
        self.modules = {name: getattr(trainer, name) for name in ('actor', 'latent_critic', 'event_critic')}
        if hasattr(trainer, 'target_latent_critic'):
            self.modules['target_latent_critic'] = trainer.target_latent_critic
        self.states = {name: deepcopy(module.state_dict()) for name, module in self.modules.items()}
        self.grads = {name: _gradients(module) for name, module in self.modules.items()}
        self.optimizers = {name: trainer.optimizers[name] for name in ('actor', 'latent_critic', 'event_critic')}
        self.optimizer_states = {name: deepcopy(opt.state_dict()) for name, opt in self.optimizers.items()}
        self._step_attributes = {}

    def __enter__(self):
        for name, optimizer in self.optimizers.items():
            self._step_attributes[name] = ('step' in optimizer.__dict__, optimizer.__dict__.get('step'))
            def reject_step(*args, _name=name, **kwargs):
                raise RuntimeError(f'pure imagination real phase attempted {_name} optimizer step')
            optimizer.step = reject_step
        return self

    def __exit__(self, exc_type, exc, traceback):
        for name, (had_attribute, previous) in self._step_attributes.items():
            optimizer = self.optimizers[name]
            if had_attribute:
                optimizer.step = previous
            else:
                del optimizer.step
        self._step_attributes.clear()
        return False

    def verify(self):
        for name, module in self.modules.items():
            if not _same_tree(self.states[name], module.state_dict()) or not _same_tree(self.grads[name], _gradients(module)):
                raise RuntimeError(f'pure imagination real phase mutated {name}')
        for name, optimizer in self.optimizers.items():
            if not _same_tree(self.optimizer_states[name], optimizer.state_dict()):
                raise RuntimeError(f'pure imagination real phase mutated {name} Adam')
        return dict(pure_imagination_real_phase_unchanged=1., real_actor_optimizer_steps=0.,
            real_critic_optimizer_steps=0., real_event_critic_optimizer_steps=0., real_target_value_updates=0.)
