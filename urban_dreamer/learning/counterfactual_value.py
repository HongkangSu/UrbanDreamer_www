"""Model-only value targets for fixed-window counterfactual policy learning."""
from copy import deepcopy

import torch


def uses_value_bootstrap(method):
    return (getattr(method, 'actor_objective', '') == 'counterfactual_actor_critic_recoverq_v1'
            and getattr(method, 'imagination_seed_scope', '') == 'full_episode_v1'
            and getattr(method, 'imagination_horizon_windows', 0) == 8)


def fit_counterfactual_value(trainer, examples, *, epochs=32):
    rows = [item for example in examples for item in example.get('value_rows', [])]
    if not rows:
        return dict(actual_steps=0, retained_steps=0, mse_before=0., mse_after=0.)
    pooled = torch.stack([row['pooled'] for row in rows]).to(trainer.device)
    contexts = torch.stack([row['context'] for row in rows]).to(trainer.device)
    targets = torch.tensor([row['target'] for row in rows], device=trainer.device, dtype=pooled.dtype)
    if not torch.isfinite(targets).all():
        raise ValueError('nonfinite imagined value target')
    critic, optimizer = trainer.latent_critic, trainer.optimizers['latent_critic']
    def loss():
        return (critic(pooled[:, None], contexts).reshape(-1)-targets).square().mean()
    from .imagined_value import _gradients, _restore_gradients
    with torch.no_grad():
        before = float(loss())
    best_loss, retained = before, 0
    best = (deepcopy(critic.state_dict()), deepcopy(optimizer.state_dict()), _gradients(critic))
    for step in range(1, epochs+1):
        optimizer.zero_grad(set_to_none=True)
        objective = loss()
        if not bool(torch.isfinite(objective)):
            raise ValueError('nonfinite imagined value fit')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            current = float(loss())
        if current < best_loss:
            best_loss, retained = current, step
            best = (deepcopy(critic.state_dict()), deepcopy(optimizer.state_dict()), _gradients(critic))
    critic.load_state_dict(best[0])
    optimizer.load_state_dict(best[1])
    _restore_gradients(critic, best[2])
    energy = float(targets.square().mean())
    return dict(actual_steps=epochs, retained_steps=retained, mse_before=before, mse_after=best_loss,
                target_energy=energy, normalized_mse_after=best_loss/max(energy,1e-8),
                examples=len(rows), target='Frozen imagined lambda0.95 returns; no real value targets')
