"""Validate one Q-gradient proposal in the model; never create teacher labels."""
from __future__ import annotations

from dataclasses import replace
import math

import torch

from .parallel_imagination import collect_parallel_imagination


def validate_actor_in_model(trainer, original_actor_state, *, batches=2, workers=2):
    """Compare original/proposed policies on fresh paired cohort simulations.

    The only proposal is the learned-Q policy-gradient update. These are not
    action targets: rejected proposals restore the prior Actor/Adam. Scores use
    physical imagined rewards over H8 with no V tail; truncation stays explicit.
    """
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
        raise ValueError('model policy validation needs positive fresh batches')
    original_config = trainer.config
    old = {name:value.detach().cpu().numpy().copy() for name,value in original_actor_state.items()}
    proposed = {name:value.detach().cpu().numpy().copy() for name,value in trainer.actor.state_dict().items()}
    plan = {'policies': [dict(label=label, feature_index=0, strength=0., actor_state=state)
        for label,state in [('oldA',old),('proposalA',proposed),('oldB',old),('proposalB',proposed)]]}
    rows = []
    try:
        trainer.config = replace(original_config, method=replace(original_config.method,
            imagination_horizon_windows=8, imagination_seed_scope='full_episode_v1', disagreement_samples=2))
        for _ in range(batches):
            with torch.no_grad():
                tracks, _ = collect_parallel_imagination(trainer, workers=workers,
                    policy_plan=plan, return_tracks=True, reward_only=True)
            scores = [-sum(window['output']['result'].active_vehicle_seconds for window in track['windows'])
                / (track['population'] * 60.) for track in tracks]
            if len(scores) != 4 or not all(math.isfinite(score) for score in scores):
                raise ValueError('model policy validation requires four finite physical returns')
            error = max(abs(scores[0]-scores[2]), abs(scores[1]-scores[3]))
            if error > 1e-7:
                raise ValueError('model policy validation duplicate control is inconsistent')
            rows.append(dict(old_return=scores[0], proposed_return=scores[1],
                delta=scores[1]-scores[0], duplicate_error=error,
                start_time=tracks[0]['windows'][0]['start'], population=tracks[0]['population'],
                windows_per_arm=[len(t['windows']) for t in tracks],
                terminal_per_arm=[bool(t['windows'][-1]['terminal']) for t in tracks]))
    finally:
        trainer.config = original_config
    return dict(accepted=all(row['delta'] >= -1e-7 for row in rows)
        and any(row['delta'] > 1e-7 for row in rows), pairs=rows,
        duplicate_error_max=max(row['duplicate_error'] for row in rows),
        model_rollouts=4*len(rows), macro_windows=sum(sum(row['windows_per_arm']) for row in rows),
        real_episodes=0, bootstrap_used=False,
        scope='Fresh model/noise cohort validation of a single Q-gradient proposal; H8 physical return, no teacher or policy search')
