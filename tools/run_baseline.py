#!/usr/bin/env python3
"""Train or evaluate one of the nine baselines on the common routing interface."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from urban_dreamer.learning.baselines import build_method
from urban_dreamer.learning.baselines.common import EpisodeRunner, evaluation_rng
from urban_dreamer.learning.baselines.generic_config import GenericConfig
from urban_dreamer.learning.baselines.generic_data import build_episode, pack_observation
from urban_dreamer.learning.baselines.heuristics import GeneticCandidatePolicy, GreedyCandidatePolicy
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.workflow.portable import FEATURE_CONTRACT, portable_data

METHODS = ('greedy', 'ga', 'iql', 'ippo', 'mappo', 'gnac', 'dreamer', 'mamba', 'mbpo')
NATIVE_METHODS = ('dreamer', 'mamba', 'mbpo')
HEURISTICS = ('greedy', 'ga')
EVALUATION_DEMAND_SEED = 314159
EVALUATION_EPISODES = 3
CHECKPOINT_FORMAT = 'www_baseline_v1'


def evaluation_identity(index):
    if index not in range(EVALUATION_EPISODES):
        raise ValueError('evaluation index must be 0, 1 or 2')
    return dict(demand_seed=EVALUATION_DEMAND_SEED, demand_episode_index=index,
                simulation_seed=EVALUATION_DEMAND_SEED + index)


def make_native_agent(name, config):
    cls = dict(dreamer='GenericDreamer', mamba='GenericMAMBA', mbpo='GenericMBPO')[name]
    module = importlib.import_module('urban_dreamer.learning.baselines.generic_' + name)
    return getattr(module, cls)(config)


class NativePolicy:
    """Attach generic learners to real observations without exposing physics."""

    def __init__(self, agent):
        self.agent, self.ids = agent, {}

    def begin_episode(self, topology=None, *, seed=None):
        self.ids = {}
        self.agent.reset()

    def select(self, vehicles, observations, flow, now, *, deterministic=False):
        ids = np.asarray([self.ids.setdefault(str(v['vehicle_id']), len(self.ids))
                          for v in vehicles], dtype=np.int64)
        actions, auxiliary = self.agent.act(pack_observation(observations.features, observations.mask),
            observations.mask, agent_ids=ids, deterministic=deterministic)
        return np.asarray(actions, dtype=np.int64).tolist(), auxiliary


class TensorEpisodeRunner(EpisodeRunner):
    def global_observation(self, state):
        # Native learners use local 44-dimensional route observations.
        return np.empty(0, dtype=np.float32)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--collect-only', action='store_true',
                        help='collect training episodes without learner updates')
    parser.add_argument('--save-episodes', action='store_true',
                        help='write real train archives under episodes/train')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path,
                        help='release checkpoint to use with --evaluate-only')
    parser.add_argument('--smoke', action='store_true',
                        help='one short eight-vehicle run; marked as smoke in every report')
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.threads < 1:
        parser.error('episodes and threads must be positive')
    if not args.smoke and not args.collect_only and args.episodes != 1000:
        parser.error('formal learning uses 1000 episodes; use --smoke for a short check')
    if args.collect_only and (not args.save_episodes or args.evaluate_only):
        parser.error('--collect-only requires --save-episodes and cannot use --evaluate-only')
    if args.checkpoint is not None and not args.evaluate_only:
        parser.error('--checkpoint is only supported with --evaluate-only')
    if args.evaluate_only and args.method not in HEURISTICS and args.checkpoint is None:
        parser.error('learned baseline evaluation requires --checkpoint')
    args.actor_lr, args.critic_lr = 2e-4, 3e-4
    args.reward_objective, args.global_reward_only = 'observed_mean', True
    if args.smoke:
        args.episodes = 1
    return args


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(portable_data(value, root=ROOT), default=json_value,
                                    allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def append_metrics(output, metrics):
    row = portable_data(metrics, root=ROOT)
    with (output / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row, default=json_value, allow_nan=False) + '\n')
    print(json.dumps(row, default=json_value, allow_nan=False), flush=True)


def standard_metrics(metrics):
    return dict(metrics,
        ATT=metrics.get('all_demand_censored_att_seconds'),
        P90=metrics.get('all_demand_censored_p90_seconds'),
        P95=metrics.get('all_demand_censored_p95_seconds'),
        demand_hash=metrics['demand_sha256'])


def create_policy(args, runner):
    if args.method == 'greedy':
        return GreedyCandidatePolicy(runner.network, candidates=runner.candidates), None
    if args.method == 'ga':
        return GeneticCandidatePolicy(runner.network, candidates=runner.candidates, seed=args.seed,
            vehicle_max_speed_mps=runner.config.simulation.vehicle_max_speed_mps), None
    if args.method in NATIVE_METHODS:
        values = dict(obs_dim=44, action_dim=4, seed=args.seed, device=args.device)
        if args.smoke:
            values.update(hidden_dim=32, batch_size=8, sequence_length=2, sequence_batch_size=2,
                mamba_sequence_batch_size=1, mamba_imagination_starts=2, imagination_starts=2,
                imagination_horizon=2, rollout_batch_size=8, model_updates=1, policy_updates=1,
                max_agents_per_chunk=8)
        learner = make_native_agent(args.method, GenericConfig(**values))
        return NativePolicy(learner), learner
    learner = build_method(args.method, args, runner)
    return learner, learner


def physical_protocol(runner):
    return dict(simulation=runner.simulation_settings, deadline_seconds=runner.deadline,
                demand_duration_seconds=runner.demand.duration_seconds)


def load_evaluation_checkpoint(path, method, learner, runner, *, smoke=False):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != CHECKPOINT_FORMAT or saved.get('method') != method
            or saved.get('feature_contract') != FEATURE_CONTRACT
            or saved.get('input_hashes') != runner.prepared.input_hashes):
        raise ValueError('checkpoint does not match the release method, features and inputs')
    if saved.get('smoke') is not smoke:
        raise ValueError('checkpoint smoke status must match the evaluation mode')
    if saved.get('physical_protocol') != physical_protocol(runner):
        raise ValueError('checkpoint physical protocol differs from the evaluation')
    if saved.get('training_seed') not in (42,43,44):
        raise ValueError('checkpoint is missing its training seed')
    if not smoke and saved.get('completed_training_episodes') != 1000:
        raise ValueError('formal evaluation requires a completed 1000-episode checkpoint')
    if method in NATIVE_METHODS:
        learner.load_inference_state(saved['learner'])
    else:
        learner.load_state_dict(saved['learner'])
    return dict(training_seed=saved['training_seed'],
                completed_training_episodes=saved['completed_training_episodes'])


def main(argv=None):
    args = arguments(argv)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty output directory for each independent baseline run')
    output.mkdir(parents=True, exist_ok=True)
    runner_type = TensorEpisodeRunner if args.method in NATIVE_METHODS else EpisodeRunner
    runner = runner_type(args)
    runner.config = replace(runner.config, run=replace(runner.config.run, output_dir=output.resolve()))
    if args.smoke:
        runner.topology = replace(runner.topology, vehicles=runner.topology.vehicles[:8])
        runner.demand = PoissonDemand.from_vehicles(runner.topology.vehicles, duration_seconds=60.)
        runner.deadline = 180.
    elif (runner.deadline != 3810. or runner.demand.duration_seconds != 1800.
            or runner.config.simulation.world_model_intervals_per_episode != 127
            or runner.config.simulation.training_completion_buffer_seconds != 0):
        raise ValueError('formal baseline runs require 1800-second demand and 127 windows ending at 3810 seconds')
    policy, learner = create_policy(args, runner)
    source = None
    if args.checkpoint is not None:
        source = load_evaluation_checkpoint(args.checkpoint, args.method, learner, runner,
                                            smoke=args.smoke)
    native = args.method in NATIVE_METHODS
    protocol = dict(method=args.method, feature_contract=FEATURE_CONTRACT,
        config=runner.config.resolved_snapshot(root=output),
        training_seed=source['training_seed'] if source else args.seed,
        online_training_episodes=(source['completed_training_episodes'] if source else
            0 if args.evaluate_only or args.method in HEURISTICS else args.episodes),
        evaluate_only=args.evaluate_only,
        collected_training_episodes=args.episodes if args.collect_only else None,
        evaluation_episodes=3, evaluation_demand_seed=EVALUATION_DEMAND_SEED,
        evaluation_indices=[0, 1, 2], evaluation_policy='deterministic_argmax',
        checkpoint_selection='final', deadline_seconds=runner.deadline,
        episode_index_base=1, smoke=args.smoke, input_hashes=runner.prepared.input_hashes)
    if learner is not None:
        protocol['learning'] = getattr(learner, 'learning_protocol', lambda: {})()
    if args.method == 'ga':
        protocol['genetic_search'] = dict(chromosome='one_common_candidate_index_per_vehicle',
            fitness='sum_current_observed_segment_travel_times', population=100, generations=50,
            crossover=.8, mutation=.1, tournament_size=5)
    write_json(output / 'protocol.json', protocol)

    def run_episode(*, index, training):
        action_seed = args.seed * 100000 + index if training else EVALUATION_DEMAND_SEED + index
        physical = (dict(demand_seed=args.seed, demand_episode_index=index - 1,
                         simulation_seed=args.seed + index - 1) if training else evaluation_identity(index))
        begin = getattr(policy, 'begin_episode', None)
        if native:
            select = lambda *values: policy.select(*values, deterministic=not training)
        elif args.method in HEURISTICS:
            select = policy.select
        else:
            select = lambda *values: policy.select(*values, explore=training)
        archive = (output / 'episodes' / 'train' / f'episode_{index:06d}.npz'
                   if training and args.save_episodes else None)
        return runner.run_episode(select, seed=action_seed, collect=training,
            row_flow=args.method == 'gnac', on_demand=begin,
            archive_path=archive, archive_metadata=dict(phase='train', episode=index, smoke=args.smoke,
                feature_contract=FEATURE_CONTRACT),
            label='stochastic' if training else 'deterministic', **physical)

    train = not args.evaluate_only and (args.method not in HEURISTICS or args.collect_only)
    if train:
        for episode in range(1, args.episodes + 1):
            rows, windows, metrics = run_episode(index=episode, training=True)
            update = {}
            if learner is not None and not args.collect_only:
                if native:
                    data = build_episode(rows, windows, obs_dim=44, action_dim=4)
                    if len(data['actions']):
                        learner.observe(data)
                        update = learner.update()
                else:
                    update = learner.on_episode(rows, windows, episode=episode)
            metrics = standard_metrics(metrics)
            # Censored ATT is also required on training episodes for the online AUC.
            if metrics['ATT'] is None:
                metrics['ATT'] = 0.
            append_metrics(output, dict(metrics, **update, phase='train', episode_index=episode,
                online_att_seconds=metrics['ATT'], online_decision_windows=127 * episode,
                smoke=args.smoke, method=args.method))
        if learner is not None and not args.collect_only:
            state = learner.inference_state() if native else learner.state_dict()
            if args.method == 'iql':
                state['replay'] = []
            temporary = output / 'final.tmp'
            torch.save(dict(format=CHECKPOINT_FORMAT, method=args.method,
                feature_contract=FEATURE_CONTRACT, input_hashes=runner.prepared.input_hashes,
                config=runner.config.resolved_snapshot(root=output), learner=state, smoke=args.smoke,
                training_seed=args.seed, completed_training_episodes=args.episodes,
                physical_protocol=physical_protocol(runner)), temporary)
            temporary.replace(output / 'final.pt')
    evaluations = []
    if not args.collect_only:
        for index in range(EVALUATION_EPISODES):
            with evaluation_rng(EVALUATION_DEMAND_SEED + index, learner or policy):
                _, _, metrics = run_episode(index=index, training=False)
            metrics = standard_metrics(metrics)
            evaluations.append(metrics)
            append_metrics(output, dict(metrics, phase='evaluation', evaluation_index=index,
                smoke=args.smoke, method=args.method))
    result = dict(status='completed', method=args.method, smoke=args.smoke,
                  protocol=protocol, evaluations=evaluations)
    if evaluations:
        result['mean'] = {key: (float(np.mean([row[key] for row in evaluations if row[key] is not None]))
                                  if any(row[key] is not None for row in evaluations) else None)
                          for key in ('ATT', 'P90', 'P95', 'completion_rate')}
    write_json(output / 'result.json', result)
    return result


if __name__ == '__main__':
    main()
