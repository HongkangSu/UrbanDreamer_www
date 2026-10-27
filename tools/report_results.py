"""Summarize final routing quality and the paper's online ATT integral."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from urban_dreamer.workflow.portable import portable_data


def online_auc(values, *, formal=True):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('online ATT requires at least two finite nonnegative episode values')
    if formal and len(values) != 1000:
        raise ValueError('formal online AUC requires all 1000 episodes')
    cumulative = np.concatenate(([0.], np.cumsum(values)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - 50)
    smoothed = (cumulative[ends] - cumulative[starts]) / (ends - starts)
    return float(np.trapezoid(smoothed, ends * 127.))


def read_run(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if line]
    if not rows or any(row.get('smoke') for row in rows):
        raise ValueError('smoke or empty runs cannot be reported as formal experiments')
    evaluation = sorted((row for row in rows if row['phase'] in ('evaluation', 'final_evaluation')),
                        key=lambda row: row['evaluation_index'])
    if [row['evaluation_index'] for row in evaluation] != [0, 1, 2]:
        raise ValueError('final evaluation must contain exactly three paired demands')
    if any(not row.get('demand_hash') for row in evaluation):
        raise ValueError('paired evaluation demand identities are missing')
    method = evaluation[0]['method']
    metrics = {key: float(np.mean([row[key] for row in evaluation]))
               for key in ('ATT', 'P90', 'P95', 'completion_rate')}
    if any(not np.isfinite(value) or value < 0 for value in metrics.values()):
        raise ValueError('final evaluation metrics must be finite and nonnegative')
    training = sorted((row for row in rows if row['phase'] in ('train', 'online_training')),
                      key=lambda row: row['episode_index'])
    auc = None
    if method not in ('greedy', 'ga'):
        if [row['episode_index'] for row in training] != list(range(1,1001)):
            raise ValueError('formal learning curves require unique episodes 1 through 1000')
        auc = online_auc([row['online_att_seconds'] for row in training])
    protocol_path = directory/'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    snapshot_path = directory/'resolved_config.json'
    config = protocol.get('config') or json.loads(snapshot_path.read_text())
    return dict(method=method, city=config['inputs']['network_name'],
                training_seed=protocol.get('training_seed', config['run']['seed']),
                metrics=metrics, online_att_auc_seconds_windows=auc,
                paired_demand_hashes=[row['demand_hash'] for row in evaluation])


def summarize(directories):
    runs = [read_run(directory) for directory in directories]
    paired = {}
    for run in runs:
        previous = paired.setdefault(run['city'], run['paired_demand_hashes'])
        if previous != run['paired_demand_hashes']:
            raise ValueError('all methods and training seeds must use the same evaluation demands within each city')
    groups = {}
    for run in runs:
        groups.setdefault((run['city'],run['method']),[]).append(run)
    aggregates = []
    for (city,method), selected in sorted(groups.items()):
        seeds = [run['training_seed'] for run in selected]
        if len(set(seeds)) != len(seeds):
            raise ValueError('a city/method/training-seed run occurs more than once')
        keys = list(selected[0]['metrics'])
        values = {key: [run['metrics'][key] for run in selected] for key in keys}
        if selected[0]['online_att_auc_seconds_windows'] is not None:
            values['online_att_auc_seconds_windows'] = [run['online_att_auc_seconds_windows'] for run in selected]
        aggregates.append(dict(city=city,method=method,training_seeds=sorted(seeds),
            complete_training_seed_set=method in ('greedy','ga') or sorted(seeds)==[42,43,44],
            mean={key:float(np.mean(value)) for key,value in values.items()},
            standard_deviation={key:float(np.std(value,ddof=1)) if len(value)>1 else None
                                for key,value in values.items()}))
    return dict(runs=runs,aggregates=aggregates,
        standard_deviation_definition='sample SD across independent training seeds; absent for one seed',
        auc_definition='50-episode trailing mean (available prefix), trapezoidal integral at windows 127 through 127000')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    result = summarize(args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(portable_data(result,root=args.output.parent),indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
