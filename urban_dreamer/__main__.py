from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .workflow.config import MainlineConfig, load_config
from .workflow.pipeline import run_mainline


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    config = _apply_overrides(
        load_config(args.config),
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    if args.validate_only:
        from .traffic.routing import validate_prepared_inputs
        from .traffic.topology import load_traffic_topology
        prepared = validate_prepared_inputs(config)
        if args.pretrained is not None:
            from .workflow.native_validation import validate_pretrained_models
            result = validate_pretrained_models(args.pretrained, config=config, prepared=prepared)
        else:
            topology = load_traffic_topology(prepared)
            result = dict(status='validated', config=config.resolved_snapshot(),
                road_segments=topology.network.edge_count, expected_vehicles=len(topology.vehicles))
    else:
        result = run_mainline(config, pretrain_only=args.pretrain_only,
                              pretrained=args.pretrained, evaluate_only=args.evaluate_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m urban_dreamer",
        description="Validate the codec, pretrain dynamics, then run or resume the mainline.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="strict mainline TOML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="override [run].output_dir",
    )
    parser.add_argument(
        "--device",
        help="override [run].device (for example cpu or cuda:0)",
    )
    parser.add_argument(
        '--pretrain-only', action='store_true',
        help='stop after validated codec and dynamics pretraining; preserve the online-start checkpoint',
    )
    parser.add_argument('--pretrained', type=Path,
                        help='native pretrained city directory or checkpoints/latest.pt file')
    parser.add_argument('--validate-only', action='store_true',
                        help='validate inputs and, when supplied, strictly load and check pretrained models')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='evaluate the supplied native checkpoint without online training')
    parser.add_argument('--seed', type=int, help='override the training seed')
    return parser


def _apply_overrides(
    config: MainlineConfig,
    *,
    output_dir: Path | None,
    device: str | None,
    seed: int | None = None,
) -> MainlineConfig:
    run = config.run
    if output_dir is not None:
        run = replace(run, output_dir=output_dir.expanduser().resolve())
    if device is not None:
        normalized_device = device.strip()
        if not normalized_device:
            raise ValueError("--device must be non-empty")
        run = replace(run, device=normalized_device)
    if seed is not None:
        if seed < 0:
            raise ValueError('--seed must be nonnegative')
        run = replace(run, seed=seed)
    from .workflow.config import _semantic_fingerprint
    updated = replace(config, run=run)
    fingerprint = _semantic_fingerprint(**{name: getattr(updated, name) for name in
        ('inputs','run','simulation','resources','method','data','codec')})
    return replace(updated, semantic_fingerprint=fingerprint)


if __name__ == "__main__":
    raise SystemExit(main())
