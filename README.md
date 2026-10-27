# UrbanDreamer

Standalone UrbanDreamer mainline and the nine requested baselines: **Greedy, GA, IQL, IPPO, MAPPO, GNAC, Dreamer, MAMBA and MBPO**. The benchmarks are Ingolstadt, Montereau-Fault-Yonne and Coulommiers.

The original working mainline is preserved. Cleanup removes redundant experiment branches and machine-specific paths and unifies experiment settings. It does not redefine features, codec labels, rewards, critics or model parameters. The paper describes a subset of the implemented features.

## Environment

The verified runtime is Python 3.13.1, NumPy 2.3.5, PyTorch 2.10.0 and libsumo/traci/sumolib 1.27.1.

```bash
conda env create -f environment.yaml
conda activate urban-dreamer
```

Alternatively install `requirements.txt` in a Python 3.13 environment. Run the following commands from this directory. Configuration paths are relative to the TOML file. GPU execution is recommended for city training; validation can run on CPU.

## Use the existing three pretrained checkpoints

Place the existing checkpoint directories in a location of your choice. The usual city layout is:

```text
pretrained/<city>/checkpoints/latest.pt
```

The loader accepts either the city directory or the checkpoint file itself. It loads the original schema-10 model states with `strict=True`; no model conversion or parameter substitution is performed. Checkpoint files themselves are not modified or copied into this code distribution.

Validate a real pretrained checkpoint and run native encode, decode, prediction, actor and critic forwards:

```bash
python -m urban_dreamer --config configs/ingolstadt_custom.toml \
  --pretrained pretrained/ingolstadt_custom --device cpu --validate-only
```

Start online training directly from the pretrained state:

```bash
python -m urban_dreamer --config configs/ingolstadt_custom.toml \
  --pretrained pretrained/ingolstadt_custom --device cuda:0 \
  --seed 42 --output-dir outputs/urban_dreamer_ingo_seed42
```

The equivalent configurations are `configs/montereau_fault_yonne.toml` and `configs/coulommiers.toml`. `--pretrain-only` loads and saves the initialized online-start state without collecting an online episode. `--evaluate-only` evaluates the supplied state without online training; these source checkpoints have an untrained actor, so this is an initialization check rather than a final policy result.

Original offline archives are not required to start online training from these checkpoints. The pretrained codec, world model, optimizer/RNG state, original counters and codec acceptance evidence are retained. Stored offline references describe actual pretraining history and explicitly cannot supply absent archive bodies.

Rerun the same command to resume a matching saved run. After initialization, the source `--pretrained` argument can be omitted. Saved runs retain their native schema-10 state and the source digest, so a different city's weights cannot silently be substituted. Online checkpoints are saved every 100 episodes and at phase completion. Uncommitted metric rows are trimmed before replaying interrupted work.

## Fresh training

Fresh initialization from archives remains available:

```bash
python tools/run_baseline.py --method greedy \
  --config configs/ingolstadt_custom.toml \
  --output-dir data/offline/ingolstadt_custom \
  --collect-only --save-episodes --seed 42

python -m urban_dreamer --config configs/ingolstadt_custom.toml \
  --device cuda:0 --seed 42 --output-dir outputs/urban_dreamer_fresh_seed42
```

Fresh training requires compatible fixed archives with at least two independent demand identities, allowing a held-out codec split. It runs representation learning, world-model initialization, online training and final evaluation. No source-data fallback opens the original project directory.

## Common experiment settings

| Setting | Value |
|---|---|
| Traffic | libsumo mesoscopic, junction control, 98 m nominal segments |
| Observation / route decision | 1 s / 30 s |
| Online budget | 1000 episodes, 127 windows per episode, deadline 3810 s |
| Demand | Homogeneous Poisson arrivals over 1800 s |
| Training seeds | 42, 43, 44 |
| Final evaluation | Three paired demands: base314159, indices0/1/2; deterministic actions |
| Mainline prediction / supervision / imagination | Euler10 / offline one-step, online recursive3 / horizon8 |
| Native active mainline | Recover-Q actor-critic; MC2/risk0 and optimization behavior |

Internal features, normalization, codec interpretation, reward/Q scaling and extra optimizer safeguards remain those of the original mainline.

## Baselines

```bash
python tools/run_baseline.py --method mappo \
  --config configs/ingolstadt_custom.toml \
  --output-dir outputs/mappo_ingo_seed42 --seed 42

python tools/run_baseline.py --method dreamer \
  --config configs/coulommiers.toml \
  --output-dir outputs/dreamer_cou_seed43 --seed 43 --device cuda:0
```

The baseline entry exposes exactly the nine named methods. Greedy and GA evaluate directly; learned methods train for 1000 episodes and evaluate their final checkpoint. Baseline inference can be repeated using `--evaluate-only --checkpoint outputs/<run>/final.pt` with a fresh output directory. Checkpoint training seed, completed budget and smoke/physical protocol are retained and checked. A short smoke checkpoint cannot be relabelled as a formal run.

The three native model-based learners keep their own dynamics. GA uses finite-candidate genetic search, with population 100, 50 generations, crossover 0.8, mutation 0.1 and tournament size 5. Its fitness sums current observed route travel times, and all outputs remain within the common candidate pool.

## Metrics and verification

Training runs write `metrics.jsonl`. Final metrics contain ATT, P90, P95, completion rates and demand hashes. Routing durations include insertion waiting and unfinished/uninserted trips censored at the evaluation deadline. These reporting metrics do not change training rewards.

```bash
python tools/report_results.py outputs/urban_dreamer_ingo_seed42 \
  outputs/mappo_ingo_seed42 --output outputs/comparison.json

python -B -m pytest -q
python -B tools/validate_release.py
```

The reporter requires all 1000 unique online episodes and three paired final demands, computes the 50-episode trailing online-ATT AUC, and rejects smoke runs. Passing three training seeds adds their mean and sample standard deviation. Test dependencies are in `requirements-dev.txt`. The toy network is a test fixture, not an additional benchmark.

The distribution audit checks a clean directory before local training outputs or datasets are created. Published source and newly saved metadata contain no machine-specific paths. Regression tests cover strict native pretrained loading, phase-specific world-model losses, short real trajectories and checkpoint restoration.

Historical experiments, additional baselines, variant studies, personal launch scripts, source archives and pretrained binary files are excluded. Full benchmark numerical reproduction and full CUDA training are not part of short-flow verification.
