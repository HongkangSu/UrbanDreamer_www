# UrbanDreamer

World-model-based multi-agent routing in SUMO, with benchmarks for **Ingolstadt, Montereau-Fault-Yonne and Coulommiers**.

Baselines: **Greedy, GA, IQL, IPPO, MAPPO, GNAC, Dreamer, MAMBA and MBPO**.

## Setup

```bash
conda env create -f environment.yaml
conda activate urban-dreamer
```

Tested with Python 3.13.1, PyTorch 2.10.0 and libsumo 1.27.1. Run commands from the repository root; paths inside TOML files resolve relative to the configuration file.

## Train UrbanDreamer

Use a native pretrained city directory containing `checkpoints/latest.pt`, or pass the checkpoint file directly:

```bash
python -m urban_dreamer --config configs/ingolstadt_custom.toml \
  --pretrained pretrained/ingolstadt_custom --device cuda:0 \
  --seed 42 --output-dir outputs/urban_dreamer_ingo_seed42
```

For a loading check without training, set `--device cpu` and add `--validate-only`. Pretrained initialization does not require the original offline archives. Rerun the same command to resume.

Use `configs/montereau_fault_yonne.toml` or `configs/coulommiers.toml` for the other cities. Run training seeds **42, 43 and 44** separately. The supplied checkpoints initialize the world model; their actors have not completed online training.

For fresh training, collect compatible archives first, then run the mainline without `--pretrained`:

```bash
python tools/run_baseline.py --method greedy \
  --config configs/ingolstadt_custom.toml \
  --output-dir data/offline/ingolstadt_custom --collect-only --save-episodes
```

## Run baselines

```bash
python tools/run_baseline.py --method mappo \
  --config configs/ingolstadt_custom.toml \
  --output-dir outputs/mappo_ingo_seed42 --seed 42
```

Replace `mappo` with any baseline listed above. Greedy and GA evaluate directly; learned methods train and evaluate their final checkpoint. Add `--smoke` for a reduced test run.

## Settings and results

Formal runs use **1,000 online episodes**, **127 × 30 s windows per episode**, a **1,800 s demand window**, and **three paired final evaluation demands**. The mainline uses soft graph filtering, two MC samples, one-step offline supervision, three-step online supervision and up to eight imagined windows. Model and checkpoint interfaces retain their native definitions.

Results are written to `metrics.jsonl`, including ATT, P90, P95 and completion rates. Summarize full runs with:

```bash
python tools/report_results.py outputs/urban_dreamer_ingo_seed42 \
  outputs/mappo_ingo_seed42 --output outputs/comparison.json
```

Install `requirements-dev.txt` for tests and run `python -B -m pytest -q`. Pretrained binaries, collected archives, outputs and local documentation are excluded from version control. Full benchmark retraining is not included in short-flow verification.
