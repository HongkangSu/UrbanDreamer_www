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
