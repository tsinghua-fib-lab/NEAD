# NEAD

Official implementation of *Abductive Artificial Intelligence Reveals the Laws Underlying Transportation Resilience*.

NEAD (Network Emergence Abductive Discovery) learns transportation-resilience dynamics with a graph neural surrogate, identifies the dominant network structure with a GNNExplainer-based decoupler, and distils the resulting low-dimensional representation into a symbolic formula.

[中文说明](README.zh.md)

## Setup

Python 3.12 is recommended. A CUDA-capable GPU is strongly recommended for model training and explanation.

```bash
conda create -p ./venv python=3.12 -y
conda activate ./venv
pip install -r requirements.txt
```

PySR also installs and uses Julia; its first run may take additional time to prepare the Julia environment.

## Data

Large datasets are distributed separately. Place the U.S., Global South, and GitHub raw networks in `data/raw/`, `data/raw_globalsouth/`, and `data/raw_github/`; place the generated U.S. and Global South scenarios in `data/augmentation/` and `data/augmentation_globalsouth/`. Small metadata and city-list files are included.

## Workflow

Run commands from the repository root. The numbered entry points correspond to the main stages of the paper:

```bash
bash scripts/1.prepare_data.sh
bash scripts/2.train_surrogate.sh
bash scripts/3.run_decoupler.sh
bash scripts/4.get_lowdim_functions.sh
bash scripts/5.extract_formula.sh
```

Data preparation and full training over 100 cities are computationally expensive. The scripts under `scripts/` also include the counterfactual capacity/OD experiments and runtime benchmarks. Experiment outputs are written to `logs/`.

Batch scripts run locally by default. To coordinate the same script across machines, start `src/utils/share/lock_server.py` and set `LOCK_SERVER=http://host:port` before running it.

## License

Released under the [MIT License](LICENSE).
