# OCT

Analysis of OCT images of cancer spheroids.

## Installation Guide

The steps below are tested for Linux and use the current project layout.

### 1. Prerequisites

- Python 3.12+
- Git
- CUDA-capable GPU for 3D training (recommended)

### 2. Clone with submodules

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install this project

Install the project in editable mode (this uses dependencies from pyproject.toml):

```bash
pip install -e .
```

### 5. Install local submodules used by training

The training code imports nd_dataloader directly, so install the local submodule package:

```bash
pip install -e submodules/nd-dataloader
```

Optional (only if you use M3d-Cam features):

```bash
pip install -e submodules/M3d-Cam
```

### 6. Validate the environment

```bash
python test_environment.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import nd_dataloader; print('nd_dataloader import ok')"
```

## Quick Start

### Preprocess 3D data

```bash
python preprocess.py --config config/experiments/config_3D.yaml
```

### Train 3D model

```bash
python main3D.py --config config/experiments/config_3D.yaml
```

### Train 2D multiview model

```bash
python main2D.py
```

### Evaluate 3D model

```bash
python evaluate3D.py config/experiments/config_3D.yaml /path/to/model_weights.pt
```

## Notes

- There is no main.py entrypoint in this repository. Use main3D.py or main2D.py.
- 3D preprocessing/training expects split CSV files under data/.
- Run outputs are written to runs/.

## License

See LICENSE for more information.