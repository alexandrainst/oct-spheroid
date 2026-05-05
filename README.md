# OCT

Analysis of OCT images of cancer spheroids.

## Installation Guide

The steps below are tested on Linux and match the current src-layout package configuration in pyproject.toml.

### 1. Prerequisites

- Python 3.12+
- Git
- CUDA-capable GPU (recommended for 3D training)

### 2. Clone repository with submodules

```bash
git clone --recurse-submodules <repo-url>
cd oct
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 3. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install the project (editable)

```bash
pip install -e .
```

### 5. Install local training dependency

3D training uses nd_dataloader from the local submodule:

```bash
pip install -e submodules/nd-dataloader
```

Optional (only if you use M3d-Cam functionality):

```bash
pip install -e submodules/M3d-Cam
```

### 6. Verify installation

```bash
python -c "import oct; print('oct import ok')"
python -c "import nd_dataloader; print('nd_dataloader import ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python test_environment.py
```

### 7. Build distribution artifacts (optional)

```bash
uv build
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
python main2D.py --config config/experiments/config_2D.yaml
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