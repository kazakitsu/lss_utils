# lss_utils

Utilities and scripts to analyze N-body simulation data, with both NumPy and JAX backends.

---

## Features

- **Mesh assignment**  
  Assign particle data to a regular grid (CIC, TSC, NGP, …) via `Mesh_Assignment`.
- **Power-spectrum & spectra measurement**  
  Compute 1D power spectra and higher-order spectra efficiently via FFTs (`Measure_Pk`, `Measure_spectra_FFT`).
- **Dual backend**  
  - **JAX** implementations for GPU/TPU acceleration  
  - Fallback to pure-NumPy if JAX is not available
- **Minimal dependencies**  
  Only requires `numpy` plus whichever backend you choose (`jax`/`jaxlib`).

---

## Prerequisites

- **Python ≥ 3.7**
- **(For GPU/TPU acceleration)** Install JAX _before_ installing **lss_utils**, following the official instructions: https://github.com/jax-ml/jax#installation

  ```bash
  # CPU-only
  pip install --upgrade pip
  pip install --upgrade "jax[cpu]"

  # GPU (CUDA 12.x) example
  pip install --upgrade pip
  pip install --upgrade "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html


## Installation

Once prerequisites are met, install **lss_utils** with pip:

```bash
git clone https://github.com/kazakitsu/lss_utils.git
cd lss_utils
pip install .


If you want to work on or modify the code locally:
```bash

git clone https://github.com/kazakitsu/lss_utils.git
cd lss_utils
pip install -e .