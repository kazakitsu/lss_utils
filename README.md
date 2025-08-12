# lss_utils

Utilities and scripts for analyzing N-body simulation data, with both NumPy and JAX backends.

---

## Features

- **Mesh assignment**  
  Assign particle data to a regular grid (NGP, CIC, TSC) using `Mesh_Assignment`.
- **Power spectrum & bispectrum measurement**  
  Compute the 1D/2D power spectrum and the 1D bispectrum efficiently using FFTs (`Measure_Pk`, `Measure_spectra_FFT`).
- **Dual backend**  
  - **JAX** implementations for GPU/TPU acceleration  
  - Fallback to pure NumPy if JAX is not available
- **Minimal dependencies**  
  Requires only `numpy`, with `jax` as an optional dependency.

---

## Prerequisites

- **Python >= 3.10**
- **NumPy >= 2.1**
- **JAX >= 0.4.3**
- **(For GPU/TPU acceleration)** Install JAX *before* installing **lss_utils**, following the [official instructions](https://github.com/jax-ml/jax#installation):

  ```bash
  # CPU-only
  pip install -U jax

  # GPU (CUDA 12.x) example
  pip install -U "jax[cuda12]"


## Installation

Once prerequisites are met, install **lss_utils** with pip:

```bash
git clone https://github.com/kazakitsu/lss_utils.git
cd lss_utils
pip install .
```

If you want to work on or modify the code locally:
```bash
git clone https://github.com/kazakitsu/lss_utils.git
cd lss_utils
pip install -e .
```

## Usage

Below is a minimal example that assigns particles to a mesh and measures the 1D power spectrum monopole.

```python
import jax
import jax.numpy as jnp
from lss_utils import Mesh_Assignment, Measure_Pk

# -------------------------------
# Configuration
# -------------------------------
boxsize = 1000.0  # Mpc/h
ng = 256          # grid size per side
dtype = jnp.float32

# -------------------------------
# Example particle catalog
# NOTE:
# - `pos` must have shape (3, N): (x, y, z) stacked along axis 0
# - coordinates should be in [0, boxsize)
# - `mass` is optional; it can be a scalar or an array of shape (N).
#   If omitted, all particles are assigned a unit mass.
# -------------------------------
key = jax.random.PRNGKey(0)
N = 100_000
pos = jax.random.uniform(key, shape=(3, N), minval=0.0, maxval=boxsize, dtype=dtype)

# -------------------------------
# Mesh assignment
# `assign_fft` returns the rfftn field in k-space.
# -------------------------------
mesh = Mesh_Assignment(
    boxsize=boxsize,
    ng=ng,
    window_order=2,   # NGP:1, CIC:2, TSC:3
    interlace=True,
    dtype=dtype,
)
field_k = mesh.assign_fft(pos)  # shape = (ng, ng, ng//2+1) in complex dtype

# -------------------------------
# Power spectrum measurement
# Measure_Pk.__call__(fieldk1, fieldk2=None, ell=0, mu_min=0.0, mu_max=1.0)
# Returns an array of shape (num_bins, 3) with columns:
#   [0] k_mean    : mean wavenumber in the bin
#   [1] P(k)      : binned power
#   [2] N_k       : mode count
# -------------------------------
k_edges = jnp.linspace(0.01, 1.0, 50, dtype=dtype)  # k-bin edges [h/Mpc]

measure_pk = Measure_Pk(
    boxsize=boxsize,
    ng=ng,
    kbin_edges=k_edges,
    ell_max=4,    # enable {0,2,4}. You can still call with ell=0 below.
    dtype=dtype,
)

# Monopole over all mu (0 <= mu <= 1)
pk = measure_pk(field_k)

print("k (cemean):", pk[:,0])
print("P(k):", pk[:,1])
print("N_k:", pk[:,2])
```

For more detailed examples, see the example/ folder (to be added).