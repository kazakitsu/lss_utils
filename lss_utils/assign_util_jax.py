#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple

import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

# --- Public API class ---
class Mesh_Assignment:
    """
    Example
    -------
    >>> mesh = Mesh_Assignment(boxsize=1000.0, ng=512, window_order=2,interlace=True)
    >>> field_k = mesh.assign_fft(pos, mass)
    """

    def __init__(
        self,
        boxsize: float,
        ng: int,
        window_order: int,
        *,
        interlace: bool = False,
        normalize: bool = True,
        max_scatter_indices: int = 100_000_000,
    ):
        self.boxsize = float(boxsize)
        self.ng = int(ng)
        self.window_order = int(window_order)
        self.interlace = bool(interlace)
        self.normalize = bool(normalize)
        self.max_scatter_indices = int(max_scatter_indices)

        # derived constants (shared across calls)
        self.cell = self.boxsize / self.ng
        self.kvec = _rfftn_kvec((self.ng,) * 3, self.boxsize)
        self.Wk = _deconvolve(self.kvec, self.boxsize, self.window_order)

        # pre‑compile inner assign
        self._single_assign = jit(
            partial(_single_assign, ng=self.ng, window_order=self.window_order)
        )

    # -------- public methods -------- #
    def assign_fft(self, pos, weight):
        """Density to Fourier space (optionally interlaced)"""
        field_r = self.assign_to_grid(pos, weight, interlace=False)

        if self.interlace:
            field_r_i = self.assign_to_grid(pos, weight, interlace=True)
            return self.fft_deconvolve(field_r, field_r_i)
        return self.fft_deconvolve(field_r)
    
    def fft_deconvolve(self, field_r, field_r_i=None):
        """FFT and deconvolve (optionally interlaced)"""
        field_k = jnp.fft.rfftn(field_r) / self.ng ** 3

        if field_r_i is not None:
            field_k_i = jnp.fft.rfftn(field_r_i) / self.ng ** 3
            # interlacing phase shift
            nvec = self.kvec * (self.boxsize / (2.0 * jnp.pi))
            phase = jnp.exp(1j * jnp.pi * jnp.sum(nvec, axis=0) / self.ng)
            field_k = 0.5 * (field_k + field_k_i * phase)

        field_k = field_k * self.Wk

        if self.normalize:
            field_k = field_k.at[0, 0, 0].set(0.0)

        return field_k

    def assign_to_grid(self, pos, weight, *, interlace: bool = False, normalize_mean: bool = True):
        """Position-space mass assignment (NGP/CIC/TSC)."""
        return _assign_to_grid(
            self.ng,
            self.window_order,
            self.cell,
            pos,
            weight,
            interlace,
            normalize_mean,
            self.max_scatter_indices,
            self._single_assign,
        )

# ---------- assign_to_grid (chunk loop) ---------- #
def _assign_to_grid(
    ng: int,
    window_order: int,
    cell: float,
    pos,
    weight,
    interlace: bool,
    normalize: bool,
    max_scatter_indices: int,
    single_assign_fn,  # pre‑compiled _single_assign
):
    """Mass-assignment with optional chunking."""
    field = jnp.zeros((ng,) * 3, dtype=pos.dtype)

    # flatten (3,Nx,Ny,Nz) -> (3,N)
    if pos.ndim == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    pos_mesh = pos / cell
    if interlace:
        pos_mesh = pos_mesh + 0.5

    num_p = pos.shape[1]
    n_shifts = {1: 1, 2: 8, 3: 27}[window_order]
    chunk_size = min(max(max_scatter_indices // n_shifts, 1), num_p)
    n_chunks = (num_p + chunk_size - 1) // chunk_size

    # pad once for cheap slicing
    pos_pad = jnp.pad(pos_mesh, ((0, 0), (0, chunk_size)), constant_values=0.0)
    wt_pad = jnp.pad(weight, (0, chunk_size), constant_values=0.0)

    def body(i, fld):
        start = i * chunk_size
        pos_ck = lax.dynamic_slice_in_dim(pos_pad, start, chunk_size, axis=-1)
        wt_ck = lax.dynamic_slice_in_dim(wt_pad, start, chunk_size, axis=0)
        return single_assign_fn(fld, pos_ck, wt_ck)

    field = lax.fori_loop(0, n_chunks, body, field)

    if normalize:
        field = field / (num_p / ng ** 3)

    return field

# ------------ one-chunk core ------------
@partial(jit, static_argnames=('ng', 'window_order'))
def _single_assign(field, pos_mesh, weight, *, ng: int, window_order: int):
    # --- base cell・fractional offset・shift list -----------------
    if window_order == 1:                                    # NGP
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = jnp.zeros_like(pos_mesh)
        shifts = jnp.array([[0, 0, 0]], jnp.int32)
    elif window_order == 2:                                  # CIC
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = pos_mesh - imesh
        shifts = jnp.array([[dx, dy, dz]
                            for dx in (0, 1)
                            for dy in (0, 1)
                            for dz in (0, 1)], jnp.int32)
    else:                                                    # TSC
        imesh = (jnp.floor(pos_mesh - 1.5).astype(jnp.int32) + 2)
        fmesh = pos_mesh - imesh
        shifts = jnp.array([[dx, dy, dz]
                            for dx in (-1, 0, 1)
                            for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)], jnp.int32)
    #periodic BC
    imesh = jnp.mod(imesh, ng)                               

    stride_y, stride_x = ng, ng * ng
    flat = field.ravel()                                     # 1‑D view
    N = pos_mesh.shape[1]

    # --- scan over shifts ----------------------------------------
    def body(flat_field, sh):
        dx, dy, dz = sh
        idx_x = (imesh[0] + dx) % ng
        idx_y = (imesh[1] + dy) % ng
        idx_z = (imesh[2] + dz) % ng
        flat_idx = idx_x * stride_x + idx_y * stride_y + idx_z

        # per-particle weight
        if window_order == 1:
            w_shift = weight
        elif window_order == 2:
            w_x = jnp.where(dx == 0, 1.0 - fmesh[0], fmesh[0])
            w_y = jnp.where(dy == 0, 1.0 - fmesh[1], fmesh[1])
            w_z = jnp.where(dz == 0, 1.0 - fmesh[2], fmesh[2])
            w_shift = w_x * w_y * w_z * weight
        else:
            w_x = jnp.where(dx == 0,
                            0.75 - fmesh[0] ** 2,
                            0.5 * (fmesh[0] ** 2 + dx * fmesh[0] + 0.25))
            w_y = jnp.where(dy == 0,
                            0.75 - fmesh[1] ** 2,
                            0.5 * (fmesh[1] ** 2 + dy * fmesh[1] + 0.25))
            w_z = jnp.where(dz == 0,
                            0.75 - fmesh[2] ** 2,
                            0.5 * (fmesh[2] ** 2 + dz * fmesh[2] + 0.25))
            w_shift = w_x * w_y * w_z * weight

        flat_field = flat_field.at[flat_idx].add(w_shift.astype(field.dtype))
        return flat_field, None

    flat, _ = lax.scan(body, flat, shifts)                   # fused loop
    return flat.reshape(field.shape)

@partial(jit, static_argnames=("shape",))
def _rfftn_kvec(shape: Tuple[int, int, int], boxsize: float, dtype=float):
    """
    Return k-vector array with shape (3, *shape[:-1], shape[-1]//2+1).
    """
    spacing = boxsize / (2.*jnp.pi) / shape[-1]
    # Create 1D frequency arrays for each dimension.
    freqs = [jnp.fft.fftfreq(n, d=spacing) for n in shape[:-1]]
    freqs.append(jnp.fft.rfftfreq(shape[-1], d=spacing))

    # Use jnp.meshgrid to create the coordinate grid.
    kvec_grid = jnp.meshgrid(*freqs, indexing='ij')
    
    # Stack the coordinate arrays to get the final (D, N1, N2, ...) shape.    
    return jnp.stack(kvec_grid, axis=0).astype(dtype)
    
@partial(jit, static_argnames=('boxsize', 'window_order',))
def _deconvolve(kvec, boxsize: float, window_order: int):
    """
    Computes the deconvolution window function to correct for the assignment shape.
    """
    ng = kvec.shape[1]
    
    # Convert physical k-vector back to grid index vector `nvec`.
    # k = n * (2 * pi / boxsize) -> n = k * boxsize / (2 * pi)
    nvec = kvec * (boxsize / (2. * jnp.pi))

    # The argument to the sinc function is n / ng.
    # jnp.sinc(x) computes sin(pi*x)/(pi*x).
    sinc_vals = jnp.sinc(nvec / ng)
    sinc_vals = jnp.where(sinc_vals == 0, 1.0, sinc_vals) # avoid 0/0
    window_ft = jnp.prod(sinc_vals, axis=0)
    
    # The deconvolution kernel is the inverse of the window's Fourier transform.
    wk = jnp.where(window_ft == 0.0, 0.0, 1.0 / window_ft)
    
    return wk ** window_order