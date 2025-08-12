#!/usr/bin/env python3
from __future__ import annotations
import sys
from functools import partial
from typing import Tuple

import numpy as np


# --- Public API class ---
class Mesh_Assignment:
    """
    Example
    -------
    >>> mesh = Mesh_Assignment(boxsize=1000.0, ng=512, window_order=2, interlace=True, dtype=np.float32)
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
        max_scatter_indices: int = 200_000_000,
        dtype=np.float32,
    ):
        self.boxsize = float(boxsize)
        self.ng = int(ng)
        self.window_order = int(window_order)
        self.interlace = bool(interlace)
        self.normalize = bool(normalize)
        self.max_scatter_indices = int(max_scatter_indices)

        # Precision
        self.real_dtype = np.dtype(dtype)
        self.complex_dtype = np.complex64 if self.real_dtype == np.float32 else np.complex128

        # Avoid upcasting issues
        self.cell = np.asarray(self.boxsize / self.ng, dtype=self.real_dtype)

        # 1D frequency indices
        self.nx = _fftfreq_index_1d(self.ng).astype(self.real_dtype)
        self.ny = self.nx
        self.nz = _rfftfreq_index_1d(self.ng).astype(self.real_dtype)

        # Interlacing phase
        pi = np.array(np.pi, dtype=self.real_dtype)
        inv_ng = np.asarray(self.ng, dtype=self.real_dtype)
        self.phase_x_1d = np.exp(1j * (pi * self.nx / inv_ng)).astype(self.complex_dtype)
        self.phase_z_1d = np.exp(1j * (pi * self.nz / inv_ng)).astype(self.complex_dtype)

        # Deconvolution weights
        self.wx, self.wy, self.wz = _deconv_weights_1d(
            self.nx, self.ny, self.nz, self.ng, self.window_order, self.real_dtype
        )

        # Precompute neighbor shifts per window (NGP/CIC/TSC)
        if self.window_order == 1:
            shifts = [(0, 0, 0)]
        elif self.window_order == 2:
            shifts = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
        else:
            shifts = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        self._shifts = np.asarray(shifts, dtype=np.int32)

    # -------- public API -------- #
    def assign_fft(self, pos, weight=1.0):
        pos = np.asarray(pos, dtype=self.real_dtype)
        weight = np.asarray(weight, dtype=self.real_dtype)
        field_r = self.assign_to_grid(pos, weight, interlace=False)
        if self.interlace:
            field_r_i = self.assign_to_grid(pos, weight, interlace=True)
            return self.fft_deconvolve(field_r, field_r_i)
        return self.fft_deconvolve(field_r)

    def fft_deconvolve(self, field_r, field_r_i=None):
        field_r = field_r.astype(self.real_dtype, copy=False)
        field_k = np.fft.rfftn(field_r, norm='forward').astype(self.complex_dtype, copy=False)

        if field_r_i is not None:
            field_r_i = field_r_i.astype(self.real_dtype, copy=False)
            field_k_i = np.fft.rfftn(field_r_i, norm='forward').astype(self.complex_dtype, copy=False)
            field_k_i = field_k_i * self.phase_x_1d[:, None, None]
            field_k_i = field_k_i * self.phase_x_1d[None, :, None]
            field_k_i = field_k_i * self.phase_z_1d[None, None, :]
            field_k = 0.5 * (field_k + field_k_i)

        field_k = field_k * self.wx[:, None, None].astype(self.complex_dtype, copy=False)
        field_k = field_k * self.wy[None, :, None].astype(self.complex_dtype, copy=False)
        field_k = field_k * self.wz[None, None, :].astype(self.complex_dtype, copy=False)

        if self.normalize:
            field_k[0, 0, 0] = 0.0
        return field_k

    def assign_to_grid(self, pos, weight=1.0, *, interlace: bool = False, normalize_mean: bool = True):
        pos = np.asarray(pos, dtype=self.real_dtype)
        weight = np.asarray(weight, dtype=self.real_dtype)
        return _assign_to_grid(
            self.ng,
            self.window_order,
            self.cell,
            pos,
            weight,
            interlace,
            normalize_mean,
            self.max_scatter_indices,
            # pass closures so we reuse precomputed shifts
            single_assign_fn=partial(_single_assign_numpy, ng=self.ng, window_order=self.window_order, shifts=self._shifts),
            dtype=self.real_dtype,
        )

    # -------- directly to grid from disp and weight -------- #
    def assign_from_disp_to_grid(
        self,
        disp_r,
        weight,
        *,
        interlace: bool = False,
        normalize_mean: bool = True,
    ):
        """disp_r: (3, ng_L, ng_L, ng_L), weight: (ng_L, ng_L, ng_L) or scalar"""
        ng_L = int(disp_r.shape[1])
        s = {1: 1, 2: 8, 3: 27}[self.window_order]
        slab = max(1, min(ng_L, self.max_scatter_indices // (s * ng_L * ng_L)))

        field_r = _assign_from_disp_to_grid_slab_numpy(
            ng=self.ng,
            window_order=self.window_order,
            slab=int(slab),
            cell=self.cell,
            disp_r=np.asarray(disp_r, dtype=self.real_dtype),
            weight=np.asarray(weight, dtype=self.real_dtype),
            interlace=bool(interlace),
            single_assign_fn=partial(_single_assign_numpy, ng=self.ng, window_order=self.window_order, shifts=self._shifts),
        )
        if normalize_mean:
            ng = np.asarray(self.ng, dtype=self.real_dtype)
            norm = ng ** 3
            n_particles = np.asarray(disp_r.shape[1], dtype=self.real_dtype) ** 3
            field_r = field_r * (norm / n_particles)
        return field_r

    def assign_from_disp_fft(self, disp_r, weight):
        field_r = self.assign_from_disp_to_grid(disp_r, weight, interlace=False)
        if self.interlace:
            field_r_i = self.assign_from_disp_to_grid(disp_r, weight, interlace=True)
            field_k = self.fft_deconvolve(field_r, field_r_i)
        else:
            field_k = self.fft_deconvolve(field_r)
        return field_k


# ---------- assign_to_grid (chunk loop) ---------- #
def _assign_to_grid(
    ng: int,
    window_order: int,
    cell,
    pos,
    weight,
    interlace: bool,
    normalize: bool,
    max_scatter_indices: int,
    single_assign_fn,
    *,
    dtype,
):
    field_dtype = np.dtype(dtype)

    pos = pos.astype(field_dtype, copy=False)
    weight = weight.astype(field_dtype, copy=False)

    field = np.zeros((ng, ng, ng), dtype=field_dtype)

    # Accept (3, N) or (3, ng, ng, ng)
    if pos.ndim == 4:
        pos = pos.reshape(3, -1)

    num_p = int(pos.shape[1])

    weight_is_scalar = (weight.ndim == 0)
    if not weight_is_scalar:
        if int(weight.size) != num_p:
            raise ValueError(f"weight length {weight.size} must equal num particles {num_p}.")
        weight = weight.reshape(-1).astype(field_dtype, copy=False)

    n_shifts = {1: 1, 2: 8, 3: 27}[window_order]
    base = max(max_scatter_indices // n_shifts, 1)
    chunk_size = min(base, num_p)

    n_chunks = (num_p + chunk_size - 1) // chunk_size
    print('n_chunks:', n_chunks, file=sys.stderr)

    # padding for positions to simplify slicing
    pad = chunk_size
    pos_pad = np.pad(pos, ((0, 0), (0, pad)), mode='constant')

    if not weight_is_scalar:
        wt_pad = np.pad(weight, (0, pad), mode='constant')

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        pos_ck = pos_pad[:, start:end]

        pos_mesh_ck = pos_ck / cell
        if interlace:
            pos_mesh_ck = pos_mesh_ck + np.array(0.5, dtype=field_dtype)

        if weight_is_scalar:
            valid = (np.arange(chunk_size) < (num_p - start)).astype(field_dtype)
            wt_ck = (weight.astype(field_dtype) * valid)  # (chunk_size,)
        else:
            wt_ck = wt_pad[start:end]

        field = single_assign_fn(field, pos_mesh_ck, wt_ck)

    if normalize:
        ngv = np.asarray(ng, dtype=field_dtype)
        norm = ngv ** 3
        n_particles = np.asarray(num_p, dtype=field_dtype)
        field = field * (norm / n_particles)

    return field


# ---------- directly scatter from displacement, using z slab (NumPy) ---------- #
def _assign_from_disp_to_grid_slab_numpy(
    *,
    ng: int,
    window_order: int,
    slab: int,
    cell: float,
    disp_r: np.ndarray,   # (3, ng_L, ng_L, ng_L)
    weight: np.ndarray,   # (ng_L, ng_L, ng_L) or scalar
    interlace: bool,
    single_assign_fn,
):
    fdtype = disp_r.dtype
    field = np.zeros((ng, ng, ng), dtype=fdtype)

    # Lagrangian size
    ng_L = disp_r.shape[1]

    # scale from Lagrangian integer to Eulerian cell units
    scale = np.asarray(ng, dtype=fdtype) / np.asarray(ng_L, dtype=fdtype)
    cell_arr = np.asarray(cell, dtype=fdtype)

    # pad z to multiple of slab
    pad_z = (-ng_L) % slab
    disp_pad = np.pad(disp_r, ((0, 0), (0, 0), (0, 0), (0, pad_z)), mode='constant')

    weight_is_scalar = (weight.ndim == 0)
    if not weight_is_scalar:
        w_pad = np.pad(weight, ((0, 0), (0, 0), (0, pad_z)), mode='constant')

    n_slices = (ng_L + pad_z) // slab
    print('n_slices:', n_slices, file=sys.stderr)

    # integer base grid (2D) — avoid building 3D base arrays
    ix_L = np.arange(ng_L, dtype=fdtype)[:, None, None]
    iy_L = np.arange(ng_L, dtype=fdtype)[None, :, None]

    for i in range(n_slices):
        z0 = i * slab

        # slice (always in-bounds thanks to padding)
        disp_ck = disp_pad[:, :, :, z0:z0 + slab]  # (3, ngL, ngL, slab)

        # (Lagrangian) z indices for this slab
        iz_L = (z0 + np.arange(slab, dtype=fdtype))[None, None, :]

        # convert to Eulerian cell units
        pos_x = ix_L * scale + disp_ck[0] / cell_arr
        pos_y = iy_L * scale + disp_ck[1] / cell_arr
        pos_z = iz_L * scale + disp_ck[2] / cell_arr

        pos_mesh_ck = np.stack([pos_x, pos_y, pos_z], axis=0).reshape(3, -1)
        if interlace:
            pos_mesh_ck = pos_mesh_ck + np.array(0.5, dtype=fdtype)

        # weights
        if weight_is_scalar:
            # mask-out padded z
            Lz = min(slab, ng_L - z0)
            valid_z = (np.arange(slab) < Lz)[None, None, :]          # (1,1,slab)
            valid = np.broadcast_to(valid_z, (ng_L, ng_L, slab)).astype(fdtype)
            wt_ck_flat = (valid * weight.astype(fdtype)).reshape(-1)
        else:
            w_ck = w_pad[:, :, z0:z0 + slab]
            wt_ck_flat = w_ck.reshape(-1)

        field = single_assign_fn(field, pos_mesh_ck, wt_ck_flat)

    return field


# ------------ one-chunk core (3D scatter version, NumPy) ------------
def _single_assign_numpy(field, pos_mesh, weight, *, ng: int, window_order: int, shifts: np.ndarray):
    """
    field: (ng, ng, ng)
    pos_mesh: (3, N) in Eulerian *cell* units
    weight: (N,) or scalar (broadcast OK)
    """
    if window_order == 1:  # NGP
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = np.zeros_like(pos_mesh)  # unused
    elif window_order == 2:  # CIC
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = pos_mesh - imesh
    else:  # TSC
        imesh = (np.floor(pos_mesh - 1.5).astype(np.int32) + 2)
        fmesh = pos_mesh - imesh

    # periodic wrap
    imesh %= ng

    # scatter-add for each neighbor shift
    for dx, dy, dz in shifts:
        idx_x = (imesh[0] + dx) % ng
        idx_y = (imesh[1] + dy) % ng
        idx_z = (imesh[2] + dz) % ng

        if window_order == 1:
            w_shift = weight
        elif window_order == 2:
            w_x = np.where(dx == 0, 1.0 - fmesh[0], fmesh[0])
            w_y = np.where(dy == 0, 1.0 - fmesh[1], fmesh[1])
            w_z = np.where(dz == 0, 1.0 - fmesh[2], fmesh[2])
            w_shift = (w_x * w_y * w_z) * weight
        else:
            # TSC 1D kernel:
            # if s==0: 0.75 - u^2
            # else    : 0.5 * (u^2 + s*u + 0.25)
            w_x = np.where(
                dx == 0,
                0.75 - fmesh[0] ** 2,
                0.5 * (fmesh[0] ** 2 + dx * fmesh[0] + 0.25),
            )
            w_y = np.where(
                dy == 0,
                0.75 - fmesh[1] ** 2,
                0.5 * (fmesh[1] ** 2 + dy * fmesh[1] + 0.25),
            )
            w_z = np.where(
                dz == 0,
                0.75 - fmesh[2] ** 2,
                0.5 * (fmesh[2] ** 2 + dz * fmesh[2] + 0.25),
            )
            w_shift = (w_x * w_y * w_z) * weight

        # atomic scatter-add (handles duplicates)
        np.add.at(field, (idx_x, idx_y, idx_z), w_shift.astype(field.dtype, copy=False))

    return field


# ---------- 1D helpers ----------
def _fftfreq_index_1d(n: int):
    # Equivalent to jnp.fft.fftfreq(n, d=1.0/n)
    return np.fft.fftfreq(n, d=1.0 / n)

def _rfftfreq_index_1d(n: int):
    return np.fft.rfftfreq(n, d=1.0 / n)

def _deconv_weights_1d(nx, ny, nz, ng: int, window_order: int, dtype):
    dtype = np.dtype(dtype)
    one = np.array(1.0, dtype=dtype)
    sx = np.sinc((nx / ng).astype(dtype))
    sy = np.sinc((ny / ng).astype(dtype))
    sz = np.sinc((nz / ng).astype(dtype))
    # avoid divide-by-zero
    sx = np.where(sx == 0, one, sx)
    sy = np.where(sy == 0, one, sy)
    sz = np.where(sz == 0, one, sz)
    px = (one / sx) ** window_order
    py = (one / sy) ** window_order
    pz = (one / sz) ** window_order
    return px.astype(dtype, copy=False), py.astype(dtype, copy=False), pz.astype(dtype, copy=False)
