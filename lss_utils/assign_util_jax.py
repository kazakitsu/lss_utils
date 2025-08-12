#!/usr/bin/env python3
from __future__ import annotations
import sys

import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# --- Public API class ---
class Mesh_Assignment:
    """
    Example
    -------
    >>> mesh = Mesh_Assignment(boxsize=1000.0, ng=512, window_order=2, interlace=True, dtype=jnp.float32)
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
        dtype=jnp.float32,
    ):
        self.boxsize = float(boxsize)
        self.ng = int(ng)
        self.window_order = int(window_order)
        self.interlace = bool(interlace)
        self.normalize = bool(normalize)
        self.max_scatter_indices = int(max_scatter_indices)

        # Precision
        self.real_dtype = jnp.dtype(dtype)
        self.complex_dtype = jnp.complex64 if self.real_dtype == jnp.float32 else jnp.complex128

        # Avoid upcasting issues
        self.cell = jnp.asarray(self.boxsize / self.ng, dtype=self.real_dtype)

        # 1D frequency indices
        self.nx = _fftfreq_index_1d(self.ng).astype(self.real_dtype)
        self.ny = self.nx
        self.nz = _rfftfreq_index_1d(self.ng).astype(self.real_dtype)

        # Interlacing phase
        pi = jnp.array(jnp.pi, dtype=self.real_dtype)
        inv_ng = jnp.asarray(self.ng, dtype=self.real_dtype)
        self.phase_x_1d = jnp.exp(1j * (pi * self.nx / inv_ng)).astype(self.complex_dtype)
        self.phase_z_1d = jnp.exp(1j * (pi * self.nz / inv_ng)).astype(self.complex_dtype)

        # Deconvolution weights
        self.wx, self.wy, self.wz = _deconv_weights_1d(
            self.nx, self.ny, self.nz, self.ng, self.window_order, self.real_dtype
        )

        # Pre-compile inner assign kernels
        self._single_assign = jit(partial(_single_assign, ng=self.ng, window_order=self.window_order))

    # -------- public API -------- #
    def assign_fft(self, pos, weight = 1.0):
        pos = jnp.asarray(pos, dtype=self.real_dtype)
        weight = jnp.asarray(weight, dtype=self.real_dtype)
        field_r = self.assign_to_grid(pos, weight, interlace=False)
        if self.interlace:
            field_r_i = self.assign_to_grid(pos, weight, interlace=True)
            return self.fft_deconvolve(field_r, field_r_i)
        return self.fft_deconvolve(field_r)

    def fft_deconvolve(self, field_r, field_r_i=None):
        field_r = field_r.astype(self.real_dtype)
        field_k = jnp.fft.rfftn(field_r, norm='forward').astype(self.complex_dtype)

        if field_r_i is not None:
            field_r_i = field_r_i.astype(self.real_dtype)
            field_k_i = jnp.fft.rfftn(field_r_i, norm='forward').astype(self.complex_dtype)
            field_k_i = field_k_i * self.phase_x_1d[:, None, None]
            field_k_i = field_k_i * self.phase_x_1d[None, :, None]
            field_k_i = field_k_i * self.phase_z_1d[None, None, :]
            field_k = 0.5 * (field_k + field_k_i)

        field_k = field_k * self.wx[:, None, None].astype(self.complex_dtype)
        field_k = field_k * self.wy[None, :, None].astype(self.complex_dtype)
        field_k = field_k * self.wz[None, None, :].astype(self.complex_dtype)

        if self.normalize:
            field_k = field_k.at[0, 0, 0].set(0.0)
        return field_k

    def assign_to_grid(self, pos, weight = 1.0, *, interlace: bool = False, normalize_mean: bool = True):
        pos = jnp.asarray(pos, dtype=self.real_dtype)
        weight = jnp.asarray(weight, dtype=self.real_dtype)
        return _assign_to_grid(
            self.ng,
            self.window_order,
            self.cell,                      # jnp scalar
            pos,
            weight,
            interlace,
            normalize_mean,
            self.max_scatter_indices,
            self._single_assign,
            dtype=self.real_dtype,
        )

    # -------- directly to grid from disp and weight -------- #
    def assign_from_disp_to_grid(self, 
                                 disp_r,
                                 weight, 
                                 *, 
                                 interlace: bool=False, 
                                 normalize_mean: bool=True):
        """disp_r: (3, ng, ng, ng), weight: (ng, ng, ng)"""
        ng_L = int(disp_r.shape[1])
        s = {1:1, 2:8, 3:27}[self.window_order]
        slab = max(1, min(ng_L, self.max_scatter_indices // (s * ng_L * ng_L)))
        field_r = _assign_from_disp_to_grid_slab(
            ng=self.ng,
            window_order=self.window_order,
            slab=int(slab),
            cell=self.cell,
            disp_r=jnp.asarray(disp_r, dtype=self.real_dtype),
            weight=jnp.asarray(weight, dtype=self.real_dtype),
            interlace=bool(interlace),
            single_assign_fn=self._single_assign,
        )
        if normalize_mean:
            ng = jnp.asarray(self.ng, dtype=self.real_dtype)
            norm = ng**3
            n_particles = jnp.asarray(disp_r.shape[1], dtype=self.real_dtype) ** 3
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
    cell,               # jnp scalar
    pos,
    weight,
    interlace: bool,
    normalize: bool,
    max_scatter_indices: int,
    single_assign_fn,
    *,
    dtype,
):
    field_dtype = jnp.dtype(dtype)

    pos = pos.astype(field_dtype)
    weight = weight.astype(field_dtype)

    field = jnp.zeros((ng,) * 3, dtype=field_dtype)

    if pos.ndim == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    num_p = int(pos.shape[1])

    weight_is_scalar = (weight.ndim == 0)
    if weight_is_scalar:
        w_scalar = jnp.squeeze(weight).astype(field_dtype)
    else:
        if int(weight.size) != num_p:
            raise ValueError(f"weight length {weight.size} must equal num particles {num_p}.")
        weight = weight.reshape(-1).astype(field_dtype)


    n_shifts = {1: 1, 2: 8, 3: 27}[window_order]

    base = max(max_scatter_indices // n_shifts, 1)
    chunk_size = min(base, num_p)

    n_chunks = (num_p + chunk_size - 1) // chunk_size
    print('n_chunks:', n_chunks, file=sys.stderr)

    # padding
    pad_val = jnp.array(0.0, dtype=pos.dtype)
    pos_pad = jnp.pad(pos, ((0, 0), (0, chunk_size)), constant_values=pad_val)

    # if weight is a scalar, no padding
    if not weight_is_scalar:
        weight = weight.reshape(-1).astype(field_dtype)
        wt_pad = jnp.pad(weight, (0, chunk_size), constant_values=jnp.array(0, dtype=field_dtype))

    def body(i, fld):
        start = i * chunk_size
        pos_ck = lax.dynamic_slice_in_dim(pos_pad, start, chunk_size, axis=-1)

        pos_mesh_ck = pos_ck / cell
        if interlace:
            half = jnp.array(0.5, dtype=field_dtype)
            pos_mesh_ck = pos_mesh_ck + half

        if weight_is_scalar:
            valid = (jnp.arange(chunk_size) < (num_p - start)).astype(field_dtype)
            wt_ck = (weight.astype(field_dtype) * valid)  # (chunk_size,)
        else:
            wt_ck = lax.dynamic_slice_in_dim(wt_pad, start, chunk_size, axis=0)

        return single_assign_fn(fld, pos_mesh_ck, wt_ck)

    field = lax.fori_loop(0, n_chunks, body, field)

    if normalize:
        ng = jnp.asarray(ng, dtype=field_dtype)
        norm = ng ** 3
        n_particles = jnp.asarray(num_p, dtype=field_dtype)
        field = field * (norm / n_particles)

    return field


# ---------- directly scatter from displacement, using z slab ---------- #
@partial(jit, static_argnames=('ng','window_order','slab','interlace','single_assign_fn'))
def _assign_from_disp_to_grid_slab(
    *, ng: int, 
    window_order: int, 
    slab: int, 
    cell: float,
    disp_r: jnp.ndarray,     # (3, ng_L, ng_L, ng_L)
    weight: jnp.ndarray,     # (ng_L, ng_L, ng_L)
    interlace: bool,
    single_assign_fn,
):
    fdtype = disp_r.dtype
    field  = jnp.zeros((ng, ng, ng), dtype=fdtype)

    # Lagrangian size
    ng_L = disp_r.shape[1]

    # scale from Lagrangian index units to Eulerian cell units
    scale = jnp.asarray(ng,   dtype=fdtype) / jnp.asarray(ng_L, dtype=fdtype)
    cell_arr = jnp.asarray(cell, dtype=fdtype)

    # pad along z so that ng_L + pad_z is a multiple of 'slab'
    pad_z = (-ng_L) % slab
    zero  = jnp.array(0, dtype=fdtype)
    disp_pad = jnp.pad(disp_r, ((0,0),(0,0),(0,0),(0,pad_z)), constant_values=zero)


    weight = jnp.asarray(weight, dtype=fdtype)
    weight_is_scalar = (weight.ndim == 0)
    if not weight_is_scalar:
        w_pad = jnp.pad(weight, ((0,0),(0,0),(0,pad_z)), constant_values=zero)

    # number of slices in Lagrangian z
    n_slices = (ng_L + pad_z) // slab
    print('n_slices:', n_slices, file=sys.stderr)

    # base integer grid in Lagrangian indices (kept 2D to avoid 3D residency)
    ix_L = jnp.arange(ng_L, dtype=fdtype)[:, None, None]
    iy_L = jnp.arange(ng_L, dtype=fdtype)[None, :, None]

    def body(i, fld):
        z0 = i * slab

        # slice ALWAYS within bounds thanks to padding
        disp_ck = lax.dynamic_slice(disp_pad, (0, 0, 0, z0), (3, ng_L, ng_L, slab))

        # z indices in Lagrangian integer units for this block
        iz_L = (z0 + jnp.arange(slab, dtype=fdtype))[None, None, :]

        # convert to Eulerian cell units: i_L * scale + disp / cell_E
        pos_x = ix_L * scale + disp_ck[0] / cell_arr
        pos_y = iy_L * scale + disp_ck[1] / cell_arr
        pos_z = iz_L * scale + disp_ck[2] / cell_arr

        pos_mesh_ck = jnp.stack([pos_x, pos_y, pos_z], axis=0).reshape(3, -1)
        if interlace:
            pos_mesh_ck = pos_mesh_ck + jnp.array(0.5, dtype=fdtype)

        if weight_is_scalar:
            Lz = jnp.minimum(slab, ng_L - z0)
            valid_z = (jnp.arange(slab) < Lz)[None, None, :] # (1, 1, slab)
            valid   = jnp.broadcast_to(valid_z, (ng_L, ng_L, slab)).astype(fdtype)
            wt_ck_flat = (valid * weight.astype(fdtype)).reshape(-1)
        else:
            w_ck    = lax.dynamic_slice(w_pad, (0, 0, z0), (ng_L, ng_L, slab))
            wt_ck_flat = w_ck.reshape(-1)

        return single_assign_fn(fld, pos_mesh_ck, wt_ck_flat)

    field = lax.fori_loop(0, n_slices, body, field)
    return field


# ------------ one-chunk core (3D scatter version) ------------
@partial(jit, static_argnames=('ng', 'window_order'))
def _single_assign(field, pos_mesh, weight, *, ng: int, window_order: int):
    if window_order == 1:  # NGP
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = jnp.zeros_like(pos_mesh)
        shifts = jnp.array([[0, 0, 0]], jnp.int32)
    elif window_order == 2:  # CIC
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = pos_mesh - imesh
        shifts = jnp.array([[dx, dy, dz]
                            for dx in (0, 1)
                            for dy in (0, 1)
                            for dz in (0, 1)], jnp.int32)
    else:  # TSC
        imesh = (jnp.floor(pos_mesh - 1.5).astype(jnp.int32) + 2)
        fmesh = pos_mesh - imesh
        shifts = jnp.array([[dx, dy, dz]
                            for dx in (-1, 0, 1)
                            for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)], jnp.int32)

    imesh = jnp.mod(imesh, ng)

    def body(fld, sh):
        dx, dy, dz = sh
        idx_x = (imesh[0] + dx) % ng
        idx_y = (imesh[1] + dy) % ng
        idx_z = (imesh[2] + dz) % ng

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

        fld = fld.at[idx_x, idx_y, idx_z].add(w_shift.astype(fld.dtype))
        return fld, None

    field, _ = lax.scan(body, field, shifts)
    return field


# ---------- 1D helpers ----------
def _fftfreq_index_1d(n: int):
    return jnp.fft.fftfreq(n, d=1.0 / n)

def _rfftfreq_index_1d(n: int):
    return jnp.fft.rfftfreq(n, d=1.0 / n)

def _deconv_weights_1d(nx, ny, nz, ng: int, window_order: int, dtype):
    dtype = jnp.dtype(dtype)
    one = jnp.array(1.0, dtype=dtype)
    sx = jnp.sinc((nx / ng).astype(dtype))
    sy = jnp.sinc((ny / ng).astype(dtype))
    sz = jnp.sinc((nz / ng).astype(dtype))
    sx = jnp.where(sx == 0, one, sx)
    sy = jnp.where(sy == 0, one, sy)
    sz = jnp.where(sz == 0, one, sz)
    px = (one / sx) ** window_order
    py = (one / sy) ** window_order
    pz = (one / sz) ** window_order
    return px.astype(dtype), py.astype(dtype), pz.astype(dtype)
