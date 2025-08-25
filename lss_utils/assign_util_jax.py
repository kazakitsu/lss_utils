#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional

import jax.numpy as jnp
from jax import jit, lax
from functools import partial

import sys

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

        # Cell size
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

        # Deconvolution weights (precast to complex for cheaper muls later)
        wx, wy, wz = _deconv_weights_1d(self.nx, self.ny, self.nz, self.ng, self.window_order, self.real_dtype)
        self.wx = wx.astype(self.complex_dtype)
        self.wy = wy.astype(self.complex_dtype)
        self.wz = wz.astype(self.complex_dtype)

        # Pre-compile both neighbor kernels with static ng/window_order
        self._assign_scan  = jit(partial(_single_assign_scan,  ng=self.ng, window_order=self.window_order))
        self._assign_fused = jit(partial(_single_assign_fused, ng=self.ng, window_order=self.window_order))

    # -------- public API -------- #
    def assign_fft(self, pos, weight=1.0, *, normalize_mean: bool=True, norm: Optional[float] = None, 
                   neighbor_mode: str="auto", fuse_updates_threshold: int=100_000_000):
        pos = jnp.asarray(pos, dtype=self.real_dtype)
        weight = jnp.asarray(weight, dtype=self.real_dtype)
        field_r = self.assign_to_grid(pos, weight,
                                      interlace=False,
                                      normalize_mean=normalize_mean,
                                      norm=norm,
                                      neighbor_mode=neighbor_mode,
                                      fuse_updates_threshold=fuse_updates_threshold)
        if self.interlace:
            field_r_i = self.assign_to_grid(pos, weight,
                                            interlace=True,
                                            normalize_mean=normalize_mean,
                                            norm=norm,
                                            neighbor_mode=neighbor_mode,
                                            fuse_updates_threshold=fuse_updates_threshold)
            return self.fft_deconvolve(field_r, field_r_i)
        return self.fft_deconvolve(field_r)

    @partial(jit, static_argnames=('self',))
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

        field_k = field_k * self.wx[:, None, None]
        field_k = field_k * self.wy[None, :, None]
        field_k = field_k * self.wz[None, None, :]
        if self.normalize:
            field_k = field_k.at[0, 0, 0].set(0.0)
        return field_k
    
    # for PT model
    @partial(jit, static_argnames=('self',))
    def fft_deconvolve_batched(self, fields_r, fields_r_i=None):
        """
        fields_r:   (m, ng, ng, ng) real
        fields_r_i: (m, ng, ng, ng) real or None
        return:     (m, ng, ng, ng//2+1) complex
        """
        field_r = fields_r.astype(self.real_dtype)
        field_k = jnp.fft.rfftn(field_r, axes=(-3, -2, -1), norm='forward').astype(self.complex_dtype)

        if fields_r_i is not None:
            field_r_i = fields_r_i.astype(self.real_dtype)
            field_k_i = jnp.fft.rfftn(field_r_i, axes=(-3, -2, -1), norm='forward').astype(self.complex_dtype)
            # interlacing phase
            field_k_i = field_k_i * self.phase_x_1d[None, :, None, None]
            field_k_i = field_k_i * self.phase_x_1d[None, None, :, None]
            field_k_i = field_k_i * self.phase_z_1d[None, None, None, :]
            field_k = 0.5 * (field_k + field_k_i)

        # deconv
        field_k = field_k * self.wx[None, :, None, None]
        field_k = field_k * self.wy[None, None, :, None]
        field_k = field_k * self.wz[None, None, None, :]

        if self.normalize:
            field_k = field_k.at[:, 0, 0, 0].set(0.0)
        return field_k

    def assign_to_grid(self,
                       pos,
                       weight=1.0,
                       *,
                       interlace: bool=False,
                       normalize_mean: bool=True,
                       norm: Optional[float] = None,
                       neighbor_mode: str="auto",
                       fuse_updates_threshold: int=100_000_000):
        """Host wrapper: decide chunking and kernel, then call jitted inner."""
        pos = jnp.asarray(pos, dtype=self.real_dtype)
        weight = jnp.asarray(weight, dtype=self.real_dtype)

        if pos.ndim == 4:
            pos = pos.reshape(3, -1)
            weight = weight.reshape(-1)

        num_p = int(pos.shape[1])
        shifts = {1:1, 2:8, 3:27}[self.window_order]

        # --- BOTH-CLIP: cap by both max_scatter_indices and fuse_updates_threshold ---
        cap        = max(1, min(int(self.max_scatter_indices), int(fuse_updates_threshold)))
        chunk_size = max(1, min(cap // shifts, num_p))   # ensure shifts*chunk_size <= fuse_updates_threshold
        n_chunks   = (num_p + chunk_size - 1) // chunk_size

        # choose neighbor kernel (auto prefers fused; now it's always safe)
        updates_per_chunk = shifts * chunk_size
        if neighbor_mode == "fused":
            use_fused = True
        elif neighbor_mode == "scan":
            use_fused = False
        else:  # "auto"
            use_fused = (updates_per_chunk <= int(fuse_updates_threshold))

        #print(f"Using {'fused' if use_fused else 'scan'} neighbor updates for {num_p} particles ", file=sys.stderr)
        single_assign_fn = self._assign_fused if use_fused else self._assign_scan

        pad = (-num_p) % chunk_size
        pad_val = jnp.array(0.0, dtype=self.real_dtype)
        pos_pad = jnp.pad(pos, ((0,0),(0,pad)), constant_values=pad_val)
        if weight.ndim == 0:
            wt_pad = weight
        else:
            wt = weight.reshape(-1).astype(self.real_dtype)
            wt_pad = jnp.pad(wt, (0,pad), constant_values=jnp.array(0, dtype=self.real_dtype))

        field0 = jnp.zeros((self.ng, )*3, dtype=self.real_dtype)

        return _assign_to_grid(field0, pos_pad, wt_pad, self.cell, num_p,
                               self.ng, self.window_order,
                               interlace, normalize_mean, norm,
                               int(n_chunks), int(chunk_size),
                               single_assign_fn)

    def assign_from_disp_to_grid(self,
                                 disp_r,
                                 weight,
                                 *,
                                 interlace: bool=False,
                                 normalize_mean: bool=True,
                                 norm: Optional[float] = None,
                                 neighbor_mode: str="auto",
                                 fuse_updates_threshold: int=100_000_000):
        """Host wrapper for slabbed displacement scattering."""
        disp_r = jnp.asarray(disp_r, dtype=self.real_dtype)
        weight = jnp.asarray(weight, dtype=self.real_dtype)

        ng_L = int(disp_r.shape[1])
        shifts = {1:1, 2:8, 3:27}[self.window_order]

        # --- BOTH-CLIP for slab size ---
        # slab <= ng_L, and also bounded by both limits so that
        # updates/slab = shifts * ng_L^2 * slab <= fuse_updates_threshold
        cap_idx   = max(1, int(self.max_scatter_indices))
        cap_fused = max(1, int(fuse_updates_threshold))
        slab_by_idx   = max(1, cap_idx   // (shifts * ng_L * ng_L))
        slab_by_fused = max(1, cap_fused // (shifts * ng_L * ng_L))
        slab = max(1, min(ng_L, slab_by_idx, slab_by_fused))

        # kernel choice (auto prefers fused; now safe by construction)
        updates_per_slab = shifts * ng_L * ng_L * slab

        if neighbor_mode == "fused":
            use_fused = True
        elif neighbor_mode == "scan":
            use_fused = False
        else:
            use_fused = (updates_per_slab <= int(fuse_updates_threshold))

        #print(f"Using {'fused' if use_fused else 'scan'} neighbor updates for {ng_L}^3 particles ", file=sys.stderr)
        single_assign_fn = self._assign_fused if use_fused else self._assign_scan

        field0 = jnp.zeros((self.ng, )*3, dtype=self.real_dtype)
        field_r = _assign_from_disp_to_grid_slab(field0,
                                                 disp_r, weight, self.cell,
                                                 self.ng, self.window_order, int(slab),
                                                 interlace, single_assign_fn)
        if normalize_mean :
            ng = jnp.asarray(self.ng, dtype=self.real_dtype)
            norm_grid = ng**3
            if norm is None:
                norm = jnp.asarray(ng_L, dtype=self.real_dtype) ** 3 ### number of particles
            field_r = field_r * (norm_grid / norm)
        return field_r

    def assign_from_disp_fft(self, disp_r, weight, **kwargs):
        field_r = self.assign_from_disp_to_grid(disp_r, weight, interlace=False, **kwargs)
        if self.interlace:
            field_r_i = self.assign_from_disp_to_grid(disp_r, weight, interlace=True, **kwargs)
            field_k = self.fft_deconvolve(field_r, field_r_i)
        else:
            field_k = self.fft_deconvolve(field_r)
        return field_k

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

# ------------ one-chunk cores (neighbor update) ------------
@partial(jit, static_argnames=('ng', 'window_order'),)
def _single_assign_scan(field, pos_mesh, weight, *, ng: int, window_order: int):
    """Scatter using a scan over neighbor shifts (minimal peak memory)."""
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

@partial(jit, static_argnames=('ng', 'window_order'),)
def _single_assign_fused(field, pos_mesh, weight, *, ng: int, window_order: int):
    """Scatter by fusing all neighbor shifts into one add (faster, more peak memory)."""
    if window_order == 1:  # NGP
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = jnp.zeros_like(pos_mesh)
        shifts = jnp.array([[0,0,0]], jnp.int32)
    elif window_order == 2:  # CIC
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = pos_mesh - imesh
        shifts = jnp.array([[dx,dy,dz] for dx in (0,1) for dy in (0,1) for dz in (0,1)], jnp.int32)
    else:  # TSC
        imesh = (jnp.floor(pos_mesh - 1.5).astype(jnp.int32) + 2)
        fmesh = pos_mesh - imesh
        shifts = jnp.array([[dx,dy,dz] for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)], jnp.int32)

    imesh = jnp.mod(imesh, ng)

    S = shifts.shape[0]
    P = pos_mesh.shape[1]
    sh = shifts.T[:, :, None]                # (3,S,1)
    base = imesh[:, None, :]                 # (3,1,P)
    idx = (base + sh) % ng                   # (3,S,P)
    idx_x, idx_y, idx_z = idx[0], idx[1], idx[2]

    if window_order == 1:
        w_all = weight[None, :]              # (1,P)
    elif window_order == 2:
        w_x = jnp.where(shifts[:,0,None]==0, 1.0 - fmesh[0][None,:], fmesh[0][None,:])
        w_y = jnp.where(shifts[:,1,None]==0, 1.0 - fmesh[1][None,:], fmesh[1][None,:])
        w_z = jnp.where(shifts[:,2,None]==0, 1.0 - fmesh[2][None,:], fmesh[2][None,:])
        w_all = (w_x * w_y * w_z) * weight[None,:]
    else:
        def w1d(fr, d):
            return jnp.where(d==0, 0.75 - fr**2, 0.5 * (fr**2 + d*fr + 0.25))
        wx = w1d(fmesh[0][None,:], shifts[:,0,None])
        wy = w1d(fmesh[1][None,:], shifts[:,1,None])
        wz = w1d(fmesh[2][None,:], shifts[:,2,None])
        w_all = (wx * wy * wz) * weight[None,:]

    field = field.at[idx_x.reshape(-1),
                     idx_y.reshape(-1),
                     idx_z.reshape(-1)].add(w_all.reshape(-1).astype(field.dtype))
    return field

# ---------- jitted inners (chunk/slab loops) ----------
@partial(jit, static_argnames=('ng','window_order','interlace',
                               'normalize_mean', 'norm', 'n_chunks','chunk_size',
                               'single_assign_fn'),
         )
def _assign_to_grid(field, pos_pad, wt_pad, cell, num_p,
                    ng: int, window_order: int,
                    interlace: bool, normalize_mean: bool, norm: Optional[float],
                    n_chunks: int, chunk_size: int,
                    single_assign_fn):
    """Chunk over particles"""
    fdtype = field.dtype
    cell = jnp.asarray(cell, dtype=fdtype)
    num_p = jnp.asarray(num_p, dtype=fdtype)

    def body(i, fld):
        start = i * chunk_size
        pos_ck = lax.dynamic_slice_in_dim(pos_pad, start, chunk_size, axis=-1)
        pos_mesh_ck = pos_ck / cell
        if interlace:
            pos_mesh_ck = pos_mesh_ck + jnp.array(0.5, dtype=fdtype)

        if wt_pad.ndim == 0:
            valid_len = jnp.maximum(0, jnp.minimum(chunk_size, num_p - start))
            valid = (jnp.arange(chunk_size, dtype=fdtype) < valid_len).astype(fdtype)
            wt_ck = (wt_pad.astype(fdtype) * valid)
        else:
            wt_ck = lax.dynamic_slice_in_dim(wt_pad, start, chunk_size, axis=0)

        return single_assign_fn(fld, pos_mesh_ck, wt_ck)

    field = lax.fori_loop(0, n_chunks, body, field)

    if normalize_mean:
        ngf = jnp.asarray(ng, dtype=fdtype)
        norm_grid = ngf**3
        if norm is None:
            norm = num_p
        field = field * (norm_grid / norm)
    return field

@partial(jit, static_argnames=('ng','window_order','slab','interlace','single_assign_fn'),
         )
def _assign_from_disp_to_grid_slab(field,
                                   disp_r, weight, cell,
                                   ng: int, window_order: int, slab: int,
                                   interlace: bool, single_assign_fn):
    """Slab along z in Lagrangian index; each slab is scattered to Eulerian grid."""
    fdtype = disp_r.dtype
    ng_L = disp_r.shape[1]
    scale = jnp.asarray(ng, dtype=fdtype) / jnp.asarray(ng_L, dtype=fdtype)
    cell_arr = jnp.asarray(cell, dtype=fdtype)

    pad_z = (-ng_L) % slab
    zero = jnp.array(0, dtype=fdtype)
    disp_pad = jnp.pad(disp_r, ((0,0),(0,0),(0,0),(0,pad_z)), constant_values=zero)

    weight = jnp.asarray(weight, dtype=fdtype)
    weight_is_scalar = (weight.ndim == 0)
    if not weight_is_scalar:
        w_pad = jnp.pad(weight, ((0,0),(0,0),(0,pad_z)), constant_values=zero)

    n_slices = (ng_L + pad_z) // slab
    ix_L = jnp.arange(ng_L, dtype=fdtype)[:, None, None]
    iy_L = jnp.arange(ng_L, dtype=fdtype)[None, :, None]

    def body(i, fld):
        z0 = i * slab
        disp_ck = lax.dynamic_slice(disp_pad, (0, 0, 0, z0), (3, ng_L, ng_L, slab))
        iz_L = (z0 + jnp.arange(slab, dtype=fdtype))[None, None, :]

        pos_x = ix_L * scale + disp_ck[0] / cell_arr
        pos_y = iy_L * scale + disp_ck[1] / cell_arr
        pos_z = iz_L * scale + disp_ck[2] / cell_arr
        pos_mesh_ck = jnp.stack([pos_x, pos_y, pos_z], axis=0).reshape(3, -1)
        if interlace:
            pos_mesh_ck = pos_mesh_ck + jnp.array(0.5, dtype=fdtype)

        if weight_is_scalar:
            Lz = jnp.minimum(slab, ng_L - z0)
            valid_z = (jnp.arange(slab) < Lz)[None, None, :]
            valid = jnp.broadcast_to(valid_z, (ng_L, ng_L, slab)).astype(fdtype)
            wt_ck_flat = (valid * weight.astype(fdtype)).reshape(-1)
        else:
            w_ck = lax.dynamic_slice(w_pad, (0, 0, z0), (ng_L, ng_L, slab))
            wt_ck_flat = w_ck.reshape(-1)

        return single_assign_fn(fld, pos_mesh_ck, wt_ck_flat)

    field = lax.fori_loop(0, n_slices, body, field)
    return field