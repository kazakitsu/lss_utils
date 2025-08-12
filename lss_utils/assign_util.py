#!/usr/bin/env python3
from __future__ import annotations
import sys
import numpy as np

# ---------- Public API class (NumPy) ----------
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
        max_scatter_indices: int = 800_000_000,
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

        # Cell size
        self.cell = np.asarray(self.boxsize / self.ng, dtype=self.real_dtype)

        # 1D frequency indices
        self.nx = _fftfreq_index_1d(self.ng).astype(self.real_dtype, copy=False)
        self.ny = self.nx
        self.nz = _rfftfreq_index_1d(self.ng).astype(self.real_dtype, copy=False)

        # Interlacing phase (complex)
        pi = np.array(np.pi, dtype=self.real_dtype)
        inv_ng = np.asarray(self.ng, dtype=self.real_dtype)
        self.phase_x_1d = np.exp(1j * (pi * self.nx / inv_ng)).astype(self.complex_dtype, copy=False)
        self.phase_z_1d = np.exp(1j * (pi * self.nz / inv_ng)).astype(self.complex_dtype, copy=False)

        # Deconvolution weights (precast to complex for cheaper muls later)
        wx, wy, wz = _deconv_weights_1d(self.nx, self.ny, self.nz, self.ng, self.window_order, self.real_dtype)
        self.wx = wx.astype(self.complex_dtype, copy=False)
        self.wy = wy.astype(self.complex_dtype, copy=False)
        self.wz = wz.astype(self.complex_dtype, copy=False)

        # Pre-bind both neighbor kernels
        self._assign_scan = _single_assign_scan
        self._assign_fused = _single_assign_fused

    # -------- public API -------- #
    def assign_fft(self, pos, weight=1.0, *, neighbor_mode: str = "auto", fuse_updates_threshold: int = 500_000_000):
        pos = np.asarray(pos, dtype=self.real_dtype)
        weight = np.asarray(weight, dtype=self.real_dtype)
        field_r = self.assign_to_grid(pos, weight,
                                      interlace=False,
                                      neighbor_mode=neighbor_mode,
                                      fuse_updates_threshold=fuse_updates_threshold)
        if self.interlace:
            field_r_i = self.assign_to_grid(pos, weight,
                                            interlace=True,
                                            neighbor_mode=neighbor_mode,
                                            fuse_updates_threshold=fuse_updates_threshold)
            return self.fft_deconvolve(field_r, field_r_i)
        return self.fft_deconvolve(field_r)

    def fft_deconvolve(self, field_r, field_r_i=None):
        """FFT + (optional) interlaced average + window deconvolution."""
        field_r = np.asarray(field_r, dtype=self.real_dtype)
        field_k = np.fft.rfftn(field_r, norm='forward').astype(self.complex_dtype, copy=False)

        if field_r_i is not None:
            field_r_i = np.asarray(field_r_i, dtype=self.real_dtype)
            field_k_i = np.fft.rfftn(field_r_i, norm='forward').astype(self.complex_dtype, copy=False)
            field_k_i = field_k_i * self.phase_x_1d[:, None, None]
            field_k_i = field_k_i * self.phase_x_1d[None, :, None]
            field_k_i = field_k_i * self.phase_z_1d[None, None, :]
            field_k = 0.5 * (field_k + field_k_i)

        field_k = field_k * self.wx[:, None, None]
        field_k = field_k * self.wy[None, :, None]
        field_k = field_k * self.wz[None, None, :]
        if self.normalize:
            field_k[(0, 0, 0)] = 0.0
        return field_k

    def assign_to_grid(self,
                       pos,
                       weight=1.0,
                       *,
                       interlace: bool = False,
                       normalize_mean: bool = True,
                       neighbor_mode: str = "auto",
                       fuse_updates_threshold: int = 500_000_000):
        """Host wrapper: decide chunking and kernel, then run NumPy inner."""
        pos = np.asarray(pos, dtype=self.real_dtype)
        weight = np.asarray(weight, dtype=self.real_dtype)

        # accept (3,ngL,ngL,ngL) too (LPT displacement style); flatten outside to keep memory low
        if pos.ndim == 4:
            pos = pos.reshape(3, -1)
            weight = weight.reshape(-1)

        num_p = int(pos.shape[1])
        shifts = {1: 1, 2: 8, 3: 27}[self.window_order]

        # particle chunk size from global cap
        cap        = max(1, min(int(self.max_scatter_indices), int(fuse_updates_threshold)))
        chunk_size = max(1, min(cap // shifts, num_p))
        n_chunks   = (num_p + chunk_size - 1) // chunk_size

        updates_per_chunk = shifts * chunk_size
        if neighbor_mode == "fused":
            use_fused = True
        elif neighbor_mode == "scan":
            use_fused = False
        else:
            use_fused = (updates_per_chunk <= int(fuse_updates_threshold))

        print(f"Using {'fused' if use_fused else 'scan'} neighbor updates for {num_p} particles", file=sys.stderr)
        single_assign_fn = self._assign_fused if use_fused else self._assign_scan

        # no need for strict padding in NumPy; slice last chunk length directly
        pos_pad = pos
        wt_pad = weight

        field0 = np.zeros((self.ng, ) * 3, dtype=self.real_dtype)

        return _assign_to_grid(field0, pos_pad, wt_pad, self.cell, num_p,
                               self.ng, self.window_order,
                               interlace, normalize_mean,
                               int(n_chunks), int(chunk_size),
                               single_assign_fn)

    def assign_from_disp_to_grid(self,
                                 disp_r,
                                 weight,
                                 *,
                                 interlace: bool = False,
                                 normalize_mean: bool = True,
                                 neighbor_mode: str = "auto",
                                 fuse_updates_threshold: int = 500_000_000):
        """Scatter directly from displacement field and optional weight."""
        disp_r = np.asarray(disp_r, dtype=self.real_dtype)
        weight = np.asarray(weight, dtype=self.real_dtype)

        ng_L = int(disp_r.shape[1])
        shifts = {1: 1, 2: 8, 3: 27}[self.window_order]

        # slab chosen from global cap (updates per slab ≈ shifts * ng_L^2 * slab)
        cap_idx      = max(1, int(self.max_scatter_indices))
        cap_fused    = max(1, int(fuse_updates_threshold))
        slab_by_idx   = max(1, cap_idx   // (shifts * ng_L * ng_L))
        slab_by_fused = max(1, cap_fused // (shifts * ng_L * ng_L))
        slab          = max(1, min(ng_L, slab_by_idx, slab_by_fused))

        updates_per_slab = shifts * ng_L * ng_L * slab
        if neighbor_mode == "fused":
            use_fused = True
        elif neighbor_mode == "scan":
            use_fused = False
        else:
            use_fused = (updates_per_slab <= int(fuse_updates_threshold))

        print(f"Using {'fused' if use_fused else 'scan'} neighbor updates for {ng_L}^3 particles", file=sys.stderr)
        single_assign_fn = self._assign_fused if use_fused else self._assign_scan

        field0 = np.zeros((self.ng, ) * 3, dtype=self.real_dtype)
        field_r = _assign_from_disp_to_grid_slab(field0,
                                                 disp_r, weight, self.cell,
                                                 self.ng, self.window_order, int(slab),
                                                 interlace, single_assign_fn)
        if normalize_mean:
            norm = (np.asarray(self.ng, dtype=self.real_dtype) ** 3) / (np.asarray(ng_L, dtype=self.real_dtype) ** 3)
            field_r = field_r * norm
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
    # Return frequency index in "index units" compatible with rfftn on size n
    return np.fft.fftfreq(n, d=1.0 / n)

def _rfftfreq_index_1d(n: int):
    # Return rfftn z-axis frequency index for size n
    return np.fft.rfftfreq(n, d=1.0 / n)

def _deconv_weights_1d(nx, ny, nz, ng: int, window_order: int, dtype):
    """Return 1D deconvolution prefactors along each axis (float arrays)."""
    dtype = np.dtype(dtype)
    one = np.array(1.0, dtype=dtype)
    sx = np.sinc((nx / ng).astype(dtype))
    sy = np.sinc((ny / ng).astype(dtype))
    sz = np.sinc((nz / ng).astype(dtype))
    # Avoid division by zero at exact zeros
    sx = np.where(sx == 0, one, sx)
    sy = np.where(sy == 0, one, sy)
    sz = np.where(sz == 0, one, sz)
    px = (one / sx) ** window_order
    py = (one / sy) ** window_order
    pz = (one / sz) ** window_order
    return px.astype(dtype, copy=False), py.astype(dtype, copy=False), pz.astype(dtype, copy=False)


# ---------- low-level neighbor kernels (NumPy) ----------
def _single_assign_scan(field: np.ndarray, pos_mesh: np.ndarray, weight: np.ndarray, *, ng: int, window_order: int):
    """Scatter using a small Python loop over neighbor shifts (minimal peak memory).
       field: (ng, ng, ng) real; pos_mesh: (3, P); weight: (P,) or scalar."""
    # integer base cell and fractional offsets
    if window_order == 1:  # NGP
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = np.zeros_like(pos_mesh, dtype=field.dtype)
        shifts = np.array([[0, 0, 0]], dtype=np.int32)
    elif window_order == 2:  # CIC
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = pos_mesh - imesh
        shifts = np.array([[dx, dy, dz]
                           for dx in (0, 1)
                           for dy in (0, 1)
                           for dz in (0, 1)], dtype=np.int32)
    else:  # TSC
        imesh = (np.floor(pos_mesh - 1.5).astype(np.int32) + 2)
        fmesh = pos_mesh - imesh
        shifts = np.array([[dx, dy, dz]
                           for dx in (-1, 0, 1)
                           for dy in (-1, 0, 1)
                           for dz in (-1, 0, 1)], dtype=np.int32)

    # periodic wrap
    imesh %= ng

    P = pos_mesh.shape[1]
    # flatten field view for faster add.at with linear indices
    flat = field.reshape(-1)

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
            # TSC weights
            def w1d(fr, d):
                return np.where(d == 0, 0.75 - fr**2, 0.5 * (fr**2 + d * fr + 0.25))
            w_x = w1d(fmesh[0], dx)
            w_y = w1d(fmesh[1], dy)
            w_z = w1d(fmesh[2], dz)
            w_shift = (w_x * w_y * w_z) * weight

        # linear index: ix*ng*ng + iy*ng + iz
        lin = (idx_x.astype(np.int64) * ng + idx_y.astype(np.int64)) * ng + idx_z.astype(np.int64)
        np.add.at(flat, lin, w_shift.astype(field.dtype, copy=False))

    return field


def _single_assign_fused(field: np.ndarray, pos_mesh: np.ndarray, weight: np.ndarray, *, ng: int, window_order: int):
    """Scatter by fusing all neighbor shifts into one add (faster but more peak memory)."""
    if window_order == 1:  # NGP
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = np.zeros_like(pos_mesh, dtype=field.dtype)
        shifts = np.array([[0, 0, 0]], dtype=np.int32)
    elif window_order == 2:  # CIC
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = pos_mesh - imesh
        shifts = np.array([[dx, dy, dz]
                           for dx in (0, 1)
                           for dy in (0, 1)
                           for dz in (0, 1)], dtype=np.int32)
    else:  # TSC
        imesh = (np.floor(pos_mesh - 1.5).astype(np.int32) + 2)
        fmesh = pos_mesh - imesh
        shifts = np.array([[dx, dy, dz]
                           for dx in (-1, 0, 1)
                           for dy in (-1, 0, 1)
                           for dz in (-1, 0, 1)], dtype=np.int32)

    imesh %= ng

    S = shifts.shape[0]
    P = pos_mesh.shape[1]

    # indices for all shifts: (3,S,P)
    sh = shifts.T[:, :, None]           # (3,S,1)
    base = imesh[:, None, :]            # (3,1,P)
    idx = (base + sh) % ng              # (3,S,P)
    idx_x, idx_y, idx_z = idx[0], idx[1], idx[2]  # (S,P)

    # weights for all shifts: (S,P)
    if window_order == 1:
        w_all = weight[None, :]
    elif window_order == 2:
        w_x = np.where(shifts[:, 0, None] == 0, 1.0 - fmesh[0][None, :], fmesh[0][None, :])
        w_y = np.where(shifts[:, 1, None] == 0, 1.0 - fmesh[1][None, :], fmesh[1][None, :])
        w_z = np.where(shifts[:, 2, None] == 0, 1.0 - fmesh[2][None, :], fmesh[2][None, :])
        w_all = (w_x * w_y * w_z) * weight[None, :]
    else:
        def w1d(fr, d):
            return np.where(d == 0, 0.75 - fr**2, 0.5 * (fr**2 + d * fr + 0.25))
        wx = w1d(fmesh[0][None, :], shifts[:, 0, None])
        wy = w1d(fmesh[1][None, :], shifts[:, 1, None])
        wz = w1d(fmesh[2][None, :], shifts[:, 2, None])
        w_all = (wx * wy * wz) * weight[None, :]

    # one fused add.at on flattened view
    flat = field.reshape(-1)
    lin = (idx_x.astype(np.int64) * ng + idx_y.astype(np.int64)) * ng + idx_z.astype(np.int64)  # (S,P)
    np.add.at(flat, lin.reshape(-1), w_all.reshape(-1).astype(field.dtype, copy=False))
    return field


# ---------- jitted-analog "inners" (NumPy loops) ----------
def _assign_to_grid(field: np.ndarray, pos_pad: np.ndarray, wt_pad, cell, num_p: int,
                    ng: int, window_order: int,
                    interlace: bool, normalize: bool,
                    n_chunks: int, chunk_size: int,
                    single_assign_fn):
    """Chunk over particles and scatter to the grid."""
    fdtype = field.dtype
    cell = np.asarray(cell, dtype=fdtype)

    for i in range(n_chunks):
        start = i * chunk_size
        stop = min(start + chunk_size, pos_pad.shape[1])
        # slice current chunk
        pos_ck = pos_pad[:, start:stop]     # (3,L)
        if pos_ck.size == 0:
            continue
        pos_mesh_ck = pos_ck / cell
        if interlace:
            pos_mesh_ck = pos_mesh_ck + np.array(0.5, dtype=fdtype)

        # weight chunk
        if np.ndim(wt_pad) == 0:
            L = pos_ck.shape[1]
            wt_ck = np.full((L,), wt_pad, dtype=fdtype)
        else:
            wt_ck = wt_pad[start:stop].astype(fdtype, copy=False)

        single_assign_fn(field, pos_mesh_ck, wt_ck, ng=ng, window_order=window_order)

    if normalize:
        norm = (np.array(ng, dtype=fdtype) ** 3) / np.array(num_p, dtype=fdtype)
        field *= norm
    return field


def _assign_from_disp_to_grid_slab(field: np.ndarray,
                                   disp_r: np.ndarray, weight, cell,
                                   ng: int, window_order: int, slab: int,
                                   interlace: bool, single_assign_fn):
    """Slab along z in Lagrangian index; each slab is scattered to Eulerian grid."""
    fdtype = field.dtype
    ng_L = int(disp_r.shape[1])
    scale = np.asarray(ng, dtype=fdtype) / np.asarray(ng_L, dtype=fdtype)
    cell_arr = np.asarray(cell, dtype=fdtype)

    pad_z = (-ng_L) % slab
    if pad_z:
        disp_pad = np.pad(disp_r, ((0, 0), (0, 0), (0, 0), (0, pad_z)), mode="constant")
    else:
        disp_pad = disp_r

    weight_is_scalar = (np.ndim(weight) == 0)
    if not weight_is_scalar:
        w_pad = np.pad(weight, ((0, 0), (0, 0), (0, pad_z)), mode="constant")

    n_slices = (ng_L + pad_z) // slab
    ix_L = np.arange(ng_L, dtype=fdtype)[:, None, None]
    iy_L = np.arange(ng_L, dtype=fdtype)[None, :, None]

    for i in range(n_slices):
        z0 = i * slab
        disp_ck = disp_pad[:, :, :, z0:z0 + slab]       # (3, ng_L, ng_L, slab)
        Lz = min(slab, ng_L - z0)

        # positions in Eulerian cell units
        pos_x = ix_L * scale + disp_ck[0] / cell_arr
        pos_y = iy_L * scale + disp_ck[1] / cell_arr
        pos_z = (z0 + np.arange(slab, dtype=fdtype))[None, None, :] * scale + disp_ck[2] / cell_arr

        if Lz != slab:
            pos_x = pos_x[:, :, :Lz]
            pos_y = pos_y[:, :, :Lz]
            pos_z = pos_z[:, :, :Lz]

        pos_mesh_ck = np.stack([pos_x, pos_y, pos_z], axis=0).reshape(3, -1)  # (3, ng_L*ng_L*Lz)
        if interlace:
            pos_mesh_ck = pos_mesh_ck + np.array(0.5, dtype=fdtype)

        if weight_is_scalar:
            wt_ck_flat = np.full((pos_mesh_ck.shape[1],), weight, dtype=fdtype)
        else:
            w_ck = w_pad[:, :, z0:z0 + Lz]
            wt_ck_flat = w_ck.reshape(-1).astype(fdtype, copy=False)

        single_assign_fn(field, pos_mesh_ck, wt_ck_flat, ng=ng, window_order=window_order)

    return field
