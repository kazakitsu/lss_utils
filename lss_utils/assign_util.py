#!/usr/bin/env python3
from __future__ import annotations
from typing import Tuple

import numpy as np

# -----------------------------------------------------------------------------
# public class
# -----------------------------------------------------------------------------

class Mesh_Assignment:
    """
    Example
    -------
    >>> mesh = Mesh_Assignment(boxsize=1000.0, ng=512, window_order=2,interlace=True)
    >>> field_k = mesh.assign_fft(pos, mass)
    """

    def __init__(self,
                 boxsize: float,
                 ng: int,
                 window_order: int,
                 *,
                 interlace: bool = False,
                 normalize: bool = True,
                 max_scatter_indices: int = 100_000_000):
        self.boxsize = float(boxsize)
        self.ng = int(ng)
        self.window_order = int(window_order)
        self.interlace = bool(interlace)
        self.normalize = bool(normalize)
        self.max_scatter_indices = int(max_scatter_indices)

        self.cell = self.boxsize / self.ng
        self.kvec = _rfftn_kvec((self.ng,) * 3, self.boxsize, dtype=np.float32)
        self.Wk = _deconvolve(self.kvec, self.boxsize, self.window_order)

    # ------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------
    # ---------------- public helpers ----------------
    def assign_to_grid(self, pos, weight, *, interlace: bool = False, normalize_mean: bool = True):
        return _assign_to_grid(self.ng,
                               self.window_order,
                               self.cell,
                               pos,
                               weight,
                               interlace,
                               normalize_mean,
                               self.max_scatter_indices)

    def fft_deconvolve(self, field_r, field_r_i=None):
        field_k = np.fft.rfftn(field_r) / self.ng ** 3
        if field_r_i is not None:
            field_k_i = np.fft.rfftn(field_r_i) / self.ng ** 3
            nvec = self.kvec * (self.boxsize / (2.0 * np.pi))
            phase = np.exp(1j * np.pi * np.sum(nvec, axis=0) / self.ng)
            field_k = 0.5 * (field_k + field_k_i * phase)
        field_k *= self.Wk
        if self.normalize:
            field_k[0, 0, 0] = 0.0
        return field_k

    def assign_fft(self, pos, weight):
        field_r = self.assign_to_grid(pos, weight, interlace=False, normalize_mean=self.normalize)
        if self.interlace:
            field_r_i = self.assign_to_grid(pos, weight, interlace=True, normalize_mean=self.normalize)
            return self.fft_deconvolve(field_r, field_r_i)
        return self.fft_deconvolve(field_r)

# -----------------------------------------------------------------------------
# grid assignment with chunking
# -----------------------------------------------------------------------------

def _assign_to_grid(ng: int,
                    window_order: int,
                    cell: float,
                    pos,
                    weight,
                    interlace: bool,
                    normalize: bool,
                    max_scatter_indices: int):
    pos = np.asarray(pos)
    weight = np.asarray(weight)
    if pos.ndim == 4:
        pos, weight = pos.reshape(3, -1), weight.reshape(-1)

    pos_mesh = pos / cell
    if interlace:
        pos_mesh += 0.5

    num_p = pos.shape[1]
    n_shifts = {1: 1, 2: 8, 3: 27}[window_order]
    chunk_sz = min(max(max_scatter_indices // n_shifts, 1), num_p)

    field = np.zeros((ng, ng, ng), dtype=pos.dtype)
    for start in range(0, num_p, chunk_sz):
        end = min(start + chunk_sz, num_p)
        field = _single_assign(field,
                               pos_mesh[:, start:end],
                               weight[start:end],
                               ng,
                               window_order)

    if normalize:
        field /= (num_p / ng ** 3)
    return field

# -----------------------------------------------------------------------------
# per‑chunk scatter
# -----------------------------------------------------------------------------

def _single_assign(field: np.ndarray,
                   pos_mesh: np.ndarray,
                   weight: np.ndarray,
                   ng: int,
                   window_order: int) -> np.ndarray:
    if pos_mesh.size == 0:
        return field

    # base cell & shift list ---------------------------------
    if window_order == 1:  # NGP
        imesh = np.floor(pos_mesh).astype(int)
        fmesh = np.zeros_like(pos_mesh)
        shifts = [(0, 0, 0)]
    elif window_order == 2:  # CIC
        imesh = np.floor(pos_mesh).astype(int)
        fmesh = pos_mesh - imesh
        shifts = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    else:  # TSC
        imesh = (np.floor(pos_mesh - 1.5).astype(int) + 2)
        fmesh = pos_mesh - imesh
        shifts = [(dx, dy, dz)
                  for dx in (-1, 0, 1)
                  for dy in (-1, 0, 1)
                  for dz in (-1, 0, 1)]

    imesh = np.mod(imesh, ng)  # periodic

    flat = field.ravel()
    stride_y, stride_x = ng, ng * ng

    for dx, dy, dz in shifts:
        idx_x = (imesh[0] + dx) % ng
        idx_y = (imesh[1] + dy) % ng
        idx_z = (imesh[2] + dz) % ng
        flat_idx = (idx_x * stride_x + idx_y * stride_y + idx_z).astype(np.int64)

        # weight per particle --------------------------------
        if window_order == 1:
            w_shift = weight
        elif window_order == 2:
            w_shift = (
                np.where(dx == 0, 1.0 - fmesh[0], fmesh[0]) *
                np.where(dy == 0, 1.0 - fmesh[1], fmesh[1]) *
                np.where(dz == 0, 1.0 - fmesh[2], fmesh[2]) * weight)
        else:
            def w_axis(f, d):
                return np.where(d == 0,
                                0.75 - f ** 2,
                                0.5 * (f ** 2 + d * f + 0.25))
            w_shift = w_axis(fmesh[0], dx) * w_axis(fmesh[1], dy) * w_axis(fmesh[2], dz) * weight

        np.add.at(flat, flat_idx, w_shift.astype(field.dtype))

    return flat.reshape(field.shape)


# -----------------------------------------------------------------------------
# k‑vector & deconvolution window
# -----------------------------------------------------------------------------

def _rfftn_kvec(shape: Tuple[int, int, int], boxsize: float, dtype=float):
    spacing = boxsize / (2.0 * np.pi) / shape[-1]
    freqs = [np.fft.fftfreq(n, d=spacing) for n in shape[:-1]]
    freqs.append(np.fft.rfftfreq(shape[-1], d=spacing))
    kvec_grid = np.meshgrid(*freqs, indexing="ij")
    return np.stack(kvec_grid, axis=0).astype(dtype)


def _deconvolve(kvec, boxsize: float, window_order: int):
    ng = kvec.shape[1]
    nvec = kvec * (boxsize / (2.0 * np.pi))
    sinc_vals = np.sinc(nvec / ng)
    sinc_vals = np.where(sinc_vals == 0.0, 1.0, sinc_vals)
    window_ft = np.prod(sinc_vals, axis=0)
    wk = np.where(window_ft == 0.0, 0.0, 1.0 / window_ft)
    return wk ** window_order
