#!/usr/bin/env python3
import numpy as np

# ---------- single-chunk core ----------
def _single_assign(field, pos_mesh, weight, ng, window_order):
    """Core scatter-add for one chunk using np.bincount (matches JAX logic)."""
    # base cell, fractional offset and shifts
    if window_order == 1:          # NGP
        imesh  = np.floor(pos_mesh).astype(np.int32)
        fmesh  = np.zeros_like(pos_mesh)
        shifts = np.array([[0, 0, 0]], np.int32)
    elif window_order == 2:        # CIC
        imesh  = np.floor(pos_mesh).astype(np.int32)
        fmesh  = pos_mesh - imesh
        shifts = np.stack(
            np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"), -1
        ).reshape(-1, 3).astype(np.int32)
    else:                          # TSC
        imesh  = (np.floor(pos_mesh - 1.5).astype(np.int32) + 2)
        fmesh  = pos_mesh - imesh
        shifts = np.stack(
            np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1
        ).reshape(-1, 3).astype(np.int32)

    imesh %= ng                               # periodic BC
    S, N = shifts.shape[0], pos_mesh.shape[1]

    # flat indices (3,S*N)
    idx = (shifts[:, :, None] + imesh[None, :, :]) % ng       # (S,3,N)
    idx = idx.transpose(1, 0, 2).reshape(3, -1)               # (3,S*N)
    flat_idx = (idx[0] * ng + idx[1]) * ng + idx[2]           # (S*N,)

    # weights (S*N,)
    def w_vec(sh):
        if window_order == 1:
            return np.ones(N, dtype=pos_mesh.dtype)
        elif window_order == 2:
            return np.prod(
                np.where(sh[:, None] == 0, 1.0 - fmesh, fmesh), axis=0
            )
        else:  # TSC
            return np.prod(
                np.where(
                    sh[:, None] == 0,
                    0.75 - fmesh ** 2,
                    0.5 * (fmesh ** 2 + sh[:, None] * fmesh + 0.25),
                ),
                axis=0,
            )

    W = np.vstack([w_vec(sh) for sh in shifts])              # (S,N)
    flat_w = (W * weight[None, :]).reshape(-1).astype(field.dtype)

    # single bincount reduce
    flat_field = field.ravel()
    flat_field += np.bincount(
        flat_idx, weights=flat_w, minlength=flat_field.size
    ).astype(field.dtype)
    return flat_field.reshape(field.shape)


# ---------- public assign (chunk + bincount) ----------
def assign(
    boxsize,
    field,
    weight,
    pos,
    num_particles,
    window_order,
    interlace=0,
    contd=0,
    max_scatter_indices=100_000_000,
):
    """
    NumPy assign with the same chunk-and-bincount strategy as the JAX version.
    Produces bit-wise identical results (up to rounding) given the same dtype.
    """
    ng = field.shape[0]
    cell = boxsize / ng

    # flatten if needed
    if pos.ndim == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    pos_mesh = pos / cell
    if interlace:
        pos_mesh += 0.5

    n_shifts = {1: 1, 2: 8, 3: 27}[window_order]
    total_scatter = num_particles * n_shifts

    # choose static chunk size identical to the JAX rule
    chunk_size = max_scatter_indices // n_shifts
    chunk_size = int(min(chunk_size, num_particles))
    n_chunks = (num_particles + chunk_size - 1) // chunk_size

    # work on a flat view to avoid extra ravel/reshape in the loop
    #flat = field.ravel()
    #for i in range(n_chunks):
    #    start = i * chunk_size
    #    end = start + chunk_size
    #    flat_chunk = _single_assign(
    #        flat.reshape(field.shape),
    #        pos_mesh[:, start:end],
    #        weight[start:end],
    #        ng,
    #        window_order,
    #    ).ravel()
    #    flat[:] = flat_chunk  # inplace update (no extra allocation)

    for i in range(n_chunks):
        start, end = i*chunk_size, (i+1)*chunk_size
        field = _single_assign(
            field,
            pos_mesh[:, start:end],
            weight[start:end],
            ng,
            window_order,
        )

    # per-file normalisation
    if contd == 0:
        flat /= num_particles / ng ** 3

    return field
