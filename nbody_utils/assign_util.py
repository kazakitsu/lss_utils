#!/usr/bin/env python3

import numpy as np


def assign(boxsize,
           field,
           weight,
           pos,
           num_particles,
           window_order,
           interlace=0,
           max_scatter_indices=100_000_000):
    """
    Deposit particles onto a 3D grid (ng x ng x ng) using NGP/CIC/TSC,
    with vectorized chunked scatter-add for memory control.

    Parameters
    ----------
    boxsize : float
        Physical size of the box.
    field : ndarray, shape (ng,ng,ng)
        Density grid to be updated in place.
    weight : float or ndarray, shape (num_particles,)
        Particle weights.
    pos : ndarray, shape (3,num_particles) or (3,Nx,Ny,Nz)
        Particle positions in the box.
    num_particles : int
        Number of particles.
    window_order : int
        1 = NGP, 2 = CIC, 3 = TSC.
    interlace : {0,1}
        If 1, shift mesh by half a cell before assignment.
    max_scatter_indices : int
        Maximum total scatter operations per chunk.

    Returns
    -------
    field : ndarray
        Updated density grid (normalized per file).
    """
    ng = field.shape[0]
    cell_size = boxsize / ng

    # flatten pos & weight if in grid form
    if pos.ndim == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    # convert to cell units
    pos_mesh = pos / cell_size
    if interlace:
        pos_mesh += 0.5

    # choose shifts and base indices
    if window_order == 1:
        imesh = np.floor(pos_mesh).astype(int)
        fmesh = np.zeros_like(pos_mesh)
        shifts = np.array([[0, 0, 0]], dtype=int)
    elif window_order == 2:
        imesh = np.floor(pos_mesh).astype(int)
        fmesh = pos_mesh - imesh
        grid = np.meshgrid([0,1],[0,1],[0,1], indexing='ij')
        shifts = np.stack(grid, axis=-1).reshape(-1,3).astype(int)
    elif window_order == 3:
        imesh = np.floor(pos_mesh - 1.5).astype(int) + 2
        fmesh = pos_mesh - imesh
        grid = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1], indexing='ij')
        shifts = np.stack(grid, axis=-1).reshape(-1,3).astype(int)
    else:
        raise ValueError(f"Unsupported window_order={window_order}")

    # periodic boundary for base index
    imesh = np.mod(imesh, ng)

    # total scatter count
    S = shifts.shape[0]
    N = num_particles
    total = S * N

    # helper: build indices and weights for a given chunk of particles
    def build_chunk(imesh_chunk, fmesh_chunk, weight_chunk):
        # vectorize base+shifts -> all indices (3, S*M)
        imesh_exp = imesh_chunk[None, ...]      # (1,3,M)
        shifts_exp = shifts[:, :, None]         # (S,3,1)
        all_idx = (imesh_exp + shifts_exp) % ng  # (S,3,M)
        all_idx = all_idx.reshape(3, -1)         # (3, S*M)

        # compute weights matrix (S,M)
        def compute_weights_vec(fmesh_c, shift):
            if window_order == 1:
                return np.ones(fmesh_c.shape[1])
            elif window_order == 2:
                w = np.where(shift[:,None]==0, 1 - fmesh_c, fmesh_c)
                return np.prod(w, axis=0)
            else:
                w = np.where(shifts[:,None]==0,
                             0.75 - fmesh_c**2,
                             0.5*(fmesh_c**2 + shifts[:,None]*fmesh_c + 0.25))
                return np.prod(w, axis=0)

        # weights per shift
        w_mat = np.vstack([compute_weights_vec(fmesh_chunk, sh) for sh in shifts])  # (S,M)
        all_w = (w_mat * weight_chunk[None,:]).reshape(-1)                           # (S*M,)

        return all_idx, all_w

    # choose chunking
    if total <= max_scatter_indices:
        # single chunk = all particles
        idxs, wts = build_chunk(imesh, fmesh, weight)
        fx, fy, fz = idxs
        np.add.at(field, (fx, fy, fz), wts)
    else:
        # chunk size in particles
        chunk_particles = max(max_scatter_indices // S, 1)
        n_chunks = (N + chunk_particles - 1) // chunk_particles
        for i in range(n_chunks):
            start = i * chunk_particles
            end = min((i+1)*chunk_particles, N)
            imesh_ck = imesh[:, start:end]
            fmesh_ck = fmesh[:, start:end]
            weight_ck = weight[start:end]
            idxs, wts = build_chunk(imesh_ck, fmesh_ck, weight_ck)
            fx, fy, fz = idxs
            np.add.at(field, (fx, fy, fz), wts)

    # normalization
    field /= (num_particles / (ng**3))
    
    return field
