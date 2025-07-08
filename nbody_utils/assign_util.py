#!/usr/bin/env python3

import numpy as np

def assign(boxsize,
           field,
           weight,
           pos,
           num_particles,
           window_order,
           interlace=0,
           contd=0,
           max_scatter_indices=100_000_000):
    """
    Deposit particles onto a 3D grid (ng x ng x ng) using NGP/CIC/TSC.

    Parameters
    ----------
    boxsize : float
        Physical size of the box.
    field : ndarray, shape (ng,ng,ng)
        Density grid to be updated in place.
    weight : ndarray, shape (num_particles,) or (Nx,Ny,Nz)
        Particle weights.
    pos : ndarray, shape (3,num_particles) or (3,Nx,Ny,Nz)
        Particle positions in the box.
    num_particles : int
        Number of particles.
    window_order : int
        1 = NGP, 2 = CIC, 3 = TSC.
    interlace : {0,1}
        If 1, shift mesh by half a cell before assignment.
    contd : {0,1}
        If 0, normalize field by num_particles/(ng**3) at the end.
    max_scatter_indices : int
        Threshold for chunked processing.

    Returns
    -------
    field : ndarray
        Updated and optionally normalized density grid.
    """
    ng = field.shape[0]
    cell_size = boxsize / ng

    # flatten pos & weight if given in (3,Nx,Ny,Nz) form
    if pos.ndim == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    # convert to cell units
    pos_mesh = pos / cell_size
    if interlace:
        pos_mesh = pos_mesh + 0.5

    # determine how many shifts per particle
    if window_order == 1:
        n_shifts = 1
    elif window_order == 2:
        n_shifts = 8
    elif window_order == 3:
        n_shifts = 27
    else:
        raise ValueError(f"Unsupported window_order={window_order}")

    total_scatter = num_particles * n_shifts

    # helper to call the inner assign function
    def single_assign(fld, pm, w):
        return assign_(fld, pm, w, ng, window_order)

    # if too many scatter indices, process in chunks
    if total_scatter > max_scatter_indices:
        chunk_size = max(max_scatter_indices // n_shifts, 1)
        num_iters  = (num_particles + chunk_size - 1) // chunk_size
        pad_len    = num_iters * chunk_size - num_particles

        # pad to full chunks
        pos_padded    = np.pad(pos_mesh,    ((0, 0), (0, pad_len)), constant_values=0.0)
        weight_padded = np.pad(weight,       (0, pad_len),         constant_values=0.0)

        fld = field
        for i in range(num_iters):
            start = i * chunk_size
            pm_ck = pos_padded[:, start:start+chunk_size]
            w_ck  = weight_padded[start:start+chunk_size]
            fld   = single_assign(fld, pm_ck, w_ck)
        field = fld
    else:
        field = single_assign(field, pos_mesh, weight)

    # optional normalization
    if not contd:
        field = field / (num_particles / (ng**3))

    return field


def assign_(field, pos_mesh, weight, ng, window_order):
    """
    Core scatter-add routine for one chunk of particles.
    """
    # choose base cell and fractional offset
    if window_order == 1:  # NGP
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = np.zeros_like(pos_mesh)
        shifts = np.array([[0, 0, 0]], dtype=np.int32)
    elif window_order == 2:  # CIC
        imesh = np.floor(pos_mesh).astype(np.int32)
        fmesh = pos_mesh - imesh
        # all 8 corners of the cube
        grid  = np.meshgrid([0,1], [0,1], [0,1], indexing='ij')
        shifts = np.stack(grid, axis=-1).reshape(-1, 3).astype(np.int32)
    elif window_order == 3:  # TSC
        # shift the floor by 1.5 then add 2 to center
        imesh = np.floor(pos_mesh - 1.5).astype(np.int32) + 2
        fmesh = pos_mesh - imesh
        # all 27 vertices in the 3×3×3 cube
        grid = np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1], indexing='ij')
        shifts = np.stack(grid, axis=-1).reshape(-1, 3).astype(np.int32)
    else:
        raise ValueError(f"Unsupported window_order={window_order}")

    # apply periodic boundary conditions
    imesh = np.mod(imesh, ng)

    def compute_weights(f, shift):
        """Compute per-shift weight for one particle."""
        if window_order == 1:
            return 1.0
        elif window_order == 2:
            w = np.where(shift==0, 1.0 - f, f)
        else:  # TSC
            # for shift==0, use triangular kernel; else parabolic wings
            w = np.where(shift == 0,
                         0.75 - f**2,
                         0.5 * (f**2 + shift * f + 0.25))
        return np.prod(w, axis=0)

    # gather all scatter indices and weights
    idx_list = []
    wgt_list = []
    N = weight.shape[0]
    for i in range(N):
        base = imesh[:, i]
        f    = fmesh[:, i]
        w0   = weight[i]
        for shift in shifts:
            idx = base + shift
            idx = np.mod(idx, ng)
            w_sh = compute_weights(f, shift) * w0
            idx_list.append(tuple(idx))
            wgt_list.append(w_sh)

    # convert to arrays
    indices = np.array(idx_list, dtype=np.int32)  # shape (n_shifts*N, 3)
    weights = np.array(wgt_list, dtype=field.dtype)

    # scatter‐add into the field
    fx, fy, fz = indices[:,0], indices[:,1], indices[:,2]
    np.add.at(field, (fx, fy, fz), weights)

    return field
