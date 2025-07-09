#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

# ------------ one-chunk core ------------
@partial(jit, static_argnames=("ng", "window_order"))
def _single_assign(field, pos_mesh, weight, ng, window_order):
    # --- choose base cell, fractional offset, and shift list ---
    if window_order == 1:            # NGP
        imesh  = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh  = jnp.zeros_like(pos_mesh)
        shifts = jnp.array([[0, 0, 0]], jnp.int32)
    elif window_order == 2:          # CIC
        imesh  = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh  = pos_mesh - imesh
        shifts = jnp.stack(jnp.meshgrid(jnp.arange(2),
                                        jnp.arange(2),
                                        jnp.arange(2),
                                        indexing="ij"), -1
                          ).reshape(-1, 3).astype(jnp.int32)
    else:                            # TSC
        imesh  = (jnp.floor(pos_mesh - 1.5).astype(jnp.int32) + 2)
        fmesh  = pos_mesh - imesh
        shifts = jnp.stack(jnp.meshgrid(jnp.arange(-1, 2),
                                        jnp.arange(-1, 2),
                                        jnp.arange(-1, 2),
                                        indexing="ij"), -1
                          ).reshape(-1, 3).astype(jnp.int32)

    # apply periodic boundary conditions to the base cell
    imesh = jnp.mod(imesh, ng)
    S, N  = shifts.shape[0], pos_mesh.shape[1]

    # --- build flat indices (S*N,) ---
    idx = (shifts[:, :, None] + imesh[None, :, :]) % ng      # (S,3,N)
    idx = idx.transpose(1, 0, 2).reshape(3, -1)              # (3,S*N)
    flat_idx = (idx[0] * ng + idx[1]) * ng + idx[2]          # (S*N,)

    # --- build per-shift weights (S*N,) ---
    def weight_for_shift(sh):
        if window_order == 1:
            w = jnp.ones(N, dtype=pos_mesh.dtype)
        elif window_order == 2:
            w = jnp.prod(jnp.where(sh[:, None] == 0, 1.0 - fmesh, fmesh), axis=0)
        else:  # TSC
            w = jnp.prod(
                jnp.where(
                    sh[:, None] == 0,
                    0.75 - fmesh ** 2,
                    0.5 * (fmesh ** 2 + sh[:, None] * fmesh + 0.25)
                ),
                axis=0,
            )
        return w

    W = vmap(weight_for_shift)(shifts)            # (S,N)
    flat_w = (W * weight[None, :]).reshape(-1)    # (S*N,)

    # --- scatter-add in one shot via jnp.bincount ---
    flat_field = field.ravel()
    flat_field = flat_field + jnp.bincount(
        flat_idx, weights=flat_w, length=flat_field.size
    ).astype(field.dtype)
    return flat_field.reshape(field.shape)


# ------------ public assign (with chunking) ------------
@partial(
    jit,
    static_argnames=(
        "num_particles",
        "window_order",
        "interlace",
        "contd",
        "max_scatter_indices",
    ),
)
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
    Mass-assignment in JAX with optional chunking and jnp.bincount.
    """
    ng = field.shape[0]
    cell = boxsize / ng

    # flatten (3,Nx,Ny,Nz) -> (3,N) if necessary
    if pos.ndim == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    # coordinates in mesh units
    pos_mesh = pos / cell
    if interlace:
        pos_mesh += 0.5

    n_shifts = {1: 1, 2: 8, 3: 27}[window_order]
    total_scatter = num_particles * n_shifts

    # decide static chunk size (Python int) so that S*chunk <= max_scatter_indices
    chunk_size = max_scatter_indices // n_shifts
    chunk_size = int(min(chunk_size, num_particles))
    n_chunks = (num_particles + chunk_size - 1) // chunk_size  # runtime value OK

    # body function applied by lax.fori_loop
    def body(i, fld):
        start = i * chunk_size
        pm_ck = lax.dynamic_slice(pos_mesh, (0, start), (3, chunk_size))
        wt_ck = lax.dynamic_slice(weight, (start,), (chunk_size,))
        return _single_assign(fld, pm_ck, wt_ck, ng, window_order)

    # loop over chunks
    field = lax.fori_loop(0, n_chunks, body, field)

    # optional per-file normalisation
    if contd == 0:
        field /= num_particles / ng ** 3

    return field


