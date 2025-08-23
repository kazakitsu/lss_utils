#!/usr/bin/env python3
from __future__ import annotations
from functools import partial
import jax.numpy as jnp
from jax import jit

# ------------------------------------------------------------
# k-axes helper: physical wave numbers on rfftn grid
def kaxes_1d(ng: int, boxsize: float, *, dtype=jnp.float32):
    """Return 1D physical k-axes for x,y (fftfreq) and z (rfftfreq)."""
    dtype = jnp.dtype(dtype)
    boxsize = jnp.asarray(boxsize, dtype)
    fac = (2.0 * jnp.pi) / boxsize
    # Use sample spacing d = 1/ng so that fac * fftfreq gives physical k.
    d = 1.0 / jnp.asarray(ng, dtype)
    kx = fac * jnp.fft.fftfreq(ng, d=d)
    ky = kx
    kz = fac * jnp.fft.rfftfreq(ng, d=d)
    return kx.astype(dtype), ky.astype(dtype), kz.astype(dtype)

# ------------------------------------------------------------
# Broadcast helper
def _k_broadcast(kx, ky, kz):
    """Broadcast 1D axes to rfftn grid shapes."""
    return kx[:, None, None], ky[None, :, None], kz[None, None, :]

# ------------------------------------------------------------
# Symmetric 3x3 * vector
@jit
def _S_matvec(S6, vx, vy, vz):
    """Given S6=(xx,xy,xz,yy,yz,zz), return w = S @ v as 3 arrays."""
    Sxx, Sxy, Sxz, Syy, Syz, Szz = S6
    wx = Sxx*vx + Sxy*vy + Sxz*vz
    wy = Sxy*vx + Syy*vy + Syz*vz
    wz = Sxz*vx + Syz*vy + Szz*vz
    return wx, wy, wz

@jit
def _trace_S(S6):
    """Trace of S from packed 6-tuple representation."""
    Sxx, _, _, Syy, _, Szz = S6
    return Sxx + Syy + Szz

# ------------------------------------------------------------
# Public API
def project_rank2_to_helicity(
    S6, kx, ky, kz, *,
    mode: str = "speed",     # "speed" or "low_mem"
    chunk_z: int = 64,       # used when mode == "low_mem"
    eps: float = 1e-12,
    m_select=None,           # None/"all", or subset like 0, "+1", (-2, 0, +2), ["+2","-2"], etc.
):
    """
    Project symmetric rank-2 tensor S_ij(k) to helicity components S_m(k).

    Parameters
    ----------
    S6 : (6, nx, ny, nz_r) array (real or complex). Order: (xx, xy, xz, yy, yz, zz).
    kx, ky, kz : 1D axes from kaxes_1d.
    mode : "speed" (vectorized) or "low_mem" (chunked over z).
    chunk_z : slab thickness when mode == "low_mem".
    eps : small threshold for pole handling when building e1.
    m_select : choose which m to compute.
       - None / "all": return [0, +1, -1, +2, -2] in this order (shape (5,...)).
       - int/str or list/tuple thereof: compute only those m in the given order.

    Returns
    -------
    out : (n_comp, nx, ny, nz_r) complex array.
          The first axis follows the order specified by m_select; when None, it is [0, +1, -1, +2, -2].
    """
    pack_order, flags = _normalize_m_select(m_select)
    need_m0, need_p1, need_m1, need_p2, need_m2 = flags

    if mode not in ("speed", "low_mem"):
        raise ValueError("mode must be 'speed' or 'low_mem'")

    if mode == "speed":
        return _project_speed_subset(
            S6, kx, ky, kz,
            need_m0=need_m0, need_p1=need_p1, need_m1=need_m1, need_p2=need_p2, need_m2=need_m2,
            eps=eps, pack_order=pack_order,
        )
    else:
        return _loop_chunked_subset(
            S6, kx, ky, kz,
            need_m0=need_m0, need_p1=need_p1, need_m1=need_m1, need_p2=need_p2, need_m2=need_m2,
            pack_order=pack_order, chunk_z=chunk_z, eps=eps,
        )

# ------------------------------------------------------------
# Basis builder: khat, e1, e2 with robust fallback near poles
@jit
def _helicity_basis_real(kx, ky, kz, eps=1e-12):
    """
    Build khat and an orthonormal real frame (e1,e2) with e1 ⟂ khat, e2 = khat x e1.
    Returns: KHX, KHY, KHZ, e1x, e1y, e1z, e2x, e2y, e2z, mask(k>0)
    """
    rtype = jnp.result_type(kx, ky, kz)
    KX, KY, KZ = _k_broadcast(kx, ky, kz)
    k2 = (KX*KX + KY*KY + KZ*KZ).astype(rtype)

    # khat with safe normalization (k=0 -> 0)
    kn = jnp.sqrt(jnp.where(k2 > 0, k2, 1.0)).astype(rtype)
    KHX = jnp.where(k2 > 0, KX/kn, 0.0)
    KHY = jnp.where(k2 > 0, KY/kn, 0.0)
    KHZ = jnp.where(k2 > 0, KZ/kn, 0.0)

    # First reference n = z-hat: c1 = n × khat = (-KHY, KHX, 0)
    c1x, c1y, c1z = -KHY, KHX, jnp.zeros_like(KHX)
    n1 = jnp.sqrt(c1x*c1x + c1y*c1y + c1z*c1z)
    use1 = (n1 > eps)

    # Fallback n = x-hat: c2 = n × khat = (0, -KHZ, KHY)
    c2x, c2y, c2z = jnp.zeros_like(KHX), -KHZ, KHY
    n2 = jnp.sqrt(c2x*c2x + c2y*c2y + c2z*c2z)

    invn1 = jnp.where(use1, 1.0/n1, 0.0)
    invn2 = jnp.where(n2 > 0, 1.0/n2, 0.0)

    e1x = jnp.where(use1, c1x*invn1, jnp.where(k2 > 0, c2x*invn2, 0.0))
    e1y = jnp.where(use1, c1y*invn1, jnp.where(k2 > 0, c2y*invn2, 0.0))
    e1z = jnp.where(use1, c1z*invn1, jnp.where(k2 > 0, c2z*invn2, 0.0))

    # e2 = khat × e1
    e2x = KHY*e1z - KHZ*e1y
    e2y = KHZ*e1x - KHX*e1z
    e2z = KHX*e1y - KHY*e1x

    mask = (k2 > 0)
    return KHX, KHY, KHZ, e1x, e1y, e1z, e2x, e2y, e2z, mask

# ------------------------------------------------------------
# Normalization of m selection
def _normalize_m_select(m_select):
    """
    Normalize m_select to:
      - pack_order: tuple of indices in canonical order space {0:+0, 1:+1, 2:-1, 3:+2, 4:-2}
      - flags: need_m0, need_p1, need_m1, need_p2, need_m2 (booleans)
    Accepts: None/'all', int, str, list/tuple of int/str.
    """
    idx_map = {0:0, +1:1, -1:2, +2:3, -2:4}
    str_map = {"0":0, "+1":1, "-1":2, "+2":3, "-2":4}

    if (m_select is None) or (m_select == "all"):
        pack_order = (0, 1, 2, 3, 4)
    else:
        if isinstance(m_select, (int, str)):
            m_list = [m_select]
        else:
            m_list = list(m_select)
        # convert to canonical indices preserving user order
        conv = []
        for m in m_list:
            if isinstance(m, str):
                m = m.strip()
                if m not in str_map:
                    raise ValueError(f"Invalid m label: {m}")
                conv.append(str_map[m])
            else:
                if m not in idx_map:
                    raise ValueError(f"Invalid m value: {m}")
                conv.append(idx_map[m])
        # remove duplicates while preserving order
        seen = set()
        pack_order = tuple(x for x in conv if not (x in seen or seen.add(x)))

    flags = tuple(i in pack_order for i in (0,1,2,3,4))
    return pack_order, flags  # pack_order, (need_m0, need_p1, need_m1, need_p2, need_m2)

# ------------------------------------------------------------
# SPEED mode: subset kernel (compute only requested m)
@partial(
    jit,
    static_argnames=("need_m0","need_p1","need_m1","need_p2","need_m2","eps","pack_order"),
)
def _project_speed_subset(
    S6, kx, ky, kz, *,
    need_m0: bool, need_p1: bool, need_m1: bool, need_p2: bool, need_m2: bool,
    eps: float, pack_order: tuple,
):
    """Vectorized subset projection on the full rfftn grid."""
    KHX, KHY, KHZ, e1x, e1y, e1z, e2x, e2y, e2z, mask = _helicity_basis_real(kx, ky, kz, eps)

    KHXc, KHYc, KHZc = KHX.astype(S6.dtype), KHY.astype(S6.dtype), KHZ.astype(S6.dtype)
    e1x_c, e1y_c, e1z_c = e1x.astype(S6.dtype), e1y.astype(S6.dtype), e1z.astype(S6.dtype)
    e2x_c, e2y_c, e2z_c = e2x.astype(S6.dtype), e2y.astype(S6.dtype), e2z.astype(S6.dtype)

    ctype = jnp.result_type(S6.dtype, jnp.complex64)
    zero = jnp.zeros((), dtype=ctype)

    # Decide which matvecs are needed
    need_Sk  = need_m0 or need_p1 or need_m1
    need_Se1 = need_p1 or need_m1 or need_p2 or need_m2
    need_Se2 = need_p1 or need_m1 or need_p2 or need_m2

    # Compute required matvecs
    if need_Sk:
        Skx, Sky, Skz = _S_matvec(S6, KHXc, KHYc, KHZc)
    if need_Se1:
        Se1x, Se1y, Se1z = _S_matvec(S6, e1x_c, e1y_c, e1z_c)
    if need_Se2:
        Se2x, Se2y, Se2z = _S_matvec(S6, e2x_c, e2y_c, e2z_c)

    # Results slots (indexed as 0: m0, 1: +1, 2: -1, 3: +2, 4: -2)
    outs = [None, None, None, None, None]

    # m=0
    if need_m0:
        trS = _trace_S(S6).astype(S6.dtype)
        s0 = jnp.sqrt(1.5) * (KHXc*Skx + KHYc*Sky + KHZc*Skz - trS/3.0)
        outs[0] = jnp.where(mask, s0.astype(ctype), zero)

    # m=±1
    if need_p1 or need_m1:
        a = (KHXc*Se1x + KHYc*Se1y + KHZc*Se1z).astype(ctype)
        b = (KHXc*Se2x + KHYc*Se2y + KHZc*Se2z).astype(ctype)
        if need_p1:
            outs[1] = jnp.where(mask, a - 1j*b, zero)
        if need_m1:
            outs[2] = jnp.where(mask, a + 1j*b, zero)

    # m=±2
    if need_p2 or need_m2:
        q11 = (e1x_c*Se1x + e1y_c*Se1y + e1z_c*Se1z).astype(ctype)
        q22 = (e2x_c*Se2x + e2y_c*Se2y + e2z_c*Se2z).astype(ctype)
        q12 = (e1x_c*Se2x + e1y_c*Se2y + e1z_c*Se2z).astype(ctype)
        d = 0.5*(q11 - q22)
        if need_p2:
            outs[3] = jnp.where(mask, d - 1j*q12, zero)
        if need_m2:
            outs[4] = jnp.where(mask, d + 1j*q12, zero)

    # Pack only requested in the user-provided order
    packed = [outs[i] for i in pack_order]
    return jnp.stack(packed, axis=0)

# ------------------------------------------------------------
# LOW-MEM mode: subset kernel for a z-slab block
@partial(
    jit,
    static_argnames=("need_m0","need_p1","need_m1","need_p2","need_m2","eps","pack_order"),
)
def _project_block_subset(
    S6_blk, kx, ky, kz_blk, *,
    need_m0: bool, need_p1: bool, need_m1: bool, need_p2: bool, need_m2: bool,
    eps: float, pack_order: tuple,
):
    """Project a z-slab block to a subset of helicity components."""
    KHX, KHY, KHZ, e1x, e1y, e1z, e2x, e2y, e2z, mask = _helicity_basis_real(kx, ky, kz_blk, eps)

    KHXc, KHYc, KHZc = KHX.astype(S6_blk.dtype), KHY.astype(S6_blk.dtype), KHZ.astype(S6_blk.dtype)
    e1x_c, e1y_c, e1z_c = e1x.astype(S6_blk.dtype), e1y.astype(S6_blk.dtype), e1z.astype(S6_blk.dtype)
    e2x_c, e2y_c, e2z_c = e2x.astype(S6_blk.dtype), e2y.astype(S6_blk.dtype), e2z.astype(S6_blk.dtype)

    ctype = jnp.result_type(S6_blk.dtype, jnp.complex64)
    zero = jnp.zeros((), dtype=ctype)

    need_Sk  = need_m0 or need_p1 or need_m1
    need_Se1 = need_p1 or need_m1 or need_p2 or need_m2
    need_Se2 = need_p1 or need_m1 or need_p2 or need_m2

    if need_Sk:
        Skx, Sky, Skz = _S_matvec(S6_blk, KHXc, KHYc, KHZc)
    if need_Se1:
        Se1x, Se1y, Se1z = _S_matvec(S6_blk, e1x_c, e1y_c, e1z_c)
    if need_Se2:
        Se2x, Se2y, Se2z = _S_matvec(S6_blk, e2x_c, e2y_c, e2z_c)

    outs = [None, None, None, None, None]

    if need_m0:
        trS = _trace_S(S6_blk).astype(S6_blk.dtype)
        s0 = jnp.sqrt(1.5) * (KHXc*Skx + KHYc*Sky + KHZc*Skz - trS/3.0)
        outs[0] = jnp.where(mask, s0.astype(ctype), zero)

    if need_p1 or need_m1:
        a = (KHXc*Se1x + KHYc*Se1y + KHZc*Se1z).astype(ctype)
        b = (KHXc*Se2x + KHYc*Se2y + KHZc*Se2z).astype(ctype)
        if need_p1:
            outs[1] = jnp.where(mask, a - 1j*b, zero)
        if need_m1:
            outs[2] = jnp.where(mask, a + 1j*b, zero)

    if need_p2 or need_m2:
        q11 = (e1x_c*Se1x + e1y_c*Se1y + e1z_c*Se1z).astype(ctype)
        q22 = (e2x_c*Se2x + e2y_c*Se2y + e2z_c*Se2z).astype(ctype)
        q12 = (e1x_c*Se2x + e1y_c*Se2y + e1z_c*Se2z).astype(ctype)
        d = 0.5*(q11 - q22)
        if need_p2:
            outs[3] = jnp.where(mask, d - 1j*q12, zero)
        if need_m2:
            outs[4] = jnp.where(mask, d + 1j*q12, zero)

    packed = [outs[i] for i in pack_order]
    return jnp.stack(packed, axis=0)

def _loop_chunked_subset(
    S6, kx, ky, kz, *,
    need_m0, need_p1, need_m1, need_p2, need_m2, pack_order,
    chunk_z: int, eps: float,
):
    """Python loop over z-chunks; each block is JIT-projected by _project_block_subset."""
    nx, ny, nz = S6.shape[1], S6.shape[2], S6.shape[3]
    # Determine output dtype once
    ctype = jnp.result_type(S6.dtype, jnp.complex64)
    ncomp = len(pack_order)
    out = jnp.zeros((ncomp, nx, ny, nz), dtype=ctype)

    z0 = 0
    while z0 < nz:
        z1 = min(z0 + chunk_z, nz)
        S_blk  = S6[..., z0:z1]
        kz_blk = kz[z0:z1]
        out_blk = _project_block_subset(
            S_blk, kx, ky, kz_blk,
            need_m0=need_m0, need_p1=need_p1, need_m1=need_m1, need_p2=need_p2, need_m2=need_m2,
            eps=eps, pack_order=pack_order,
        )
        out = out.at[..., z0:z1].set(out_blk)
        z0 = z1
    return out