#!/usr/bin/env python3
from __future__ import annotations
from typing import Literal, Dict, Optional, Tuple, List

import jax.numpy as jnp
from jax import jit, vmap, lax, ops
from functools import partial

# ------------------------------------------------------------
# 1D k-helpers (avoid huge captured constants)
# ------------------------------------------------------------

def _k1d(ng: int, boxsize: float, *, dtype):
    """Return 1D physical wavenumbers for full/rfft axes, with squared variants."""
    dtype = jnp.dtype(dtype)
    fac = (2.0 * jnp.pi) / boxsize
    kx = fac * jnp.fft.fftfreq(ng, d=1.0 / ng).astype(dtype)           # (ng,)
    ky = kx                                                            # (ng,)
    kz = fac * jnp.fft.rfftfreq(ng, d=1.0 / ng).astype(dtype)          # (ng // 2 + 1,)
    return kx, ky, kz

# ------------------------------------------------------------
# Power spectrum estimator (streaming / low-mem)
# ------------------------------------------------------------

class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_edges, ell_max=0, leg_fac=True, *, dtype=jnp.float32):
        """Memory-savvy P(k): precompute only 1D/2D k-structures and stream over kz-planes."""
        self.boxsize = jnp.asarray(boxsize, dtype)
        self.kbin_edges = jnp.asarray(kbin_edges, dtype=dtype)
        self.num_bins = int(self.kbin_edges.shape[0] - 1)
        self.vol = self.boxsize ** 3
        self.ng = int(ng)
        self.dtype = jnp.dtype(dtype)
        self.ell_max = int(ell_max)
        self.leg_fac = bool(leg_fac)

        # 1D k and squared
        kx, ky, kz = _k1d(self.ng, self.boxsize, dtype=self.dtype)
        self.kx2 = (kx * kx).astype(self.dtype)
        self.ky2 = (ky * ky).astype(self.dtype)
        self.kz2 = (kz * kz).astype(self.dtype)

        # 2D kxy^2 reused across all kz-planes (memory ~ ng^2)
        self.kxy2 = (self.kx2[:, None] + self.ky2[None, :]).astype(self.dtype)  # (ng, ng)

        # Bin centers
        self.kbin_centers = 0.5 * (self.kbin_edges[1:] + self.kbin_edges[:-1])

        # Precompute mean-k and counts via streaming over kz planes
        self.k_mean, self.Nk = self._precompute_bin_stats()

        # Legendre normalization factors (scalar per even ell)
        self._leg_scalar = {ell: (2*ell + 1) if self.leg_fac else 1.0
                            for ell in range(0, self.ell_max + 1, 2)}

    @partial(jit, static_argnames=('self',))
    def _precompute_bin_stats(self):
        """Stream over kz planes to get Nk (degeneracy-aware) and mean-k per bin."""
        num_bins = self.num_bins
        edges = self.kbin_edges
        ng = self.ng
        kxy2 = self.kxy2
        kz2 = self.kz2
        dtype = self.dtype

        def plane_body(p, carry):
            k_sum, n_sum = carry
            deg = jnp.where((p == 0) | ((ng % 2 == 0) & (p == ng // 2)), 1.0, 2.0).astype(dtype)
            k2 = kxy2 + kz2[p]
            kmag = jnp.sqrt(jnp.maximum(k2, 0.0))
            mask_dc = ~( (p == 0) & (kmag == 0.0) )
            kidx = jnp.digitize(kmag.ravel(), edges, right=True)
            w = (deg * mask_dc.ravel()).astype(dtype)
            k_tot = ops.segment_sum(kmag.ravel() * w, kidx, num_bins + 2)[1:-1]
            N_tot = ops.segment_sum(w, kidx, num_bins + 2)[1:-1]
            return (k_sum + k_tot, n_sum + N_tot)

        k0 = jnp.zeros((num_bins,), dtype=dtype)
        n0 = jnp.zeros((num_bins,), dtype=dtype)
        k_sum, n_sum = lax.fori_loop(0, self.kz2.shape[0], plane_body, (k0, n0))
        k_mean = jnp.where(n_sum > 0, k_sum / n_sum, 0.0)
        return k_mean, n_sum

    @partial(jit, static_argnames=('self', 'ell'))
    def __call__(self, fieldk1, fieldk2=None, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Compute auto/cross P(k) for multipole ell (0,2,4,...) and mu-range.
        Streams over kz-planes; avoids 3D kmag storage.
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        dtype = self.dtype
        ctype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        fieldk1 = fieldk1.astype(ctype)
        fieldk2 = fieldk2.astype(ctype)

        ng = self.ng
        edges = self.kbin_edges
        num_bins = self.num_bins
        kxy2 = self.kxy2
        kz2 = self.kz2

        ell = int(ell)
        if ell % 2 != 0 or ell < 0 or ell > self.ell_max:
            raise ValueError(f"ell={ell} not supported; must be even in [0, {self.ell_max}]")
        leg_scalar = jnp.asarray(self._leg_scalar[ell], dtype=dtype)

        mu2_min = jnp.asarray(mu_min**2, dtype=dtype)
        mu2_max = jnp.asarray(mu_max**2, dtype=dtype)

        def plane_body(p, carry):
            P_sum, N_sum = carry
            deg = jnp.where((p == 0) | ((ng % 2 == 0) & (p == ng // 2)), 1.0, 2.0).astype(dtype)
            fk1 = fieldk1[..., p]
            fk2 = fieldk2[..., p]
            cross = (fk1 * jnp.conj(fk2)).real.astype(dtype)

            k2 = kxy2 + kz2[p]
            kmag = jnp.sqrt(jnp.maximum(k2, 0.0))
            mu2 = jnp.where(k2 > 0, kz2[p] / k2, 0.0).astype(dtype)

            # Legendre factor
            if ell == 0:
                leg = jnp.ones_like(mu2) * leg_scalar
            elif ell == 2:
                leg = leg_scalar * 0.5 * (3.0 * mu2 - 1.0)
            else:
                leg = leg_scalar * 0.125 * (35.0 * mu2**2 - 30.0 * mu2 + 3.0)

            # mu-range mask + exclude DC at (0,0,0)
            mask_mu = (mu2 >= mu2_min) & (mu2 <= mu2_max)
            mask_dc = ~((p == 0) & (kmag == 0.0))
            mask = (mask_mu & mask_dc).astype(dtype)

            kidx = jnp.digitize(kmag.ravel(), edges, right=True)
            wP = (deg * (cross * leg * mask).ravel()).astype(dtype)
            wN = (deg * mask.ravel()).astype(dtype)
            P_add = ops.segment_sum(wP, kidx, num_bins + 2)[1:-1]
            N_add = ops.segment_sum(wN, kidx, num_bins + 2)[1:-1]
            return (P_sum + P_add, N_sum + N_add)

        P0 = jnp.zeros((num_bins,), dtype=dtype)
        N0 = jnp.zeros((num_bins,), dtype=dtype)
        P_sum, N_sum = lax.fori_loop(0, kz2.shape[0], plane_body, (P0, N0))

        Pk_binned = jnp.where(N_sum > 0, P_sum / N_sum, 0.0)
        out = jnp.stack([self.k_mean, Pk_binned * self.vol, N_sum], axis=1)
        return out

    @partial(jit, static_argnames=('self',))
    def compute_mu_mean(self, mu_min=0.0, mu_max=1.0):
        """Return ⟨mu⟩ over mu in [mu_min, mu_max]; rfftn implies kz >= 0 thus mu >= 0 (no sign cancellation)."""
        dtype = self.dtype
        mu2_min = jnp.asarray(mu_min**2, dtype=dtype)
        mu2_max = jnp.asarray(mu_max**2, dtype=dtype)
        ng = self.ng
        kxy2 = self.kxy2
        kz2  = self.kz2

        def plane_body(p, carry):
            num, den = carry
            deg = jnp.where((p == 0) | ((ng % 2 == 0) & (p == ng // 2)), 1.0, 2.0).astype(dtype)
            k2 = kxy2 + kz2[p]
            kmag = jnp.sqrt(jnp.maximum(k2, 0.0))
            mu = jnp.where(k2 > 0, jnp.sqrt(kz2[p] / k2), 0.0).astype(dtype)
            mask = ((mu*mu >= mu2_min) & (mu*mu <= mu2_max)).astype(dtype)
            mask = jnp.where((p == 0) & (kmag == 0.0), 0.0, mask)
            num += jnp.sum(mu * mask) * deg
            den += jnp.sum(mask) * deg
            return (num, den)

        num0 = jnp.array(0.0, dtype=dtype)
        den0 = jnp.array(0.0, dtype=dtype)
        num, den = lax.fori_loop(0, kz2.shape[0], plane_body, (num0, den0))
        return jnp.where(den > 0, num / den, 0.0)

# ============================================================
# FFT-based P(k) & B(k)
# ============================================================

class Measure_spectra_FFT:
    """
    FFT-based estimators for P(k) and B(k) with three memory/speed modes.

      - mode="speed"   : Precompute and cache all bin-filtered real-space fields I_i (per field) and N_i once.
      - mode="chunked" : Build I/N only for a small set of bins at a time (bin_chunk).
      - mode="low_mem" : Build I/N on-the-fly for each bin/triangle.

    Triangle selection (order_mode):
      - "auto"   : choose the minimal independent set based on field equality
                   (all-same -> unique; exactly-two-same -> pair symmetry; all-distinct -> all-ordered)
      - "unique" : always unique i<=j<=k
      - "all"    : always all ordered (i,j,k) that satisfy the triangle condition

    Note:
      * Triangle index lists are built lazily on first use and cached. Changing k-bin edges requires a new instance.
    """

    # ---------- low-level jitted kernel (no static self capture) ----------
    @staticmethod
    @jit
    def _filter_kernel(fieldk, kx2, ky2, kz2, kmin2, kmax2):
        """Band-pass in k-space by scalar [kmin,kmax), then irfftn -> real."""
        k2 = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :])
        mask = (kmin2 <= k2) & (k2 < kmax2)
        fr   = jnp.fft.irfftn(fieldk * mask, norm='forward')
        return fr

    # ---------- public wrapper ----------
    def _filter_rfft_bandpass(self, fieldk, kmin, kmax):
        kmin2 = jnp.asarray(kmin, dtype=self.dtype) ** 2
        kmax2 = jnp.asarray(kmax, dtype=self.dtype) ** 2
        out = self._filter_kernel(fieldk, self.kx2, self.ky2, self.kz2, kmin2, kmax2)
        return out.astype(self.dtype)

    # ---------------------------------------------------------------------
    # ctor
    # ---------------------------------------------------------------------
    def __init__(self, boxsize: float, ng: int, kbin_edges, *,
                 dtype=jnp.float32, bispec: bool=True, open_triangle: bool=False):
        self.boxsize = jnp.array(boxsize, dtype=dtype)
        self.ng      = int(ng)
        self.dtype   = jnp.dtype(dtype)
        self.open_triangle = bool(open_triangle)
        self._enable_bispec = bool(bispec)

        self.kbin_edges   = jnp.asarray(kbin_edges, dtype=self.dtype)
        self.kbin_centers = 0.5 * (self.kbin_edges[1:] + self.kbin_edges[:-1])
        self.num_bins     = int(self.kbin_edges.shape[0] - 1)
        self.vol          = self.boxsize ** 3

        # Build k^2 1D axes
        kx, ky, kz = _k1d(self.ng, self.boxsize, dtype=self.dtype)
        self.kx2 = (kx * kx).astype(self.dtype)
        self.ky2 = (ky * ky).astype(self.dtype)
        self.kz2 = (kz * kz).astype(self.dtype)

        # Lazy triangle caches (None until first use)
        self._tri_cache: Dict[str, Optional[jnp.ndarray]] = {
            "unique": None,   # i<=j<=k
            "all":    None,   # all ordered
            "i_le_j": None,   # for f1==f2!=f3
            "j_le_k": None,   # for f2==f3!=f1
            "i_le_k": None,   # for f1==f3!=f2
        }

        # Speed-mode full stacks
        self._I_full: Dict[str, Optional[jnp.ndarray]] = {"f1": None, "f2": None, "f3": None}
        self._N_full: Optional[jnp.ndarray] = None
        self._sig: Dict[str, Optional[Tuple[int, Tuple[int, ...], str]]] = {"f1": None, "f2": None, "f3": None}
        self._skip_prepare: bool = False

    # ---------------------------------------------------------------------
    # helpers: build stacks for a list of bins (vmap over bins)
    # ---------------------------------------------------------------------
    def _build_stack_for_bins(self, fieldk, bins_1d: jnp.ndarray) -> jnp.ndarray:
        """Return stacked real-space filtered fields for bins in `bins_1d`."""
        ctype = jnp.complex64 if self.dtype == jnp.float32 else jnp.complex128
        fieldk = fieldk.astype(ctype)
        kmin = self.kbin_edges[bins_1d]
        kmax = self.kbin_edges[bins_1d + 1]
        return vmap(lambda a, b: self._filter_rfft_bandpass(fieldk, a, b))(kmin, kmax)

    # ---------------------------------------------------------------------
    # triangle builders (lazy)
    # ---------------------------------------------------------------------
    def _triangle_ok(self, i: int, j: int, k: int) -> bool:
        """Triangle condition with either closed (centers) or open (bin edges) rule."""
        if self.open_triangle:
            kmin = self.kbin_edges[:-1]; kmax = self.kbin_edges[1:]
            k1min, k1max = float(kmin[i]), float(kmax[i])
            k2min, k2max = float(kmin[j]), float(kmax[j])
            k3min, k3max = float(kmin[k]), float(kmax[k])
            return ((k1max + k2max > k3min) and
                    (k2max + k3max > k1min) and
                    (k3max + k1max > k2min))
        else:
            k1 = float(self.kbin_centers[i]); k2 = float(self.kbin_centers[j]); k3 = float(self.kbin_centers[k])
            return (k3 >= abs(k1 - k2)) and (k3 <= (k1 + k2))

    def _build_triangles(self, pattern: str) -> jnp.ndarray:
        """Build triangle list for a given pattern and return int32 array (n,3)."""
        if not self._enable_bispec:
            return jnp.zeros((0, 3), dtype=jnp.int32)

        B = self.num_bins
        out: List[Tuple[int, int, int]] = []

        if pattern == "unique":  # i<=j<=k
            for i in range(B):
                for j in range(i, B):
                    for k in range(j, B):
                        if self._triangle_ok(i, j, k):
                            out.append((i, j, k))

        elif pattern == "all":   # all ordered
            for i in range(B):
                for j in range(B):
                    for k in range(B):
                        if self._triangle_ok(i, j, k):
                            out.append((i, j, k))

        elif pattern == "i_le_j":  # i<=j, k free
            for i in range(B):
                for j in range(i, B):
                    for k in range(B):
                        if self._triangle_ok(i, j, k):
                            out.append((i, j, k))

        elif pattern == "j_le_k":  # j<=k, i free
            for i in range(B):
                for j in range(B):
                    for k in range(j, B):
                        if self._triangle_ok(i, j, k):
                            out.append((i, j, k))

        elif pattern == "i_le_k":  # i<=k, j free
            for i in range(B):
                for k in range(i, B):
                    for j in range(B):
                        if self._triangle_ok(i, j, k):
                            out.append((i, j, k))
        else:
            raise ValueError(f"Unknown triangle pattern: {pattern}")

        return (jnp.asarray(out, dtype=jnp.int32) if out
                else jnp.zeros((0, 3), dtype=jnp.int32))

    def _get_triangles(self, pattern: str) -> jnp.ndarray:
        """Return (and cache) triangle list for the requested pattern."""
        arr = self._tri_cache.get(pattern)
        if arr is None:
            arr = self._build_triangles(pattern)
            self._tri_cache[pattern] = arr
        return arr

    # ---------------------------------------------------------------------
    # cache preparation
    # ---------------------------------------------------------------------
    def _prepare_cache(self, fieldk1, fieldk2=None, fieldk3=None, *,
                       mode: Literal["speed", "chunked", "low_mem"] = "speed"):
        """Prepare bin-filtered stacks depending on mode."""
        if mode in ('chunked', 'low_mem'):
            self._I_full = {"f1": None, "f2": None, "f3": None}
            self._N_full = None
            self._sig = {"f1": None, "f2": None, "f3": None}
            return

        if mode == "speed":
            bins_all = jnp.arange(self.num_bins, dtype=jnp.int32)
            same12 = (fieldk2 is None) or (fieldk2 is fieldk1)
            same23 = (fieldk3 is None) or (fieldk3 is fieldk2)
            same13 = (fieldk3 is None) or (fieldk3 is fieldk1)
            tag2 = "f1" if same12 else "f2"
            tag3 = "f1" if same13 else ("f2" if same23 else "f3")

            sig1 = self._sig_from(fieldk1)
            if self._sig["f1"] != sig1:
                self._I_full["f1"] = self._build_stack_for_bins(fieldk1, bins_all)
                self._sig["f1"] = sig1

            if tag2 == "f2":
                sig2 = self._sig_from(fieldk2)
                if self._sig["f2"] != sig2:
                    self._I_full["f2"] = self._build_stack_for_bins(fieldk2, bins_all)
                    self._sig["f2"] = sig2

            if tag3 == "f3":
                sig3 = self._sig_from(fieldk3)
                if self._sig["f3"] != sig3:
                    self._I_full["f3"] = self._build_stack_for_bins(fieldk3, bins_all)
                    self._sig["f3"] = sig3

            if self._N_full is None:
                onesk = jnp.ones_like(fieldk1)
                self._N_full = self._build_stack_for_bins(onesk, bins_all)

    def _sig_from(self, arr) -> Tuple[int, Tuple[int, ...], str]:
        return (id(arr), tuple(arr.shape), str(arr.dtype))

    # ---------------------------------------------------------------------
    # P(k)
    # ---------------------------------------------------------------------
    def compute_pk(self, fieldk1, fieldk2=None, *,
                   mode: Literal["speed", "chunked", "low_mem"] = "speed",
                   bin_chunk: Optional[int] = None) -> jnp.ndarray:
        if fieldk2 is None:
            fieldk2 = fieldk1

        norm = jnp.asarray(self.ng, dtype=self.dtype) ** 3

        if mode == "speed":
            if not self._skip_prepare:
                self._prepare_cache(fieldk1, fieldk2, None, mode="speed")
            I1 = self._I_full["f1"]
            I2 = I1 if (fieldk2 is fieldk1) else self._I_full["f2"]
            N  = self._N_full
            num = jnp.sum(I1 * I2, axis=(1, 2, 3))
            den = jnp.sum(N  * N,  axis=(1, 2, 3))
            P   = jnp.where(den > 0, num / den, 0.0)
            return jnp.stack([self.kbin_centers, (P * self.vol).real, den/norm], axis=1).astype(self.dtype)

        elif mode == "chunked":
            if bin_chunk is None or bin_chunk <= 0:
                raise ValueError("chunked PK requires bin_chunk > 0.")
            rows = []
            onesk = jnp.ones_like(fieldk1)
            for s in range(0, self.num_bins, int(bin_chunk)):
                e = min(s + int(bin_chunk), self.num_bins)
                bins = jnp.arange(s, e, dtype=jnp.int32)
                I1 = self._build_stack_for_bins(fieldk1, bins)
                I2 = I1 if (fieldk2 is fieldk1) else self._build_stack_for_bins(fieldk2, bins)
                N  = self._build_stack_for_bins(onesk,   bins)
                num = jnp.sum(I1 * I2, axis=(1, 2, 3))
                den = jnp.sum(N  * N,  axis=(1, 2, 3))
                P   = jnp.where(den > 0, num / den, 0.0)
                rows.append(jnp.stack([self.kbin_centers[bins], (P * self.vol).real, den/norm], axis=1))
            return jnp.concatenate(rows, axis=0).astype(self.dtype)

        else:  # low_mem
            onesk = jnp.ones_like(fieldk1)

            def one_bin(i):
                i = jnp.int32(i)
                kmin = self.kbin_edges[i]; kmax = self.kbin_edges[i + 1]
                I1 = self._filter_rfft_bandpass(fieldk1, kmin, kmax)
                I2 = I1 if (fieldk2 is fieldk1) else self._filter_rfft_bandpass(fieldk2, kmin, kmax)
                N  = self._filter_rfft_bandpass(onesk,   kmin, kmax)
                num = jnp.sum(I1 * I2)
                den = jnp.sum(N  * N)
                P   = jnp.where(den > 0, num / den, 0.0)
                return jnp.stack([ self.kbin_centers[i], (P * self.vol).real, den/norm], axis=0).astype(self.dtype)

            LMAP_CHUNK = 256
            rows = []
            for s in range(0, int(self.num_bins), LMAP_CHUNK):
                e = min(s + LMAP_CHUNK, int(self.num_bins))
                idx = jnp.arange(s, e, dtype=jnp.int32)
                rows.append(lax.map(one_bin, idx))
            return jnp.concatenate(rows, axis=0) if rows else jnp.zeros((0, 3), dtype=self.dtype)

    # ---------------------------------------------------------------------
    # B(k)
    # ---------------------------------------------------------------------
    def compute_bk(self, fieldk1, fieldk2=None, fieldk3=None, *,
                   mode: Literal["speed", "chunked", "low_mem"] = "speed",
                   bin_chunk: Optional[int] = None,
                   order_mode: Literal["auto", "unique", "all"] = "auto",
                   ) -> jnp.ndarray:
        """
        Return (nrows, 5): [k1, k2, k3, B(k)*V^2, norm].

        Triangle selection:
          - order_mode="auto" (default):
              * all-equal fields       -> unique (i<=j<=k)
              * exactly-two-equal      -> pair symmetry (i<=j / j<=k / i<=k)
              * all-distinct           -> all ordered
          - order_mode="unique": always unique (i<=j<=k)
          - order_mode="all":    always all ordered
        """
        if fieldk2 is None: fieldk2 = fieldk1
        if fieldk3 is None: fieldk3 = fieldk2

        same12 = (fieldk2 is fieldk1)
        same23 = (fieldk3 is fieldk2)
        same13 = (fieldk3 is fieldk1)
        same_all = same12 and same23

        # decide pattern + get (lazy) triangle list
        if order_mode == "unique":
            pattern = "unique"
        elif order_mode == "all":
            pattern = "all"
        else:  # "auto"
            if same_all:
                pattern = "unique"
            elif same12 and (not same13):    # f1==f2!=f3 -> i<=j
                pattern = "i_le_j"
            elif same23 and (not same12):    # f2==f3!=f1 -> j<=k
                pattern = "j_le_k"
            elif same13 and (not same12):    # f1==f3!=f2 -> i<=k
                pattern = "i_le_k"
            else:
                pattern = "all"

        tri = self._get_triangles(pattern)
        ntri = int(tri.shape[0])
        if ntri == 0:
            return jnp.zeros((0, 5), dtype=self.dtype)

        onesk = jnp.ones_like(fieldk1)
        tag2 = "f1" if same12 else "f2"
        tag3 = "f1" if same13 else ("f2" if same23 else "f3")

        norm = jnp.asarray(self.ng, dtype=self.dtype) ** 3

        if mode == "speed":
            if not self._skip_prepare:
                self._prepare_cache(fieldk1, fieldk2, fieldk3, mode="speed")
            I1s = self._I_full["f1"]
            I2s = I1s if tag2 == "f1" else self._I_full["f2"]
            I3s = I1s if tag3 == "f1" else (self._I_full["f2"] if tag3 == "f2" else self._I_full["f3"])
            Ns  = self._N_full

            def one_triangle(triple):
                i = jnp.int32(triple[0]); j = jnp.int32(triple[1]); k = jnp.int32(triple[2])
                num = jnp.sum(I1s[i] * I2s[j] * I3s[k])
                den = jnp.sum(Ns[i]  * Ns[j]  * Ns[k])
                B   = jnp.where(den > 0, num / den, 0.0)
                return jnp.stack([ self.kbin_centers[i],
                                   self.kbin_centers[j],
                                   self.kbin_centers[k],
                                   (B * (self.vol**2)).real,
                                   den / norm ], axis=0).astype(self.dtype)

            return lax.map(one_triangle, tri)

        elif mode == "chunked":
            if bin_chunk is None or bin_chunk <= 0:
                raise ValueError("chunked BK requires bin_chunk > 0.")

            Bins = int(self.num_bins)
            C    = int(bin_chunk)
            nch  = (Bins + C - 1) // C

            cache_I = {"f1": {}, "f2": {}, "f3": {}}
            cache_N = {}

            def bins_of(ch):
                s = ch * C
                e = min(s + C, Bins)
                return s, e, jnp.arange(s, e, dtype=jnp.int32)

            def get_I(tag, ch):
                if ch in cache_I[tag]:
                    return cache_I[tag][ch]
                s, e, bins = bins_of(ch)
                if tag == "f1": I = self._build_stack_for_bins(fieldk1, bins)
                elif tag == "f2": I = self._build_stack_for_bins(fieldk2, bins)
                else: I = self._build_stack_for_bins(fieldk3, bins)
                cache_I[tag][ch] = I
                return I

            def get_N(ch):
                if ch in cache_N:
                    return cache_N[ch]
                s, e, bins = bins_of(ch)
                N = self._build_stack_for_bins(onesk, bins)
                cache_N[ch] = N
                return N

            rows_list, idx_list = [], []

            # chunk-loop bounds depend on pattern
            for a in range(nch):
                b_start = a if pattern in ("unique", "i_le_j") else 0
                for b in range(b_start, nch):
                    if pattern in ("unique", "j_le_k"):
                        c_start = b
                    elif pattern == "i_le_k":
                        c_start = a
                    else:  # "all" or "i_le_j"
                        c_start = 0
                    for c in range(c_start, nch):
                        sA, eA, _ = bins_of(a)
                        sB, eB, _ = bins_of(b)
                        sC, eC, _ = bins_of(c)

                        i_in = (tri[:, 0] >= sA) & (tri[:, 0] < eA)
                        j_in = (tri[:, 1] >= sB) & (tri[:, 1] < eB)
                        k_in = (tri[:, 2] >= sC) & (tri[:, 2] < eC)
                        mask = i_in & j_in & k_in
                        idx  = jnp.where(mask, size=None, fill_value=0)[0]
                        if int(idx.shape[0]) == 0:
                            continue

                        tri_sub = tri[idx]
                        li = tri_sub[:, 0] - jnp.int32(sA)
                        lj = tri_sub[:, 1] - jnp.int32(sB)
                        lk = tri_sub[:, 2] - jnp.int32(sC)

                        I1 = get_I("f1", a)
                        I2 = I1 if (same12 and a == b) else get_I("f2", b)
                        if same13 and a == c: I3 = I1
                        elif same23 and b == c: I3 = I2
                        else: I3 = get_I("f3", c)
                        NA, NB, NC = get_N(a), get_N(b), get_N(c)

                        def one_row(local_ids):
                            li_, lj_, lk_ = jnp.int32(local_ids[0]), jnp.int32(local_ids[1]), jnp.int32(local_ids[2])
                            num = jnp.sum(I1[li_] * I2[lj_] * I3[lk_])
                            den = jnp.sum(NA[li_] * NB[lj_] * NC[lk_])
                            Bv  = jnp.where(den > 0, num / den, 0.0)
                            gi  = jnp.int32(sA) + li_
                            gj  = jnp.int32(sB) + lj_
                            gk  = jnp.int32(sC) + lk_
                            return jnp.stack([ self.kbin_centers[gi],
                                               self.kbin_centers[gj],
                                               self.kbin_centers[gk],
                                               (Bv * (self.vol**2)).real,
                                               den / norm ], axis=0).astype(self.dtype)

                        local_triples = jnp.stack([li, lj, lk], axis=1)
                        rows = lax.map(one_row, local_triples)
                        rows_list.append(rows)
                        idx_list.append(idx)

            all_rows = jnp.concatenate(rows_list, axis=0) if rows_list else jnp.zeros((0, 5), dtype=self.dtype)
            all_idx  = jnp.concatenate(idx_list,  axis=0) if idx_list  else jnp.zeros((0,), dtype=jnp.int32)
            order    = jnp.argsort(all_idx)
            return all_rows[order]

        else:  # low_mem
            # Cast field-equality flags to JAX booleans once
            same12_b = jnp.bool_(same12)
            same13_b = jnp.bool_(same13)
            same23_b = jnp.bool_(same23)

            onesk = jnp.ones_like(fieldk1)

            def one_triangle(triple):
                i = jnp.int32(triple[0]); j = jnp.int32(triple[1]); k = jnp.int32(triple[2])

                kmin_i, kmax_i = self.kbin_edges[i], self.kbin_edges[i + 1]
                kmin_j, kmax_j = self.kbin_edges[j], self.kbin_edges[j + 1]
                kmin_k, kmax_k = self.kbin_edges[k], self.kbin_edges[k + 1]

                # Always compute I_i
                I_i = self._filter_rfft_bandpass(fieldk1, kmin_i, kmax_i)

                # I_j: reuse I_i if (field2 is field1) and (i==j); otherwise compute from field2
                cond_ij = jnp.logical_and(same12_b, jnp.equal(i, j))
                I_j = lax.cond(
                    cond_ij,
                    lambda _: I_i,
                    lambda _: self._filter_rfft_bandpass(fieldk2, kmin_j, kmax_j),
                    operand=None
                )

                # I_k: try reuse from I_i or I_j based on equalities; otherwise compute from field3
                cond_ik = jnp.logical_and(same13_b, jnp.equal(i, k))
                I_k = lax.cond(
                    cond_ik,
                    lambda _: I_i,
                    lambda _: lax.cond(
                        jnp.logical_and(same23_b, jnp.equal(j, k)),
                        lambda __: I_j,
                        lambda __: self._filter_rfft_bandpass(fieldk3, kmin_k, kmax_k),
                        operand=None
                    ),
                    operand=None
                )

                # N_i, N_j, N_k similarly with reuse when bin indices match
                N_i = self._filter_rfft_bandpass(onesk, kmin_i, kmax_i)
                N_j = lax.cond(
                    jnp.equal(i, j),
                    lambda _: N_i,
                    lambda _: self._filter_rfft_bandpass(onesk, kmin_j, kmax_j),
                    operand=None
                )
                N_k = lax.cond(
                    jnp.equal(i, k),
                    lambda _: N_i,
                    lambda _: lax.cond(
                        jnp.equal(j, k),
                        lambda __: N_j,
                        lambda __: self._filter_rfft_bandpass(onesk, kmin_k, kmax_k),
                        operand=None
                    ),
                    operand=None
                )

                num = jnp.sum(I_i * I_j * I_k)
                den = jnp.sum(N_i * N_j * N_k)
                B   = jnp.where(den > 0, num / den, 0.0)

                return jnp.stack([
                    self.kbin_centers[i],
                    self.kbin_centers[j],
                    self.kbin_centers[k],
                    (B * (self.vol**2)).real,
                    den / norm
                ], axis=0).astype(self.dtype)

            # Chunk over triangles to keep compile time moderate
            LMAP_CHUNK = 256
            rows = []
            for s in range(0, ntri, LMAP_CHUNK):
                e = min(s + LMAP_CHUNK, ntri)
                rows.append(lax.map(one_triangle, tri[s:e, :]))
            return jnp.concatenate(rows, axis=0) if rows else jnp.zeros((0, 5), dtype=self.dtype)

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------
    def compute_pk_bk(self, fieldk1, fieldk2=None, fieldk3=None, *,
                      mode: Literal["speed", "chunked", "low_mem"] = "speed",
                      bin_chunk: Optional[int] = None,
                      order_mode: Literal["auto", "unique", "all"] = "auto",
                      ):
        """Compute P(k) and B(k) with consistent mode/chunk settings."""
        if mode == "speed":
            self._prepare_cache(fieldk1, fieldk2, fieldk3, mode="speed")
            self._skip_prepare = True
        try:
            pk = self.compute_pk(fieldk1, fieldk2, mode=mode, bin_chunk=bin_chunk)
            bk = self.compute_bk(fieldk1, fieldk2, fieldk3,
                                 mode=mode, bin_chunk=bin_chunk,
                                 order_mode=order_mode,
                                 )
        finally:
            self._skip_prepare = False
        return pk, bk
