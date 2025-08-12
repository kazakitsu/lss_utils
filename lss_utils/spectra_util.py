#!/usr/bin/env python3
from __future__ import annotations
from typing import Literal, Dict, Optional, Tuple, List

import numpy as np


# ------------------------------------------------------------
# 1D k-helpers (avoid huge captured constants)
# ------------------------------------------------------------
def _k1d(ng: int, boxsize: float, *, dtype):
    """Return 1D physical wavenumbers for full/rfft axes."""
    dtype = np.dtype(dtype)
    fac = (2.0 * np.pi) / boxsize
    kx = fac * np.fft.fftfreq(ng, d=1.0 / ng).astype(dtype)            # (ng,)
    ky = kx                                                             # reuse
    kz = fac * np.fft.rfftfreq(ng, d=1.0 / ng).astype(dtype)           # (ng//2+1,)
    return kx, ky, kz


# ------------------------------------------------------------
# Power spectrum estimator (streaming / low-mem)
# ------------------------------------------------------------
class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_edges, ell_max=0, leg_fac=True, *, dtype=np.float32):
        """Memory-savvy P(k): precompute only 1D/2D k-structures and stream over kz-planes."""
        self.boxsize = np.asarray(boxsize, dtype=dtype)
        self.kbin_edges = np.asarray(kbin_edges, dtype=dtype)
        self.num_bins = int(self.kbin_edges.shape[0] - 1)
        self.vol = self.boxsize ** 3
        self.ng = int(ng)
        self.dtype = np.dtype(dtype)
        self.ell_max = int(ell_max)
        self.leg_fac = bool(leg_fac)

        # 1D k and squared
        kx, ky, kz = _k1d(self.ng, float(self.boxsize), dtype=self.dtype)
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
        self._leg_scalar = {ell: (2 * ell + 1) if self.leg_fac else 1.0
                            for ell in range(0, self.ell_max + 1, 2)}

    def _precompute_bin_stats(self):
        """Stream over kz planes to get Nk (degeneracy-aware) and mean-k per bin."""
        num_bins = self.num_bins
        edges = self.kbin_edges
        ng = self.ng
        kxy2 = self.kxy2
        kz2 = self.kz2
        dtype = self.dtype

        k_sum = np.zeros((num_bins,), dtype=dtype)
        n_sum = np.zeros((num_bins,), dtype=dtype)

        for p in range(kz2.shape[0]):
            # rfft degeneracy: p==0 and (if even) Nyquist plane have weight 1, else 2
            deg = np.array(1.0 if (p == 0 or ((ng % 2 == 0) and (p == ng // 2))) else 2.0, dtype=dtype)

            k2 = kxy2 + kz2[p]
            kmag = np.sqrt(np.maximum(k2, 0.0))
            # exclude DC at (0,0,0) which appears only in p==0 plane
            mask_dc = ~((p == 0) & (k2 == 0.0))

            kidx = np.digitize(kmag.ravel(), edges, right=True)  # 0..num_bins+1
            w = (deg * mask_dc.ravel().astype(dtype)).astype(dtype)

            # segment_sum equivalent with bincount
            k_tot = np.bincount(kidx, weights=(kmag.ravel() * w), minlength=num_bins + 2)[1:-1]
            N_tot = np.bincount(kidx, weights=w, minlength=num_bins + 2)[1:-1]

            k_sum += k_tot
            n_sum += N_tot

        k_mean = np.where(n_sum > 0, k_sum / n_sum, 0.0).astype(dtype)
        return k_mean, n_sum

    def __call__(self, fieldk1, fieldk2=None, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Compute auto/cross P(k) for multipole ell (0,2,4,...) and mu-range.
        Streams over kz-planes; avoids 3D kmag storage.
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        dtype = self.dtype
        ctype = np.complex64 if dtype == np.float32 else np.complex128
        fieldk1 = fieldk1.astype(ctype, copy=False)
        fieldk2 = fieldk2.astype(ctype, copy=False)

        ng = self.ng
        edges = self.kbin_edges
        num_bins = self.num_bins
        kxy2 = self.kxy2
        kz2 = self.kz2

        ell = int(ell)
        if ell % 2 != 0 or ell < 0 or ell > self.ell_max:
            raise ValueError(f"ell={ell} not supported; must be even in [0, {self.ell_max}]")
        leg_scalar = np.asarray(self._leg_scalar[ell], dtype=dtype)

        mu2_min = np.asarray(mu_min ** 2, dtype=dtype)
        mu2_max = np.asarray(mu_max ** 2, dtype=dtype)

        P_sum = np.zeros((num_bins,), dtype=dtype)
        N_sum = np.zeros((num_bins,), dtype=dtype)

        for p in range(kz2.shape[0]):
            deg = np.array(1.0 if (p == 0 or ((ng % 2 == 0) and (p == ng // 2))) else 2.0, dtype=dtype)
            fk1 = fieldk1[..., p]
            fk2 = fieldk2[..., p]
            cross = (fk1 * np.conj(fk2)).real.astype(dtype, copy=False)

            k2 = kxy2 + kz2[p]
            kmag = np.sqrt(np.maximum(k2, 0.0))
            # mu^2 = kz^2 / (kx^2 + ky^2 + kz^2); safe divide
            with np.errstate(divide='ignore', invalid='ignore'):
                mu2 = np.zeros_like(k2, dtype=dtype)
                np.divide(kz2[p], k2, out=mu2, where=(k2 > 0))

            # Legendre factor
            if ell == 0:
                leg = np.ones_like(mu2, dtype=dtype) * leg_scalar
            elif ell == 2:
                leg = leg_scalar * 0.5 * (3.0 * mu2 - 1.0)
            else:  # ell == 4 (if used)
                leg = leg_scalar * 0.125 * (35.0 * mu2 * mu2 - 30.0 * mu2 + 3.0)

            # mu-range mask + exclude DC at (0,0,0)
            mask_mu = (mu2 >= mu2_min) & (mu2 <= mu2_max)
            mask_dc = ~((p == 0) & (kmag == 0.0))
            mask = (mask_mu & mask_dc).astype(dtype, copy=False)

            kidx = np.digitize(kmag.ravel(), edges, right=True)
            wP = (deg * (cross * leg * mask).ravel()).astype(dtype, copy=False)
            wN = (deg * mask.ravel()).astype(dtype, copy=False)

            P_add = np.bincount(kidx, weights=wP, minlength=num_bins + 2)[1:-1]
            N_add = np.bincount(kidx, weights=wN, minlength=num_bins + 2)[1:-1]

            P_sum += P_add
            N_sum += N_add

        Pk_binned = np.where(N_sum > 0, P_sum / N_sum, 0.0)
        out = np.stack([self.k_mean, Pk_binned * self.vol, N_sum], axis=1).astype(dtype)
        return out

    def compute_mu_mean(self, mu_min=0.0, mu_max=1.0):
        """Return ⟨mu⟩ over mu in [mu_min, mu_max]; rfftn implies kz>=0 thus mu>=0 (no sign cancellation)."""
        dtype = self.dtype
        mu2_min = np.asarray(mu_min ** 2, dtype=dtype)
        mu2_max = np.asarray(mu_max ** 2, dtype=dtype)
        ng = self.ng
        kxy2 = self.kxy2
        kz2 = self.kz2

        num = np.array(0.0, dtype=dtype)
        den = np.array(0.0, dtype=dtype)

        for p in range(kz2.shape[0]):
            deg = np.array(1.0 if (p == 0 or ((ng % 2 == 0) and (p == ng // 2))) else 2.0, dtype=dtype)
            k2 = kxy2 + kz2[p]
            kmag = np.sqrt(np.maximum(k2, 0.0))
            with np.errstate(divide='ignore', invalid='ignore'):
                tmp = np.zeros_like(k2, dtype=dtype)
                np.divide(kz2[p], k2, out=tmp, where=(k2 > 0))
                mu = np.zeros_like(k2, dtype=dtype)
                np.sqrt(tmp, out=mu, where=(k2 > 0))
            mask = ((mu * mu >= mu2_min) & (mu * mu <= mu2_max)).astype(dtype, copy=False)
            mask = np.where((p == 0) & (kmag == 0.0), 0.0, mask)
            num += np.sum(mu * mask) * deg
            den += np.sum(mask) * deg

        return (num / den) if (den > 0) else np.array(0.0, dtype=dtype)


# ============================================================
# FFT-based P(k) & B(k)  (NumPy implementation, order_mode only)
# ============================================================
class Measure_spectra_FFT:
    """
    FFT-based estimators for P(k) and B(k) with three memory/speed modes:

      - mode="speed"   : Precompute and cache all bin-filtered real-space fields I_i (per field) and N_i once.
                         Fastest runtime (few FFTs), highest memory (≈ (#bins) x (#fields+1) x ng^3 x sizeof(float)).
      - mode="chunked" : Build I/N only for a small set of bins at a time (bin_chunk). Medium memory / medium speed.
      - mode="low_mem" : Build I/N on-the-fly for each bin/triangle. Lowest memory / slowest runtime.

    Triangle enumeration is controlled by `order_mode`:
      - "auto"   : Use the minimal independent set based on field equalities:
                   * all-same       -> i<=j<=k
                   * f1==f2!=f3     -> i<=j      (k free)
                   * f1==f3!=f2     -> i<=k      (j free)
                   * f2==f3!=f1     -> j<=k      (i free)
                   * all-different  -> i<=j<=k
      - "unique" : Always i<=j<=k (ignore additional symmetries).
      - "all"    : All ordered triangles (i,j,k) satisfying the triangle condition.
    """

    # ---------- low-level kernel (NumPy) ----------
    @staticmethod
    def _filter_kernel(fieldk, kx2, ky2, kz2, kmin2, kmax2):
        """Band-pass in k-space by scalar [kmin,kmax), then irfftn -> real (float)."""
        k2 = (kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :])
        mask = (kmin2 <= k2) & (k2 < kmax2)
        fr = np.fft.irfftn(fieldk * mask, norm='forward')
        return fr  # real float

    # ---------- public wrapper ----------
    def _filter_rfft_bandpass(self, fieldk, kmin, kmax):
        """Call the kernel with instance's k-squared 1D axes (no large captured arrays)."""
        kmin2 = np.asarray(kmin, dtype=self.dtype) ** 2
        kmax2 = np.asarray(kmax, dtype=self.dtype) ** 2
        out = self._filter_kernel(fieldk, self.kx2, self.ky2, self.kz2, kmin2, kmax2)
        return out.astype(self.dtype, copy=False)

    # ---------------------------------------------------------------------
    # ctor
    # ---------------------------------------------------------------------
    def __init__(self, boxsize: float, ng: int, kbin_edges, *,
                 dtype=np.float32, bispec: bool = True, open_triangle: bool = False):
        self.boxsize = np.asarray(boxsize, dtype=dtype)
        self.ng = int(ng)
        self.dtype = np.dtype(dtype)

        self.kbin_edges = np.asarray(kbin_edges, dtype=self.dtype)
        self.kbin_centers = 0.5 * (self.kbin_edges[1:] + self.kbin_edges[:-1])
        self.num_bins = int(self.kbin_edges.shape[0] - 1)
        self.vol = self.boxsize ** 3

        # 1D squared axes to avoid materializing kmag
        kx, ky, kz = _k1d(self.ng, float(self.boxsize), dtype=self.dtype)
        self.kx2 = (kx * kx).astype(self.dtype)        # (ng,)
        self.ky2 = (ky * ky).astype(self.dtype)        # (ng,)
        self.kz2 = (kz * kz).astype(self.dtype)        # (ng//2+1,)

        self.open_triangle = bool(open_triangle)

        # Lazy triangle caches keyed by mode: "unique", "all", "eq12", "eq13", "eq23"
        self._tri_cache: Dict[str, Optional[np.ndarray]] = {
            "unique": None, "all": None, "eq12": None, "eq13": None, "eq23": None
        }

        # Speed-mode full stacks and signatures
        self._I_full: Dict[str, Optional[np.ndarray]] = {"f1": None, "f2": None, "f3": None}
        self._N_full: Optional[np.ndarray] = None
        self._sig: Dict[str, Optional[Tuple[int, Tuple[int, ...], str]]] = {"f1": None, "f2": None, "f3": None}

        # Flag to skip prepare when compute_pk_bk() prewarmed
        self._skip_prepare: bool = False

    # ---------------------------------------------------------------------
    # triangle builders (lazy)
    # ---------------------------------------------------------------------
    def _triangle_ok(self, i: int, j: int, k: int) -> bool:
        """Triangle condition with bin centers (or open interval with edges)."""
        if self.open_triangle:
            k1min, k1max = float(self.kbin_edges[i]),     float(self.kbin_edges[i + 1])
            k2min, k2max = float(self.kbin_edges[j]),     float(self.kbin_edges[j + 1])
            k3min, k3max = float(self.kbin_edges[k]),     float(self.kbin_edges[k + 1])
            return ((k1max + k2max > k3min) and
                    (k2max + k3max > k1min) and
                    (k3max + k1max > k2min))
        else:
            k1, k2, k3 = float(self.kbin_centers[i]), float(self.kbin_centers[j]), float(self.kbin_centers[k])
            return (k3 >= abs(k1 - k2)) and (k3 <= (k1 + k2))

    def _build_tri_unique(self) -> np.ndarray:
        rows: List[Tuple[int, int, int]] = []
        for i in range(self.num_bins):
            for j in range(i, self.num_bins):
                for k in range(j, self.num_bins):
                    if self._triangle_ok(i, j, k):
                        rows.append((i, j, k))
        return np.asarray(rows, dtype=np.int32) if rows else np.zeros((0, 3), dtype=np.int32)

    def _build_tri_all(self) -> np.ndarray:
        rows: List[Tuple[int, int, int]] = []
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                for k in range(self.num_bins):
                    if self._triangle_ok(i, j, k):
                        rows.append((i, j, k))
        return np.asarray(rows, dtype=np.int32) if rows else np.zeros((0, 3), dtype=np.int32)

    def _build_tri_eq12(self) -> np.ndarray:
        """Triangles for symmetry f1==f2 (i<=j; k free)."""
        rows: List[Tuple[int, int, int]] = []
        for i in range(self.num_bins):
            for j in range(i, self.num_bins):          # i <= j
                for k in range(self.num_bins):         # k free
                    if self._triangle_ok(i, j, k):
                        rows.append((i, j, k))
        return np.asarray(rows, dtype=np.int32) if rows else np.zeros((0, 3), dtype=np.int32)

    def _build_tri_eq13(self) -> np.ndarray:
        """Triangles for symmetry f1==f3 (i<=k; j free)."""
        rows: List[Tuple[int, int, int]] = []
        for i in range(self.num_bins):
            for k in range(i, self.num_bins):          # i <= k
                for j in range(self.num_bins):         # j free
                    if self._triangle_ok(i, j, k):
                        rows.append((i, j, k))
        return np.asarray(rows, dtype=np.int32) if rows else np.zeros((0, 3), dtype=np.int32)

    def _build_tri_eq23(self) -> np.ndarray:
        """Triangles for symmetry f2==f3 (j<=k; i free)."""
        rows: List[Tuple[int, int, int]] = []
        for j in range(self.num_bins):
            for k in range(j, self.num_bins):          # j <= k
                for i in range(self.num_bins):         # i free
                    if self._triangle_ok(i, j, k):
                        rows.append((i, j, k))
        return np.asarray(rows, dtype=np.int32) if rows else np.zeros((0, 3), dtype=np.int32)

    def _get_triangles(self, order_mode: Literal["auto", "unique", "all"],
                       same12: bool, same13: bool, same23: bool) -> np.ndarray:
        """Return a triangle list consistent with order_mode and field equalities (lazy build+cache)."""
        if order_mode == "all":
            if self._tri_cache["all"] is None:
                self._tri_cache["all"] = self._build_tri_all()
            return self._tri_cache["all"]

        if order_mode == "unique":
            if self._tri_cache["unique"] is None:
                self._tri_cache["unique"] = self._build_tri_unique()
            return self._tri_cache["unique"]

        # order_mode == "auto"
        if same12 and not (same13 or same23):  # only f1==f2
            key = "eq12"
            if self._tri_cache[key] is None:
                self._tri_cache[key] = self._build_tri_eq12()
            return self._tri_cache[key]
        if same13 and not (same12 or same23):  # only f1==f3
            key = "eq13"
            if self._tri_cache[key] is None:
                self._tri_cache[key] = self._build_tri_eq13()
            return self._tri_cache[key]
        if same23 and not (same12 or same13):  # only f2==f3
            key = "eq23"
            if self._tri_cache[key] is None:
                self._tri_cache[key] = self._build_tri_eq23()
            return self._tri_cache[key]

        # all-same or all-different -> unique i<=j<=k
        if same12 and same23:
            if self._tri_cache["unique"] is None:
                self._tri_cache["unique"] = self._build_tri_unique()
            return self._tri_cache["unique"]
        else: 
            if self._tri_cache["all"] is None:
                self._tri_cache["all"] = self._build_tri_all()
            return self._tri_cache["all"]

    # ---------------------------------------------------------------------
    # helpers: build stacks for a list of bins
    # ---------------------------------------------------------------------
    def _build_stack_for_bins(self, fieldk, bins_1d: np.ndarray) -> np.ndarray:
        """Return stacked real-space filtered fields for bins in `bins_1d` (shape: (K, ng, ng, ng))."""
        ctype = np.complex64 if self.dtype == np.float32 else np.complex128
        fk = fieldk.astype(ctype, copy=False)
        out_list = []
        for i in bins_1d.tolist():
            kmin = self.kbin_edges[i]
            kmax = self.kbin_edges[i + 1]
            out_list.append(self._filter_rfft_bandpass(fk, kmin, kmax))
        return np.stack(out_list, axis=0).astype(self.dtype, copy=False)

    # ---------------------------------------------------------------------
    # cache preparation
    # ---------------------------------------------------------------------
    def _prepare_cache(self, fieldk1, fieldk2=None, fieldk3=None, *,
                       mode: Literal["speed", "chunked", "low_mem"] = "speed"):
        """Prepare bin-filtered stacks depending on mode."""
        if mode in ("chunked", "low_mem"):
            self._I_full = {"f1": None, "f2": None, "f3": None}
            self._N_full = None
            self._sig = {"f1": None, "f2": None, "f3": None}
            return

        # mode == "speed"
        bins_all = np.arange(self.num_bins, dtype=np.int32)

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
            onesk = np.ones_like(fieldk1)
            self._N_full = self._build_stack_for_bins(onesk, bins_all)

    def _sig_from(self, arr) -> Tuple[int, Tuple[int, ...], str]:
        return (id(arr), tuple(arr.shape), str(arr.dtype))

    # ---------------------------------------------------------------------
    # P(k): speed / chunked / low_mem
    # ---------------------------------------------------------------------
    def compute_pk(self, fieldk1, fieldk2=None, *,
                   mode: Literal["speed", "chunked", "low_mem"] = "speed",
                   bin_chunk: Optional[int] = None) -> np.ndarray:
        """Return (num_bins, 3): [k_center, P(k)*V, norm]."""
        if fieldk2 is None:
            fieldk2 = fieldk1

        norm = np.asarray(self.ng, dtype=self.dtype) ** 3

        if mode == "speed":
            if not self._skip_prepare:
                self._prepare_cache(fieldk1, fieldk2, None, mode="speed")
            I1 = self._I_full["f1"]
            I2 = I1 if (fieldk2 is fieldk1) else self._I_full["f2"]
            N = self._N_full
            num = np.sum(I1 * I2, axis=(1, 2, 3))
            den = np.sum(N * N, axis=(1, 2, 3))
            P = np.where(den > 0, num / den, 0.0)
            return np.stack([self.kbin_centers, (P * self.vol).real, den/norm], axis=1).astype(self.dtype)

        elif mode == "chunked":
            if bin_chunk is None or bin_chunk <= 0:
                raise ValueError("chunked PK requires bin_chunk > 0.")
            rows = []
            onesk = np.ones_like(fieldk1)
            for s in range(0, self.num_bins, int(bin_chunk)):
                e = min(s + int(bin_chunk), self.num_bins)
                bins = np.arange(s, e, dtype=np.int32)
                I1 = self._build_stack_for_bins(fieldk1, bins)
                I2 = I1 if (fieldk2 is fieldk1) else self._build_stack_for_bins(fieldk2, bins)
                N = self._build_stack_for_bins(onesk, bins)
                num = np.sum(I1 * I2, axis=(1, 2, 3))
                den = np.sum(N * N, axis=(1, 2, 3))
                P = np.where(den > 0, num / den, 0.0)
                rows.append(np.stack([self.kbin_centers[bins], (P * self.vol).real, den/norm], axis=1))
            return np.concatenate(rows, axis=0).astype(self.dtype)

        else:  # low_mem
            onesk = np.ones_like(fieldk1)
            out_rows = []
            for i in range(self.num_bins):
                kmin = self.kbin_edges[i]; kmax = self.kbin_edges[i + 1]
                I1 = self._filter_rfft_bandpass(fieldk1, kmin, kmax)
                I2 = I1 if (fieldk2 is fieldk1) else self._filter_rfft_bandpass(fieldk2, kmin, kmax)
                N = self._filter_rfft_bandpass(onesk, kmin, kmax)
                num = np.sum(I1 * I2)
                den = np.sum(N * N)
                P = (num / den) if (den > 0) else 0.0
                out_rows.append(np.array([self.kbin_centers[i], (P * self.vol).real, den/norm], dtype=self.dtype))
            return np.stack(out_rows, axis=0)

    # ---------------------------------------------------------------------
    # B(k): speed / chunked / low_mem  (order_mode controls enumeration)
    # ---------------------------------------------------------------------
    def compute_bk(self, fieldk1, fieldk2=None, fieldk3=None, *,
                   mode: Literal["speed", "chunked", "low_mem"] = "speed",
                   bin_chunk: Optional[int] = None,
                   order_mode: Literal["auto", "unique", "all"] = "auto") -> np.ndarray:
        """
        Return (nrows, 5): [k1, k2, k3, B(k)*V^2, norm].

        order_mode:
          - "auto"   : minimal independent set based on field equalities (see class docstring).
          - "unique" : always i<=j<=k.
          - "all"    : all ordered triangles.
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        if fieldk3 is None:
            fieldk3 = fieldk2

        same12 = (fieldk2 is fieldk1)
        same13 = (fieldk3 is fieldk1)
        same23 = (fieldk3 is fieldk2)

        tri = self._get_triangles(order_mode, same12, same13, same23)
        ntri = int(tri.shape[0])
        onesk = np.ones_like(fieldk1)

        tag2 = "f1" if same12 else "f2"
        tag3 = "f1" if same13 else ("f2" if same23 else "f3")

        norm = np.asarray(self.ng, dtype=self.dtype) ** 3

        if mode == "speed":
            if not self._skip_prepare:
                self._prepare_cache(fieldk1, fieldk2, fieldk3, mode="speed")
            I1s = self._I_full["f1"]
            I2s = I1s if tag2 == "f1" else self._I_full["f2"]
            I3s = I1s if tag3 == "f1" else (self._I_full["f2"] if tag3 == "f2" else self._I_full["f3"])
            Ns = self._N_full

            if ntri == 0:
                return np.zeros((0, 5), dtype=self.dtype)

            out_rows = []
            for t in range(ntri):
                i, j, k = int(tri[t, 0]), int(tri[t, 1]), int(tri[t, 2])
                num = np.sum(I1s[i] * I2s[j] * I3s[k])
                den = np.sum(Ns[i] * Ns[j] * Ns[k])
                Bv = (num / den) if (den > 0) else 0.0
                out_rows.append(np.array([self.kbin_centers[i],
                                          self.kbin_centers[j],
                                          self.kbin_centers[k],
                                          (Bv * (self.vol ** 2)).real,
                                          den / norm], dtype=self.dtype))
            return np.stack(out_rows, axis=0)

        elif mode == "chunked":
            if bin_chunk is None or bin_chunk <= 0:
                raise ValueError("chunked BK requires bin_chunk > 0.")

            B = int(self.num_bins)
            C = int(bin_chunk)
            nch = (B + C - 1) // C  # number of chunks

            # Per-chunk tiny caches so we reuse I/N for the same chunk across triplets
            cache_I = {"f1": {}, "f2": {}, "f3": {}}  # dict[ch] -> (len_chunk, ng, ng, ng)
            cache_N = {}                              # dict[ch] -> (len_chunk, ng, ng, ng)

            def bins_of(ch):
                s = ch * C
                e = min(s + C, B)
                return s, e, np.arange(s, e, dtype=np.int32)

            def get_I(tag, ch):
                if ch in cache_I[tag]:
                    return cache_I[tag][ch]
                s, e, bins = bins_of(ch)
                if tag == "f1":
                    I = self._build_stack_for_bins(fieldk1, bins)
                elif tag == "f2":
                    I = self._build_stack_for_bins(fieldk2, bins)
                else:
                    I = self._build_stack_for_bins(fieldk3, bins)
                cache_I[tag][ch] = I
                return I

            def get_N(ch):
                if ch in cache_N:
                    return cache_N[ch]
                s, e, bins = bins_of(ch)
                N = self._build_stack_for_bins(onesk, bins)
                cache_N[ch] = N
                return N

            if ntri == 0:
                return np.zeros((0, 5), dtype=self.dtype)

            rows_all = []
            idx_all = []

            # Choose minimal (a,b,c) loop ranges based on triangle mode
            def ab_c_ranges():
                # yields (a,b,c) triplets to scan
                if order_mode == "all":
                    for a in range(nch):
                        for b in range(nch):
                            for c in range(nch):
                                yield a, b, c
                elif order_mode == "unique" or (order_mode == "auto" and (not (same12 or same13 or same23))):
                    # unique i<=j<=k
                    for a in range(nch):
                        for b in range(a, nch):
                            for c in range(b, nch):
                                yield a, b, c
                elif order_mode == "auto" and same12 and not (same13 or same23):
                    # eq12: i<=j, k free
                    for a in range(nch):
                        for b in range(a, nch):
                            for c in range(nch):
                                yield a, b, c
                elif order_mode == "auto" and same13 and not (same12 or same23):
                    # eq13: i<=k, b free
                    for a in range(nch):
                        for c in range(a, nch):
                            for b in range(nch):
                                yield a, b, c
                elif order_mode == "auto" and same23 and not (same12 or same13):
                    # eq23: j<=k, a free
                    for b in range(nch):
                        for c in range(b, nch):
                            for a in range(nch):
                                yield a, b, c
                else:
                    # all-same -> unique; safety fallback
                    for a in range(nch):
                        for b in range(a, nch):
                            for c in range(b, nch):
                                yield a, b, c

            # Scan chosen chunk triplets
            for a, b, c in ab_c_ranges():
                sA, eA, binsA = bins_of(a)
                sB, eB, binsB = bins_of(b)
                sC, eC, binsC = bins_of(c)

                # Pick subset of triangles (i∈A, j∈B, k∈C)
                i_in = (tri[:, 0] >= sA) & (tri[:, 0] < eA)
                j_in = (tri[:, 1] >= sB) & (tri[:, 1] < eB)
                k_in = (tri[:, 2] >= sC) & (tri[:, 2] < eC)
                idx = np.where(i_in & j_in & k_in)[0]
                if idx.size == 0:
                    continue

                tri_sub = tri[idx]          # (nsub, 3)
                li = tri_sub[:, 0] - sA     # local indices
                lj = tri_sub[:, 1] - sB
                lk = tri_sub[:, 2] - sC

                I1 = get_I("f1", a)
                I2 = I1 if (same12 and a == b) else get_I("f2", b)
                if same13 and (a == c):
                    I3 = I1
                elif same23 and (b == c):
                    I3 = I2
                else:
                    I3 = get_I("f3", c)

                NA = get_N(a); NB = get_N(b); NC = get_N(c)

                rows = []
                for t in range(idx.size):
                    li_, lj_, lk_ = int(li[t]), int(lj[t]), int(lk[t])
                    num = np.sum(I1[li_] * I2[lj_] * I3[lk_])
                    den = np.sum(NA[li_] * NB[lj_] * NC[lk_])
                    Bv = (num / den) if (den > 0) else 0.0
                    gi = sA + li_; gj = sB + lj_; gk = sC + lk_
                    rows.append(np.array([self.kbin_centers[gi],
                                          self.kbin_centers[gj],
                                          self.kbin_centers[gk],
                                          (Bv * (self.vol ** 2)).real,
                                          den / norm], dtype=self.dtype))
                rows_all.append(np.stack(rows, axis=0))
                idx_all.append(idx)

            if not rows_all:
                return np.zeros((0, 5), dtype=self.dtype)
            all_rows = np.concatenate(rows_all, axis=0)
            all_idx = np.concatenate(idx_all, axis=0)
            order = np.argsort(all_idx)
            return all_rows[order]

        else:  # low_mem
            if ntri == 0:
                return np.zeros((0, 5), dtype=self.dtype)

            rows = []
            for t in range(ntri):
                i, j, k = int(tri[t, 0]), int(tri[t, 1]), int(tri[t, 2])

                kmin_i, kmax_i = self.kbin_edges[i], self.kbin_edges[i + 1]
                kmin_j, kmax_j = self.kbin_edges[j], self.kbin_edges[j + 1]
                kmin_k, kmax_k = self.kbin_edges[k], self.kbin_edges[k + 1]

                I_i = self._filter_rfft_bandpass(fieldk1, kmin_i, kmax_i)
                I_j = I_i if (same12 and i == j) else self._filter_rfft_bandpass(fieldk2, kmin_j, kmax_j)
                if same13 and (i == k):
                    I_k = I_i
                elif same23 and (j == k):
                    I_k = I_j
                else:
                    I_k = self._filter_rfft_bandpass(fieldk3, kmin_k, kmax_k)

                N_i = self._filter_rfft_bandpass(np.ones_like(fieldk1), kmin_i, kmax_i)
                N_j = N_i if (i == j) else self._filter_rfft_bandpass(np.ones_like(fieldk1), kmin_j, kmax_j)
                if i == k:
                    N_k = N_i
                elif j == k:
                    N_k = N_j
                else:
                    N_k = self._filter_rfft_bandpass(np.ones_like(fieldk1), kmin_k, kmax_k)

                num = np.sum(I_i * I_j * I_k)
                den = np.sum(N_i * N_j * N_k)
                Bv = (num / den) if (den > 0) else 0.0

                rows.append(np.array([self.kbin_centers[i],
                                      self.kbin_centers[j],
                                      self.kbin_centers[k],
                                      (Bv * (self.vol ** 2)).real,
                                      den / norm], dtype=self.dtype))
            return np.stack(rows, axis=0)

    # ---------------------------------------------------------------------
    # Convenience: compute both with a shared strategy
    # ---------------------------------------------------------------------
    def compute_pk_bk(self, fieldk1, fieldk2=None, fieldk3=None, *,
                      mode: Literal["speed", "chunked", "low_mem"] = "speed",
                      bin_chunk: Optional[int] = None,
                      order_mode: Literal["auto", "unique", "all"] = "auto"):
        """Compute P(k) and B(k) with consistent mode/chunk settings."""
        if mode == "speed":
            # Prewarm once, then tell children to skip their internal prepare in this call
            self._prepare_cache(fieldk1, fieldk2, fieldk3, mode="speed")
            self._skip_prepare = True
        try:
            pk = self.compute_pk(fieldk1, fieldk2, mode=mode, bin_chunk=bin_chunk)
            bk = self.compute_bk(fieldk1, fieldk2, fieldk3,
                                 mode=mode, bin_chunk=bin_chunk,
                                 order_mode=order_mode)
        finally:
            self._skip_prepare = False
        return pk, bk
