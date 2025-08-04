#!/usr/bin/env python3

from __future__ import annotations
from typing import Tuple

import numpy as np


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def rfftn_kvec(shape: Tuple[int, int, int], boxsize: float, dtype=float):
    """
    Generate the Cartesian wave-vector grid that matches numpy.fft.rfftn.
    """
    spacing = boxsize / (2.0 * np.pi) / shape[-1]
    freqs = [np.fft.fftfreq(n, d=spacing) for n in shape[:-1]]
    freqs.append(np.fft.rfftfreq(shape[-1], d=spacing))
    kvec_grid = np.meshgrid(*freqs, indexing="ij")
    return np.stack(kvec_grid, axis=0).astype(dtype)


# ------------------------------------------------------------
# P(k) estimator
# ------------------------------------------------------------
class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_edges, ell_max: int = 0, leg_fac: bool = True):
        self.boxsize = boxsize
        self.ng = ng
        self.vol = boxsize**3
        self.kbin_edges = np.asarray(kbin_edges)
        self.num_bins = len(self.kbin_edges) - 1

        # --- grid bookkeeping ------------------------------------------------
        kvec = rfftn_kvec((ng,) * 3, boxsize)
        k2 = (kvec**2).sum(axis=0)
        kz2 = kvec[2] ** 2
        kmag = np.sqrt(k2)

        self.kmag_1d = kmag.ravel()
        self.kidx = np.digitize(self.kmag_1d, self.kbin_edges, right=True)

        Nk = np.full_like(k2, 2, dtype=np.int32)         # Hermitian symmetry
        Nk[..., 0] = 1
        if ng % 2 == 0:
            Nk[..., -1] = 1
        self.Nk_1d = Nk.ravel()

        # mean k in each bin and mode counts ----------------------------------
        k_tot = np.bincount(self.kidx, 
                            weights=self.kmag_1d * self.Nk_1d, 
                            minlength=self.num_bins + 2)[1:-1]
        N_tot = np.bincount(self.kidx, 
                            weights=self.Nk_1d, 
                            minlength=self.num_bins + 2)[1:-1]

        self.k_mean = k_tot / np.maximum(N_tot, 1)
        self.Nk = N_tot

        # mu2 grid -------------------------------------------------------------
        mu2 = np.where(k2 == 0.0, 0.0, kz2 / k2)
        self.mu2_1d = mu2.ravel()

        # pre-compute Legendre factors ---------------------------------------
        legs = []
        for ell in range(0, ell_max + 1, 2):
            fac = (2 * ell + 1) if leg_fac else 1.0
            if ell == 0:
                legs.append(fac * np.ones_like(self.mu2_1d))
            elif ell == 2:
                legs.append(fac * 0.5 * (3 * self.mu2_1d - 1))
            elif ell == 4:
                legs.append(
                    fac * 0.125 * (35 * self.mu2_1d**2 - 30 * self.mu2_1d + 3)
                )
        self.legendre_stack = np.stack(legs, axis=0)

    def compute_mu_mean(self, mu_min=0.0, mu_max=1.0):
        """
        Computes the mean mu value for the given mu range.
        """
        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        mu_mean = np.sum(self.mu2_1d * mask * self.Nk_1d) / np.sum(mask * self.Nk_1d)
        return mu_mean

    # ------------------------------------------------------------------------
    def _compute(self, Pk1d, mask):
        w_P = (Pk1d * mask * self.Nk_1d).real
        w_N = self.Nk_1d * mask

        P_sum = np.bincount(self.kidx, 
                            weights=w_P, 
                            minlength=self.num_bins + 2)[1:-1]
        N_sum = np.bincount(self.kidx, 
                            weights=w_N, 
                            minlength=self.num_bins + 2)[1:-1]
        
        Pk_binned = np.where(N_sum > 0, P_sum / N_sum, 0.0)

        return np.column_stack((self.k_mean, Pk_binned * self.vol, N_sum))

    # ------------------------------------------------------------------------
    def __call__(
        self, fieldk1, fieldk2=None, *, ell: int = 0, mu_min: float = 0.0, mu_max: float = 1.0
    ):
        if fieldk2 is None:
            fieldk2 = fieldk1

        Pk1d = fieldk1.ravel() * np.conjugate(fieldk2.ravel())
        Pk1d *= self.legendre_stack[ell // 2]
        Pk1d[0] = 0.0  # remove DC

        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        return self._compute(Pk1d, mask)


# ------------------------------------------------------------
# P(k) & B(k) estimator (FFT method)
# ------------------------------------------------------------
import numpy as np


class Measure_spectra_FFT:
    """
    FFT-based power- and bispectrum estimator.

    """

    def __init__(self, boxsize, ng, kbin_edges,
                 bispec: bool = True, open_triangle: bool = False):
        self.boxsize = float(boxsize)
        self.ng = int(ng)
        self.vol = self.boxsize ** 3

        self.kbin_edges = np.asarray(kbin_edges, dtype=float)
        self.kbin_centers = 0.5 * (self.kbin_edges[1:] + self.kbin_edges[:-1])
        self.num_bins = len(self.kbin_edges) - 1

        kvec = rfftn_kvec((ng,) * 3, self.boxsize)
        self.kmag = np.sqrt((kvec**2).sum(axis=0))

        # triangle list -----------------------------------------------------
        if bispec:
            tri = []
            kmin_bins, kmax_bins = self.kbin_edges[:-1], self.kbin_edges[1:]
            for i in range(self.num_bins):
                for j in range(i, self.num_bins):
                    for k in range(j, self.num_bins):      # k1 ≤ k2 ≤ k3
                        k1c, k2c, k3c = (self.kbin_centers[i],
                                         self.kbin_centers[j],
                                         self.kbin_centers[k])
                        if open_triangle:
                            cond = (
                                kmax_bins[i] + kmax_bins[j] > kmin_bins[k] and
                                kmax_bins[j] + kmax_bins[k] > kmin_bins[i] and
                                kmax_bins[k] + kmax_bins[i] > kmin_bins[j]
                            )
                        else:
                            cond = (k3c >= abs(k1c - k2c)) and (k3c <= (k1c + k2c))
                        if cond:
                            tri.append((i, j, k))
            self.triangle_idxs = np.asarray(tri, dtype=np.int32)
        else:
            self.triangle_idxs = np.empty((0, 3), dtype=np.int32)

    # --------------------------------------------------------------------- #
    # internals
    # --------------------------------------------------------------------- #
    def _filter_field(self, fieldk, kmin, kmax):
        """
        Select Fourier modes with kmin ≤ |k| < kmax and IRFFT to real space.
        """
        mask = (kmin <= self.kmag) & (self.kmag < kmax)
        return np.fft.irfftn(fieldk * mask) * (self.ng ** 3)

    # --------------------------------------------------------------------- #
    # user API
    # --------------------------------------------------------------------- #
    def compute_pk_bk(self, fieldk):
        """Return (P(k) table, B(k) table)"""
        return self.compute_pk(fieldk), self.compute_bk(fieldk)

    # --------------------------------------------------------------------- #
    def compute_pk(self, fieldk1, fieldk2=None):
        """
        FFT-shell power spectrum.

        returns (num_bins, 3) = [k_center,  P(k)*V,  normalisation]
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        onesk = np.ones_like(fieldk1)

        out = np.empty((self.num_bins, 2))

        for i in range(self.num_bins):
            kmin, kmax = self.kbin_edges[i], self.kbin_edges[i + 1]

            I1r = self._filter_field(fieldk1, kmin, kmax)
            I2r = self._filter_field(fieldk2, kmin, kmax)
            Nr = self._filter_field(onesk, kmin, kmax)

            num = np.sum(I1r * I2r)
            den = np.sum(Nr * Nr)
            out[i] = ((num / den * self.vol).real if den > 0.0 else 0.0, den)

        return np.column_stack((self.kbin_centers, out))

    # --------------------------------------------------------------------- #
    def compute_bk(self, fieldk1, fieldk2=None, fieldk3=None, *,
                   batch_size: int = 16):
        """
        Bispectrum with small per-shell cache (IRFFT once per shell).

        returns (n_tri, 5) =
        [k1, k2, k3,  B(k1,k2,k3)*V^2,  normalisation]
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        if fieldk3 is None:
            fieldk3 = fieldk2
        onesk = np.ones_like(fieldk1)

        # FIFO caches -------------------------------------------------------
        cache_f1, cache_f2, cache_f3, cache_N = {}, {}, {}, {}
        max_keep = 3   # <=3 shells kept per cache -> memory O( few × grid )

        def _shell(fieldk, idx, cache):
            """Fetch real-space shell from cache or compute it."""
            if idx not in cache:
                kmin = self.kbin_edges[idx]
                kmax = self.kbin_edges[idx + 1]
                cache[idx] = self._filter_field(fieldk, kmin, kmax)
                if len(cache) > max_keep:
                    cache.pop(next(iter(cache)))
            return cache[idx]

        n_tri = len(self.triangle_idxs)
        bk_tab = np.empty((n_tri, 2))

        # small outer batch to reduce Python overhead -----------------------
        for start in range(0, n_tri, batch_size):
            end = min(start + batch_size, n_tri)
            for local, (i, j, k) in enumerate(self.triangle_idxs[start:end]):
                I1 = _shell(fieldk1, i, cache_f1)
                I2 = _shell(fieldk2, j, cache_f2)
                I3 = _shell(fieldk3, k, cache_f3)

                N1 = _shell(onesk, i, cache_N)
                N2 = _shell(onesk, j, cache_N)
                N3 = _shell(onesk, k, cache_N)

                num = np.sum(I1 * I2 * I3)
                den = np.sum(N1 * N2 * N3)
                bk_tab[start + local] = (
                    (num / den * self.vol ** 2).real if den > 0.0 else 0.0,
                    den,
                )

        k123 = np.column_stack(
            (
                self.kbin_centers[self.triangle_idxs[:, 0]],
                self.kbin_centers[self.triangle_idxs[:, 1]],
                self.kbin_centers[self.triangle_idxs[:, 2]],
            )
        )
        return np.hstack((k123, bk_tab))







