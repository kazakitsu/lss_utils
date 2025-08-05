#!/usr/bin/env python3

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

@partial(jit, static_argnames=('shape', 'dtype'))
def rfftn_kvec(shape, boxsize, dtype=float):
    """
    Generate wavevectors for `jax.numpy.fft.rfftn`.
    JAX version using jnp.meshgrid.
    """
    # full FFT frequencies for all but last axis

    spacing = boxsize / (2.*jnp.pi) / shape[-1]
    # Create 1D frequency arrays for each dimension.
    freqs = [jnp.fft.fftfreq(n, d=spacing) for n in shape[:-1]]
    freqs.append(jnp.fft.rfftfreq(shape[-1], d=spacing))

    # Use jnp.meshgrid to create the coordinate grid.
    kvec_grid = jnp.meshgrid(*freqs, indexing='ij')
    
    # Stack the coordinate arrays to get the final (D, N1, N2, ...) shape.
    kvec = jnp.stack(kvec_grid, axis=0)
    return kvec.astype(dtype)

class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_edges, ell_max=0, leg_fac=True):
        self.boxsize = boxsize
        self.kbin_edges = jnp.array(kbin_edges)
        self.num_bins = self.kbin_edges.shape[0] - 1
        self.vol = boxsize**3
        self.ng = ng

        # precompute k-vector grid
        kvec = rfftn_kvec((ng,)*3, boxsize)
        k2 = (kvec**2).sum(axis=0)
        k_is_zero = (k2 == 0.0)
        
        self.kmag_1d = jnp.sqrt(k2).ravel()

        # per-mode digitized k-index and counts
        self.kidx = jnp.digitize(self.kmag_1d, self.kbin_edges, right=True)

        Nk = jnp.full_like(k2, 2, dtype=jnp.int32)
        Nk = Nk.at[...,0].set(1)
        if self.ng % 2 == 0:
            Nk = Nk.at[..., -1].set(1)
        self.Nk_1d = Nk.ravel()

        # mean k and total counts per bin
        k_tot = jnp.bincount(self.kidx, 
                             weights=self.kmag_1d * self.Nk_1d,
                             length=self.num_bins+2)[1:-1]
        N_tot = jnp.bincount(self.kidx, 
                             weights=self.Nk_1d,
                             length=self.num_bins+2)[1:-1]
        
        self.k_mean = k_tot / jnp.maximum(N_tot, 1)
        self.Nk = N_tot

        # precompute mu^2 per mode
        mu2 = jnp.where(k_is_zero, 0.0, kvec[2]**2 / k2)
        self.mu2_1d = mu2.ravel()

        # build Legendre factors for monopole/quadrupole/etc.
        self.leg_fac = leg_fac
        legs = []
        for ell in range(0, ell_max+1, 2):
            fac = (2*ell + 1) if leg_fac else 1
            if ell == 0:
                legs.append(fac * jnp.ones_like(self.mu2_1d))
            elif ell == 2:
                legs.append(fac * 0.5 * (3*self.mu2_1d - 1))
            elif ell == 4:
                legs.append(fac * 0.125 * (35*self.mu2_1d**2 - 30*self.mu2_1d + 3))
        self.legendre_stack = jnp.stack(legs, axis=0)

    def compute_mu_mean(self, mu_min=0.0, mu_max=1.0):
        """
        Computes the mean mu value for the given mu range.
        """
        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        mu_mean = jnp.sum(self.mu2_1d * mask * self.Nk_1d) / jnp.sum(mask * self.Nk_1d)
        return mu_mean

    def _compute(self, Pk1d, mask):
        # apply mask and weights
        w_P = (Pk1d * mask * self.Nk_1d).real
        w_N = self.Nk_1d * mask

        P_sum = jnp.bincount(self.kidx, weights=w_P, length=self.num_bins + 2)[1:-1]
        N_sum = jnp.bincount(self.kidx, weights=w_N, length=self.num_bins + 2)[1:-1]

        Pk_binned = jnp.where(N_sum > 0, P_sum / N_sum, 0.0)

        return jnp.array([self.k_mean, Pk_binned * self.vol, N_sum]).T

    @partial(jit, static_argnames=('self', 'ell'))
    def __call__(self, fieldk1, fieldk2=None, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Computes auto or cross-power spectrum for a given multipole,
        with an optional mu range.
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        
        Pk1d = fieldk1.ravel() * jnp.conj(fieldk2.ravel())
        
        # Apply Legendre polynomial for multipole selection
        Pk1d = Pk1d * self.legendre_stack[ell // 2]
        Pk1d = Pk1d.at[0].set(0.0) # Exclude DC mode

        # Create mask based on the mu range
        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        
        return self._compute(Pk1d, mask)


class Measure_spectra_FFT:
    def __init__(self, boxsize, ng, kbin_edges, bispec=True, open_triangle=False):
        self.boxsize = boxsize
        self.kbin_edges = jnp.array(kbin_edges)
        self.vol     = self.boxsize**3
        self.ng      = ng
        kvec         = rfftn_kvec((ng,)*3, self.boxsize)
        self.kmag = jnp.sqrt(jnp.sum(kvec**2, axis=0))
        self.kbin_centers = 0.5 * (self.kbin_edges[1:] + self.kbin_edges[:-1])
        self.num_bins = self.kbin_edges.shape[0] - 1

        if bispec:
            triangle_idx_list = []

            kmin_bins = self.kbin_edges[:-1]
            kmax_bins = self.kbin_edges[1:]

            for i in range(self.num_bins):
                for j in range(i, self.num_bins):
                    for k in range(j, self.num_bins):   ### k1 <= k2 <= k3
                        k1 = float(self.kbin_centers[i])
                        k2 = float(self.kbin_centers[j])
                        k3 = float(self.kbin_centers[k])

                        k1min = float(kmin_bins[i])
                        k1max = float(kmax_bins[i])
                        k2min = float(kmin_bins[j])
                        k2max = float(kmax_bins[j])
                        k3min = float(kmin_bins[k])
                        k3max = float(kmax_bins[k])
                        
                        if open_triangle:
                            if ((k1max + k2max > k3min) and 
                                (k2max + k3max > k1min) and 
                                (k3max + k1max > k2min)):
                                triangle_idx_list.append((i, j, k))
                        else:
                            if k3 >= abs(k1 - k2) and k3 <= (k1 + k2):
                                triangle_idx_list.append((i, j, k))
            self.triangle_idxs = jnp.array(triangle_idx_list)

    @partial(jit, static_argnames=('self',))
    def _filter_field(self, fieldk, kmin, kmax):
        """Helper to filter a field in k-space and transform to real space."""
        mask = (kmin <= self.kmag) & (self.kmag < kmax)
        return jnp.fft.irfftn(fieldk * mask) * (self.ng**3)

    @partial(jit, static_argnames=('self',))
    def compute_pk_bk(self, fieldk):
        """Computes both P(k) and B(k) for a given field."""
        pk_results = self.compute_pk(fieldk)
        bk_results = self.compute_bk(fieldk)
        return pk_results, bk_results

    @partial(jit, static_argnames=('self', 'batch_size'))
    def compute_bk(self, fieldk1, fieldk2=None, fieldk3=None, batch_size=16):
        """
        Computes the auto or cross-bispectrum using a memory-safe batching method.
        """
        if fieldk2 is None: fieldk2 = fieldk1
        if fieldk3 is None: fieldk3 = fieldk2
        onesk = jnp.ones_like(fieldk1)

        def compute_single_triangle(indices):
            """Computes Bk for one (i, j, k) triplet of k-bins."""
            i, j, k = indices
            kmin_i, kmax_i = self.kbin_edges[i], self.kbin_edges[i+1]
            kmin_j, kmax_j = self.kbin_edges[j], self.kbin_edges[j+1]
            kmin_k, kmax_k = self.kbin_edges[k], self.kbin_edges[k+1]

            I1r = self._filter_field(fieldk1, kmin_i, kmax_i)
            I2r = self._filter_field(fieldk2, kmin_j, kmax_j)
            I3r = self._filter_field(fieldk3, kmin_k, kmax_k)
            bispec_num = jnp.sum(I1r * I2r * I3r)

            N1r = self._filter_field(onesk, kmin_i, kmax_i)
            N2r = self._filter_field(onesk, kmin_j, kmax_j)
            N3r = self._filter_field(onesk, kmin_k, kmax_k)
            norm = jnp.sum(N1r * N2r * N3r)
            
            bispec = jnp.where(norm > 0, bispec_num / norm, 0.0)
            # Return a single array for cleaner vmap handling
            return jnp.array([(bispec * self.vol**2).real, norm])

        # Use lax.fori_loop for JIT-compatible batching
        n_triangles = self.triangle_idxs.shape[0]
        n_batches = (n_triangles + batch_size - 1) // batch_size
        
        def body_fn(i, results_carry):
            start = i * batch_size
            # Use dynamic_slice for reading indices
            batch_idxs = lax.dynamic_slice_in_dim(self.triangle_idxs, start, batch_size, axis=0)
            # vmap computes results for the batch
            batch_results = vmap(compute_single_triangle)(batch_idxs)
            # *** CORRECTED LINE ***
            # Use lax.dynamic_update_slice for writing results at a dynamic start index
            return lax.dynamic_update_slice(results_carry, batch_results, (start, 0))

        initial_results = jnp.zeros((n_triangles, 2))
        bk_tab = lax.fori_loop(0, n_batches, body_fn, initial_results)
        
        k123 = jnp.array([self.kbin_centers[self.triangle_idxs[:,0]], 
                          self.kbin_centers[self.triangle_idxs[:,1]], 
                          self.kbin_centers[self.triangle_idxs[:,2]]]).T
        
        return jnp.hstack([k123, bk_tab])

    @partial(jit, static_argnames=('self',))
    def compute_pk(self, fieldk1, fieldk2=None):
        """Computes the auto or cross-power spectrum using the FFT method."""
        if fieldk2 is None: fieldk2 = fieldk1
        onesk = jnp.ones_like(fieldk1)

        def compute_power_for_bin(i):
            kmin, kmax = self.kbin_edges[i], self.kbin_edges[i+1]
            
            I1r = self._filter_field(fieldk1, kmin, kmax)
            I2r = self._filter_field(fieldk2, kmin, kmax)
            Nr  = self._filter_field(onesk, kmin, kmax)
            
            power_numerator = jnp.sum(I1r * I2r)
            normalization = jnp.sum(Nr * Nr)
            
            power = jnp.where(normalization > 0, power_numerator / normalization, 0.0)
            # Return a single array for cleaner vmap handling
            return jnp.array([(power * self.vol).real, normalization])

        # vmap over all bins for efficient computation
        results = vmap(compute_power_for_bin)(jnp.arange(self.num_bins))
        return jnp.hstack([self.kbin_centers[:, None], results])
