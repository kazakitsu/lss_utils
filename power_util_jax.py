#!/usr/bin/env python3

import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

def rfftn_kvec(shape, boxsize, dtype=float):
    """
    Generate wavevectors for `jax.numpy.fft.rfftn`
    """
    kvec = [jnp.fft.fftfreq(n, d=1./shape[-1]).astype(dtype) for n in shape[:-1]]
    kvec.append(jnp.fft.rfftfreq(shape[-1], d=1./shape[-1]).astype(dtype))
    kvec = jnp.meshgrid(*kvec, indexing='ij')
    kvec = jnp.stack(kvec, axis=0)
    return kvec * (2 * jnp.pi / boxsize)

class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_1d, ell_max=0, leg_fac=True):
        self.boxsize = boxsize
        self.kbin_1d = jnp.array(kbin_1d)
        self.num_bins = self.kbin_1d.shape[0] - 1
        self.vol = boxsize**3
        self.ng = ng

        # precompute k-vector grid
        kvec = rfftn_kvec([ng]*3, boxsize)
        k2 = (kvec**2).sum(axis=0)
        self.kmag_1d = jnp.sqrt(k2).ravel()

        # per-mode digitized k-index and counts
        self.kidx = jnp.digitize(self.kmag_1d, self.kbin_1d, right=True)
        Nk = jnp.full_like(k2, 2, dtype=jnp.int32)
        Nk = Nk.at[...,0].set(1)
        if k2.shape[-1] % 2 == 0:
            Nk = Nk.at[..., -1].set(1)
        self.Nk_1d = Nk.ravel()

        # mean k and total counts per bin
        k_tot = jnp.bincount(self.kidx, weights=self.kmag_1d * self.Nk_1d,
                             length=self.num_bins+1)[1:]
        N_tot = jnp.bincount(self.kidx, weights=self.Nk_1d,
                             length=self.num_bins+1)[1:]
        self.k_mean = k_tot / N_tot
        self.Nk = N_tot

        # precompute mu^2 per mode
        mu2 = (kvec[2]**2) / k2
        mu2 = mu2.at[0,0,0].set(0.0)
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

    def _compute(self, Pk1d, mask, kbin_edges):
        # apply mask and weights
        k_arr  = self.kmag_1d
        mu2    = self.mu2_1d
        Nk_arr = self.Nk_1d

        # select modes
        sel = mask
        k_sel   = k_arr[sel]
        P_sel   = (Pk1d * Nk_arr)[sel]
        Nk_sel  = Nk_arr[sel]

        # digitize into k-bins
        kidx = jnp.digitize(k_sel, kbin_edges, right=True)

        # accumulate
        k_sum = jnp.bincount(kidx, weights=k_sel * Nk_sel,
                             length=kbin_edges.shape[0])[1:]
        P_sum = jnp.bincount(kidx, weights=P_sel.real,
                             length=kbin_edges.shape[0])[1:]
        N_sum = jnp.bincount(kidx, weights=Nk_sel,
                             length=kbin_edges.shape[0])[1:]

        # normalize
        k_mean = k_sum / N_sum
        Pk_out = P_sum / N_sum * self.vol
        return k_mean, Pk_out, N_sum

    @partial(jit, static_argnames=('self','ell','kbin_1d','mu_min','mu_max'))
    def pk_auto(self, fieldk, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Auto-spectrum multipole with optional mu-range.
        If mu_min<0 or mu_max>1, defaults to full range.
        """
        Pk1d = (fieldk.ravel() * fieldk.ravel().conj())
        leg = self.legendre_stack[ell//2]
        Pk1d = Pk1d * leg * self.Nk_1d
        Pk1d = Pk1d.at[0].set(0.0)

        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        return self._compute(Pk1d, mask, self.kbin_1d)

    @partial(jit, static_argnames=('self','ell','kbin_1d','mu_min','mu_max'))
    def pk_cross(self, fieldk1, fieldk2, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Cross-spectrum multipole with optional mu-range.
        """
        Pk1d = (fieldk1.ravel() * fieldk2.ravel().conj())
        leg = self.legendre_stack[ell//2]
        Pk1d = Pk1d * leg * self.Nk_1d
        Pk1d = Pk1d.at[0].set(0.0)

        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        return self._compute(Pk1d, mask, self.kbin_1d)


class Measure_spectra_FFT:
    def __init__(self, boxsize, ng, kbin_1d, bispec=True, open_triangle=False):
        self.boxsize = boxsize
        self.kbin_1d = jnp.array(kbin_1d)
        self.vol     = self.boxsize**3
        self.ng      = ng
        kvec         = rfftn_kvec([ng,]*3, self.boxsize)
        self.kmag    = jnp.sqrt(rfftn_k2(kvec))
        self.kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        self.num_bins = self.kbin_1d.shape[0] - 1

        if bispec:
            triangle_idx_list = []

            kmin_bins = self.kbin_1d[:-1]
            kmax_bins = self.kbin_1d[1:]

            for i in range(self.num_bins):
                for j in range(i, self.num_bins):
                    for k in range(j, self.num_bins):
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
            self.k123 = jnp.array([self.kbin_centers[self.triangle_idxs[:,0]], 
                                   self.kbin_centers[self.triangle_idxs[:,1]], 
                                   self.kbin_centers[self.triangle_idxs[:,2]]]).T

    def filter_field(self, fieldk, kmin, kmax):
        mask = (kmin <= self.kmag) & (self.kmag < kmax)
        fieldk_filtered = fieldk * mask
        fieldr = jnp.fft.irfftn(fieldk_filtered) * (self.ng**3)
        return fieldr

    def measure_pk_bk(self, fieldk,):
        num_bins = len(self.kbin_1d) - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])

        onesk = jnp.ones_like(fieldk)

        def filter_all_fields(kmin, kmax):
            fieldr = self.filter_field(fieldk, kmin, kmax)
            onesr   = self.filter_field(onesk,  kmin, kmax)
            return fieldr, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        Ix_fields, norm_fields = vmap(filter_all_fields)(kmins, kmaxs)

        def compute_triangle(idx):
            i, j, k = idx
            k1 = kbin_centers[i]
            k2 = kbin_centers[j]
            k3 = kbin_centers[k]

            valid = jnp.logical_and(k3 >= jnp.abs(k1 - k2), k3 <= (k1 + k2))

            def true_fn(_):
                product_field = Ix_fields[i] * Ix_fields[j] * Ix_fields[k]
                bispec = jnp.sum(product_field) * self.vol * self.vol
                normalization = norm_fields[i] * norm_fields[j] * norm_fields[k]
                num_triangles = jnp.sum(normalization)
                bispec /= num_triangles
                return jnp.array([k1, k2, k3, bispec.real, num_triangles / self.ng**3])
            return lax.cond(valid, true_fn, lambda _: jnp.zeros(5), operand=None)

        triangles = vmap(compute_triangle)(self.triangle_idxs)

        def compute_power(i):
            product_field = Ix_fields[i] * Ix_fields[i]
            power = jnp.sum(product_field) * self.vol
            normalization = norm_fields[i] ** 2
            num_modes = jnp.sum(normalization)
            power /= num_modes
            return jnp.array([kbin_centers[i], power.real, num_modes / self.ng**3])
        
        lines = vmap(compute_power)(jnp.arange(num_bins))

        return lines, triangles

    @partial(jit, static_argnames=('self', 'batch_size'))
    def measure_bk(self, fieldk, batch_size=20):
        num_bins = len(self.kbin_1d) - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])

        onesk = jnp.ones_like(fieldk)

        def filter_all_fields(kmin, kmax):
            fieldr = self.filter_field(fieldk, kmin, kmax)
            onesr  = self.filter_field(onesk,  kmin, kmax)
            return fieldr, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        Ix_fields, norm_fields = vmap(filter_all_fields)(kmins, kmaxs)

        def compute_triangle(idx):
            i, j, k = idx
            k1 = kbin_centers[i]
            k2 = kbin_centers[j]
            k3 = kbin_centers[k]

            valid = jnp.logical_and(k3 >= jnp.abs(k1 - k2), k3 <= (k1 + k2))

            def true_fn(_):
                product_field = Ix_fields[i] * Ix_fields[j] * Ix_fields[k]
                bispec = jnp.sum(product_field) * self.vol * self.vol
                normalization = norm_fields[i] * norm_fields[j] * norm_fields[k]
                num_triangles = jnp.sum(normalization)
                bispec /= num_triangles
                return jnp.array([k1, k2, k3, bispec.real, num_triangles / self.ng**3])
            return lax.cond(valid, true_fn, lambda _: jnp.zeros(5), operand=None)

        n_triangles = self.triangle_idxs.shape[0]
        triangles_batches = []
        for start in range(0, n_triangles, batch_size):
            end = start + batch_size
            batch_idxs = self.triangle_idxs[start:end]
            batch_triangles = vmap(compute_triangle)(batch_idxs)
            triangles_batches.append(batch_triangles)
        triangles = jnp.concatenate(triangles_batches, axis=0)

        return triangles

    @partial(jit, static_argnames=('self'))
    def measure_pk(self, fieldk):
        num_bins = self.kbin_1d.shape[0] - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        onesk = jnp.ones_like(fieldk)

        def filter_two_fields(kmin, kmax):
            fieldr = self.filter_field(fieldk, kmin, kmax)
            onesr  = self.filter_field(onesk,  kmin, kmax)
            return fieldr, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        Ix_fields, norm_fields = vmap(filter_two_fields)(kmins, kmaxs)

        def compute_power(i):
            product_field = Ix_fields[i] * Ix_fields[i]
            power = jnp.sum(product_field) * self.vol
            normalization = norm_fields[i] ** 2
            num_modes = jnp.sum(normalization)
            power /= num_modes
            return jnp.array([kbin_centers[i], power.real, num_modes / self.ng**3])
        
        lines = vmap(compute_power)(jnp.arange(num_bins))
        return lines


    @partial(jit, static_argnames=('self',))
    def measure_bispectrum_(self, field1k, field2k, field3k):
        num_bins = len(self.kbin_1d) - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])

        onesk = jnp.ones_like(field1k)

        def filter_all_fields(kmin, kmax):
            field1r = self.filter_field(field1k, kmin, kmax)
            field2r = self.filter_field(field2k, kmin, kmax)
            field3r = self.filter_field(field3k, kmin, kmax)
            onesr   = self.filter_field(onesk,  kmin, kmax)
            return field1r, field2r, field3r, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        I1x_fields, I2x_fields, I3x_fields, norm_fields = vmap(filter_all_fields)(kmins, kmaxs)

        def compute_triangle(idx):
            i, j, k = idx
            k1 = kbin_centers[i]
            k2 = kbin_centers[j]
            k3 = kbin_centers[k]

            valid = jnp.logical_and(k3 >= jnp.abs(k1 - k2), k3 <= (k1 + k2))

            def true_fn(_):
                product_field = I1x_fields[i] * I2x_fields[j] * I3x_fields[k]
                bispec = jnp.sum(product_field) * self.vol * self.vol
                normalization = norm_fields[i] * norm_fields[j] * norm_fields[k]
                num_triangles = jnp.sum(normalization)
                bispec /= num_triangles
                return jnp.array([k1, k2, k3, bispec.real, num_triangles / self.ng**3])
            return lax.cond(valid, true_fn, lambda _: jnp.zeros(5), operand=None)

        triangles = vmap(compute_triangle)(self.triangle_idxs)

        return triangles


    @partial(jit, static_argnames=('self',))
    def measure_power_(self, field1k, field2k):
        num_bins = self.kbin_1d.shape[0] - 1
        kbin_centers = 0.5 * (self.kbin_1d[1:] + self.kbin_1d[:-1])
        onesk = jnp.ones_like(field1k)

        def filter_two_fields(kmin, kmax):
            field1r = self.filter_field(field1k, kmin, kmax)
            field2r = self.filter_field(field2k, kmin, kmax)
            onesr   = self.filter_field(onesk,  kmin, kmax)
            return field1r, field2r, onesr

        kmins = self.kbin_1d[:-1]
        kmaxs = self.kbin_1d[1:]
        I1x_fields, I2x_fields, norm_fields = vmap(filter_two_fields)(kmins, kmaxs)

        def compute_power(i):
            product_field = I1x_fields[i] * I2x_fields[i]
            power = jnp.sum(product_field) * self.vol
            normalization = norm_fields[i] ** 2
            num_modes = jnp.sum(normalization)
            power /= num_modes
            return jnp.array([kbin_centers[i], power.real, num_modes / self.ng**3])
        
        lines = vmap(compute_power)(jnp.arange(num_bins))
        return lines

