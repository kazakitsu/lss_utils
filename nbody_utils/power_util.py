import numpy as np


def rfftn_kvec(shape, boxsize, dtype=float):
    """
    Generate wavevectors for numpy.fft.rfftn
    """
    # full FFT frequencies for all but last axis
    grids = []
    for n in shape[:-1]:
        k = np.fft.fftfreq(n, d=1.0/shape[-1]).astype(dtype)
        grids.append(k)
    # real FFT frequencies for last axis
    k = np.fft.rfftfreq(shape[-1], d=1.0/shape[-1]).astype(dtype)
    grids.append(k)
    # meshgrid in ij indexing
    mesh = np.meshgrid(*grids, indexing='ij')
    kvec = np.stack(mesh, axis=0)
    return kvec * (2 * np.pi / boxsize)


class Measure_Pk:
    def __init__(self, boxsize, ng, kbin_1d, ell_max=0, leg_fac=True):
        self.boxsize = boxsize
        self.ng = ng
        self.vol = boxsize**3
        self.kbin_1d = np.array(kbin_1d)
        self.num_bins = len(self.kbin_1d) - 1

        # precompute k-vector grid
        kvec = rfftn_kvec([ng, ng, ng], boxsize)
        k2 = np.sum(kvec**2, axis=0)
        kmag = np.sqrt(k2)
        self.kmag_1d = kmag.ravel()

        # digitize into k-bins
        self.kidx = np.digitize(self.kmag_1d, self.kbin_1d, right=True)

        # count modes (real-to-complex symmetry)
        Nk = np.full_like(k2, 2, dtype=int)
        Nk[...,0] = 1 
        if k2.shape[-1] % 2 == 0:
            Nk[..., -1] = 1
        self.Nk_1d = Nk.ravel()

        # mean k and mode counts per bin
        k_tot = np.bincount(self.kidx, weights=self.kmag_1d * self.Nk_1d, minlength=self.num_bins+1)[1:]
        N_tot = np.bincount(self.kidx, weights=self.Nk_1d, minlength=self.num_bins+1)[1:]
        self.k_mean = k_tot / N_tot
        self.Nk = N_tot

        # compute mu^2 per mode
        mu2 = (kvec[2]**2) / np.where(k2==0, 1, k2)
        mu2.flat[0] = 0.0
        self.mu2_1d = mu2.ravel()

        # build Legendre factors
        self.legendre_stack = []
        for ell in range(0, ell_max+1, 2):
            fac = (2*ell + 1) if leg_fac else 1.0
            if ell == 0:
                P = fac * np.ones_like(self.mu2_1d)
            elif ell == 2:
                P = fac * 0.5 * (3*self.mu2_1d - 1)
            elif ell == 4:
                P = fac * 0.125 * (35*self.mu2_1d**2 - 30*self.mu2_1d + 3)
            else:
                continue
            self.legendre_stack.append(P)
        self.legendre_stack = np.array(self.legendre_stack)

    def _compute(self, Pk1d, mask, kbin_edges):
        # apply mask
        k_arr = self.kmag_1d
        Nk_arr = self.Nk_1d

        sel = mask
        k_sel = k_arr[sel]
        P_sel = (Pk1d * Nk_arr)[sel]
        Nk_sel = Nk_arr[sel]

        # k-bin indices
        kidx = np.digitize(k_sel, kbin_edges, right=True)

        # accumulate sums
        k_sum = np.bincount(kidx, weights=k_sel * Nk_sel, minlength=len(kbin_edges))[1:]
        P_sum = np.bincount(kidx, weights=P_sel.real, minlength=len(kbin_edges))[1:]
        N_sum = np.bincount(kidx, weights=Nk_sel, minlength=len(kbin_edges))[1:]

        # normalize
        k_mean = k_sum / N_sum
        Pk_out = (P_sum / N_sum) * self.vol
        return k_mean, Pk_out, N_sum

    def pk_auto(self, fieldk, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Compute auto-spectrum multipole with optional mu-range.
        """
        # flatten
        F = fieldk.ravel()
        Pk1d = F * np.conj(F)
        # apply Legendre and mode count
        leg = self.legendre_stack[ell//2]
        Pk1d = Pk1d * leg
        Pk1d[0] = 0.0

        # mu mask
        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        return self._compute(Pk1d, mask, self.kbin_1d)

    def pk_cross(self, fieldk1, fieldk2, ell=0, mu_min=0.0, mu_max=1.0):
        """
        Compute cross-spectrum multipole with optional mu-range.
        """
        Pk1d = fieldk1.ravel() * np.conj(fieldk2.ravel())
        leg = self.legendre_stack[ell//2]
        Pk1d = Pk1d * leg
        Pk1d[0] = 0.0

        mask = (self.mu2_1d >= mu_min**2) & (self.mu2_1d <= mu_max**2)
        return self._compute(Pk1d, mask, self.kbin_1d)


class Measure_spectra_FFT:
    def __init__(self, boxsize, ng, kbin_1d, bispec=True, open_triangle=False):
        self.boxsize = boxsize
        self.vol = boxsize**3
        self.ng = ng
        self.kbin_1d = np.array(kbin_1d)
        self.kbin_centers = 0.5*(self.kbin_1d[1:]+self.kbin_1d[:-1])
        self.num_bins = len(self.kbin_1d) - 1

        # k-magnitude grid
        kvec = rfftn_kvec([ng,ng,ng], boxsize)
        k2 = np.sum(kvec**2, axis=0)
        self.kmag = np.sqrt(k2)

        # triangle indices
        if bispec:
            tris = []
            kmin = self.kbin_1d[:-1]
            kmax = self.kbin_1d[1:]
            for i in range(self.num_bins):
                for j in range(i, self.num_bins):
                    for k in range(j, self.num_bins):
                        k1c = self.kbin_centers[i]
                        k2c = self.kbin_centers[j]
                        k3c = self.kbin_centers[k]
                        if open_triangle:
                            cond = (kmin[i]+kmin[j] > kmax[k]) and \
                                   (kmin[j]+kmin[k] > kmax[i]) and \
                                   (kmin[k]+kmin[i] > kmax[j])
                        else:
                            cond = (k3c >= abs(k1c-k2c)) and (k3c <= k1c+k2c)
                        if cond:
                            tris.append((i,j,k))
            self.triangle_idxs = np.array(tris, dtype=int)

    def filter_field(self, fieldk, kmin, kmax):
        mask = (self.kmag >= kmin) & (self.kmag < kmax)
        fld = fieldk * mask
        # inverse real FFT
        fieldr = np.fft.irfftn(fld) * (self.ng**3)
        return fieldr, mask.astype(float)

    def measure_pk(self, fieldk):
        """
        Measure P(k) for each bin.
        Returns array of shape (num_bins, 3): [k_center, Pk, mode_density]
        """
        results = []
        for i in range(self.num_bins):
            kmin = self.kbin_1d[i]
            kmax = self.kbin_1d[i+1]
            fi, norm = self.filter_field(fieldk, kmin, kmax)
            product = fi * fi
            Psum = np.sum(product)
            Nsum = np.sum(norm)
            Pk = (Psum / Nsum) * self.vol if Nsum>0 else 0.0
            results.append([self.kbin_centers[i], Pk, Nsum/(self.ng**3)])
        return np.array(results)

    def measure_bk(self, fieldk):
        """
        Measure bispectrum for each triangle bin.
        Returns array of shape (n_tri, 5): [k1,k2,k3,bk,num_triangles_density]
        """
        fi, norm = [], []
        for i in range(self.num_bins):
            kmin = self.kbin_1d[i]
            kmax = self.kbin_1d[i+1]
            f, m = self.filter_field(fieldk, kmin, kmax)
            fi.append(f)
            norm.append(m)

        tris = []
        for (i,j,k) in self.triangle_idxs:
            k1c, k2c, k3c = self.kbin_centers[i], self.kbin_centers[j], self.kbin_centers[k]
            f_prod = fi[i] * fi[j] * fi[k]
            bispec = np.sum(f_prod) * (self.vol**2)
            num_tri = np.sum(norm[i] * norm[j] * norm[k])
            bk = bispec/num_tri if num_tri>0 else 0.0
            tris.append([k1c, k2c, k3c, bk.real, num_tri/(self.ng**3)])
        return np.array(tris)

    def measure_power(self, fieldk1, fieldk2=None):
        """
        Measure cross- or auto-power between fieldk1 and fieldk2.
        If fieldk2 is None, computes auto-power.
        Returns array shape (num_bins,3).
        """
        if fieldk2 is None:
            fieldk2 = fieldk1
        results = []
        for i in range(self.num_bins):
            kmin = self.kbin_1d[i]
            kmax = self.kbin_1d[i+1]
            f1, m1 = self.filter_field(fieldk1, kmin, kmax)
            f2, m2 = self.filter_field(fieldk2, kmin, kmax)
            Psum = np.sum(f1 * f2)
            Nsum = np.sum(m1 * m2)
            Pk = (Psum / Nsum) * self.vol if Nsum>0 else 0.0
            results.append([self.kbin_centers[i], Pk.real, Nsum/(self.ng**3)])
        return np.array(results)

    def measure_pk_bk(self, fieldk):
        """
        Compute both power spectrum and bispectrum for a single field.
        Returns a tuple (pk_lines, tris_array).
        """
        pk_lines = self.measure_pk(fieldk)
        tris_array = self.measure_bk(fieldk)
        return pk_lines, tris_array

    def measure_bispectrum_(self, field1k, field2k, field3k):
        """
        Alias for bispectrum measurement with three fields.
        Returns array of shape (n_tri, 5): [k1,k2,k3,bk,num_triangles_density]
        """
        fi1, fi2, fi3, norm = [], [], [], []
        for i in range(self.num_bins):
            kmin = self.kbin_1d[i]
            kmax = self.kbin_1d[i+1]
            f1, m = self.filter_field(field1k, kmin, kmax)
            f2, _ = self.filter_field(field2k, kmin, kmax)
            f3, _ = self.filter_field(field3k, kmin, kmax)
            fi1.append(f1)
            fi2.append(f2)
            fi3.append(f3)
            norm.append(m)

        tris = []
        for (i,j,k) in self.triangle_idxs:
            k1c, k2c, k3c = self.kbin_centers[i], self.kbin_centers[j], self.kbin_centers[k]
            prod = fi1[i] * fi2[j] * fi3[k]
            bispec = np.sum(prod) * (self.vol**2)
            num_tri = np.sum(norm[i] * norm[j] * norm[k])
            bk = bispec/num_tri if num_tri>0 else 0.0
            tris.append([k1c, k2c, k3c, bk.real, num_tri/(self.ng**3)])
        return np.array(tris)
