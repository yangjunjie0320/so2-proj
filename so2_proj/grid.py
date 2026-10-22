"""Cylindrical integration grid around a bond axis and FFT projection onto |m| channels."""

from dataclasses import dataclass

import numpy as np


def bond_frame(axis):
    """Orthonormal frame (ex, ey, ez) with ez along ``axis``."""
    ez = np.asarray(axis, float)
    if ez.shape != (3,) or not np.all(np.isfinite(ez)) or np.linalg.norm(ez) < 1e-12:
        raise ValueError("Bond axis must be a finite nonzero 3-vector")
    ez = ez / np.linalg.norm(ez)
    t = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = t - (t @ ez) * ez
    ex /= np.linalg.norm(ex)
    return ex, np.cross(ez, ex), ez


@dataclass
class CylinderGrid:
    """Product grid z x rho x phi around the A->B axis, origin at atom A.

    z: Gauss-Legendre on segments split at on-axis nuclei; rho: Gauss-Legendre with a
    cubic map (dense near the axis); phi: uniform, so that an FFT along phi is exact
    for |m| < nphi / 2.
    """

    z: np.ndarray
    wz: np.ndarray
    rho: np.ndarray
    wrho: np.ndarray  # includes the cylindrical Jacobian rho
    phi: np.ndarray
    origin: np.ndarray
    frame: tuple

    @property
    def wphi(self):
        return 2 * np.pi / len(self.phi)

    @property
    def npts(self):
        return len(self.z) * len(self.rho) * len(self.phi)

    @classmethod
    def build(cls, coords, A, B, nz=160, nrho=80, nphi=32, pad=6.0, tol=1e-6):
        coords = np.asarray(coords, float)
        for name, value in (("nz", nz), ("nrho", nrho), ("nphi", nphi)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(pad) or pad <= 0:
            raise ValueError("pad must be positive and finite")
        ex, ey, ez = bond_frame(coords[B] - coords[A])
        rel = coords - coords[A]
        zc = rel @ ez
        perp = np.linalg.norm(rel - np.outer(zc, ez), axis=1)
        zmin, zmax = zc.min() - pad, zc.max() + pad
        rhomax = perp.max() + pad
        breaks = np.unique(np.round(np.r_[zmin, zc[perp < tol], zmax], 10))
        lens = np.diff(breaks)
        npts = np.maximum(16, np.round(nz * lens / lens.sum()).astype(int))
        zs, wz = [], []
        for (a, b), m in zip(zip(breaks[:-1], breaks[1:]), npts):
            x, w = np.polynomial.legendre.leggauss(m)
            zs.append(0.5 * (b - a) * x + 0.5 * (a + b))
            wz.append(0.5 * (b - a) * w)
        x, w = np.polynomial.legendre.leggauss(nrho)
        u = 0.5 * (x + 1)
        rho = rhomax * u ** 3
        wrho = 0.5 * w * 3 * rhomax * u ** 2 * rho
        phi = 2 * np.pi * np.arange(nphi) / nphi
        return cls(np.concatenate(zs), np.concatenate(wz), rho, wrho, phi,
                   coords[A].copy(), (ex, ey, ez))

    def slabs(self, zchunk=8):
        """Yield (points (n,3), weights (n,), shape (nz_chunk, nrho, nphi)) slab by slab."""
        if not isinstance(zchunk, (int, np.integer)) or zchunk < 1:
            raise ValueError("zchunk must be a positive integer")
        ex, ey, ez = self.frame
        c, s = np.cos(self.phi), np.sin(self.phi)
        nr, nf = len(self.rho), len(self.phi)
        for z0 in range(0, len(self.z), zchunk):
            zz, wzz = self.z[z0:z0 + zchunk], self.wz[z0:z0 + zchunk]
            pts = (self.origin[None, None, None, :]
                   + zz[:, None, None, None] * ez
                   + (self.rho[None, :, None, None] * c[None, None, :, None]) * ex
                   + (self.rho[None, :, None, None] * s[None, None, :, None]) * ey)
            w = wzz[:, None, None] * self.wrho[None, :, None] * self.wphi
            shape = (len(zz), nr, nf)
            yield pts.reshape(-1, 3), np.broadcast_to(w, shape).reshape(-1), shape


def channel_components(values, shape, lmax):
    """Split orbital values into real |m| channels by an FFT along phi.

    values : (npts, norb) on a slab of shape (nz, nrho, nphi).
    Returns a list of lmax + 2 arrays (npts, norb): Q_0 f, ..., Q_lmax f, and the
    remainder f - sum_l Q_l f (all |m| > lmax). Q_l = Pi_l + Pi_{-l} for l > 0.
    """
    nz, nr, nf = shape
    if not isinstance(lmax, (int, np.integer)) or lmax < 0 or nf <= 2 * lmax:
        raise ValueError("FFT sampling requires nonnegative lmax and nphi > 2*lmax")
    values = np.asarray(values)
    if np.iscomplexobj(values):
        raise NotImplementedError("Complex orbitals are unsupported")
    if values.ndim != 2 or values.shape[0] != nz * nr * nf:
        raise ValueError("Orbital values do not match the grid slab shape")
    norb = values.shape[1]
    F = np.fft.fft(values.reshape(nz, nr, nf, norb), axis=2)
    out = []
    for m in range(lmax + 1):
        G = np.zeros_like(F)
        G[:, :, m] = F[:, :, m]
        if m:
            G[:, :, nf - m] = F[:, :, nf - m]
        out.append(np.fft.ifft(G, axis=2).real.reshape(-1, norb))
    out.append(values - sum(out))
    return out
