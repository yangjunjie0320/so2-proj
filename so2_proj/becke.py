"""Becke fuzzy-atom weights (Becke, J. Chem. Phys. 88, 2547 (1988))."""

import numpy as np
from pyscf.data import radii


def becke_weights(coords, charges, points, k=3, adjust=True):
    """Normalized Becke cell weights w_A(r).

    Parameters
    ----------
    coords : (natm, 3) array, atomic positions in bohr.
    charges : (natm,) int array, nuclear charges (used for Bragg-Slater radii).
    points : (npts, 3) array, grid points in bohr.
    k : number of iterations of the Becke smoothing polynomial (k=3 is Becke's choice;
        larger k gives harder cells).
    adjust : apply Becke's atomic-size adjustment with Bragg-Slater radii.

    Returns
    -------
    (natm, npts) array with sum over atoms equal to one at every point.
    """
    coords = np.asarray(coords, float)
    natm = len(coords)
    dist = np.linalg.norm(points[None, :, :] - coords[:, None, :], axis=2)
    rad = radii.BRAGG[np.asarray(charges, int)]
    cell = np.ones((natm, len(points)))
    for a in range(natm):
        for b in range(natm):
            if a == b:
                continue
            rab = np.linalg.norm(coords[a] - coords[b])
            mu = (dist[a] - dist[b]) / rab
            if adjust:
                chi = rad[a] / rad[b]
                u = (chi - 1) / (chi + 1)
                aab = np.clip(u / (u * u - 1), -0.5, 0.5)
                mu = mu + aab * (1 - mu * mu)
            for _ in range(k):
                mu = 1.5 * mu - 0.5 * mu ** 3
            cell[a] *= 0.5 * (1 - mu)
    return cell / cell.sum(axis=0)
