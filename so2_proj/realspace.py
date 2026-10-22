"""Real-space atomic weights and SO(2) channels in real orthonormal orbitals.

Full cells project only the ket: W_pq(l) = <p|w Q_l|q>. These matrices
are generally not symmetric. Averaged cells use <Q_l p|wbar|Q_l q>.
"""
import logging
import numpy as np
from .becke import becke_weights
from .grid import CylinderGrid, channel_components

log = logging.getLogger(__name__)


def realspace_channel_weights(orbitals, coords, charges, A, B, lmax=3, becke_k=3,
                              becke_adjust=True, zchunk=8, cell=None, **grid_kw):
    """Integrate channel and unresolved weights, and the full orbital Gram matrix.

    orbitals(points) takes bohr coordinates and returns real (npoints, norbitals)
    values. cell=None computes both conventions; public methods select one.
    The final channel contains every sampled Fourier mode above lmax.
    """
    if cell not in (None, "full", "avg"):
        raise ValueError("cell must be 'full' or 'avg'")
    if not isinstance(lmax, (int, np.integer)) or lmax < 0:
        raise ValueError("lmax must be a nonnegative integer")
    if grid_kw.get("nphi", 32) <= 2 * lmax:
        raise ValueError("FFT sampling requires nphi > 2*lmax")
    grid = CylinderGrid.build(coords, A, B, **grid_kw)
    cells = ("full", "avg") if cell is None else (cell,)
    result = None
    K = lmax + 2
    for pts, w, shape in grid.slabs(zchunk):
        psi = np.asarray(orbitals(pts))
        if np.iscomplexobj(psi):
            raise NotImplementedError("Complex orbitals are unsupported")
        if psi.ndim != 2 or psi.shape[0] != len(pts) or not np.all(np.isfinite(psi)):
            raise ValueError("Orbitals must return finite real values of shape (npoints, norbitals)")
        if result is None:
            n = psi.shape[1]
            result = {name: tuple(np.zeros((K, n, n)) for _ in range(2)) for name in cells}
            result.update({name + "_total": tuple(np.zeros((n, n)) for _ in range(2)) for name in cells})
            result["overlap"] = np.zeros((n, n))
        result["overlap"] += psi.T @ (w[:, None] * psi)
        comps = channel_components(psi, shape, lmax)
        atomic = becke_weights(coords, charges, pts, k=becke_k, adjust=becke_adjust)
        for endpoint, atom in enumerate((A, B)):
            for name in cells:
                weight = atomic[atom]
                if name == "avg":
                    weight = np.broadcast_to(weight.reshape(shape).mean(axis=2, keepdims=True), shape).reshape(-1)
                weight = w * weight
                left = weight[:, None] * psi
                result[name + "_total"][endpoint][:] += left.T @ psi
                for k, component in enumerate(comps):
                    bra = left if name == "full" else weight[:, None] * component
                    result[name][endpoint][k] += bra.T @ component
    result["norm"] = np.diag(result["overlap"]).copy()
    result["npoints"] = grid.npts
    log.info("cylinder grid: %d points, max overlap error %.3g", grid.npts,
             np.max(np.abs(result["overlap"] - np.eye(len(result["norm"])))))
    return result
