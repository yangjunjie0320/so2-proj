"""Isolated adapter for the new GPAW PW engine (tested with GPAW 26.7).

Only all-electron *valence* orbitals are reconstructed.  Frozen core orbitals
are not supplied by the band wave functions.  Coordinates and amplitudes here
are in atomic units; GPAW's internal PW arrays already use those units.
"""

import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def geometry(source):
    """Read an existing calculator's geometry without running a calculation."""
    from ase.units import Bohr

    try:
        atoms = source.atoms
        coords = np.asarray(atoms.positions, dtype=float) / Bohr
        charges = np.asarray(atoms.numbers, dtype=int)
    except (AttributeError, TypeError) as err:
        raise TypeError("GPAW needs a calculator with an existing geometry") from err
    return coords.copy(), charges.copy()


def occupied_state(source):
    """Validate the supported calculation and copy spin-summed occupations."""
    from ase.units import Bohr
    from gpaw.new.ase_interface import ASECalculator

    if not isinstance(source, ASECalculator):
        raise NotImplementedError("Only the new GPAW calculator is supported")
    try:
        dft = source.dft
    except AttributeError as err:
        raise ValueError("Run the GPAW calculation before requesting a bond order") from err
    if source.comm.size != 1:
        raise NotImplementedError("GPAW bond-order postprocessing currently requires one MPI rank")
    if source.params.mode.name != "pw":
        raise NotImplementedError("GPAW bond orders currently require PW mode")
    ibz = dft.ibzwfs
    if not ibz.collinear or ibz.ncomponents != 1:
        raise NotImplementedError("Only spin-paired GPAW calculations are supported")
    kpoints = np.asarray(source.get_ibz_k_points())
    weights = np.asarray(source.get_k_point_weights())
    if (kpoints.shape != (1, 3) or not np.allclose(kpoints, 0.0, atol=1e-12)
            or weights.shape != (1,) or not np.allclose(weights, 1.0)):
        raise NotImplementedError("Only a single Gamma point is supported")
    if not dft.results or not ibz.has_wave_functions():
        raise ValueError("GPAW needs a completed calculation with saved wave functions (mode='all')")
    # GPAW 26.7's raw=False getter scales its returned array in place.  Copy
    # raw occupations first so repeated bond-order calls cannot alter the SCF.
    occ = 2.0 * np.array(source.get_occupation_numbers(raw=True), dtype=float, copy=True)
    if (not np.isfinite(occ).all() or not np.all(np.isclose(occ, 0.0, atol=1e-8, rtol=0)
                                               | np.isclose(occ, 2.0, atol=1e-8, rtol=0))):
        raise NotImplementedError("Only closed-shell occupations of zero or two are supported")
    if not np.isclose(occ.sum(), source.get_number_of_electrons(), atol=1e-7, rtol=0):
        raise ValueError("GPAW occupations do not sum to its valence electron count")
    indices = np.flatnonzero(occ > 1.0)
    if not len(indices):
        raise ValueError("GPAW has no occupied valence orbitals")
    cell = np.asarray(source.atoms.cell, dtype=float) / Bohr
    lengths = np.diag(cell)
    if not np.allclose(cell, np.diag(lengths), atol=1e-10) or np.any(lengths <= 0):
        raise NotImplementedError("GPAW currently requires an axis-aligned orthorhombic cell")
    return indices, occ[indices], lengths


def check_cylinder(coords, A, B, lengths, pad):
    """Require the whole integration cylinder, including rotations, in the box."""
    origin = coords[A]
    axis = coords[B] - origin
    axis = axis / np.linalg.norm(axis)
    relative = coords - origin
    z = relative @ axis
    rho = np.linalg.norm(relative - z[:, None] * axis, axis=1)
    zmin, zmax = z.min() - pad, z.max() + pad
    radius = rho.max() + pad
    radial_extent = radius * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
    lower = origin + np.minimum(zmin * axis, zmax * axis) - radial_extent
    upper = origin + np.maximum(zmin * axis, zmax * axis) + radial_extent
    if np.any(lower < -1e-10) or np.any(upper > lengths + 1e-10):
        raise ValueError("The integration cylinder extends outside the GPAW cell; "
                         "use a larger vacuum box or reduce pad after checking convergence")
    return lower, upper


class PAWOrbitals:
    """Evaluate real PAW-reconstructed valence orbitals at arbitrary bohr points.

    Interpolate the smooth PW part on a fine grid, and evaluate the atom-centred
    partial-wave corrections directly at each requested point.  No periodic
    images are sampled: evaluation outside the original cell is an error.
    """

    def __init__(self, source, indices, lengths, h_fine):
        from gpaw.sphere.spherical_harmonics import Y

        self._harmonic = Y
        self.indices = np.asarray(indices)
        self.norb = len(indices)
        self.lengths = np.asarray(lengths)
        stop = int(indices.max()) + 1
        wfs = source.dft.ibzwfs.get_wfs_on_master(kpt=0, spin=0, n1=0, n2=stop)
        wave_cell = np.asarray(wfs.psit_nX.desc.cell_cv)
        if not np.allclose(wave_cell, np.diag(self.lengths), rtol=0, atol=1e-10):
            raise ValueError("The GPAW cell differs from the stored wave functions; rerun GPAW first")
        positions = np.asarray(wfs.relpos_ac) * self.lengths
        coords, _ = geometry(source)
        if not np.allclose(positions, coords, rtol=0, atol=1e-10):
            raise ValueError("The GPAW geometry differs from the stored wave functions; rerun GPAW first")
        self.coefficients = []
        for n in indices:
            pseudo = wfs.psit_nX[int(n)].to_pbc_grid()
            grid = pseudo.desc.uniform_grid_with_grid_spacing(h_fine)
            fine = np.asarray(pseudo.interpolate(grid=grid).data)
            if np.iscomplexobj(fine) and np.max(np.abs(fine.imag)) > 1e-10:
                raise NotImplementedError("Only real GPAW orbitals are supported")
            self.coefficients.append(ndimage.spline_filter(fine.real, order=3,
                                                          mode="grid-wrap"))
        self.shape = np.array(self.coefficients[0].shape)
        self.spacing = self.lengths / self.shape
        corrections = wfs.setups.partial_wave_corrections()
        self.atoms = []
        for a, splines in enumerate(corrections):
            projections = np.asarray(wfs.P_ani[a])[indices]
            if np.iscomplexobj(projections) and np.max(np.abs(projections.imag)) > 1e-10:
                raise NotImplementedError("Only real GPAW PAW projections are supported")
            terms, i = [], 0
            for spline in splines:
                l = spline.get_angular_momentum_number()
                for m in range(2 * l + 1):
                    terms.append((spline, l * l + m, i))
                    i += 1
            if i != projections.shape[1]:
                raise ValueError("GPAW partial waves and projector coefficients disagree")
            cutoff = max((s.get_cutoff() for s in splines), default=0.0)
            if np.any(positions[a] - cutoff < 0) or np.any(positions[a] + cutoff > self.lengths):
                raise ValueError("A PAW augmentation sphere crosses the cell boundary; enlarge the vacuum box")
            self.atoms.append((positions[a], cutoff, terms, projections.real.copy()))
        self.frozen_core_electrons = float(sum(s.Nc for s in wfs.setups))
        log.info("GPAW fine grid %s; spacing %s bohr; %d occupied valence orbitals",
                 self.shape, self.spacing, self.norb)

    def _points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Orbital evaluation points must have shape (n, 3) and be finite")
        if np.any(points < 0) or np.any(points >= self.lengths):
            raise ValueError("Orbital evaluation points lie outside the GPAW cell")
        return points

    def pseudo(self, points):
        points = self._points(points)
        indices = (points / self.spacing).T
        return np.stack([ndimage.map_coordinates(c, indices, order=3, mode="grid-wrap",
                                                 prefilter=False)
                         for c in self.coefficients], axis=1)

    def correction(self, points):
        points = self._points(points)
        result = np.zeros((len(points), self.norb))
        for position, cutoff, terms, projections in self.atoms:
            delta = points - position
            radius = np.linalg.norm(delta, axis=1)
            selected = np.flatnonzero(radius < cutoff)
            if not len(selected):
                continue
            xyz, r = delta[selected], radius[selected]
            values = np.zeros((len(selected), projections.shape[1]))
            for spline, L, i in terms:
                values[:, i] = spline.map(r) * self._harmonic(L, *xyz.T)
            result[selected] += values @ projections.T
        return result

    def __call__(self, points):
        return self.pseudo(points) + self.correction(points)
