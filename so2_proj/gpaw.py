"""One-body channel bond orders for isolated molecules calculated with GPAW."""

import numpy as np

from .base import BaseBondOrder
from .bondorder import channel_matrices, total_bond_orders
from .realspace import realspace_channel_weights


class BondOrderForGPAW(BaseBondOrder):
    """Use a completed, spin-paired Gamma-point PW calculator.

    ``one_body()`` uses PAW-reconstructed occupied valence orbitals.  Frozen
    cores are excluded.  The molecule must be in an orthorhombic vacuum box
    large enough to contain the complete integration cylinder.  ``h_fine``
    and all grid lengths are in bohr.  GPAW is imported only on use.
    """

    def __init__(self, source, A, B, **parameters):
        from ._gpaw_adapter import geometry

        coords, charges = geometry(source)
        super().__init__(source, A, B, coords=coords, charges=charges, **parameters)

    def one_body(self):
        """Return the B1 channel matrix, including the final remainder channel."""
        from ._gpaw_adapter import PAWOrbitals, check_cylinder, geometry, occupied_state

        self._validate_parameters()
        if self.partition != "becke":
            raise NotImplementedError("The GPAW interface supports the Becke partition only")
        coords, charges = geometry(self.source)
        if (not np.array_equal(charges, self.charges)
                or not np.allclose(coords, self.coords, rtol=0, atol=1e-10)):
            raise ValueError("The GPAW geometry changed; construct a new BondOrder object")
        indices, occupations, lengths = occupied_state(self.source)
        grid_options = self._grid_options()
        bounds = check_cylinder(coords, self.A, self.B, lengths, grid_options["pad"])
        orbitals = PAWOrbitals(self.source, indices, lengths, self.h_fine)
        weights = realspace_channel_weights(
            orbitals, coords, charges, self.A, self.B, lmax=self.lmax,
            becke_k=self.becke_k, becke_adjust=self.becke_adjust,
            cell=self.cell, **grid_options)
        WA, WB = weights[self.cell]
        WAt, WBt = weights[self.cell + "_total"]
        density = np.diag(occupations)
        matrix = channel_matrices(WA, WB, density)[0]
        total = total_bond_orders(WAt, WBt, density)[0]
        gram = weights["overlap"]
        diagnostics = {
            "orbital_basis": "paw_all_electron_valence",
            "valence_only": True,
            "frozen_core_electrons": orbitals.frozen_core_electrons,
            "valence_electrons": float(occupations.sum()),
            "occupations": occupations.copy(),
            "band_indices": indices.copy(),
            "overlap": gram.copy(),
            "overlap_error": float(np.max(np.abs(gram - np.eye(len(indices))))),
            "orbital_norms": weights["norm"].copy(),
            "fine_grid_shape": orbitals.shape.copy(),
            "fine_grid_spacing": orbitals.spacing.copy(),
            "cylinder_bounds": np.asarray(bounds),
            "npoints": weights["npoints"],
        }
        return self._record(matrix, total, diagnostics, kind="one_body")

    def two_body(self, *args, **kwargs):
        """GPAW two-body densities are outside the supported interface."""
        raise NotImplementedError("The GPAW interface supports one-body bond orders only")
