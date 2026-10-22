"""PySCF molecular AO interface, independent of the electronic-structure method."""
import numpy as np
from pyscf import gto
from .aosb import aosb_channel_weights
from .base import BaseBondOrder
from .bondorder import channel_matrices, total_bond_orders
from .density import orthogonalize_dm1, prepare_dm2
from .realspace import realspace_channel_weights


class BondOrderForPySCF(BaseBondOrder):
    """Analyze real spin-summed AO densities belonging to a PySCF Mole.

    Only the molecule is stored. Neither construction nor analysis runs SCF.
    Options are ordinary mutable attributes and weights are recomputed per call.
    """
    def __init__(self, mol, A, B, **parameters):
        if not isinstance(mol, gto.MoleBase):
            raise TypeError("BondOrderForPySCF expects a PySCF Mole, not an SCF/CAS object")
        if hasattr(mol, "pbc_intor"):
            raise NotImplementedError("Periodic PySCF cells are unsupported")
        if not mol._built:
            raise ValueError("Build the PySCF molecule before creating BondOrder")
        self.mol = mol
        super().__init__(mol, A, B, coords=mol.atom_coords(),
                         charges=mol.atom_charges(), **parameters)

    def _prepare_geometry(self):
        self.coords = self.mol.atom_coords()
        self.charges = self.mol.atom_charges()
        self._validate_parameters()
        if self.mol.spin != 0:
            raise NotImplementedError("Only spin-unpolarized molecular densities are supported")

    def _weights(self, X):
        """Return channels, unresolved weights, and compact integration diagnostics."""
        S = self.mol.intor("int1e_ovlp")
        if self.partition == "mulliken":
            channels, totals = aosb_channel_weights(self.mol, X, self.A, self.B, self.lmax)
            overlap = X.T @ S @ X
            npoints = 0
        else:
            evaluator = "GTOval_cart" if self.mol.cart else "GTOval_sph"
            weights = realspace_channel_weights(
                lambda points: self.mol.eval_gto(evaluator, points) @ X,
                self.coords, self.charges, self.A, self.B, lmax=self.lmax,
                becke_k=self.becke_k, becke_adjust=self.becke_adjust,
                cell=self.cell, **self._grid_options())
            channels, totals = weights[self.cell], weights[self.cell + "_total"]
            overlap, npoints = weights["overlap"], weights["npoints"]
        diagnostics = dict(
            electron_space="molecular AO electrons", nelectron=self.mol.nelectron,
            npoints=npoints,
            overlap_error=float(np.max(np.abs(overlap - np.eye(len(X))))),
            weight_sum_error=float(max(np.max(np.abs(W.sum(axis=0) - W0))
                                       for W, W0 in zip(channels, totals))))
        return channels, totals, diagnostics

    def one_body(self, dm1):
        """Return the (lmax+2, lmax+2) one-body matrix from a full AO dm1."""
        self._prepare_geometry()
        P, X = orthogonalize_dm1(dm1, self.mol.intor("int1e_ovlp"),
                                nelectron=self.mol.nelectron)
        (WA, WB), (WAt, WBt), diagnostics = self._weights(X)
        matrix = channel_matrices(WA, WB, P)[0]
        total = total_bond_orders(WAt, WBt, P)[0]
        return self._record(matrix, total, diagnostics, kind="one_body")

    def two_body(self, dm2, *, coeff, dm1, indices=None, completion="full"):
        """Return the two-body matrix using an explicitly specified orbital space.

        dm1 is always full AO, real and spin-summed. dm2 uses PySCF FCI's
        G[p,q,r,s]=<a†_p a†_r a_s a_q> convention in coeff[:, indices] order.
        'full' requires coverage of occupied density support; 'cas' permits a
        fixed doubly occupied core and empty orbitals outside the selected space.
        Coefficients are checked in the AO overlap metric, never reordered or
        orthogonalized. No full AO four-index tensor is formed.
        """
        self._prepare_geometry()
        density = prepare_dm2(dm2, coeff, dm1=dm1,
                              overlap=self.mol.intor("int1e_ovlp"),
                              indices=indices, completion=completion,
                              nelectron=self.mol.nelectron)
        (WA, WB), (WAt, WBt), diagnostics = self._weights(density.X)
        matrix = channel_matrices(WA, WB, density.P)[0]
        total = total_bond_orders(WAt, WBt, density.P)[0]
        T = density.T
        WAs, WBs = T.T @ WA @ T, T.T @ WB @ T
        matrix -= 2 * np.einsum("pqrs,aqp,bsr->ab", density.cumulant, WAs, WBs,
                                optimize=True)
        total -= 2 * np.einsum("pqrs,qp,sr->", density.cumulant,
                               T.T @ WAt @ T, T.T @ WBt @ T, optimize=True)
        diagnostics.update(completion=completion, selected_orbitals=T.shape[1],
                           selected_electrons=float(np.trace(density.Ps)))
        return self._record(matrix, total, diagnostics, kind="two_body")
