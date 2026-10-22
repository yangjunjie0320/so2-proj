"""Mulliken partition x SO(2) projection (the AOSB decomposition of Sizova et al.).

For the Mulliken partition W^A = D_A S the channel weight is W^A(l) = Q_A(l) S, where
Q_A(l) is the projector onto |m| = l (about the bond axis) inside every shell of atom A.
Shell projectors are built from rotation matrices of real spherical harmonics, so the
bond may point in any direction; no change of coordinates is needed.
"""

import numpy as np
from functools import lru_cache
from pyscf import gto
from scipy.spatial.transform import Rotation

LMAX_SHELL = 6
@lru_cache(maxsize=1)
def _probe():
    """Build the harmonic reference only when AOSB rotations are needed."""
    pts = np.random.default_rng(7).normal(size=(300, 3))
    mol = gto.M(atom="H 0 0 0", basis={"H": [[l, [1.0, 1.0]] for l in range(LMAX_SHELL + 1)]},
                spin=1, verbose=0)
    return mol, pts


def _probe_values(pts):
    v = _probe()[0].eval_gto("GTOval_sph", pts)
    out, i = [], 0
    for l in range(LMAX_SHELL + 1):
        out.append(v[:, i:i + 2 * l + 1])
        i += 2 * l + 1
    return out


@lru_cache(maxsize=1)
def _reference_values():
    return _probe_values(_probe()[1])


def real_sph_rotation(R):
    """D^(l)(R) for l = 0..LMAX_SHELL in PySCF's real-spherical order.

    Defined by [R f](r) = f(R^{-1} r) and R Y_i = sum_j Y_j D_ji; obtained by least squares
    on random points, which fixes the ordering and sign conventions automatically.
    """
    Yr = _probe_values(_probe()[1] @ R)
    Y0 = _reference_values()
    return [np.linalg.lstsq(Y0[l], Yr[l], rcond=None)[0] for l in range(LMAX_SHELL + 1)]


def shell_channel_projectors(l, axis, lmax):
    """Projectors Q_0..Q_lmax (and the |m| > lmax remainder) inside one shell of angular momentum l."""
    n = 2 * l + 1
    if l == 0:
        out = [np.eye(1)] + [np.zeros((1, 1))] * lmax
        return out + [np.zeros((1, 1))]
    nth = 2 * l + 1
    thetas = 2 * np.pi * np.arange(nth) / nth
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    Ds = [real_sph_rotation(Rotation.from_rotvec(axis * t).as_matrix())[l] for t in thetas]
    out = []
    for m in range(lmax + 1):
        if m > l:
            out.append(np.zeros((n, n)))
            continue
        c = (1.0 if m == 0 else 2.0) * np.cos(m * thetas) / nth
        out.append(sum(ci * Di for ci, Di in zip(c, Ds)))
    out.append(np.eye(n) - sum(out))
    return out


def aosb_projectors(mol, atom, axis, lmax):
    """AO-space matrices Q_atom(l), l = 0..lmax, plus the remainder (l > lmax)."""
    if mol.cart:
        raise ValueError("AOSB needs pure spherical AOs (mol.cart = False)")
    K = lmax + 2
    Q = np.zeros((K, mol.nao, mol.nao))
    loc = mol.ao_loc_nr()
    for ib in range(mol.nbas):
        if mol.bas_atom(ib) != atom:
            continue
        l = mol.bas_angular(ib)
        if l > LMAX_SHELL:
            raise NotImplementedError(f"shells with l > {LMAX_SHELL}")
        P = shell_channel_projectors(l, axis, lmax)
        for c in range(mol.bas_nctr(ib)):
            p0 = loc[ib] + c * (2 * l + 1)
            sl = slice(p0, p0 + 2 * l + 1)
            for k in range(K):
                Q[k, sl, sl] = P[k]
    return Q


def aosb_channel_weights(mol, C, A, B, lmax=3):
    """Orbital-basis channel weights C^T Q_X(l) S C for X = A, B, and the unprojected D_X S."""
    coords = mol.atom_coords()
    axis = coords[B] - coords[A]
    S = mol.intor("int1e_ovlp")
    out, tot = [], []
    for X in (A, B):
        Q = aosb_projectors(mol, X, axis, lmax)
        out.append(np.einsum("mi,kmn,nj->kij", C, Q @ S, C))
        D = np.diag([1.0 if mol.bas_atom(ib) == X else 0.0 for ib in range(mol.nbas)
                     for _ in range((2 * mol.bas_angular(ib) + 1) * mol.bas_nctr(ib))])
        tot.append(C.T @ D @ S @ C)
    return tuple(out), tuple(tot)
