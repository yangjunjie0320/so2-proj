"""One- and two-body channel bond-order matrices from channel weights.

All quantities are in an orthonormal orbital basis {psi_p}:
  P_pq       one-particle density matrix (spin-summed, trace N)
  G_pqrs     two-particle density matrix, G_pqrs = <a+_p a+_r a_s a_q> (PySCF convention,
             trace N(N-1))
  W^X(l)_pq  channel weight of atom X
Formulas
  N_X(l)       = sum_pq P_qp W^X_pq(l)
  B1(lA, lB)   = Tr[P W^A(lA) P W^B(lB)]
  B2(lA, lB)   = 2 [N_A(lA) N_B(lB) - sum_pqrs G_pqrs W^A_qp(lA) W^B_sr(lB)]
For a closed-shell determinant B2 = B1.
"""

import numpy as np


def channel_matrices(WA, WB, P, G=None):
    """Return (B1, B2, NA, NB). Without ``G``, B2 is explicitly ``None``."""
    NA = np.einsum("qp,kpq->k", P, WA)
    NB = np.einsum("qp,kpq->k", P, WB)
    PWA = np.einsum("ij,kjl->kil", P, WA)
    PWB = np.einsum("ij,kjl->kil", P, WB)
    B1 = np.einsum("aij,bji->ab", PWA, PWB)
    if G is None:
        return B1, None, NA, NB
    pair = np.einsum("pqrs,aqp,bsr->ab", G, WA, WB)
    B2 = 2 * (np.outer(NA, NB) - pair)
    return B1, B2, NA, NB


def total_bond_orders(WA, WB, P, G=None):
    """Unresolved (B1, B2) from unprojected weights of shape (n, n)."""
    B1, B2, _, _ = channel_matrices(WA[None], WB[None], P, G)
    return float(B1[0, 0]), None if B2 is None else float(B2[0, 0])


def closed_shell_rdm2(P):
    """G_pqrs = P_pq P_rs - P_ps P_rq / 2 for a closed-shell determinant (orthonormal basis)."""
    return np.einsum("pq,rs->pqrs", P, P) - 0.5 * np.einsum("ps,rq->pqrs", P, P)
