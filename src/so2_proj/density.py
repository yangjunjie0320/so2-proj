"""Real, spin-summed density matrices and their orbital-space contracts.

The public one-particle density is in the molecule's nonorthogonal AO basis.
Two-particle densities use orthonormal orbitals and the PySCF convention
``G[p,q,r,s] = <a+_p a+_r a_s a_q>`` with trace ``N * (N - 1)``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoBodyDensity:
    """Densities and maps needed for a full or CAS two-body contraction.

    ``X`` maps the complete orthonormal basis to AOs; ``T`` maps the selected
    two-body orbitals to that orthonormal basis. ``P`` and ``Ps`` are the full
    and selected one-particle densities, respectively. The cumulant is stored
    only in the selected orbital space.
    """

    P: np.ndarray
    X: np.ndarray
    T: np.ndarray
    Ps: np.ndarray
    cumulant: np.ndarray


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise NotImplementedError(f"complex {name} is not supported")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a real numeric array") from err
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _close(actual, expected, atol):
    return np.allclose(actual, expected, atol=atol, rtol=atol)


def _tolerance(atol):
    if isinstance(atol, (bool, np.bool_)) or not np.isscalar(atol):
        raise ValueError("atol must be a positive finite number")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be a positive finite number")


def _particle_number(value, name, atol):
    rounded = round(float(value))
    if rounded < 0 or not _close(value, rounded, atol):
        raise ValueError(f"{name} must be a nonnegative integer")
    return rounded


def orthogonalize_dm1(dm1, overlap, *, nelectron=None, atol=1e-8):
    """Return ``(P, X)`` for a complete real AO one-particle density.

    ``X = S**(-1/2)`` and ``P = X.T @ S @ dm1 @ S @ X``; thus
    ``X.T @ S @ X = I`` and ``dm1 = X @ P @ X.T``. ``dm1`` must be a
    symmetric spin-summed matrix with natural occupations between zero and
    two and an integer particle number. An optional ``nelectron`` checks the
    molecule's electron count. Explicit spin blocks and complex inputs are
    unsupported.

    Overlap eigenvalues at or below ``1e-12`` times the largest eigenvalue
    are rejected. No AO directions or small density occupations are dropped.
    Antisymmetric roundoff within ``atol`` is averaged before transformation.
    """
    _tolerance(atol)
    D = _real_array(dm1, "dm1")
    if D.ndim == 3:
        raise NotImplementedError("spin-resolved dm1 is not supported; supply a spin-summed matrix")
    S = _real_array(overlap, "overlap")
    if S.ndim != 2 or S.shape[0] != S.shape[1] or not S.shape[0]:
        raise ValueError("overlap must be a nonempty square matrix")
    if D.shape != S.shape:
        raise ValueError("dm1 must be a square AO matrix matching overlap")
    if not _close(S, S.T, atol):
        raise ValueError("overlap must be symmetric")
    if not _close(D, D.T, atol):
        raise ValueError("dm1 must be symmetric")
    S = (S + S.T) / 2
    D = (D + D.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise ValueError("overlap is singular or nearly linearly dependent; no AO directions are discarded")
    X = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    sqrt_S = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    P = sqrt_S @ D @ sqrt_S
    P = (P + P.T) / 2
    occupations = np.linalg.eigvalsh(P)
    if occupations[0] < -atol or occupations[-1] > 2 + atol:
        raise ValueError("dm1 natural occupations must be between zero and two")
    actual = _particle_number(np.trace(P), "dm1 electron count", atol)
    if nelectron is not None:
        try:
            expected = float(nelectron)
        except (TypeError, ValueError) as err:
            raise ValueError("nelectron must be a nonnegative integer") from err
        if not np.isfinite(expected):
            raise ValueError("nelectron must be a nonnegative integer")
        expected = _particle_number(expected, "nelectron", atol)
        if actual != expected:
            raise ValueError("dm1 electron count does not match nelectron")
    return P, X


def _select_coeff(coeff, nao, indices):
    C = _real_array(coeff, "coeff")
    if C.ndim != 2 or C.shape[0] != nao:
        raise ValueError("coeff must have shape (nao, norb)")
    if indices is None:
        return C
    if isinstance(indices, slice):
        try:
            selected = np.arange(C.shape[1])[indices]
        except (TypeError, ValueError) as err:
            raise ValueError("indices must be a valid orbital slice") from err
    else:
        selected = np.asarray(indices)
        if selected.ndim != 1 or (selected.size and selected.dtype.kind not in "iu"):
            raise ValueError("indices must be a one-dimensional sequence of integer orbital indices")
        selected = selected.astype(np.intp)
    if np.any(selected < 0) or np.any(selected >= C.shape[1]):
        raise ValueError("indices are outside the coeff column range")
    if np.unique(selected).size != selected.size:
        raise ValueError("indices must not contain duplicate orbitals")
    return C[:, selected]


def prepare_dm2(dm2, coeff, *, dm1, overlap, indices=None, completion="full",
                nelectron=None, atol=1e-8):
    """Validate and prepare a full-state or CAS spin-summed two-body density.

    ``dm1`` always describes the complete electronic state in the AO basis.
    ``coeff[:, indices]`` supplies the ordered orthonormal orbitals for all
    four ``dm2`` axes; ``indices=None`` selects all coefficient columns.
    Integer sequences preserve their order and cannot contain duplicates.

    With ``completion='full'`` the selected orbitals must contain the entire
    support of ``dm1``. With ``completion='cas'`` the omitted orbitals must
    comprise fixed doubly occupied core and empty orbitals, without one-body
    coupling to the active space. An arbitrary truncated two-body tensor
    cannot be completed by this assumption.

    Returned ``cumulant`` is ``G - P[p,q]*P[r,s] + P[p,s]*P[r,q]/2`` in the
    selected space. For arbitrary, possibly nonsymmetric channel weights,
    ``B2 = B1(full P) - 2*einsum('pqrs,aqp,bsr', cumulant, WA_s, WB_s)``.
    These checks establish basic RDM consistency, not N-representability.
    """
    if completion not in ("full", "cas"):
        raise ValueError("completion must be 'full' or 'cas'")
    P, X = orthogonalize_dm1(dm1, overlap, nelectron=nelectron, atol=atol)
    S = _real_array(overlap, "overlap")
    S = (S + S.T) / 2
    C = _select_coeff(coeff, P.shape[0], indices)
    if not _close(C.T @ S @ C, np.eye(C.shape[1]), atol):
        raise ValueError("selected coeff orbitals must be orthonormal in the AO overlap metric")
    T = X.T @ S @ C
    Ps = T.T @ P @ T
    Ps = (Ps + Ps.T) / 2
    residual = P - T @ Ps @ T.T
    if completion == "full":
        if not _close(residual, np.zeros_like(residual), atol):
            raise ValueError("full dm2 orbital space must contain the complete dm1 support")
    else:
        if not _close(residual @ T, np.zeros_like(T), atol):
            raise ValueError("CAS dm1 must not couple active and inactive orbital spaces")
        if not _close(residual @ residual, 2 * residual, atol):
            raise ValueError("CAS inactive density must contain only doubly occupied core and empty orbitals")
    nselected = _particle_number(np.trace(Ps), "selected-space electron count", atol)
    G = _real_array(dm2, "dm2")
    if G.ndim == 5:
        raise NotImplementedError("spin-resolved dm2 is not supported; supply a spin-summed tensor")
    if G.shape != (C.shape[1],) * 4:
        raise ValueError("dm2 must have four axes matching the selected coeff orbital count")
    if nselected < 2 and not _close(G, np.zeros_like(G), atol):
        raise ValueError("dm2 must be zero when its selected space has fewer than two electrons")
    if not _close(G, G.transpose(2, 3, 0, 1), atol):
        raise ValueError("dm2 must satisfy particle-pair exchange symmetry")
    if not _close(G, G.transpose(1, 0, 3, 2), atol):
        raise ValueError("dm2 must satisfy real Hermitian symmetry")
    pair_trace = np.einsum("pprr->", G)
    if not _close(pair_trace, nselected * (nselected - 1), atol):
        raise ValueError("dm2 trace must equal N * (N - 1) in its selected orbital space")
    contraction = np.einsum("pqrr->pq", G)
    if not _close(contraction, (nselected - 1) * Ps, atol):
        raise ValueError("dm2 partial trace does not match dm1 in its selected orbital space")
    cumulant = G.copy()
    cumulant -= np.einsum("pq,rs->pqrs", Ps, Ps)
    cumulant += 0.5 * np.einsum("ps,rq->pqrs", Ps, Ps)
    return TwoBodyDensity(P, X, T, Ps, cumulant)
