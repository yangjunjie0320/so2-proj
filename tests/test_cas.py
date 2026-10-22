"""CAS completion compared with independently evaluated complete-space CI RDMs."""

import numpy as np
import pytest
from pyscf import fci, gto

from so2_proj import BondOrder
from so2_proj.aosb import aosb_channel_weights
from so2_proj.density import prepare_dm2


def correlated_cas_rdms():
    """One core, two correlated active orbitals, and one empty orbital.

    The complete CI vector is assembled by determinant addresses, not by
    any RDM reconstruction or cumulant formula under test.
    """
    active_ci = np.array([[0.8, 0.15], [0.15, 0.55]])
    active_ci /= np.linalg.norm(active_ci)
    active_P, active_G = fci.direct_spin1.make_rdm12(active_ci, 2, (1, 1))
    complete_ci = np.zeros((6, 6))  # C(4, 2) determinants for each spin.
    for i in range(2):
        alpha = 1 | (1 << (i + 1))
        alpha_address = fci.cistring.str2addr(4, 2, alpha)
        for j in range(2):
            beta = 1 | (1 << (j + 1))
            beta_address = fci.cistring.str2addr(4, 2, beta)
            complete_ci[alpha_address, beta_address] = active_ci[i, j]
    complete_P, complete_G = fci.direct_spin1.make_rdm12(complete_ci, 4, (2, 2))
    return active_P, active_G, complete_P, complete_G


def raw_two_body(P, G, WA, WB):
    NA = np.einsum("qp,apq->a", P, WA)
    NB = np.einsum("qp,bpq->b", P, WB)
    return 2 * (np.outer(NA, NB) - np.einsum("pqrs,aqp,bsr->ab", G, WA, WB))


def cumulant_two_body(data, WA, WB):
    B1 = np.einsum("ij,ajk,kl,bli->ab", data.P, WA, data.P, WB)
    WA_selected = np.einsum("pi,apq,qj->aij", data.T, WA, data.T)
    WB_selected = np.einsum("pi,bpq,qj->bij", data.T, WB, data.T)
    return B1 - 2 * np.einsum("pqrs,aqp,bsr->ab", data.cumulant, WA_selected, WB_selected)


@pytest.fixture
def nonorthogonal_cas():
    active_P, active_G, full_P, full_G = correlated_cas_rdms()
    rng = np.random.default_rng(713)
    A = rng.normal(size=(4, 4))
    S = A @ A.T + 2 * np.eye(4)
    values, vectors = np.linalg.eigh(S)
    X = (vectors / np.sqrt(values)) @ vectors.T
    U, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    C = X @ U
    D = C @ full_P @ C.T
    WA = rng.normal(size=(3, 4, 4))
    WB = rng.normal(size=(3, 4, 4))
    # Full-cell weights need not be symmetric; keep every matrix entry.
    WA_orth = np.einsum("ip,apq,jq->aij", U, WA, U)
    WB_orth = np.einsum("ip,bpq,jq->bij", U, WB, U)
    reference = raw_two_body(full_P, full_G, WA, WB)
    return active_P, active_G, full_P, full_G, S, C, D, WA_orth, WB_orth, reference


@pytest.mark.parametrize("representation", ["reordered_indices", "selected_coeff", "rotated_active"])
def test_cas_completion_matches_explicit_complete_ci(nonorthogonal_cas, representation):
    Pa, Ga, Pf, Gf, S, C, D, WA, WB, reference = nonorthogonal_cas
    np.testing.assert_allclose(Pf[1:3, 1:3], Pa, atol=1e-14)
    assert np.all((np.linalg.eigvalsh(Pa) > 0.01) & (np.linalg.eigvalsh(Pa) < 1.99))
    assert np.linalg.norm(WA - WA.transpose(0, 2, 1)) > 1
    if representation == "reordered_indices":
        order = [1, 0]
        Ga = Ga[np.ix_(order, order, order, order)]
        coefficients, indices = C, [2, 1]
        expected_P = Pa[np.ix_(order, order)]
    elif representation == "selected_coeff":
        coefficients, indices = C[:, 1:3], None
        expected_P = Pa
    else:
        angle = 0.37
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        Ga = np.einsum("pi,qj,rk,sl,pqrs->ijkl", R, R, R, R, Ga)
        coefficients, indices = C[:, 1:3] @ R, None
        expected_P = R.T @ Pa @ R
    data = prepare_dm2(
        Ga, coefficients, dm1=D, overlap=S, indices=indices,
        completion="cas", nelectron=4,
    )
    np.testing.assert_allclose(data.Ps, expected_P, atol=2e-13)
    np.testing.assert_allclose(cumulant_two_body(data, WA, WB), reference, atol=2e-12)
    B1 = np.einsum("ij,ajk,kl,bli->ab", data.P, WA, data.P, WB)
    assert np.max(np.abs(B1 - reference)) > 0.1


@pytest.mark.parametrize("omit_empty", [False, True])
def test_full_correlated_density_matches_complete_ci(nonorthogonal_cas, omit_empty):
    _, _, _, G, S, C, D, WA, WB, reference = nonorthogonal_cas
    if omit_empty:
        # A rectangular coefficient matrix is complete for this electronic state.
        C = C[:, :3]
        G = G[:3, :3, :3, :3]
    data = prepare_dm2(G, C, dm1=D, overlap=S, nelectron=4)
    np.testing.assert_allclose(cumulant_two_body(data, WA, WB), reference, atol=2e-12)


def test_active_tensor_is_not_implicitly_a_complete_state(nonorthogonal_cas):
    _, G, _, _, S, C, D, *_ = nonorthogonal_cas
    with pytest.raises(ValueError, match="complete dm1 support"):
        prepare_dm2(G, C, dm1=D, overlap=S, indices=[1, 2])


def test_arbitrary_correlated_subblock_cannot_be_cas_completed(nonorthogonal_cas):
    _, G, _, _, S, C, D, *_ = nonorthogonal_cas
    with pytest.raises(ValueError, match="CAS"):
        prepare_dm2(
            G[:1, :1, :1, :1], C, dm1=D, overlap=S,
            indices=[1], completion="cas",
        )


@pytest.mark.parametrize("weight_source", ["nonsymmetric_channels", "mulliken"])
def test_public_two_body_cas_matches_complete_ci(monkeypatch, weight_source):
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 1.6 0; H 1.8 0 0",
        basis="sto-3g", unit="Bohr", verbose=0,
    )
    S = mol.intor("int1e_ovlp")
    values, vectors = np.linalg.eigh(S)
    X = (vectors / np.sqrt(values)) @ vectors.T
    rng = np.random.default_rng(51)
    U, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    C = X @ U
    _, Ga, Pf, Gf = correlated_cas_rdms()
    D = C @ Pf @ C.T
    bo = BondOrder(mol, 0, 1, partition="mulliken", lmax=1)
    if weight_source == "nonsymmetric_channels":
        WA, WB = rng.normal(size=(2, 3, 4, 4))
        # Store AO covariant weights so the mock also tests the production X.
        WA_ao = np.einsum("ip,apq,jq->aij", S @ C, WA, S @ C)
        WB_ao = np.einsum("ip,bpq,jq->bij", S @ C, WB, S @ C)

        def weights(coefficients):
            A = coefficients.T @ WA_ao @ coefficients
            B = coefficients.T @ WB_ao @ coefficients
            return (A, B), (A.sum(axis=0), B.sum(axis=0)), {}

        monkeypatch.setattr(bo, "_weights", weights)
    else:
        (WA, WB), _ = aosb_channel_weights(mol, C, 0, 1, lmax=1)
    reference = raw_two_body(Pf, Gf, WA, WB)
    reverse = [1, 0]
    Ga = Ga[np.ix_(reverse, reverse, reverse, reverse)]
    result = bo.two_body(Ga, coeff=C, dm1=D, indices=[2, 1], completion="cas")
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, reference, atol=3e-12)
    summary = bo.analyze()
    assert summary["total"] == pytest.approx(reference.sum(), abs=3e-12)
    assert abs(summary["sum_error"]) < 3e-12
    full_result = bo.two_body(Gf, coeff=C, dm1=D)
    np.testing.assert_allclose(full_result, reference, atol=3e-12)
