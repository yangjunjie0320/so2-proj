"""Density contracts, AO metrics, and explicit completion assumptions."""

import numpy as np
import pytest

from so2_proj.density import orthogonalize_dm1, prepare_dm2


def determinant_rdm2(P):
    return np.einsum("pq,rs->pqrs", P, P) - 0.5 * np.einsum("ps,rq->pqrs", P, P)


def nonorthogonal_determinant():
    S = np.array([[1.0, 0.2, -0.1], [0.2, 1.0, 0.15], [-0.1, 0.15, 1.0]])
    values, vectors = np.linalg.eigh(S)
    X = (vectors / np.sqrt(values)) @ vectors.T
    U, _ = np.linalg.qr(np.random.default_rng(37).normal(size=(3, 3)))
    C = X @ U
    P = np.diag([2.0, 2.0, 0.0])
    return S, C, C @ P @ C.T


def test_symmetric_orthogonalization_preserves_complete_ao_density():
    S, C, D = nonorthogonal_determinant()
    P, X = orthogonalize_dm1(D, S, nelectron=4)
    np.testing.assert_allclose(X, X.T, atol=1e-13)
    np.testing.assert_allclose(X.T @ S @ X, np.eye(3), atol=1e-13)
    np.testing.assert_allclose(X @ P @ X.T, D, atol=1e-13)
    np.testing.assert_allclose(C.T @ S @ D @ S @ C, np.diag([2, 2, 0]), atol=1e-13)
    assert np.trace(P) == pytest.approx(4)


@pytest.mark.parametrize(
    "density, overlap, exception, message",
    [
        (np.ones((2, 2, 2)), np.eye(2), NotImplementedError, "spin-resolved"),
        (np.eye(2, dtype=complex), np.eye(2), NotImplementedError, "complex"),
        (np.eye(2), np.eye(2, dtype=complex), NotImplementedError, "complex"),
        (np.eye(3), np.eye(2), ValueError, "matching overlap"),
        (np.array([[1, 0.1], [0, 1]]), np.eye(2), ValueError, "dm1 must be symmetric"),
        (np.eye(2), np.array([[1, 0.1], [0, 1]]), ValueError, "overlap must be symmetric"),
        (np.eye(2), np.diag([1.0, 1e-14]), ValueError, "linearly dependent"),
        (np.eye(2), np.diag([1.0, -1.0]), ValueError, "linearly dependent"),
        (np.diag([-0.1, 2.1]), np.eye(2), ValueError, "occupations"),
        (np.diag([2.1, 1.9]), np.eye(2), ValueError, "occupations"),
        (np.diag([1.0, 0.5]), np.eye(2), ValueError, "electron count"),
        (np.diag([np.nan, 2.0]), np.eye(2), ValueError, "finite"),
        (np.eye(2), np.diag([np.inf, 1.0]), ValueError, "finite"),
    ],
)
def test_invalid_one_body_inputs_are_explicit(density, overlap, exception, message):
    with pytest.raises(exception, match=message):
        orthogonalize_dm1(density, overlap)


def test_full_electron_count_is_checked_against_molecule():
    with pytest.raises(ValueError, match="does not match nelectron"):
        orthogonalize_dm1(np.diag([2.0, 0.0]), np.eye(2), nelectron=4)


def test_rectangular_full_orbital_space_keeps_order_and_does_not_mutate_inputs():
    S, C, D = nonorthogonal_determinant()
    G = determinant_rdm2(2 * np.eye(2))
    copies = tuple(a.copy() for a in (S, C, D, G))
    data = prepare_dm2(G, C, dm1=D, overlap=S, indices=[1, 0], nelectron=4)
    assert data.T.shape == (3, 2)
    np.testing.assert_allclose(data.X @ data.T, C[:, [1, 0]], atol=1e-13)
    np.testing.assert_allclose(data.Ps, 2 * np.eye(2), atol=1e-13)
    np.testing.assert_allclose(data.cumulant, 0, atol=1e-13)
    for actual, before in zip((S, C, D, G), copies):
        np.testing.assert_array_equal(actual, before)


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3], [0.0], [True, False, True], 1])
def test_invalid_orbital_indices_are_rejected(indices):
    S, C, D = nonorthogonal_determinant()
    with pytest.raises(ValueError, match="indices"):
        prepare_dm2(np.zeros((2,) * 4), C, dm1=D, overlap=S, indices=indices)


def test_slices_preserve_the_selected_coefficient_order():
    S, C, D = nonorthogonal_determinant()
    data = prepare_dm2(
        determinant_rdm2(2 * np.eye(2)), C,
        dm1=D, overlap=S, indices=slice(1, None, -1),
    )
    np.testing.assert_allclose(data.X @ data.T, C[:, [1, 0]], atol=1e-13)


def test_coefficients_are_not_silently_orthonormalized():
    with pytest.raises(ValueError, match="orthonormal"):
        prepare_dm2(
            determinant_rdm2(2 * np.eye(2)), 2 * np.eye(2),
            dm1=2 * np.eye(2), overlap=np.eye(2),
        )


def test_full_completion_rejects_omitted_occupied_orbitals():
    with pytest.raises(ValueError, match="complete dm1 support"):
        prepare_dm2(
            np.full((1,) * 4, 2.0), np.eye(2)[:, :1],
            dm1=2 * np.eye(2), overlap=np.eye(2),
        )


def test_cas_completion_rejects_correlated_inactive_orbitals():
    with pytest.raises(ValueError, match="doubly occupied core"):
        prepare_dm2(
            np.full((1,) * 4, 2.0), np.eye(3), indices=[2],
            dm1=np.diag([1.0, 1.0, 2.0]), overlap=np.eye(3), completion="cas",
        )


def test_cas_completion_rejects_active_core_coupling():
    angle = 0.23
    U = np.eye(4)
    U[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    D = U @ np.diag([2.0, 1.4, 0.6, 0.0]) @ U.T
    with pytest.raises(ValueError, match="couple active and inactive"):
        prepare_dm2(
            np.zeros((2,) * 4), np.eye(4), indices=[1, 2],
            dm1=D, overlap=np.eye(4), completion="cas",
        )


@pytest.mark.parametrize("completion", ["auto", None, "FULL"])
def test_completion_is_explicit(completion):
    with pytest.raises(ValueError, match="completion"):
        prepare_dm2(
            np.full((1,) * 4, 2.0), np.eye(1),
            dm1=2 * np.eye(1), overlap=np.eye(1), completion=completion,
        )


def test_two_body_trace_and_partial_trace_are_separate_checks():
    P = np.diag([2.0, 0.0])
    G = determinant_rdm2(P)
    with pytest.raises(ValueError, match="trace must equal"):
        prepare_dm2(G / 2, np.eye(2), dm1=P, overlap=np.eye(2))
    with pytest.raises(ValueError, match="partial trace"):
        prepare_dm2(G, np.eye(2), dm1=P[::-1, ::-1], overlap=np.eye(2))


@pytest.mark.parametrize(
    "G, exception, message",
    [
        (np.zeros((2, 2)), ValueError, "four axes"),
        (np.zeros((3, 2, 2, 2, 2)), NotImplementedError, "spin-resolved"),
        (np.zeros((2,) * 4, dtype=complex), NotImplementedError, "complex"),
        (np.full((2,) * 4, np.nan), ValueError, "finite"),
    ],
)
def test_invalid_two_body_arrays_are_explicit(G, exception, message):
    with pytest.raises(exception, match=message):
        prepare_dm2(G, np.eye(2), dm1=np.diag([2.0, 0.0]), overlap=np.eye(2))


def test_two_body_symmetries_do_not_assume_individual_pair_transposes():
    P = np.diag([2.0, 0.0])
    G = determinant_rdm2(P)
    G[0, 0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="particle-pair exchange"):
        prepare_dm2(G, np.eye(2), dm1=P, overlap=np.eye(2))
    G[0, 1, 0, 0] = 0.1
    with pytest.raises(ValueError, match="Hermitian"):
        prepare_dm2(G, np.eye(2), dm1=P, overlap=np.eye(2))


@pytest.mark.parametrize("nelectron", [0, 1])
def test_zero_and_single_electron_spaces_require_an_entirely_zero_dm2(nelectron):
    G = np.zeros((2,) * 4)
    # These coherences obey both symmetries and have zero partial traces,
    # but no electron pair exists in either space.
    G[0, 1, 0, 1] = G[1, 0, 1, 0] = 1.0
    P = np.diag([float(nelectron), 0.0])
    with pytest.raises(ValueError, match="fewer than two electrons"):
        prepare_dm2(G, np.eye(2), dm1=P, overlap=np.eye(2))
    data = prepare_dm2(np.zeros_like(G), np.eye(2), dm1=P, overlap=np.eye(2))
    np.testing.assert_allclose(data.Ps, P)


def test_zero_orbital_cas_space_keeps_the_closed_shell_core():
    data = prepare_dm2(
        np.empty((0,) * 4), np.empty((2, 0)), dm1=2 * np.eye(2),
        overlap=np.eye(2), completion="cas",
    )
    assert data.T.shape == (2, 0)
    assert data.Ps.shape == (0, 0)
    assert data.cumulant.shape == (0, 0, 0, 0)
    np.testing.assert_allclose(data.P, 2 * np.eye(2))
