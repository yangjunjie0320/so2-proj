"""Algebraic and Fourier checks independent of an electronic-structure calculation."""

import numpy as np
from numpy.testing import assert_allclose

from so2_proj.bondorder import channel_matrices, closed_shell_rdm2, total_bond_orders
from so2_proj.grid import CylinderGrid, channel_components
from so2_proj.realspace import realspace_channel_weights


def test_nonsymmetric_weights_and_explicit_determinant_rdm2():
    rng = np.random.default_rng(17)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    density = rotation @ np.diag([2.0, 2.0, 0.0, 0.0]) @ rotation.T
    wa = rng.normal(size=(3, 4, 4))
    wb = rng.normal(size=(3, 4, 4))
    # Construct the determinant tensor explicitly, independently of the helper.
    pair_density = np.empty((4, 4, 4, 4))
    for p, q, r, s in np.ndindex(pair_density.shape):
        pair_density[p, q, r, s] = (
            density[p, q] * density[r, s] - density[p, s] * density[r, q] / 2
        )
    assert_allclose(closed_shell_rdm2(density), pair_density, atol=1e-14)
    b1, b2, na, nb = channel_matrices(wa, wb, density, pair_density)
    reference = np.array(
        [[np.trace(density @ a @ density @ b) for b in wb] for a in wa]
    )
    assert_allclose(b1, reference, atol=2e-13)
    assert_allclose(b2, reference, atol=2e-13)
    assert_allclose(na, [np.trace(density @ a) for a in wa], atol=1e-14)
    assert_allclose(nb, [np.trace(density @ b) for b in wb], atol=1e-14)
    # A transpose on only one end must not accidentally go unnoticed.
    wrong = np.array(
        [[np.trace(density @ a.T @ density @ b) for b in wb] for a in wa]
    )
    assert not np.allclose(reference, wrong)


def test_channel_sum_and_no_implicit_two_body_result():
    rng = np.random.default_rng(9)
    density = np.diag([2.0, 2.0, 0.0])
    wa, wb = rng.normal(size=(2, 4, 3, 3))
    b1, b2, _, _ = channel_matrices(wa, wb, density)
    assert b2 is None
    total1, total2 = total_bond_orders(wa.sum(axis=0), wb.sum(axis=0), density)
    assert total2 is None
    assert_allclose(b1.sum(), total1, atol=1e-13)
    pair_density = closed_shell_rdm2(density)
    _, b2, _, _ = channel_matrices(wa, wb, density, pair_density)
    _, total2 = total_bond_orders(wa.sum(axis=0), wb.sum(axis=0), density, pair_density)
    assert_allclose(b2.sum(), total2, atol=1e-13)


def test_fft_channels_keep_both_signs_and_remainder():
    phi = 2 * np.pi * np.arange(16) / 16
    modes = np.stack(
        [np.ones_like(phi), np.cos(phi), np.sin(2 * phi), np.cos(3 * phi)], axis=1
    )
    components = channel_components(modes, (1, 1, 16), lmax=2)
    assert len(components) == 4
    for channel, component in enumerate(components):
        expected = np.zeros_like(modes)
        expected[:, channel] = modes[:, channel]
        assert_allclose(component, expected, atol=2e-15)
    assert_allclose(sum(components), modes, atol=1e-15)


def test_cylinder_grid_uses_original_coordinates_and_bond_axis():
    coords = np.array([[1.2, -0.4, 2.0], [2.3, 0.7, 3.1]])
    grid = CylinderGrid.build(coords, 0, 1, nz=32, nrho=8, nphi=16, pad=2)
    ex, ey, ez = grid.frame
    frame = np.stack([ex, ey, ez])
    assert_allclose(frame @ frame.T, np.eye(3), atol=1e-15)
    assert_allclose(ez, (coords[1] - coords[0]) / np.linalg.norm(coords[1] - coords[0]))
    assert_allclose(grid.origin, coords[0])
    points, weights, shape = next(grid.slabs(zchunk=2))
    local = (points - coords[0]) @ frame.T
    assert points.shape == (np.prod(shape), 3)
    assert weights.shape == (np.prod(shape),)
    assert np.all(weights > 0)
    assert_allclose(local[:, 2].reshape(shape)[:, 0, 0], grid.z[:2])
    assert_allclose(np.linalg.norm(local[:, :2], axis=1).reshape(shape)[0, :, 0], grid.rho)


def test_realspace_full_cell_keeps_asymmetry_and_reports_complete_overlap():
    # A third off-axis atom makes the first atom's Becke cell non-axisymmetric.
    coords = np.array([[0, 0, 0], [0, 0, 1.5], [1.5, 0.3, 0.5]])

    def orbitals(points):
        gaussian = np.pi ** (-0.75) * np.exp(-np.sum(points**2, axis=1) / 2)
        return np.column_stack([gaussian, np.sqrt(2) * points[:, 0] * gaussian])

    result = realspace_channel_weights(
        orbitals, coords, [1, 1, 1], 0, 1, lmax=0,
        nz=48, nrho=32, nphi=16, pad=6,
    )
    assert result["overlap"].shape == (2, 2)
    assert_allclose(result["overlap"], np.eye(2), atol=3e-5)
    assert_allclose(np.diag(result["overlap"]), result["norm"], atol=1e-14)
    assert result["npoints"] > 0
    full_a, _ = result["full"]
    assert abs(full_a[0, 1, 0] - full_a[0, 0, 1]) > 1e-3
    for kind in ("full", "avg"):
        for weights, total in zip(result[kind], result[kind + "_total"]):
            assert weights.shape == (2, 2, 2)
            assert_allclose(weights.sum(axis=0), total, atol=3e-13)
            if kind == "avg":
                assert_allclose(weights, weights.transpose(0, 2, 1), atol=1e-14)
                assert np.min(np.linalg.eigvalsh(weights)) > -1e-13
