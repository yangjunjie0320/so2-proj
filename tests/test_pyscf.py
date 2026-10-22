"""Small-molecule checks against independently evaluated Mayer bond orders."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyscf import gto, lib, scf
from scipy.spatial.transform import Rotation

from so2_proj import BondOrder
from so2_proj.aosb import aosb_channel_weights
from so2_proj.bondorder import channel_matrices


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(previous)


def rhf(atom):
    mol = gto.M(atom=atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    assert mf.converged
    return mol, mf


@pytest.fixture(scope="module")
def h2():
    return rhf("H 0 0 0; H 0 0 0.74")


@pytest.fixture(scope="module")
def n2():
    return rhf("N 0 0 -0.55; N 0 0 0.55")


@pytest.fixture(scope="module")
def ethylene():
    return rhf(
        "C 0 0 -0.67; C 0 0 0.67; "
        "H 0.92 0 -1.23; H -0.92 0 -1.23; "
        "H 0.92 0 1.23; H -0.92 0 1.23"
    )


@pytest.fixture(scope="module")
def rotated_ethylene(ethylene):
    mol, _ = ethylene
    rotation = Rotation.from_rotvec([0.37, -0.51, 0.22]).as_matrix()
    coordinates = mol.atom_coords(unit="Angstrom") @ rotation.T + [0.4, -0.7, 0.2]
    return rhf([(mol.atom_symbol(i), r) for i, r in enumerate(coordinates)])


def mayer_total(mol, density, a=0, b=1):
    ps = density @ mol.intor("int1e_ovlp")
    ia = np.arange(*mol.aoslice_by_atom()[a, 2:])
    ib = np.arange(*mol.aoslice_by_atom()[b, 2:])
    return np.sum(ps[np.ix_(ia, ib)] * ps[np.ix_(ib, ia)].T)


@pytest.mark.parametrize("molecule", ["h2", "n2", "ethylene"])
def test_mulliken_total_matches_mayer(molecule, request):
    mol, mf = request.getfixturevalue(molecule)
    density = mf.make_rdm1()
    bond = BondOrder(mol, 0, 1, partition="mulliken", lmax=2)
    matrix = bond.one_body(density)
    assert matrix.shape == (4, 4)
    assert_allclose(matrix.sum(), mayer_total(mol, density), atol=3e-10)
    info = bond.analyze()
    assert_allclose(info["total"], mayer_total(mol, density), atol=3e-10)
    assert abs(info["sum_error"]) < 3e-10


def test_h2_has_only_sigma_channel(h2):
    mol, mf = h2
    matrix = BondOrder(mol, 0, 1, partition="mulliken", lmax=2).one_body(mf.make_rdm1())
    assert_allclose(matrix[0, 0], 1.0, atol=2e-11)
    matrix[0, 0] = 0
    assert_allclose(matrix, 0, atol=1e-13)


def test_linear_n2_channels_do_not_mix(n2):
    mol, mf = n2
    matrix = BondOrder(mol, 0, 1, partition="mulliken", lmax=2).one_body(mf.make_rdm1())
    assert_allclose(matrix - np.diag(np.diag(matrix)), 0, atol=3e-10)
    assert matrix[0, 0] > 0
    assert matrix[1, 1] > 0
    assert_allclose(matrix[2:, :], 0, atol=3e-11)


def test_planar_ethylene_sigma_pi_cross_blocks_vanish(ethylene):
    mol, mf = ethylene
    matrix = BondOrder(mol, 0, 1, partition="mulliken", lmax=2).one_body(mf.make_rdm1())
    assert_allclose([matrix[0, 1], matrix[1, 0]], 0, atol=3e-10)


def test_mulliken_rotation_and_translation_invariance(n2):
    mol, mf = n2
    expected = BondOrder(mol, 0, 1, partition="mulliken", lmax=2).one_body(mf.make_rdm1())
    rotation = Rotation.from_rotvec([0.37, -0.51, 0.22]).as_matrix()
    coordinates = mol.atom_coords(unit="Angstrom") @ rotation.T + [0.4, -0.7, 0.2]
    rotated, rotated_mf = rhf([(mol.atom_symbol(i), r) for i, r in enumerate(coordinates)])
    actual = BondOrder(rotated, 0, 1, partition="mulliken", lmax=2).one_body(rotated_mf.make_rdm1())
    assert_allclose(actual, expected, atol=3e-9)


def test_aosb_weights_keep_remainder_and_sum(n2):
    mol, mf = n2
    (wa, wb), (ta, tb) = aosb_channel_weights(mol, mf.mo_coeff, 0, 1, lmax=0)
    assert wa.shape == (2, mol.nao, mol.nao)
    assert_allclose(wa.sum(axis=0), ta, atol=2e-13)
    assert_allclose(wb.sum(axis=0), tb, atol=2e-13)
    matrix, _, _, _ = channel_matrices(wa, wb, np.diag(mf.mo_occ))
    assert matrix[1, 1] > 0  # The omitted pi channel remains in the residual block.
    assert_allclose(matrix.sum(), mayer_total(mol, mf.make_rdm1()), atol=3e-10)


@pytest.mark.parametrize("cell", ["full", "avg"])
def test_becke_h2_channel_sum_and_axis_symmetry(h2, cell):
    mol, mf = h2
    bond = BondOrder(
        mol, 0, 1, partition="becke", cell=cell, lmax=1,
        nz=48, nrho=24, nphi=16, pad=6.0,
    )
    matrix = bond.one_body(mf.make_rdm1())
    assert matrix.shape == (3, 3)
    info = bond.analyze()
    assert_allclose(matrix.sum(), info["total"], atol=2e-10)
    assert_allclose(matrix[1:, :], 0, atol=2e-12)
    assert_allclose(matrix[:, 1:], 0, atol=2e-12)
    assert np.isfinite(matrix).all()
    assert matrix[0, 0] > 0


@pytest.mark.parametrize("cell", ["full", "avg"])
@pytest.mark.parametrize("molecule", ["n2", "ethylene"])
def test_becke_sigma_pi_selection_rules(molecule, cell, request):
    mol, mf = request.getfixturevalue(molecule)
    bond = BondOrder(
        mol, 0, 1, partition="becke", cell=cell, lmax=2,
        nz=64, nrho=32, nphi=32, pad=6.0,
    )
    matrix = bond.one_body(mf.make_rdm1())
    assert_allclose([matrix[0, 1], matrix[1, 0]], 0, atol=3e-10)
    assert_allclose(matrix.sum(), bond.analyze()["total"], atol=3e-10)
    if molecule == "n2":
        assert_allclose(matrix - np.diag(np.diag(matrix)), 0, atol=3e-10)


@pytest.mark.parametrize("cell", ["full", "avg"])
def test_becke_nonlinear_molecule_rotation_invariance(ethylene, rotated_ethylene, cell):
    options = dict(
        partition="becke", cell=cell, lmax=2, nz=64, nrho=32, nphi=32, pad=6.0,
    )
    matrices = []
    for mol, mf in (ethylene, rotated_ethylene):
        matrices.append(BondOrder(mol, 0, 1, **options).one_body(mf.make_rdm1()))
    # Discrete azimuthal sampling changes phase under an arbitrary rotation.
    # At 32 angles the measured maximum difference is below 3e-8 for both cells;
    # 64 angles reduces it below 1e-10. Keep the cheaper grid with explicit margin.
    assert_allclose(matrices[0], matrices[1], atol=2e-7, rtol=0)


def test_cartesian_aos_are_rejected():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", cart=True, verbose=0)
    with pytest.raises((ValueError, NotImplementedError), match="spherical|cart"):
        BondOrder(mol, 0, 1, partition="mulliken").one_body(np.eye(mol.nao))
