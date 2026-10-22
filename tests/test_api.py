"""Public input contracts, reusable configuration, and diagnostic snapshots."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pyscf import gto, lib, scf

from so2_proj import BondOrder, BondOrderForPySCF


@pytest.fixture(scope="module")
def molecule_and_density():
    previous = lib.num_threads()
    lib.num_threads(1)
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-11)
    assert mf.converged
    yield mol, mf.make_rdm1()
    lib.num_threads(previous)


def test_factory_uses_molecule_and_explicit_density(molecule_and_density):
    mol, density = molecule_and_density
    bond = BondOrder(mol, 0, 1, partition="mulliken", lmax=1)
    assert isinstance(bond, BondOrderForPySCF)
    assert not hasattr(bond, "kernel")
    result = bond.one_body(density)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert_allclose(result[0, 0], 1, atol=2e-11)
    info = bond.analyze()
    assert info["kind"] == "one_body"
    assert info["parameters"]["partition"] == "mulliken"
    assert info["parameters"]["A"] == 0
    assert info["parameters"]["B"] == 1
    assert isinstance(info["diagnostics"], dict)
    assert_allclose(info["channel_sum"], result.sum())


def test_parameter_changes_recompute_and_diagnostics_are_snapshots(molecule_and_density):
    mol, density = molecule_and_density
    bond = BondOrder(mol, 0, 1, partition="mulliken", lmax=2)
    first = bond.one_body(density)
    first_info = bond.analyze()
    bond.lmax = 0
    # Merely editing configuration must not relabel the preceding calculation.
    assert bond.analyze()["parameters"]["lmax"] == 2
    second = bond.one_body(density)
    assert first.shape == (4, 4)
    assert second.shape == (2, 2)
    assert first_info["parameters"]["lmax"] == 2
    assert bond.analyze()["parameters"]["lmax"] == 0
    assert_allclose(first.sum(), second.sum(), atol=2e-12)
    # User changes to a returned report cannot change the stored report.
    first_info["parameters"]["lmax"] = 99
    latest = bond.analyze()
    latest["parameters"]["lmax"] = 88
    assert bond.analyze()["parameters"]["lmax"] == 0


def test_partition_change_uses_new_configuration(molecule_and_density):
    mol, density = molecule_and_density
    bond = BondOrder(
        mol, 0, 1, partition="mulliken", lmax=1,
        nz=40, nrho=20, nphi=16, pad=6.0,
    )
    bond.one_body(density)
    bond.partition = "becke"
    bond.cell = "avg"
    actual = bond.one_body(density)
    info = bond.analyze()
    assert info["parameters"]["partition"] == "becke"
    assert info["parameters"]["cell"] == "avg"
    assert_allclose(actual.sum(), info["total"], atol=2e-10)


def test_spin_resolved_density_is_not_silently_spin_summed(molecule_and_density):
    mol, density = molecule_and_density
    bond = BondOrder(mol, 0, 1, partition="mulliken")
    with pytest.raises(NotImplementedError):
        bond.one_body(np.stack([density / 2, density / 2]))


def test_invalid_density_shape(molecule_and_density):
    mol, _ = molecule_and_density
    bond = BondOrder(mol, 0, 1, partition="mulliken")
    with pytest.raises(ValueError):
        bond.one_body(np.zeros((mol.nao + 1, mol.nao + 1)))


def test_complex_density_is_rejected(molecule_and_density):
    mol, density = molecule_and_density
    complex_density = density.astype(complex)
    complex_density[0, 1] += 0.01j
    complex_density[1, 0] -= 0.01j
    with pytest.raises((ValueError, NotImplementedError)):
        BondOrder(mol, 0, 1, partition="mulliken").one_body(complex_density)


def test_invalid_atom_pair(molecule_and_density):
    mol, density = molecule_and_density
    for a, b in [(0, 0), (-1, 1), (0, 2)]:
        with pytest.raises((ValueError, IndexError)):
            BondOrder(mol, a, b, partition="mulliken").one_body(density)


def test_pyscf_api_does_not_import_optional_gpaw():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockGPAW(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'gpaw' or fullname.startswith('gpaw.'):
                    raise AssertionError('The PySCF API imported optional GPAW')

        sys.meta_path.insert(0, BlockGPAW())
        import so2_proj
        from pyscf import gto
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        bond = so2_proj.BondOrder(mol, 0, 1, partition='mulliken')
        assert isinstance(bond, so2_proj.BondOrderForPySCF)
        assert not any(name == 'gpaw' or name.startswith('gpaw.') for name in sys.modules)
        """
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
