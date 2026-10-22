"""Real GPAW integration tests; skipped when the optional dependency is absent."""

import numpy as np
import pytest

gpaw = pytest.importorskip("gpaw")
pytest.importorskip("ase")
pytestmark = pytest.mark.gpaw

from ase import Atoms
from so2_proj import BondOrder
from so2_proj.gpaw import BondOrderForGPAW
from so2_proj._gpaw_adapter import PAWOrbitals, check_cylinder, occupied_state


@pytest.fixture(scope="module")
def h2_calc():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.center(vacuum=4.0)
    atoms.pbc = True  # One molecule in a plane-wave vacuum box.
    atoms.calc = gpaw.GPAW(mode=gpaw.PW(300), xc="PBE", nbands=1,
                          kpts=(1, 1, 1), symmetry="off", txt=None,
                          occupations={"name": "fermi-dirac", "width": 0.0})
    atoms.get_potential_energy()
    return atoms.calc


def test_real_paw_h2_convergence_and_repeated_calls(h2_calc, monkeypatch):
    def forbid_scf(*args, **kwargs):
        pytest.fail("Bond-order postprocessing must not run a GPAW calculation")

    monkeypatch.setattr(h2_calc, "iconverge", forbid_scf)
    monkeypatch.setattr(h2_calc, "calculate_property", forbid_scf)
    raw_before = h2_calc.get_occupation_numbers(raw=True).copy()
    coarse = BondOrder(h2_calc, 0, 1, lmax=1, nz=48, nrho=24,
                       nphi=16, pad=6.0)
    assert isinstance(coarse, BondOrderForGPAW)
    fine = BondOrderForGPAW(h2_calc, 0, 1, lmax=1, nz=96, nrho=48,
                           nphi=32, pad=6.0)
    b_coarse, b_fine = coarse.one_body(), fine.one_body()
    report = fine.analyze()
    assert b_fine.shape == (3, 3)
    assert report["parameters"]["h_fine"] == 0.08
    assert abs(report["sum_error"]) < 1e-12
    assert report["diagnostics"]["overlap_error"] < 5e-3
    assert coarse.analyze()["diagnostics"]["overlap_error"] < 1e-2
    assert report["diagnostics"]["valence_only"] is True
    assert report["diagnostics"]["valence_electrons"] == pytest.approx(2.0)
    assert report["diagnostics"]["frozen_core_electrons"] == pytest.approx(0.0)
    assert b_fine.sum() == pytest.approx(1.0, abs=1e-2)
    assert b_fine[0, 0] == pytest.approx(b_fine.sum(), abs=1e-4)
    assert abs(b_coarse.sum() - b_fine.sum()) < 5e-3
    # Changing parameters reruns integration, and never rescales GPAW's state.
    fine.cell = "avg"
    b_avg = fine.one_body()
    assert b_avg == pytest.approx(b_fine, abs=1e-4)
    assert h2_calc.get_occupation_numbers(raw=True) == pytest.approx(raw_before, abs=0)


def test_paw_reconstruction_and_box_boundary(h2_calc):
    indices, occupations, lengths = occupied_state(h2_calc)
    orbitals = PAWOrbitals(h2_calc, indices, lengths, h_fine=0.3)
    from ase.units import Bohr

    points = np.asarray(h2_calc.atoms.positions) / Bohr + [0.1, 0.0, 0.0]
    corrected = orbitals(points)
    assert np.max(np.abs(orbitals.correction(points))) > 1e-5
    assert corrected == pytest.approx(orbitals.pseudo(points) + orbitals.correction(points))
    assert occupations == pytest.approx([2.0])
    with pytest.raises(ValueError, match="outside"):
        orbitals(np.array([[-0.01, 1.0, 1.0]]))
    too_large = BondOrderForGPAW(h2_calc, 0, 1, pad=20.0)
    with pytest.raises(ValueError, match="cylinder"):
        too_large.one_body()


def test_cylinder_bounds_include_every_rotation():
    # A diagonal axis fits along the box diagonal, but the cylinder radius
    # extends beyond it; checking only the bond endpoints would miss this.
    coords = np.array([[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]])
    with pytest.raises(ValueError, match="cylinder"):
        check_cylinder(coords, 0, 1, np.full(3, 10.0), pad=2.0)


def test_gpaw_rejects_unsupported_requests(h2_calc, monkeypatch):
    bo = BondOrderForGPAW(h2_calc, 0, 1)
    with pytest.raises(NotImplementedError, match="one-body"):
        bo.two_body()
    bo.partition = "mulliken"
    with pytest.raises(NotImplementedError, match="Becke"):
        bo.one_body()
    bo.partition = "becke"
    with monkeypatch.context() as patch:
        patch.setattr(h2_calc, "get_ibz_k_points", lambda: np.array([[0.1, 0, 0]]))
        with pytest.raises(NotImplementedError, match="Gamma"):
            bo.one_body()
    with monkeypatch.context() as patch:
        patch.setattr(h2_calc.dft.ibzwfs, "ncomponents", 2)
        with pytest.raises(NotImplementedError, match="spin-paired"):
            bo.one_body()
    with monkeypatch.context() as patch:
        patch.setattr(h2_calc, "get_occupation_numbers", lambda **kw: np.array([0.5]))
        with pytest.raises(NotImplementedError, match="occupations"):
            bo.one_body()
    with monkeypatch.context() as patch:
        patch.setattr(h2_calc.dft.ibzwfs, "has_wave_functions", lambda: False)
        with pytest.raises(ValueError, match="wave functions"):
            bo.one_body()
    with monkeypatch.context() as patch:
        changed_atoms = h2_calc.atoms.copy()
        changed_atoms.cell *= 1.1
        patch.setattr(h2_calc, "_atoms", changed_atoms)
        with pytest.raises(ValueError, match="cell differs"):
            bo.one_body()


def test_restart_with_wavefunctions(h2_calc, tmp_path):
    path = tmp_path / "h2.gpw"
    h2_calc.write(path, mode="all")
    restored = gpaw.GPAW(path, txt=None)
    bo = BondOrderForGPAW(restored, 0, 1, lmax=1, nz=48, nrho=24,
                         nphi=16, h_fine=0.5)
    assert bo.one_body().sum() == pytest.approx(1.0, abs=1e-2)
