"""Planar ethylene: PBE/PW PAW valence one-body C-C bond orders.

The geometry matches 01-pyscf-pbe-ethylene.py. Standard C/H PAW datasets give
12 valence electrons in six occupied orbitals; the four carbon 1s electrons
are frozen and excluded from B1. PySCF's example includes all 16 electrons.
Run serially with the GPAW extra and C.PBE/H.PBE datasets installed.
"""

import numpy as np
from ase import Atoms
from gpaw import GPAW, PW

from so2_proj import BondOrder

np.set_printoptions(precision=8, suppress=True, linewidth=100)

# Geometry in Angstrom; a vacuum box contains the whole integration cylinder.
atoms = Atoms("C2H4", positions=[
    [0, 0, -0.67], [0, 0, 0.67],
    [0.92, 0, -1.23], [-0.92, 0, -1.23],
    [0.92, 0, 1.23], [-0.92, 0, 1.23],
])
atoms.center(vacuum=5.0)
atoms.pbc = True

# Six doubly occupied spatial orbitals: 2*4 + 4*1 = 12 valence electrons.
atoms.calc = GPAW(mode=PW(600), xc="PBE", nbands=6, kpts=(1, 1, 1),
                  symmetry="off", txt=None,
                  convergence={"energy": 1e-7, "density": 1e-6},
                  occupations={"name": "fermi-dirac", "width": 0.0})
energy = atoms.get_potential_energy()
print(f"PBE/PW600: E = {energy:.10f} eV; "
      f"SCF iterations = {atoms.calc.get_number_of_iterations()}")

# BondOrder reads the existing orbitals and occupations without rerunning SCF.
bond = BondOrder(atoms.calc, 0, 1)
bond.partition = "becke"
bond.lmax = 3
bond.nz, bond.nrho, bond.nphi = 160, 80, 32
bond.pad = 6.0       # bohr; increasing this may require a larger vacuum box.
bond.h_fine = 0.08   # bohr; interpolation spacing for the smooth PAW orbitals.

for cell in ("full", "avg"):
    bond.cell = cell
    matrix = bond.one_body()
    print(f"\nBecke / {cell} B1 [sigma, pi, delta, phi, remainder]:")
    print(matrix)
    report = bond.analyze()
    diagnostics = report["diagnostics"]
    print(f"{diagnostics['valence_electrons']:g} valence electrons; "
          f"{diagnostics['frozen_core_electrons']:g} frozen-core electrons excluded")
    print(f"total = {report['total']:.10f}; sum error = {report['sum_error']:.3e}; "
          f"max Gram error = {diagnostics['overlap_error']:.3e}")
    print(f"sigma-pi cross blocks = {matrix[0, 1]:.3e}, {matrix[1, 0]:.3e}")
