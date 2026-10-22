"""Planar ethylene: PBE/PW PAW valence one-body C-C bond orders.

Standard C/H PAW datasets give six occupied orbitals and 12 valence electrons;
the four carbon 1s electrons are frozen and excluded from B1. Run serially.
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
atoms.get_potential_energy()

# BondOrder reads the existing orbitals and occupations without rerunning SCF.
bo = BondOrder(atoms.calc, 0, 1)
b1 = bo.one_body()
print("B1 [sigma, pi, delta, phi, remainder]:\n", b1)
print(f"Total B1 = {b1.sum():.10f}")
