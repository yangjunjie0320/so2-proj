"""B1 and B2 for 90-degree twisted ethylene: singlet pi/pi* CAS(2,2)/cc-pVDZ."""

import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.mcscf import avas

from so2_proj import BondOrder

np.set_printoptions(precision=8, suppress=True, linewidth=100)

# C--C is the z axis; the two CH2 groups lie in the xz and yz planes.
mol = gto.M(
    atom="""
        C  0.00  0.00 -0.67
        C  0.00  0.00  0.67
        H  0.92  0.00 -1.23
        H -0.92  0.00 -1.23
        H  0.00  0.92  1.23
        H  0.00 -0.92  1.23
    """,
    basis="cc-pvdz", unit="Angstrom", spin=0, verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-11, max_cycle=150)
assert mf.converged, "RHF did not converge"

# Select the local pi directions, normal to each CH2 plane, using AVAS.
ncas, nelecas, mo = avas.avas(mf, ["0 C 2py", "1 C 2px"], threshold=0.2)
assert (ncas, nelecas) == (2, 2), "The pi targets must select CAS(2,2)"
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver = fci.direct_spin0.FCI(mol)  # Singlet solver.
mc.run(mo, conv_tol=1e-9, max_cycle_macro=100)
assert mc.converged, "CASSCF did not converge"

dm1 = mc.make_rdm1()  # Complete 16-electron AO density.
_, active_dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
active = list(range(mc.ncore, mc.ncore + mc.ncas))

# Complete the two-electron active 2-RDM with seven doubly occupied inactive orbitals.
bo = BondOrder(mol, 0, 1)
b1 = bo.one_body(dm1)
b2 = bo.two_body(
    active_dm2, coeff=mc.mo_coeff, dm1=dm1, indices=active, completion="cas",
)
print("B1 [sigma, pi, delta, phi, remainder]:\n", b1)
print("B2:\n", b2)
print("B1 / B2 totals:", b1.sum(), b2.sum())
