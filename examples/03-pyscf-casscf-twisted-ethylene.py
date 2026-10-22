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
mf = scf.RHF(mol)
mf.conv_tol = 1e-11
mf.max_cycle = 150
mf.kernel()
assert mf.converged, "RHF did not converge"

# Select the local pi directions, normal to each CH2 plane, using AVAS.
# Rotating the molecule to a different torsion also requires rotating these targets.
targets = ["0 C 2py", "1 C 2px"]
ncas, nelecas, mo = avas.avas(mf, targets, threshold=0.2)
assert (ncas, nelecas) == (2, 2), "The pi targets must select CAS(2,2)"
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver = fci.direct_spin0.FCI(mol)  # Singlet solver.
mc.conv_tol = 1e-9
mc.max_cycle_macro = 100
mc.kernel(mo)
assert mc.converged, "CASSCF did not converge"

dm1 = mc.make_rdm1()  # Complete 16-electron AO density.
active_dm1, active_dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
active = list(range(mc.ncore, mc.ncore + mc.ncas))
print("RHF / CASSCF energies (Ha):", mf.e_tot, mc.e_tot)
print("Active natural occupations:", np.linalg.eigvalsh(active_dm1)[::-1])

# Complete the two-electron active 2-RDM with seven doubly occupied inactive orbitals.
# Converge the integration grid for quantitative comparisons.
for partition, cell in (("becke", "full"), ("becke", "avg"), ("mulliken", "full")):
    bond = BondOrder(mol, 0, 1)
    bond.partition = partition
    bond.cell = cell
    b1 = bond.one_body(dm1)
    info1 = bond.analyze()
    b2 = bond.two_body(
        active_dm2, coeff=mc.mo_coeff, dm1=dm1, indices=active, completion="cas",
    )
    info2 = bond.analyze()
    print(f"\n{partition}/{cell}: sigma, pi, delta, phi, remainder")
    print("B1:\n", b1)
    print("B2:\n", b2)
    print("B2 - B1:\n", b2 - b1)
    print("B1 / B2 totals:", info1["total"], info2["total"])
    print("Sum errors:", info1["sum_error"], info2["sum_error"])
    print("Overlap error:", info2["diagnostics"]["overlap_error"])
