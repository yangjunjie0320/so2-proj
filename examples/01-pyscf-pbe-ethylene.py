#!/usr/bin/env python
"""Planar ethylene: PBE/cc-pVXZ one-body C-C bond orders.

The AO density includes all 16 electrons, including the carbon 1s shells.
"""

import numpy as np
from pyscf import dft, gto
from so2_proj import BondOrder

np.set_printoptions(precision=8, suppress=True)

mol = gto.M(
    atom='''
        C   0.00  0.00  -0.67
        C   0.00  0.00   0.67
        H   0.92  0.00  -1.23
        H  -0.92  0.00  -1.23
        H   0.92  0.00   1.23
        H  -0.92  0.00   1.23
    ''',
    unit='Angstrom',
    basis='cc-pvdz',  # Change to cc-pvtz or cc-pvqz for a basis comparison.
)

mf = dft.RKS(mol)
mf.xc = 'PBE'
mf.grids.level = 4
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged

# Complete, real, spin-summed AO density.
dm1 = mf.make_rdm1()
bond = BondOrder(mol, 0, 1)
bond.nphi = 64  # Resolve the off-axis hydrogen basis functions.
bond.pad = 8.0  # bohr; check range and grid convergence as the basis grows.

# Rows and columns: sigma, pi, delta, phi, remainder.
for partition, cell in [('mulliken', 'full'), ('becke', 'full'), ('becke', 'avg')]:
    bond.partition = partition
    bond.cell = cell
    b1 = bond.one_body(dm1)
    print(f'\n{partition} / {cell}: B1')
    print(b1)
    report = bond.analyze()
    print('Total = %.10f; sum error = %.2e; overlap error = %.2e' % (
        report['total'], report['sum_error'], report['diagnostics']['overlap_error']))
