#!/usr/bin/env python
"""Planar ethylene: PBE/cc-pVDZ one-body C-C bond order (all 16 electrons)."""

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
    basis='cc-pvdz',  # Also try cc-pvtz or cc-pvqz.
)
mf = dft.RKS(mol)
mf.xc = 'PBE'
mf.grids.level = 4
mf.run(conv_tol=1e-10)
assert mf.converged

bo = BondOrder(mol, 0, 1, nphi=64, pad=8.0)
b1 = bo.one_body(mf.make_rdm1())
print('B1 [sigma, pi, delta, phi, remainder]:\n', b1)
print('Total:', b1.sum())
