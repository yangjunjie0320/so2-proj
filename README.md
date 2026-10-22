# so2-proj

Resolve a two-center bond order into angular-momentum channels about the bond axis:

```text
atomic partition × SO(2) projection → B(l_A, l_B)
```

The diagonal blocks describe sigma, pi, delta, … contributions; off-diagonal blocks
describe cross-channel contributions. Atomic partitions are Becke fuzzy cells
(`partition="becke"`) or Mulliken/AOSB (`partition="mulliken"`). Becke cells can be
used in full (`cell="full"`) or averaged over the azimuthal angle (`cell="avg"`).
The full and averaged cells define different quantities for a general molecule.

All matrices have shape `(lmax + 2, lmax + 2)`. Indices `0..lmax` represent
`|m| = 0..lmax`; the final index contains **all remaining channels**. Summing every
entry, including that remainder and the off-diagonal entries, gives the unresolved
bond order for the same partition and electronic state. Internal lengths are in
**bohr**. Atom indices are zero based; the axis points from A to B.

## Quick start

Clone the repository, then install and run the first example:

```bash
git clone https://github.com/yangjunjie0320/so2-proj.git
cd so2-proj
uv sync --extra test
uv run python examples/01-pyscf-pbe-ethylene.py
```

The core API takes a built PySCF molecule and an explicit density. For planar
ethylene at PBE/cc-pVDZ:

```python
from pyscf import dft, gto
from so2_proj import BondOrder

mol = gto.M(
    atom=(
        "C 0 0 -0.67; C 0 0 0.67; "
        "H 0.92 0 -1.23; H -0.92 0 -1.23; "
        "H 0.92 0 1.23; H -0.92 0 1.23"
    ),
    unit="Angstrom", basis="cc-pvdz",
)
mf = dft.RKS(mol)
mf.xc = "PBE"
mf.grids.level = 4
mf.conv_tol = 1e-10
mf.kernel()
assert mf.converged

bond = BondOrder(mol, 0, 1, nphi=64, pad=8.0)
b1 = bond.one_body(mf.make_rdm1())
print(b1)  # sigma, pi, delta, phi, remainder on both axes
print(bond.analyze())
```

The electronic-structure calculation belongs to PySCF; `BondOrder` only analyzes
its supplied density. The default partition is a full Becke cell. Set
`partition="mulliken"` for an AOSB decomposition without real-space quadrature.

## Installation

Python 3.13 or newer is required. PySCF, NumPy, and SciPy are core dependencies.
All installation and example commands below run from the cloned repository's
root. `uv sync --extra test` creates the project environment with test dependencies.
For an existing Python environment, the editable-install equivalent is:

```bash
python -m pip install -e '.[test]'
```

GPAW is optional and is not imported by the PySCF API. Install both the test and
GPAW extras with:

```bash
uv sync --extra test --extra gpaw
```

The GPAW extra targets version 26.7.0. Its compiled extension requires a compatible
compiler, BLAS, and LibXC installation when no suitable wheel is available. PAW
datasets must also be installed and discoverable through `gpaw-data` or
`GPAW_SETUP_PATH`. A successful core installation does not establish that GPAW or
its datasets are usable. There are no machine-specific setup paths in this package.

## Ethylene examples

The three scripts analyze the C–C bond of ethylene. They explicitly run the
electronic-structure calculation before passing its results to `so2_proj`.
They follow the direct, top-level style of PySCF examples: edit parameters in
the source and run the script.

| Script | Calculation | Geometry | Electronic space |
| --- | --- | --- | --- |
| [01-pyscf-pbe-ethylene.py](examples/01-pyscf-pbe-ethylene.py) | Restricted PBE / cc-pVDZ; selectable cc-pVXZ basis | Planar | All 16 electrons, B1 |
| [02-gpaw-pbe-ethylene.py](examples/02-gpaw-pbe-ethylene.py) | PBE / plane waves with PAW | Planar molecule in a vacuum box | 12 valence electrons, B1 |
| [03-pyscf-casscf-twisted-ethylene.py](examples/03-pyscf-casscf-twisted-ethylene.py) | Singlet CASSCF(2,2) / cc-pVDZ | 90° torsion about C–C | All 16 electrons, B1 and B2 |

Run the examples with:

```bash
uv run python examples/01-pyscf-pbe-ethylene.py
uv run python examples/02-gpaw-pbe-ethylene.py
uv run python examples/03-pyscf-casscf-twisted-ethylene.py
```

The PySCF PBE example defaults to `cc-pvdz`. To use `cc-pvtz` or `cc-pvqz`, edit
the molecule's basis assignment (`mol.basis` or the `basis` argument) before it
is built. Its `BondOrder` settings are `nphi=64` and `pad=8.0` bohr; change these
and other grid attributes in the source. Larger basis sets can require a larger
integration region or finer quadrature. Inspect `diagnostics["overlap_error"]`
when changing the basis or grid: a small channel `sum_error` alone does not
demonstrate converged integration.

The GPAW example requires the optional GPAW dependency and H/C PBE datasets.
Edit `PW(600)` to change the 600 eV plane-wave cutoff. The vacuum is 5 angstrom
on each side. `BondOrder` uses `nz=160`, `nrho=80`, `nphi=32`, `pad=6.0` bohr,
and `h_fine=0.08` bohr. These attributes control the bond-analysis grid and PAW
interpolation, independently of the plane-wave cutoff. Increase the vacuum
along with `pad` so the full integration cylinder remains inside the box.

The GPAW PAW calculation freezes the two carbon 1s shells, excluding four
electrons. Its 12-electron valence B1 and PySCF's 16-electron all-electron B1
therefore describe different electronic spaces; matching PBE and molecular
geometry does not make the two results directly equivalent.

The CASSCF script fixes the torsion at 90° and uses AVAS to select the two carbon 2p
directions normal to their respective CH₂ planes: the local orbitals of the
π/π* active space. The remaining seven occupied orbitals, including the σ bonds
and carbon 1s shells, form the doubly occupied inactive space.
`completion="cas"` combines its active-space 2-RDM with the complete AO 1-RDM,
so B2 includes all 16 electrons rather than
only the two active electrons. The script reports B1 and B2 from the same
correlated state. A minimal π CAS illustrates the correlation dependence; it
does not establish active-space or basis-set convergence.

## PySCF API

`BondOrder(mol, A, B, **parameters)` returns `BondOrderForPySCF`.
`one_body(dm1)` receives the **complete, real, spin-summed AO density** in PySCF's
`make_rdm1()` convention. It does not receive MO occupations or an SCF object.
The AO overlap is included when transforming that density to an orthonormal basis;
AO directions and small natural occupations are not discarded. The density must
have the molecule's electron count and natural occupations between zero and two.

The package analyzes supplied densities. It has no `kernel()` method and never
starts an SCF, DFT, or CASSCF calculation. Real closed-shell/singlet inputs are the
supported scope; spin-resolved densities, open-shell molecules, and complex data
are rejected. Mulliken/AOSB additionally requires pure spherical AOs
(`mol.cart = False`) and currently supports shells through angular momentum 6.

`one_body()` and `two_body()` directly return NumPy matrices. `analyze()` returns
a copied report about the last successful call: `kind`, `parameters`, `total`,
`channel_sum`, `sum_error`, and `diagnostics`. It also logs that report through
Python's `logging` module. Before the first calculation, `analyze()` raises an error.

Parameters are mutable attributes. Each calculation recomputes its weights:

```python
bond.partition = "becke"
bond.cell = "avg"
bond.nphi = 64
b1_avg = bond.one_body(mf.make_rdm1())
```

Editing attributes does not change a previous diagnostic snapshot. Construct a new
object after changing the molecular geometry or AO basis.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `partition` | `"becke"` | `"becke"` or `"mulliken"` |
| `cell` | `"full"` | `"full"` or `"avg"`; Mulliken requires `"full"` |
| `lmax` | `3` | Highest explicitly retained `|m|` |
| `becke_k` | `3` | Number of Becke smoothing iterations |
| `becke_adjust` | `True` | Atomic-size adjustment using Bragg radii |
| `nz`, `nrho`, `nphi` | `160`, `80`, `32` | Cylindrical quadrature sizes; `nz` is a target across segments |
| `pad` | `6.0` | Integration padding in bohr |
| `zchunk` | `8` | Number of z planes evaluated together |
| `h_fine` | `0.08` | GPAW interpolation spacing in bohr |

Becke projection requires `nphi > 2*lmax`. Increase the integration range and
quadrature sizes to establish numerical convergence. A small `sum_error` checks
the channel decomposition on the chosen grid; it does not by itself establish
accurate real-space integration. The full orbital-overlap diagnostic checks more
than the individual orbital norms.

## Two-body input and CAS completion

```python
b2 = bond.two_body(
    dm2, coeff=coeff, dm1=dm1, indices=None, completion="full"
)
```

`dm1` always remains the **complete AO density**. `coeff` has shape `(nao, norb)`;
the selected columns `coeff[:, indices]` are orthonormal under the AO overlap.
`indices=None` selects every column. An explicit sequence preserves its order, so
the four axes of `dm2` must use exactly that same orbital order.

`dm2[p,q,r,s] = <a†_p a†_r a_s a_q>` follows PySCF's spin-summed `make_rdm2()`
convention. In its selected orbital space its trace is `N(N-1)`, and contracting
the last pair gives `(N-1)` times the selected one-particle density.

- `completion="full"`: the selected orbitals must cover the entire support of
  `dm1`. Empty virtual orbitals may be omitted. No occupied core is added.
- `completion="cas"`: the supplied tensor is the complete active-space 2-RDM.
  Omitted orbitals must be doubly occupied core or empty virtuals, with no
  active–inactive one-particle coupling. The active cumulant is combined with
  the complete `dm1` to include core and core–active contributions. This does
  not complete an arbitrary truncated tensor.

For the converged twisted-ethylene CAS(2,2) state in the
[CASSCF example](examples/03-pyscf-casscf-twisted-ethylene.py),
include the inactive occupied orbitals as follows:

```python
# mc is the converged CASSCF object; mol is its ethylene/cc-pVDZ molecule.
_, active_dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
active = list(range(mc.ncore, mc.ncore + mc.ncas))
bond = BondOrder(mol, 0, 1, partition="mulliken")
b2 = bond.two_body(
    active_dm2, coeff=mc.mo_coeff, dm1=mc.make_rdm1(),
    indices=active, completion="cas",
)
```

The linked script contains the complete runnable calculation. Validation checks
dimensions, orbital orthonormality, traces, symmetry, and consistency with `dm1`;
these checks do not prove general N-representability.

## Optional GPAW API

`BondOrder(calc, A, B, **parameters)` returns `BondOrderForGPAW` for an already
calculated GPAW object. Call `one_body()` **without a density argument**: the
adapter reads the calculator's occupations and reconstructs its PAW valence
orbitals at the integration points. `two_body()` is not implemented for GPAW.

The supported case is GPAW 26.7.0's new plane-wave implementation, running in
serial at Γ with paired spins, real orbitals, and occupations zero or two. It
uses an axis-aligned orthorhombic vacuum box for an isolated molecule. The full
integration cylinder and PAW augmentation spheres must fit inside the box;
increase the vacuum or reduce the requested integration extent if they do not.
The adapter does not wrap the molecule across the boundary or silently zero its
orbitals outside the box. General periodic solids, non-Γ k points, fractional
occupations, spin polarization, and Mulliken/AOSB are outside this adapter's scope.

GPAW results include reconstructed **valence orbitals only**, not frozen-core
orbitals. Match that electronic space before comparing with an all-electron
calculation. Diagnostics identify the valence scope and excluded core count.

## Development and validation

Run the core suite without GPAW calculations, or select the real GPAW checks:

```bash
uv run pytest -q -m 'not gpaw'
uv run pytest -q -m gpaw
```

With the optional dependency and PAW datasets available, `uv run pytest -q` runs
both groups. The core suite covers channel sum rules, nonsymmetric weights,
determinant B1/B2 agreement, molecular selection rules, rotational invariance,
Mayer agreement, and input/diagnostic contracts. CAS tests compare completion
with independently embedded full-CI tensors and direct two-body contractions.
GPAW tests use real H₂ calculations to check PAW reconstruction, integration
refinement, restart handling, and unsupported-input boundaries.

These checks establish the tested cases, not numerical convergence for every
basis, molecule, active space, or PAW setup. Converge the relevant quadrature and
electronic-structure parameters before drawing quantitative conclusions.

## Definitions and references

For orthonormal real orbitals, the channel weights are

```text
full: W_A(l)[p,q] = <p | w_A Q_l | q>
avg:  W_A(l)[p,q] = <Q_l p | wbar_A | Q_l q>
B1(l,k) = Tr[P W_A(l) P W_B(k)]
B2(l,k) = 2 [N_A(l) N_B(k) - sum_pqrs G[p,q,r,s] W_A(l)[q,p] W_B(k)[s,r]]
```

Full-cell weights are generally nonsymmetric: the projection acts on the ket.
The averaged-cell weights are symmetric positive semidefinite. For a closed-shell
determinant, `G[p,q,r,s] = P[p,q]P[r,s] - P[p,s]P[r,q]/2` and `B2 = B1`.
For correlated states the two bond orders need not agree. Omitting a 2-RDM does
not create a two-body result from a general one-particle density.

The underlying bond-order definitions and AOSB connection follow:

- I. Mayer, *Chem. Phys. Lett.* **97**, 270 (1983); *Int. J. Quantum Chem.*
  **29**, 477 (1986).
- I. Mayer and P. Salvador, *Chem. Phys. Lett.* **383**, 368 (2004).
- X. Fradera, M. A. Austen, and R. F. W. Bader, *J. Phys. Chem. A* **103**, 304 (1999).
- O. V. Sizova, L. V. Skripnikov, and A. Yu. Sokolov, *J. Mol. Struct.: THEOCHEM*
  **870**, 1 (2008); O. V. Sizova et al., *Int. J. Quantum Chem.* **109**, 2581 (2009).
