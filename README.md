# so2-proj

Resolve a two-center bond order into angular-momentum channels about the bond
axis using an atomic partition and SO(2) projection. The default is a full Becke
cell; averaged Becke cells and Mulliken/AOSB are also available.

Results have shape `(lmax + 2, lmax + 2)`: the default `lmax=3` gives a **5×5**
matrix for sigma, pi, delta, phi, and a remainder containing all higher channels.
Sum every entry, including cross-channel blocks and the remainder, for the total.
Atom indices are zero based; internal lengths are in **bohr**.

## Install

Python 3.13 or newer is required. From a local clone:

```bash
git clone https://github.com/yangjunjie0320/so2-proj.git
cd so2-proj
uv sync --extra test
```

`uv sync --extra test --extra gpaw` adds the optional GPAW 26.7.0 adapter.

GPAW needs its compiled dependencies and PAW datasets. The ethylene example
requires H/C PBE datasets discoverable through `gpaw-data` or `GPAW_SETUP_PATH`.

## API

Given a built PySCF molecule `mol` and its density `dm1`:

```python
from so2_proj import BondOrder

bo = BondOrder(mol, 0, 1)
b1 = bo.one_body(dm1)
report = bo.analyze()
```

`dm1` is the **complete, real, spin-summed AO density**, as returned by PySCF's
`make_rdm1()`. Closed-shell/singlet data are supported. `BondOrder` analyzes
supplied data; it never runs SCF or CASSCF and has no `kernel()` method.

For a PySCF spin-summed orbital-space 2-RDM:

```python
b2 = bo.two_body(
    dm2, coeff=coeff, dm1=dm1, indices=None, completion="full"
)
```

The required `dm1` is still the **full AO density**, including inactive electrons.
`coeff` has shape `(nao, norb)`. Selected columns `C = coeff[:, indices]` must
satisfy `C.T @ S @ C = I`; all four `dm2` axes use that same column order.
`indices=None` selects all columns. Coefficients are not silently reordered.

- `completion="full"` requires the selected orbitals to cover all occupied
  support of `dm1`; no occupied core is added.
- `completion="cas"` permits omitted doubly occupied inactive and empty orbitals,
  with no active–inactive one-particle coupling; it completes their contributions.

```python
# calc already contains the GPAW wavefunctions and occupations.
bo = BondOrder(calc, 0, 1)
b1 = bo.one_body()
```

GPAW supports Becke B1 for serial, spin-paired, real Γ-point PW calculations
with integer occupations in an orthorhombic molecular vacuum box. The integration
cylinder must fit inside the box. GPAW `two_body()` is not implemented.

PAW results contain valence electrons only. For ethylene, GPAW includes **12
valence electrons** and freezes four carbon 1s electrons; the PySCF examples
include **all 16 electrons**. These electronic spaces are not directly equivalent.

## Examples

- [PySCF PBE ethylene](examples/01-pyscf-pbe-ethylene.py): planar cc-pVDZ B1;
  edit the basis for cc-pVTZ/QZ, with `nphi=64` and `pad=8.0` bohr as starting settings.
- [GPAW PBE ethylene](examples/02-gpaw-pbe-ethylene.py): planar PW600 B1 with
  5 Å vacuum on each side, using the default bond-analysis settings.
- [PySCF CASSCF twisted ethylene](examples/03-pyscf-casscf-twisted-ethylene.py):
  90° singlet π/π* CAS(2,2)/cc-pVDZ B1 and B2, completing seven doubly occupied inactive orbitals.

Run with `uv run python examples/01-pyscf-pbe-ethylene.py` (or another filename).

## Checks

```bash
uv run pytest -q -m 'not gpaw'
uv run pytest -q -m gpaw
```

Converge the grid and range; inspect `overlap_error` as well as `sum_error`.
A small channel sum error alone does not establish accurate integration.
