"""Molecular bond orders resolved by atomic partition and SO(2) channel."""
from .pyscf import BondOrderForPySCF

__all__ = ["BondOrder", "BondOrderForPySCF", "BondOrderForGPAW"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "BondOrderForGPAW":
        from .gpaw import BondOrderForGPAW
        return BondOrderForGPAW
    raise AttributeError(name)


def BondOrder(source, A, B, **parameters):
    """Return a concrete analyzer for a PySCF Mole or a completed GPAW calculator.

    Construction never runs electronic structure. Call one_body(dm1) for PySCF,
    one_body() for GPAW, or two_body(dm2, coeff=..., dm1=...) for PySCF.
    """
    from pyscf.gto import MoleBase
    if isinstance(source, MoleBase):
        return BondOrderForPySCF(source, A, B, **parameters)
    if any(cls.__module__.startswith("gpaw.") for cls in type(source).__mro__):
        from .gpaw import BondOrderForGPAW
        return BondOrderForGPAW(source, A, B, **parameters)
    raise TypeError("BondOrder expects a PySCF Mole or a completed GPAW calculator")
