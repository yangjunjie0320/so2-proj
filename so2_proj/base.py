"""Shared geometry, mutable options, and small analysis summaries."""
from copy import deepcopy
import logging
from numbers import Integral, Real
import numpy as np

log = logging.getLogger(__name__)


class BaseBondOrder:
    """Common configuration; each analysis recomputes its numerical weights.

    Coordinates and all integration lengths are in bohr. Subclasses provide the
    orbital representation without running an electronic-structure calculation.
    """
    defaults = dict(partition="becke", cell="full", lmax=3, becke_k=3,
                    becke_adjust=True, nz=160, nrho=80, nphi=32, pad=6.0,
                    zchunk=8, h_fine=0.08)

    def __init__(self, source, A, B, *, coords, charges, **parameters):
        unknown = parameters.keys() - self.defaults.keys()
        if unknown:
            raise TypeError(f"Unknown BondOrder parameter(s): {', '.join(sorted(unknown))}")
        self.source = source
        self.A, self.B = A, B
        self.coords = np.asarray(coords, dtype=float).copy()
        self.charges = np.asarray(charges).copy()
        for key, default in self.defaults.items():
            setattr(self, key, parameters.get(key, default))
        self._last_analysis = None
        self._validate_parameters()

    def _validate_parameters(self):
        if self.partition not in ("becke", "mulliken"):
            raise ValueError("partition must be 'becke' or 'mulliken'")
        if self.cell not in ("full", "avg"):
            raise ValueError("cell must be 'full' or 'avg'")
        if self.partition == "mulliken" and self.cell != "full":
            raise ValueError("Mulliken/AOSB does not support an averaged cell")
        for key in ("lmax", "becke_k", "nz", "nrho", "nphi", "zchunk"):
            value = getattr(self, key)
            minimum = 0 if key in ("lmax", "becke_k") else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.partition == "becke" and self.nphi <= 2 * self.lmax:
            raise ValueError("FFT sampling requires nphi > 2*lmax")
        for key in ("pad", "h_fine"):
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive length in bohr")
        if not isinstance(self.becke_adjust, (bool, np.bool_)):
            raise ValueError("becke_adjust must be a boolean")
        if self.coords.ndim != 2 or self.coords.shape[1] != 3 or not np.all(np.isfinite(self.coords)):
            raise ValueError("Atomic coordinates must be finite with shape (natm, 3)")
        natm = len(self.coords)
        if self.charges.shape != (natm,) or np.any(self.charges <= 0):
            raise ValueError("Positive nuclear charges are required (ghost atoms are unsupported)")
        for name in ("A", "B"):
            index = getattr(self, name)
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or not 0 <= index < natm:
                raise ValueError(f"{name} must be an atom index in [0, {natm})")
        if self.A == self.B:
            raise ValueError("A and B must be distinct atoms")
        distances = np.linalg.norm(self.coords[:, None] - self.coords[None, :], axis=2)
        if np.any(distances[np.triu_indices(natm, 1)] < 1e-10):
            raise ValueError("Coincident atomic positions are unsupported")

    def _grid_options(self):
        return {key: getattr(self, key) for key in ("nz", "nrho", "nphi", "pad", "zchunk")}

    def _parameter_snapshot(self):
        return deepcopy(dict(A=self.A, B=self.B,
                             **{key: getattr(self, key) for key in self.defaults}))

    def _record(self, matrix, total, diagnostics, *, kind):
        channel_sum = float(np.sum(matrix))
        self._last_analysis = dict(kind=kind, parameters=self._parameter_snapshot(),
                                   total=float(total), channel_sum=channel_sum,
                                   sum_error=channel_sum - float(total),
                                   diagnostics=deepcopy(diagnostics))
        return np.asarray(matrix)

    def analyze(self):
        """Return and log diagnostics for the last successful analysis.

        The snapshot describes that call even if attributes have since changed.
        No grid, orbital samples, or channel-weight tensors are retained.
        """
        if self._last_analysis is None:
            raise RuntimeError("Call one_body() or two_body() before analyze()")
        result = deepcopy(self._last_analysis)
        log.info("%s: total=%.12g, channel sum=%.12g, sum error=%.3g; %s",
                 result["kind"], result["total"], result["channel_sum"],
                 result["sum_error"], result["diagnostics"])
        return result
