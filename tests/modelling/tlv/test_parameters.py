"""
TLV model parameter container (Thesis Sec. 1.2.1, Eq. 1.3).

Ten physical parameters per print-quality group. Each is defined at two
anchor temperatures (20C and 30C, converted to Kelvin) and linearly
interpolated in between via Eq. 1.3. Field order below is the canonical
order used throughout the optimisation pipeline (TLVBounds, scaling) --
if you change it here, update those modules too.
"""
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

T_20_KELVIN = 293.15
T_30_KELVIN = 303.15


@dataclass(frozen=True)
class TLVParameters:
    """
    Physical TLV model parameters, fit separately per print-quality group.

    Units (Table 1.1):
        A20, A30   : MPa^-n * s^-(m+1)
        n20, n30   : dimensionless
        m20, m30   : dimensionless
        Ee20, Ee30 : MPa
        Ev20, Ev30 : MPa
    """
    A20: float
    A30: float
    n20: float
    n30: float
    m20: float
    m30: float
    Ee20: float
    Ee30: float
    Ev20: float
    Ev30: float

    @staticmethod
    def _interp(x20: float, x30: float, T_kelvin) -> float:
        """Eq. 1.3 -- linear interpolation between the 20C and 30C anchor values."""
        return x20 + (x30 - x20) / (T_30_KELVIN - T_20_KELVIN) * (T_kelvin - T_20_KELVIN)

    def at_temperature(self, T_kelvin) -> dict:
        """
        Evaluate every temperature-dependent parameter at T_kelvin via Eq. 1.3.
        T_kelvin may be scalar or array; returns matching scalar/array values.

        Returns:
            dict with keys 'A', 'n', 'm', 'Ee', 'Ev'.
        """
        return {
            "A": self._interp(self.A20, self.A30, T_kelvin),
            "n": self._interp(self.n20, self.n30, T_kelvin),
            "m": self._interp(self.m20, self.m30, T_kelvin),
            "Ee": self._interp(self.Ee20, self.Ee30, T_kelvin),
            "Ev": self._interp(self.Ev20, self.Ev30, T_kelvin),
        }

    def dEe_dT(self) -> float:
        """Constant slope dEe/dT implied by the linear parametrisation (Eq. 1.3)."""
        return (self.Ee30 - self.Ee20) / (T_30_KELVIN - T_20_KELVIN)

    def dEv_dT(self) -> float:
        """Constant slope dEv/dT implied by the linear parametrisation (Eq. 1.3)."""
        return (self.Ev30 - self.Ev20) / (T_30_KELVIN - T_20_KELVIN)

    def to_array(self) -> npt.NDArray[np.float64]:
        """Ordered array for optimisation routines. Order MUST stay in sync
        with TLVBounds.lower_array/upper_array and the scaling module."""
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64]) -> "TLVParameters":
        names = [f.name for f in fields(cls)]
        return cls(**dict(zip(names, arr)))