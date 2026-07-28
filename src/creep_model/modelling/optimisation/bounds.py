"""
Parameter bounds (Thesis Table 1.1) + [0, 1] normalisation for DE
(Sec. 1.3.3, 1.3.4).
"""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from creep_model.domain import CreepTest
from creep_model.modelling.tlv.parameters import TLVParameters


@dataclass(frozen=True)
class TLVBounds:
    """
    Lower/upper physical bounds for each of the 10 TLV parameters (Table 1.1).
    Ee/Ev upper bounds are DATA-DEPENDENT (max(sigma/eps) over the group
    being fit) -- always build these via from_group_data(), don't rely on
    the dataclass defaults directly.
    """
    A_lower: float = 0.0
    A_upper: float = 1e-5
    n_lower: float = 0.0
    n_upper: float = 5.0
    m_lower: float = -5.0
    m_upper: float = 0.0
    Ee_lower: float = 0.0   #MPa
    Ee_upper: float = 500.0 #MPa
    Ev_lower: float = 0.0   #MPa
    Ev_upper: float = 500.0 #MPa

    @classmethod
    def from_group_data(cls, tests: list[CreepTest], A_upper: float = 1e-5) -> "TLVBounds":
        """
        Builds bounds with Ee_upper / Ev_upper set to max(sigma/eps) across
        `tests` (Table 1.1), rather than left at +inf. `tests` should be the
        TRIMMED tests for a single print-quality group.
        """
        max_ratio = max(
            test.applied_stress_MPa / max(float(test.strain_series.max()), 1e-12)
            for test in tests
            if not test.is_empty
        )
        return cls(A_upper=A_upper, Ee_upper=max_ratio, Ev_upper=max_ratio)

    # Order below MUST match TLVParameters field order exactly:
    # A20, A30, n20, n30, m20, m30, Ee20, Ee30, Ev20, Ev30

    def lower_array(self) -> npt.NDArray[np.float64]:
        return np.array([
            self.A_lower, self.A_lower,
            self.n_lower, self.n_lower,
            self.m_lower, self.m_lower,
            self.Ee_lower, self.Ee_lower,
            self.Ev_lower, self.Ev_lower,
        ])

    def upper_array(self) -> npt.NDArray[np.float64]:
        return np.array([
            self.A_upper, self.A_upper,
            self.n_upper, self.n_upper,
            self.m_upper, self.m_upper,
            self.Ee_upper, self.Ee_upper,
            self.Ev_upper, self.Ev_upper,
        ])

    def normalize(self, params: TLVParameters) -> npt.NDArray[np.float64]:
        """Physical params -> [0, 1]^10, for use as a DE starting point or
        for inspecting a candidate solution."""
        x = params.to_array()
        lo, hi = self.lower_array(), self.upper_array()
        return (x - lo) / (hi - lo)

    def denormalize(self, x_normalized: npt.NDArray[np.float64]) -> TLVParameters:
        """[0, 1]^10 -> physical TLVParameters. This is what DE's search
        space actually operates over (Sec. 1.3.3)."""
        lo, hi = self.lower_array(), self.upper_array()
        x_physical = lo + np.asarray(x_normalized) * (hi - lo)
        return TLVParameters.from_array(x_physical)

    def as_unit_bounds(self) -> list[tuple[float, float]]:
        """Bounds list in the [0, 1]^10 form scipy.optimize.differential_evolution expects."""
        return [(0.0, 1.0)] * 10
