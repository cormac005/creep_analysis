"""
Order-of-magnitude scaling before the LM stage (Thesis Sec. 1.3.3).

Applied AFTER DE returns its best physical-unit parameter set, and BEFORE
handing that to scipy.optimize.least_squares -- LM's step-size heuristics
assume parameters are roughly comparable in scale, which is badly violated
here (A ~ 1e-5 vs Ee ~ 1e2-1e3 MPa).
"""
import numpy as np
import numpy.typing as npt


def compute_scale_factors(x_physical: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Per-parameter scale factor ~= its order of magnitude, so x_physical /
    scale sits O(1) for every parameter regardless of native units.
    Args:
        x_physical: 1D array of physical-unit parameters (e.g. TLVParameters.to_array()).
    Outputs:
        1D array of scale factors, same shape as x_physical.
    """
    safe_x = np.where(np.abs(x_physical) > 0, np.abs(x_physical), 1e-12)
    return 10.0 ** np.floor(np.log10(safe_x))


def scale(x_physical: npt.NDArray[np.float64], scale_factors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x_physical / scale_factors


def unscale(x_scaled: npt.NDArray[np.float64], scale_factors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x_scaled * scale_factors
