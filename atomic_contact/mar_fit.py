"""Fit atomic-contact current-voltage traces with a MAR pincode model.

The measured current is modeled as the sum of independent MAR channels.  A
Gaussian distribution of bias voltage with standard deviation ``sigmaV_mV``
can be applied to the summed current.  This represents voltage noise or
finite voltage resolution; it is not an orthogonal-distance-regression error
bar on the measured voltage coordinate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from superconductivity.models.mar import get_Imar_nA

FloatArray = NDArray[np.float64]
ParameterSpec = tuple[float, float, float, bool]

NOISE_PADDING_SIGMA = 6.0
DEFAULT_PARAMETERS: dict[str, ParameterSpec] = {
    "T_K": (0.025, 0.0, 1.0, False),
    "Delta_meV": (0.180, 0.10, 0.30, False),
    "gamma_meV": (1.0e-4, 0.0, 0.05, False),
    "sigmaV_mV": (0.0, 0.0, 0.10, False),
    "V_offset_mV": (0.0, -0.10, 0.10, False),
    "I_offset_nA": (0.0, -100.0, 100.0, False),
}
JACOBIAN_STEPS = {
    "tau": 1.0e-3,
    "T_K": 1.0e-3,
    "Delta_meV": 1.0e-4,
    "gamma_meV": 1.0e-5,
    "sigmaV_mV": 1.0e-4,
    "V_offset_mV": 1.0e-4,
    "I_offset_nA": 1.0e-3,
}


@dataclass(frozen=True)
class MARFitResult:
    """Result of :func:`fit_mar`.

    Attributes
    ----------
    values
        Best-fit parameter values, including ``tau_1``, ``tau_2``, ... .
    errors
        Approximate one-standard-deviation errors from the local Jacobian.
        Fixed parameters and parameters with an indeterminate covariance have
        ``nan`` errors.
    V_mV, I_exp_nA, I_fit_nA
        Finite data used for the fit and the model evaluated at those points.
    residual_nA
        Unweighted current residual ``I_fit_nA - I_exp_nA``.
    success, message
        Termination information returned by SciPy.
    cost, reduced_chi2
        Half the weighted residual sum of squares and its reduced value.
    """

    values: dict[str, float]
    errors: dict[str, float]
    V_mV: FloatArray
    I_exp_nA: FloatArray
    I_fit_nA: FloatArray
    residual_nA: FloatArray
    success: bool
    message: str
    cost: float
    reduced_chi2: float
    nfev: int

    @property
    def tau(self) -> FloatArray:
        """Return fitted transmissions in descending order."""
        names = sorted(
            (name for name in self.values if name.startswith("tau_")),
            key=lambda name: int(name.removeprefix("tau_")),
        )
        return np.sort([self.values[name] for name in names])[::-1]


def gaussian_voltage_broaden(
    V_mV: ArrayLike,
    I_nA: ArrayLike,
    sigmaV_mV: float,
) -> FloatArray:
    """Convolve a current on a uniform voltage grid with Gaussian noise.

    Parameters
    ----------
    V_mV, I_nA
        Strictly increasing, uniformly spaced voltage axis and current.
    sigmaV_mV
        Standard deviation of the bias-voltage distribution in mV.  Zero
        returns a copy of the input current.
    """
    V = _voltage_axis(V_mV, uniform=True)
    I = _current_axis(I_nA, V)
    sigma = _nonnegative_scalar(sigmaV_mV, "sigmaV_mV")
    if sigma == 0.0:
        return I.copy()

    step = float(np.diff(V)[0])
    sigma_bins = sigma / step
    radius = max(int(np.ceil(NOISE_PADDING_SIGMA * sigma_bins)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(I, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def mar_current_nA(
    V_mV: ArrayLike,
    tau: Sequence[float] | float,
    *,
    T_K: float = 0.025,
    Delta_meV: float = 0.180,
    gamma_meV: float = 1.0e-4,
    sigmaV_mV: float = 0.0,
    V_offset_mV: float = 0.0,
    I_offset_nA: float = 0.0,
    caching: bool = True,
) -> FloatArray:
    """Evaluate a voltage-broadened, multichannel MAR current in nA.

    The MAR current is evaluated on a padded uniform grid when voltage noise
    is nonzero.  Padding is important: convolving only the requested interval
    would create a boundary artifact in the returned current.
    """
    V_requested = _voltage_axis(V_mV)
    transmissions = np.asarray(tau, dtype=np.float64).reshape(-1)
    if transmissions.size == 0:
        raise ValueError("tau must contain at least one transmission.")
    if not np.all(np.isfinite(transmissions)):
        raise ValueError("tau must be finite.")
    if np.any((transmissions < 0.0) | (transmissions > 1.0)):
        raise ValueError("each transmission must lie in [0, 1].")

    sigma = _nonnegative_scalar(sigmaV_mV, "sigmaV_mV")
    shifted_voltage = V_requested - float(V_offset_mV)
    support = _support_grid(shifted_voltage, sigma)
    current = np.asarray(
        get_Imar_nA(
            V_mV=support,
            tau=transmissions,
            T_K=float(T_K),
            Delta_meV=float(Delta_meV),
            gamma_meV=float(gamma_meV),
            caching=caching,
        ),
        dtype=np.float64,
    )
    if sigma > 0.0:
        current = gaussian_voltage_broaden(support, current, sigma)
    return np.interp(shifted_voltage, support, current) + float(I_offset_nA)


def fit_mar(
    V_mV: ArrayLike,
    I_nA: ArrayLike,
    tau: Sequence[float],
    *,
    parameters: Mapping[str, ParameterSpec] | None = None,
    sigmaI_nA: ArrayLike | float | None = None,
    max_nfev: int = 300,
    verbose: int = 0,
) -> MARFitResult:
    """Fit a multichannel MAR model to one current-voltage trace.

    Parameters
    ----------
    V_mV, I_nA
        Experimental voltage and current.  Non-finite pairs are discarded.
    tau
        Initial channel transmissions.  The number of entries fixes the
        number of channels.  Every transmission is varied in ``[0, 1]``.
    parameters
        Optional mapping from parameter name to
        ``(initial, lower, upper, vary)``.  Unspecified parameters use
        :data:`DEFAULT_PARAMETERS`.  Supported names are ``T_K``,
        ``Delta_meV``, ``gamma_meV``, ``sigmaV_mV``, ``V_offset_mV``, and
        ``I_offset_nA``.
    sigmaI_nA
        Current standard deviation used to weight residuals.  This can be a
        positive scalar or one value per input point.  It is independent of
        the physical voltage broadening ``sigmaV_mV``.
    max_nfev, verbose
        Forwarded to :func:`scipy.optimize.least_squares`.

    Notes
    -----
    Temperature, Dynes broadening, and voltage broadening can all smooth MAR
    structure and are therefore correlated.  Fit only the quantities that
    the data constrain, or compare nested fits with the same voltage window.
    """
    V_all = np.asarray(V_mV, dtype=np.float64).reshape(-1)
    I_all = np.asarray(I_nA, dtype=np.float64).reshape(-1)
    if V_all.shape != I_all.shape:
        raise ValueError("V_mV and I_nA must have the same shape.")

    sigma_all = _sigma_current(sigmaI_nA, V_all.shape)
    finite = np.isfinite(V_all) & np.isfinite(I_all) & np.isfinite(sigma_all)
    V = V_all[finite]
    I = I_all[finite]
    sigma_I = sigma_all[finite]
    order = np.argsort(V)
    V, I, sigma_I = V[order], I[order], sigma_I[order]
    _voltage_axis(V)

    tau_initial = np.asarray(tau, dtype=np.float64).reshape(-1)
    if tau_initial.size == 0 or np.any((tau_initial < 0) | (tau_initial > 1)):
        raise ValueError("tau must be a non-empty sequence with values in [0, 1].")

    specs = dict(DEFAULT_PARAMETERS)
    if parameters is not None:
        unknown = set(parameters) - set(specs)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        specs.update(parameters)
    _validate_parameter_specs(specs)

    names = [f"tau_{index + 1}" for index in range(tau_initial.size)]
    values = dict(zip(names, tau_initial, strict=True))
    lower = [0.0] * tau_initial.size
    upper = [1.0] * tau_initial.size
    for name, (initial, low, high, vary) in specs.items():
        values[name] = float(initial)
        if vary:
            names.append(name)
            lower.append(float(low))
            upper.append(float(high))

    x0 = np.array([values[name] for name in names], dtype=np.float64)

    def unpack(x: FloatArray) -> dict[str, float]:
        result = values.copy()
        result.update(zip(names, x, strict=True))
        return result

    def evaluate(x: FloatArray) -> FloatArray:
        current = unpack(x)
        fitted_tau = [current[f"tau_{index + 1}"] for index in range(tau_initial.size)]
        return mar_current_nA(
            V,
            fitted_tau,
            T_K=current["T_K"],
            Delta_meV=current["Delta_meV"],
            gamma_meV=current["gamma_meV"],
            sigmaV_mV=current["sigmaV_mV"],
            V_offset_mV=current["V_offset_mV"],
            I_offset_nA=current["I_offset_nA"],
            caching=True,
        )

    def residual_function(x: FloatArray) -> FloatArray:
        return (evaluate(x) - I) / sigma_I

    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    jacobian_steps = np.array(
        [
            JACOBIAN_STEPS["tau"] if name.startswith("tau_") else JACOBIAN_STEPS[name]
            for name in names
        ],
        dtype=np.float64,
    )

    def jacobian(x: FloatArray) -> FloatArray:
        """Differentiate above the MAR backend's parameter quantization."""
        columns = []
        for index, step in enumerate(jacobian_steps):
            distance_down = x[index] - lower_array[index]
            distance_up = upper_array[index] - x[index]
            if distance_down >= step and distance_up >= step:
                x_down, x_up = x.copy(), x.copy()
                x_down[index] -= step
                x_up[index] += step
                column = (residual_function(x_up) - residual_function(x_down)) / (
                    2.0 * step
                )
            elif distance_up >= step:
                x_up = x.copy()
                x_up[index] += step
                column = (residual_function(x_up) - residual_function(x)) / step
            elif distance_down >= step:
                x_down = x.copy()
                x_down[index] -= step
                column = (residual_function(x) - residual_function(x_down)) / step
            else:
                column = np.zeros_like(I)
            columns.append(column)
        return np.column_stack(columns)

    solution = least_squares(
        residual_function,
        jac=jacobian,
        x0=x0,
        bounds=(lower_array, upper_array),
        max_nfev=int(max_nfev),
        verbose=int(verbose),
        x_scale="jac",
    )
    fitted_values = unpack(solution.x)
    I_fit = evaluate(solution.x)
    residual = I_fit - I
    degrees_of_freedom = V.size - solution.x.size
    reduced_chi2 = (
        float(2.0 * solution.cost / degrees_of_freedom)
        if degrees_of_freedom > 0
        else np.nan
    )
    errors = {name: np.nan for name in fitted_values}
    errors.update(
        zip(
            names,
            _parameter_errors(solution.jac, reduced_chi2),
            strict=True,
        )
    )
    return MARFitResult(
        values=fitted_values,
        errors=errors,
        V_mV=V,
        I_exp_nA=I,
        I_fit_nA=I_fit,
        residual_nA=residual,
        success=bool(solution.success),
        message=str(solution.message),
        cost=float(solution.cost),
        reduced_chi2=reduced_chi2,
        nfev=int(solution.nfev),
    )


def _support_grid(V_mV: FloatArray, sigmaV_mV: float) -> FloatArray:
    if sigmaV_mV == 0.0:
        return V_mV.copy()
    differences = np.diff(V_mV)
    step = float(np.min(differences))
    padding = max(NOISE_PADDING_SIGMA * sigmaV_mV, 2.0 * step)
    start = V_mV[0] - padding
    stop = V_mV[-1] + padding
    count = int(np.ceil((stop - start) / step)) + 1
    return np.linspace(start, stop, count, dtype=np.float64)


def _voltage_axis(V_mV: ArrayLike, *, uniform: bool = False) -> FloatArray:
    V = np.asarray(V_mV, dtype=np.float64).reshape(-1)
    if V.size < 2 or not np.all(np.isfinite(V)):
        raise ValueError("V_mV must contain at least two finite points.")
    differences = np.diff(V)
    if np.any(differences <= 0.0):
        raise ValueError("V_mV must be strictly increasing.")
    if uniform and not np.allclose(
        differences,
        differences[0],
        rtol=1.0e-7,
        atol=1.0e-12,
    ):
        raise ValueError("V_mV must be uniformly spaced.")
    return V


def _current_axis(I_nA: ArrayLike, V_mV: FloatArray) -> FloatArray:
    I = np.asarray(I_nA, dtype=np.float64).reshape(-1)
    if I.shape != V_mV.shape or not np.all(np.isfinite(I)):
        raise ValueError("I_nA must be finite and have the same shape as V_mV.")
    return I


def _nonnegative_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite, nonnegative scalar.")
    return scalar


def _sigma_current(
    sigmaI_nA: ArrayLike | float | None,
    shape: tuple[int, ...],
) -> FloatArray:
    if sigmaI_nA is None:
        return np.ones(shape, dtype=np.float64)
    sigma = np.asarray(sigmaI_nA, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(shape, float(sigma), dtype=np.float64)
    else:
        sigma = sigma.reshape(-1)
    if sigma.shape != shape:
        raise ValueError("sigmaI_nA must be scalar or match V_mV and I_nA.")
    if np.any(sigma <= 0.0):
        raise ValueError("finite sigmaI_nA values must be positive.")
    return sigma


def _validate_parameter_specs(specs: Mapping[str, ParameterSpec]) -> None:
    for name, spec in specs.items():
        if len(spec) != 4:
            raise ValueError(f"{name} must be (initial, lower, upper, vary).")
        initial, lower, upper, _ = spec
        if not np.all(np.isfinite([initial, lower, upper])):
            raise ValueError(f"{name} bounds and initial value must be finite.")
        if lower > upper or not lower <= initial <= upper:
            raise ValueError(f"{name} must satisfy lower <= initial <= upper.")


def _parameter_errors(jacobian: FloatArray, reduced_chi2: float) -> FloatArray:
    if not np.isfinite(reduced_chi2):
        return np.full(jacobian.shape[1], np.nan, dtype=np.float64)
    try:
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * reduced_chi2
    except np.linalg.LinAlgError:
        return np.full(jacobian.shape[1], np.nan, dtype=np.float64)
    diagonal = np.diag(covariance)
    return np.sqrt(np.where(diagonal >= 0.0, diagonal, np.nan))


__all__ = [
    "DEFAULT_PARAMETERS",
    "MARFitResult",
    "fit_mar",
    "gaussian_voltage_broaden",
    "mar_current_nA",
]
