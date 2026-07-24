"""Apply Gaussian voltage noise to sampled current-voltage curves."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from superconductivity.models.basics.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)

_EXTRAPOLATION_POINTS = 25
_NOISE_ORDER = 32


def apply_sigmaV(
    V_mV: ArrayLike = ...,
    I_nA: ArrayLike = ...,
    sigmaV_mV: float = ...,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Average current over Gaussian voltage fluctuations.

    This uses the same padded-support, linear-extrapolation, and discrete
    Gaussian-kernel algorithm as ``mar_fit.prepare_mar_database``.

    Parameters
    ----------
    V_mV
        One-dimensional, finite, strictly increasing voltage values in mV.
    I_nA
        Current values in nA.  The length of ``axis`` must equal the length
        of ``V_mV``; all other dimensions are treated as independent curves.
    sigmaV_mV
        Standard deviation of the Gaussian voltage noise in mV.  Zero returns
        an unchanged floating-point copy.
    axis
        Axis of ``I_nA`` corresponding to ``V_mV``.

    Returns
    -------
    numpy.ndarray
        Voltage-noise-averaged currents with the same shape as ``I_nA``.
    """
    if V_mV is Ellipsis or I_nA is Ellipsis or sigmaV_mV is Ellipsis:
        raise TypeError("V_mV, I_nA, and sigmaV_mV must be provided.")

    voltage = np.asarray(V_mV, dtype=np.float64)
    current = np.asarray(I_nA, dtype=np.float64)
    sigma = float(sigmaV_mV)

    if voltage.ndim != 1 or voltage.size < 2:
        raise ValueError("V_mV must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(voltage)):
        raise ValueError("V_mV must contain only finite values.")
    if np.any(np.diff(voltage) <= 0.0):
        raise ValueError("V_mV must be strictly increasing.")
    if current.ndim == 0:
        raise ValueError("I_nA must have at least one dimension.")
    if not np.all(np.isfinite(current)):
        raise ValueError("I_nA must contain only finite values.")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigmaV_mV must be finite and nonnegative.")

    axis = np.core.numeric.normalize_axis_index(axis, current.ndim)
    if current.shape[axis] != voltage.size:
        raise ValueError(
            f"I_nA.shape[{axis}] must equal V_mV.size "
            f"({current.shape[axis]} != {voltage.size})."
        )
    if sigma == 0.0:
        return current.copy()

    support = make_bias_support_grid(voltage, sigma)
    current_last = np.moveaxis(current, axis, -1)
    original_shape = current_last.shape
    current_bank = current_last.reshape(-1, voltage.size)
    extended = _extend_current_bank(current_bank, voltage, support)

    broadened = np.empty_like(current_bank)
    for index, current_support in enumerate(extended):
        filtered = apply_voltage_noise(
            support,
            current_support,
            sigma,
            order=_NOISE_ORDER,
        )
        broadened[index] = np.interp(voltage, support, filtered)

    broadened = broadened.reshape(original_shape)
    return np.moveaxis(broadened, -1, axis)


def _extend_current_bank(
    currents: NDArray[np.float64],
    voltage: NDArray[np.float64],
    support: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Match the boundary extrapolation used by the MAR fit database."""
    count = min(_EXTRAPOLATION_POINTS, voltage.size)
    extended = np.stack(
        [np.interp(support, voltage, row) for row in currents]
    )
    for edge, boundary_voltage, outside, boundary in (
        (slice(0, count), voltage[0], support < voltage[0], currents[:, 0]),
        (
            slice(-count, None),
            voltage[-1],
            support > voltage[-1],
            currents[:, -1],
        ),
    ):
        x = voltage[edge]
        centered = x - np.mean(x)
        slopes = currents[:, edge] @ centered / np.sum(centered**2)
        extended[:, outside] = boundary[:, None] + slopes[:, None] * (
            support[outside] - boundary_voltage
        )
    return extended


__all__ = ["apply_sigmaV"]
