"""Utilities for the threshold-resolved photon-assisted MAR model."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_weights(
    Vbias: ArrayLike,
    Delta_meV: float | None = None,
    mmax: int = 10,
    *,
    return_unused: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return smooth voltage-dependent weights for MAR orders 1 through mmax.

    Each MAR order is represented by a Gaussian on the inverse-voltage axis

    ``2 Delta / abs(V)``

    and is centered on the integer corresponding to that MAR order.

    Parameters
    ----------
    Vbias:
        One-dimensional, uniformly increasing bias-voltage grid. If
        ``Delta_meV`` is omitted, the voltage must already be reduced as
        ``V / Delta``. Otherwise, ``Vbias`` and ``Delta_meV`` must use the
        same voltage units (normally millivolts and millielectronvolts).
    Delta_meV:
        Superconducting gap in millielectronvolts. Passing it converts the
        supplied physical voltage to reduced voltage. The default assumes
        that ``Vbias`` is already reduced.
    mmax:
        Largest explicitly returned MAR order.
    return_unused:
        If true, also return the total weight assigned to orders above
        ``mmax`` and to the unresolved central overflow.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        The resolved weights have shape ``(mmax, Vbias.size)``. Row ``m - 1``
        contains the weight of MAR order ``m``. If ``return_unused`` is true,
        the second returned array has shape ``Vbias.shape`` and contains all
        unused weight. Consequently, ``weights.sum(axis=0) + unused_weight``
        equals one.
    """
    voltage = np.asarray(Vbias, dtype=np.float64)
    if voltage.ndim != 1:
        raise ValueError("Vbias must be one-dimensional.")
    if voltage.size < 2:
        raise ValueError("Vbias must contain at least two points.")
    if not np.all(np.isfinite(voltage)):
        raise ValueError("Vbias must contain only finite values.")
    if isinstance(mmax, (bool, np.bool_)) or not isinstance(
        mmax,
        (int, np.integer),
    ):
        raise TypeError("mmax must be an integer.")
    if mmax < 1:
        raise ValueError("mmax must be positive.")

    if Delta_meV is not None:
        gap_meV = float(Delta_meV)
        if not np.isfinite(gap_meV) or gap_meV <= 0.0:
            raise ValueError("Delta_meV must be finite and positive.")
        voltage = voltage / gap_meV

    voltage_steps = np.diff(voltage)
    if np.any(voltage_steps <= 0.0):
        raise ValueError("Vbias must be strictly increasing.")
    step = float(np.mean(voltage_steps))
    if not np.allclose(voltage_steps, step, rtol=1e-6, atol=1e-12):
        raise ValueError("Vbias must be uniformly spaced.")

    # Construct orders down to the voltage-grid resolution. Orders above
    # mmax participate in normalization and are then left unresolved.
    construction_mmax = max(
        mmax + 1,
        int(np.ceil(2.0 / step - 1e-12)),
    )
    construction_orders = np.arange(
        1,
        construction_mmax + 1,
        dtype=np.float64,
    )
    abs_voltage = np.abs(voltage)
    inverse_voltage = np.divide(
        2.0,
        abs_voltage,
        out=np.full_like(abs_voltage, np.inf),
        where=abs_voltage != 0.0,
    )
    width = 0.5
    distance = inverse_voltage[None, :] - construction_orders[:, None]
    weights = np.exp(-0.5 * (distance / width) ** 2)

    # Represent all orders above the grid-resolved range by the rising half of
    # the next Gaussian and a unit plateau towards zero voltage. This gives an
    # equal crossover halfway between the last resolved and first unresolved
    # integer order.
    overflow_distance = np.maximum(
        construction_mmax + 1.0 - inverse_voltage,
        0.0,
    )
    overflow_weight = np.exp(-0.5 * (overflow_distance / width) ** 2)
    normalization = np.sum(weights, axis=0) + overflow_weight
    weights /= normalization[None, :]
    overflow_weight /= normalization

    unused_weight = np.sum(weights[mmax:], axis=0) + overflow_weight
    resolved_weights = np.asarray(weights[:mmax], dtype=np.float64)
    if return_unused:
        return resolved_weights, np.asarray(unused_weight, dtype=np.float64)
    return resolved_weights


__all__ = ["get_weights"]
