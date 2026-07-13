"""Cached grid fitting for atomic-contact multiple Andreev reflection.

The expensive, unbroadened single-channel MAR currents are obtained as:

``I[tau, T, Delta, gamma, sigmaV, V]``.

``get_Imar_nA(..., caching=True)`` owns persistence in the shared native MAR
HDF5 cache.  This module assembles the requested curves into a dense in-memory
array and precalculates voltage-noise broadening from them.  ``sigmaV_mV`` is
therefore a fit-grid dimension, but not a native HDF5-cache dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from superconductivity.models.mar import get_Imar_nA
from superconductivity.models.basics.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)
from superconductivity.utilities.constants import G0_muS
from tqdm.auto import tqdm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]


@dataclass(frozen=True)
class MARGrid:
    """Axes of the in-memory MAR current bank.

    Parameters
    ----------
    V_mV
        Arbitrarily spaced, strictly increasing voltage grid.  Include
        sufficient voltage range for the fit; sigma broadening uses only the
        available support at the database boundaries.
    tau, T_K, Delta_meV, gamma_meV, sigmaV_mV
        Arbitrarily spaced single-channel MAR parameter axes.  They are
        sorted, rounded to the native MAR-cache precision, and deduplicated.
        One-point axes are allowed.
    """

    V_mV: FloatArray
    tau: FloatArray
    T_K: FloatArray
    Delta_meV: FloatArray
    gamma_meV: FloatArray
    sigmaV_mV: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "V_mV", _axis(self.V_mV, "V_mV"))
        object.__setattr__(self, "tau", _parameter_axis(self.tau, "tau", 4))
        object.__setattr__(self, "T_K", _parameter_axis(self.T_K, "T_K", 4))
        object.__setattr__(
            self,
            "Delta_meV",
            _parameter_axis(self.Delta_meV, "Delta_meV", 6),
        )
        object.__setattr__(
            self,
            "gamma_meV",
            _parameter_axis(self.gamma_meV, "gamma_meV", 9),
        )
        object.__setattr__(
            self,
            "sigmaV_mV",
            _parameter_axis(self.sigmaV_mV, "sigmaV_mV", 9),
        )
        if self.V_mV.size < 2:
            raise ValueError("V_mV must contain at least two values.")
        if np.any((self.tau < 0.0) | (self.tau > 1.0)):
            raise ValueError("tau must lie in [0, 1].")
        if (
            np.any(self.T_K < 0.0)
            or np.any(self.gamma_meV < 0.0)
            or np.any(self.sigmaV_mV < 0.0)
        ):
            raise ValueError("T_K, gamma_meV, and sigmaV_mV must be nonnegative.")
        if np.any(self.Delta_meV <= 0.0):
            raise ValueError("Delta_meV must be positive.")

    @property
    def current_shape(self) -> tuple[int, ...]:
        """Shape of the cached current array."""
        return (
            self.tau.size,
            self.T_K.size,
            self.Delta_meV.size,
            self.gamma_meV.size,
            self.sigmaV_mV.size,
            self.V_mV.size,
        )


@dataclass(frozen=True)
class MARDatabase:
    """A voltage-broadened single-channel MAR current bank and its grid."""

    grid: MARGrid
    I_nA: FloatArray
    curves_requested: int = 0
    loading_time_s: float = 0.0


@dataclass(frozen=True)
class MARGridFitResult:
    """Best point in a complete pincode and global-parameter grid search."""

    tau: FloatArray
    T_K: float
    Delta_meV: float
    gamma_meV: float
    sigmaV_mV: float
    chi2: float
    reduced_chi2: float
    V_mV: FloatArray
    I_exp_nA: FloatArray
    Ifit_nA: FloatArray
    residual_nA: FloatArray
    pincode_index: int
    parameter_index: tuple[int, int, int, int]
    pincodes_tested: int
    grid_points_tested: int
    fitting_time_s: float


def prepare_mar_trace(
    V_mV,
    I_nA,
    Vnan_mV=0.03,
    VGN_mV=0.5,
):
    """Return a fit mask and estimate GN/G0 from the outer IV plateaus."""
    V_mV = np.asarray(V_mV, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)

    if V_mV.shape != I_nA.shape:
        raise ValueError("V_mV and I_nA must have the same shape.")

    finite = np.isfinite(V_mV) & np.isfinite(I_nA)
    mask = finite & (np.abs(V_mV) > Vnan_mV)
    plateau = finite & (np.abs(V_mV) >= VGN_mV)

    slopes_uS = []
    for polarity in (-1, 1):
        select = plateau & (np.sign(V_mV) == polarity)
        if np.count_nonzero(select) >= 2:
            slope_uS, _ = np.polyfit(
                V_mV[select],
                I_nA[select],
                deg=1,
            )
            slopes_uS.append(slope_uS)

    if not slopes_uS:
        raise ValueError("Not enough finite points on the outer IV plateaus.")

    GN_G0 = float(np.mean(slopes_uS) / G0_muS)
    return mask, GN_G0


def prepare_mar_database(
    grid: MARGrid,
    *,
    progress: bool = True,
) -> MARDatabase:
    """Assemble a dense grid using the native MAR cache.

    Every call to :func:`get_Imar_nA` uses ``caching=True``.  The MAR frontend
    checks its shared HDF5 cache for the parameter tuple and requested voltage
    points, evaluates only missing points, and merges them into that cache.
    Consequently, this function does not create or manage another cache.

    Parameters
    ----------
    grid
        Requested voltage and MAR parameter axes.
    progress
        Show a notebook-aware progress bar while loading the grid.
    """
    V_support_mV = make_bias_support_grid(
        grid.V_mV,
        float(np.max(grid.sigmaV_mV)),
    )
    base_shape = (
        grid.tau.size,
        grid.T_K.size,
        grid.Delta_meV.size,
        grid.gamma_meV.size,
        V_support_mV.size,
    )
    base_I_nA = np.empty(base_shape, dtype=np.float64)
    parameter_indices = list(np.ndindex(base_shape[:-1]))
    total = len(parameter_indices)
    start = perf_counter()
    iterator = tqdm(
        parameter_indices,
        desc="MAR curves",
        unit="curve",
        disable=not progress,
    )
    for i_tau, i_T, i_delta, i_gamma in iterator:
        base_I_nA[i_tau, i_T, i_delta, i_gamma] = get_Imar_nA(
            V_mV=V_support_mV,
            tau=float(grid.tau[i_tau]),
            T_K=float(grid.T_K[i_T]),
            Delta_meV=float(grid.Delta_meV[i_delta]),
            gamma_meV=float(grid.gamma_meV[i_gamma]),
            caching=True,
        )

    I_nA = np.empty(grid.current_shape, dtype=np.float64)
    global_indices = np.ndindex(
        grid.T_K.size,
        grid.Delta_meV.size,
        grid.gamma_meV.size,
    )
    broadening_jobs = [
        (i_T, i_delta, i_gamma, i_sigma)
        for i_T, i_delta, i_gamma in global_indices
        for i_sigma in range(grid.sigmaV_mV.size)
    ]
    broadening_iterator = tqdm(
        broadening_jobs,
        desc="Voltage noise",
        unit="bank",
        disable=not progress,
    )
    for i_T, i_delta, i_gamma, i_sigma in broadening_iterator:
        sigma = float(grid.sigmaV_mV[i_sigma])
        for i_tau in range(grid.tau.size):
            if sigma == 0.0:
                I_nA[i_tau, i_T, i_delta, i_gamma, i_sigma, :] = np.interp(
                    grid.V_mV,
                    V_support_mV,
                    base_I_nA[i_tau, i_T, i_delta, i_gamma, :],
                )
                continue
            broadened_support = apply_voltage_noise(
                V_support_mV,
                base_I_nA[i_tau, i_T, i_delta, i_gamma, :],
                sigma,
                order=32,
            )
            I_nA[i_tau, i_T, i_delta, i_gamma, i_sigma, :] = np.interp(
                grid.V_mV,
                V_support_mV,
                broadened_support,
            )

    elapsed = perf_counter() - start
    return MARDatabase(
        grid=grid,
        I_nA=I_nA,
        curves_requested=total,
        loading_time_s=elapsed,
    )


def fit_mar_grid(
    I_nA: ArrayLike,
    database: MARDatabase,
    *,
    n_channels: int,
    tau_sum_bounds: tuple[float, float] = (0.0, np.inf),
    sigmaG_G0: ArrayLike | float | None = None,
    voltage_bounds_mV: tuple[float, float] | None = None,
    batch_size: int = 50_000,
    progress: bool = True,
) -> MARGridFitResult:
    """Search the full MAR, voltage-noise, and pincode grid.

    ``I_nA`` must already be mapped onto ``database.grid.V_mV``.  The fitted
    quantity is ``I_nA / (V_mV * G0_muS)``.  Non-finite current or uncertainty
    entries and zero voltage are ignored directly; no interpolation is
    performed.  All combinations with replacement of ``n_channels`` are
    tested, optionally restricted by total transmission.

    Voltage broadening is already precalculated along the grid's
    ``sigmaV_mV`` axis.  All pincodes are scored in batches.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be at least one.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    grid = database.grid
    if database.I_nA.shape != grid.current_shape:
        raise ValueError("database current shape does not match its grid.")
    if not np.all(np.isfinite(database.I_nA)):
        raise ValueError("database contains missing or non-finite IVs.")

    I_data = np.asarray(I_nA, dtype=np.float64)
    if I_data.shape != grid.V_mV.shape:
        raise ValueError(
            "I_nA must be a 1D trace mapped onto database.grid.V_mV "
            f"with shape {grid.V_mV.shape}; got {I_data.shape}."
        )
    sigma_data = _fit_uncertainty(sigmaG_G0, grid.V_mV.shape)
    fit_mask = (
        np.isfinite(I_data)
        & np.isfinite(sigma_data)
        & (sigma_data > 0.0)
        & (grid.V_mV != 0.0)
    )
    if voltage_bounds_mV is not None:
        low, high = map(float, voltage_bounds_mV)
        if low >= high:
            raise ValueError("voltage_bounds_mV must satisfy low < high.")
        fit_mask &= (grid.V_mV >= low) & (grid.V_mV <= high)
    V_fit = grid.V_mV[fit_mask]
    if V_fit.size < 2:
        raise ValueError("fewer than two finite current samples lie in the fit window.")
    I_exp = I_data[fit_mask]
    G_exp = I_exp / (V_fit * G0_muS)
    sigma_G = sigma_data[fit_mask]

    pincodes = generate_pincodes(
        grid.tau,
        n_channels,
        tau_sum_bounds=tau_sum_bounds,
        progress=progress,
    )
    if pincodes.shape[0] == 0:
        raise ValueError("no pincodes satisfy tau_sum_bounds.")

    best_chi2 = np.inf
    best_indices: tuple[int, int, int, int, int] | None = None
    best_current: FloatArray | None = None
    start = perf_counter()
    total_global = (
        grid.T_K.size * grid.Delta_meV.size * grid.gamma_meV.size * grid.sigmaV_mV.size
    )
    global_indices = np.ndindex(
        grid.T_K.size,
        grid.Delta_meV.size,
        grid.gamma_meV.size,
        grid.sigmaV_mV.size,
    )
    fit_progress = tqdm(
        total=total_global * pincodes.shape[0],
        desc="MAR fit",
        unit="candidate",
        disable=not progress,
    )
    for i_T, i_delta, i_gamma, i_sigma in global_indices:
        bank = database.I_nA[:, i_T, i_delta, i_gamma, i_sigma, :][:, fit_mask]
        G_bank = bank / (V_fit[None, :] * G0_muS)
        chi2, i_pincode, current = _best_pincode(
            G_bank,
            pincodes,
            G_exp,
            sigma_G,
            batch_size,
            progress_bar=fit_progress,
        )
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_indices = (
                i_T,
                i_delta,
                i_gamma,
                i_sigma,
                i_pincode,
            )
            best_current = current
            fit_progress.set_postfix(
                best_chi2=f"{best_chi2:.4g}",
                refresh=False,
            )
    fit_progress.close()

    if best_indices is None or best_current is None:
        raise RuntimeError("MAR grid search produced no result.")
    i_T, i_delta, i_gamma, i_sigma, i_pincode = best_indices
    tau_fit = np.sort(grid.tau[pincodes[i_pincode]])[::-1]
    Ifit_nA = _evaluate_mar_with_voltage_noise(
        grid.V_mV,
        tau=tau_fit,
        T_K=float(grid.T_K[i_T]),
        Delta_meV=float(grid.Delta_meV[i_delta]),
        gamma_meV=float(grid.gamma_meV[i_gamma]),
        sigmaV_mV=float(grid.sigmaV_mV[i_sigma]),
    )
    residual = np.full_like(I_data, np.nan)
    residual[fit_mask] = Ifit_nA[fit_mask] - I_data[fit_mask]
    degrees_of_freedom = V_fit.size
    return MARGridFitResult(
        tau=tau_fit,
        T_K=float(grid.T_K[i_T]),
        Delta_meV=float(grid.Delta_meV[i_delta]),
        gamma_meV=float(grid.gamma_meV[i_gamma]),
        sigmaV_mV=float(grid.sigmaV_mV[i_sigma]),
        chi2=float(best_chi2),
        reduced_chi2=float(best_chi2 / degrees_of_freedom),
        V_mV=grid.V_mV.copy(),
        I_exp_nA=I_data.copy(),
        Ifit_nA=Ifit_nA,
        residual_nA=residual,
        pincode_index=i_pincode,
        parameter_index=(i_T, i_delta, i_gamma, i_sigma),
        pincodes_tested=pincodes.shape[0],
        grid_points_tested=total_global * pincodes.shape[0],
        fitting_time_s=perf_counter() - start,
    )


def generate_pincodes(
    tau: ArrayLike,
    n_channels: int,
    *,
    tau_sum_bounds: tuple[float, float] = (0.0, np.inf),
    progress: bool = False,
) -> IntArray:
    """Return nondecreasing transmission-index combinations in bounds."""
    tau_grid = _parameter_axis(tau, "tau", 4)
    if n_channels < 1:
        raise ValueError("n_channels must be at least one.")
    lower, upper = map(float, tau_sum_bounds)
    if lower > upper:
        raise ValueError("tau_sum_bounds must satisfy lower <= upper.")
    candidates = combinations_with_replacement(
        range(tau_grid.size),
        n_channels,
    )
    total = comb(tau_grid.size + n_channels - 1, n_channels)
    iterator = tqdm(
        candidates,
        total=total,
        desc="Pincodes",
        unit="candidate",
        disable=not progress,
    )
    accepted = []
    for indices in iterator:
        transmission = sum(tau_grid[index] for index in indices)
        if lower <= transmission <= upper:
            accepted.append(indices)
    if not accepted:
        return np.empty((0, n_channels), dtype=np.int32)
    return np.asarray(accepted, dtype=np.int32)


def _evaluate_mar_with_voltage_noise(
    V_mV: FloatArray,
    *,
    tau: FloatArray,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    sigmaV_mV: float,
) -> FloatArray:
    """Recalculate one complete pincode with padded voltage broadening."""
    if sigmaV_mV == 0.0:
        return np.asarray(
            get_Imar_nA(
                V_mV=V_mV,
                tau=tau,
                T_K=T_K,
                Delta_meV=Delta_meV,
                gamma_meV=gamma_meV,
                caching=True,
            ),
            dtype=np.float64,
        )

    V_support_mV = make_bias_support_grid(V_mV, sigmaV_mV)
    I_support_nA = get_Imar_nA(
        V_mV=V_support_mV,
        tau=tau,
        T_K=T_K,
        Delta_meV=Delta_meV,
        gamma_meV=gamma_meV,
        caching=True,
    )
    broadened_support_nA = apply_voltage_noise(
        V_support_mV,
        np.asarray(I_support_nA, dtype=np.float64),
        sigmaV_mV,
        order=32,
    )
    return np.interp(V_mV, V_support_mV, broadened_support_nA)


def broaden_current_bank(
    V_mV: ArrayLike,
    I_tau_nA: ArrayLike,
    sigmaV_mV: float,
) -> FloatArray:
    """Apply the shared BCS voltage-noise kernel to a current bank.

    This is a low-level operation on the supplied support.  To avoid physical
    boundary artifacts, the support must already extend beyond the desired
    output interval.  :func:`prepare_mar_database` handles that padding and
    cropping automatically.
    """
    V = _axis(V_mV, "V_mV")
    currents = np.asarray(I_tau_nA, dtype=np.float64)
    if currents.ndim != 2 or currents.shape[1] != V.size:
        raise ValueError("I_tau_nA must have shape (n_tau, n_voltage).")
    if not np.all(np.isfinite(currents)):
        raise ValueError("I_tau_nA must be finite.")
    sigma = float(sigmaV_mV)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigmaV_mV must be finite and nonnegative.")
    if sigma == 0.0:
        return currents.copy()
    return np.stack(
        [apply_voltage_noise(V, current, sigma, order=32) for current in currents]
    )


def _best_pincode(
    bank: FloatArray,
    pincodes: IntArray,
    target: FloatArray,
    sigma: FloatArray,
    batch_size: int,
    progress_bar=None,
) -> tuple[float, int, FloatArray]:
    best_chi2 = np.inf
    best_index = -1
    best_current: FloatArray | None = None
    for start in range(0, pincodes.shape[0], batch_size):
        stop = min(start + batch_size, pincodes.shape[0])
        models = np.sum(bank[pincodes[start:stop]], axis=1)
        residuals = (models - target[None, :]) / sigma[None, :]
        chi2 = np.einsum("ij,ij->i", residuals, residuals)
        local = int(np.argmin(chi2))
        if chi2[local] < best_chi2:
            best_chi2 = float(chi2[local])
            best_index = start + local
            best_current = models[local].copy()
        if progress_bar is not None:
            progress_bar.update(stop - start)
    if best_current is None:
        raise RuntimeError("pincode search received no candidates.")
    return best_chi2, best_index, best_current


def _fit_uncertainty(
    sigmaG_G0: ArrayLike | float | None,
    shape: tuple[int, ...],
) -> FloatArray:
    if sigmaG_G0 is None:
        return np.ones(shape, dtype=np.float64)
    sigma = np.asarray(sigmaG_G0, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(shape, float(sigma), dtype=np.float64)
    if sigma.shape != shape:
        raise ValueError(
            "sigmaG_G0 must be scalar or have the same shape as " "database.grid.V_mV."
        )
    return sigma


def _axis(values: ArrayLike, name: str, *, uniform: bool = False) -> FloatArray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values.")
    if axis.size > 1:
        differences = np.diff(axis)
        if np.any(differences <= 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
        if uniform and not np.allclose(
            differences,
            differences[0],
            rtol=1.0e-7,
            atol=1.0e-12,
        ):
            raise ValueError(f"{name} must be uniformly spaced.")
    elif uniform:
        raise ValueError(f"{name} must contain at least two values.")
    return axis


def _parameter_axis(
    values: ArrayLike,
    name: str,
    decimals: int,
) -> FloatArray:
    """Normalize an arbitrary parameter grid to native cache precision."""
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values.")
    return np.unique(np.round(axis, decimals=decimals))


__all__ = [
    "MARDatabase",
    "MARGrid",
    "MARGridFitResult",
    "broaden_current_bank",
    "fit_mar_grid",
    "generate_pincodes",
    "prepare_mar_database",
]
