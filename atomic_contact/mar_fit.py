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
from functools import lru_cache
from itertools import product
from math import comb
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from superconductivity.models.basics.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)
from superconductivity.models.mar import get_Imar_nA
from superconductivity.models.mar.core import (
    SymmetricHAParams,
    quantize_voltage_mV,
)
from superconductivity.models.mar.core.cache import locked_h5_file
from superconductivity.models.mar.models.ha_sym import (
    CACHE_FILE as HA_SYM_CACHE_FILE,
)
from superconductivity.models.mar.models.ha_sym import (
    CACHE_ROOT_GROUP as HA_SYM_CACHE_ROOT,
)
from superconductivity.utilities.constants import G0_muS
from tqdm.auto import tqdm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
FitBackend = Literal["auto", "jax", "numpy"]


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
            raise ValueError(
                "T_K, gamma_meV, and sigmaV_mV must be nonnegative."
            )
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


def _bulk_read_mar_cache(
    V_mV: FloatArray,
    grid: MARGrid,
    output: FloatArray,
    parameter_indices: list[tuple[int, ...]],
    *,
    desc: str,
    progress: bool,
) -> list[tuple[int, ...]]:
    """Fill complete symmetric-HA curves during one HDF5 session."""
    nonzero = V_mV != 0.0
    V_lookup_q = quantize_voltage_mV(np.abs(V_mV[nonzero]))
    missing = []
    iterator = tqdm(
        parameter_indices,
        desc=desc,
        unit="curve",
        disable=not progress,
    )
    with locked_h5_file(HA_SYM_CACHE_FILE, "a") as handle:
        for indices in iterator:
            i_tau, i_T, i_delta, i_gamma = indices
            tau = float(grid.tau[i_tau])
            if tau == 0.0:
                output[indices] = 0.0
                continue

            params = SymmetricHAParams.from_raw(
                tau=tau,
                T_K=float(grid.T_K[i_T]),
                Delta_meV=float(grid.Delta_meV[i_delta]),
                gamma_meV=float(grid.gamma_meV[i_gamma]),
                gamma_meV_min=1.0e-4,
            )
            group_path = f"{HA_SYM_CACHE_ROOT}/{params.cache_key()}"
            if group_path not in handle:
                missing.append(indices)
                continue

            group = handle[group_path]
            V_cached_q = np.asarray(group["V_q"][...], dtype=np.int64)
            I_cached_nA = np.asarray(group["I_nA"][...], dtype=np.float64)
            positions = np.searchsorted(V_cached_q, V_lookup_q)
            complete = np.all(positions < V_cached_q.size)
            if complete:
                complete = np.array_equal(
                    V_cached_q[positions],
                    V_lookup_q,
                )
            if not complete:
                missing.append(indices)
                continue

            current = np.zeros_like(V_mV)
            current[nonzero] = np.sign(V_mV[nonzero]) * I_cached_nA[positions]
            output[indices] = current
    return missing


def prepare_mar_trace(
    V_mV,
    I_nA,
    Vnan_mV=0.03,
):
    """Return a fit mask and estimate GN/G0 from the outer IV plateaus."""
    V_mV = np.asarray(V_mV, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)

    if V_mV.shape != I_nA.shape:
        raise ValueError("V_mV and I_nA must have the same shape.")

    finite = np.isfinite(V_mV) & np.isfinite(I_nA)
    mask = finite & (np.abs(V_mV) > Vnan_mV)
    return mask


def estimate_GN_bounds(
    V_mV,
    I_nA,
    VGN_mV=0.5,
    n_sections=4,
    confidence=2.0,
):
    V_mV = np.asarray(V_mV, dtype=np.float64)
    I_nA = np.asarray(I_nA, dtype=np.float64)

    finite = np.isfinite(V_mV) & np.isfinite(I_nA)
    slopes_G0 = []

    for polarity in (-1, 1):
        select = finite & (polarity * V_mV >= VGN_mV)
        indices = np.flatnonzero(select)

        for section in np.array_split(indices, n_sections):
            if section.size < 2:
                continue

            slope_uS, _ = np.polyfit(
                V_mV[section],
                I_nA[section],
                deg=1,
            )
            slopes_G0.append(slope_uS / G0_muS)

    if len(slopes_G0) < 2:
        raise ValueError("Not enough plateau sections to estimate bounds.")

    slopes_G0 = np.asarray(slopes_G0)
    GN_G0 = float(np.median(slopes_G0))

    sigma_GN_G0 = float(1.4826 * np.median(np.abs(slopes_G0 - GN_G0)))
    delta_GN_G0 = max(
        confidence * sigma_GN_G0,
        0.02 * GN_G0,
        0.02,
    )

    bounds = (
        max(0.0, GN_G0 - delta_GN_G0),
        GN_G0 + delta_GN_G0,
    )
    return GN_G0, bounds


def prepare_mar_database(
    grid: MARGrid,
    *,
    extrapolation_points: int = 25,
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
    extrapolation_points
        Number of points at each voltage edge used to linearly extrapolate
        the IV onto the Gaussian-convolution padding.  This avoids expensive
        MAR calculations outside ``grid.V_mV``.
    progress
        Show a notebook-aware progress bar while loading the grid.
    """
    V_support_mV = make_bias_support_grid(
        grid.V_mV,
        float(np.max(grid.sigmaV_mV)),
    )
    if extrapolation_points < 2:
        raise ValueError("extrapolation_points must be at least two.")
    base_shape = (
        grid.tau.size,
        grid.T_K.size,
        grid.Delta_meV.size,
        grid.gamma_meV.size,
        grid.V_mV.size,
    )
    base_I_nA = np.full(base_shape, np.nan, dtype=np.float64)
    parameter_indices = list(np.ndindex(base_shape[:-1]))
    total = len(parameter_indices)
    start = perf_counter()
    missing = _bulk_read_mar_cache(
        grid.V_mV,
        grid,
        base_I_nA,
        parameter_indices,
        desc="MAR cache read",
        progress=progress,
    )
    iterator = tqdm(
        missing,
        desc="Missing MAR curves",
        unit="curve",
        disable=not progress or not missing,
    )
    for i_tau, i_T, i_delta, i_gamma in iterator:
        base_I_nA[i_tau, i_T, i_delta, i_gamma] = get_Imar_nA(
            V_mV=grid.V_mV,
            tau=float(grid.tau[i_tau]),
            T_K=float(grid.T_K[i_T]),
            Delta_meV=float(grid.Delta_meV[i_delta]),
            gamma_meV=float(grid.gamma_meV[i_gamma]),
            caching=True,
        )
    if missing:
        _bulk_read_mar_cache(
            grid.V_mV,
            grid,
            base_I_nA,
            parameter_indices,
            desc="MAR cache reread",
            progress=progress,
        )
    if not np.all(np.isfinite(base_I_nA)):
        raise RuntimeError("MAR cache remains incomplete after calculation.")

    I_nA = np.empty(grid.current_shape, dtype=np.float64)
    global_indices = list(
        np.ndindex(
            grid.T_K.size,
            grid.Delta_meV.size,
            grid.gamma_meV.size,
        )
    )
    broadening_progress = tqdm(
        total=len(global_indices) * grid.sigmaV_mV.size,
        desc="Voltage noise",
        unit="bank",
        disable=not progress,
    )
    for i_T, i_delta, i_gamma in global_indices:
        base_bank = base_I_nA[:, i_T, i_delta, i_gamma, :]
        support_bank = _extend_current_bank(
            grid.V_mV,
            base_bank,
            V_support_mV,
            extrapolation_points,
        )
        for i_sigma, sigma in enumerate(grid.sigmaV_mV):
            for i_tau in range(grid.tau.size):
                if sigma == 0.0:
                    I_nA[i_tau, i_T, i_delta, i_gamma, i_sigma, :] = base_bank[
                        i_tau
                    ]
                    continue
                broadened_support = apply_voltage_noise(
                    V_support_mV,
                    support_bank[i_tau],
                    float(sigma),
                    order=32,
                )
                I_nA[i_tau, i_T, i_delta, i_gamma, i_sigma, :] = np.interp(
                    grid.V_mV,
                    V_support_mV,
                    broadened_support,
                )
            broadening_progress.update(1)
    broadening_progress.close()

    elapsed = perf_counter() - start
    return MARDatabase(
        grid=grid,
        I_nA=I_nA,
        curves_requested=total,
        loading_time_s=elapsed,
    )


def _selected_grid_indices(
    axis: FloatArray,
    fixed_value: float | None,
    name: str,
) -> NDArray[np.int64]:
    """Return all indices or the index of one existing grid value."""
    if fixed_value is None:
        return np.arange(axis.size, dtype=np.int64)
    value = float(fixed_value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite when fixed.")
    matches = np.flatnonzero(
        np.isclose(
            axis,
            value,
            rtol=0.0,
            atol=np.finfo(np.float64).eps * max(1.0, abs(value)) * 8.0,
        )
    )
    if matches.size == 0:
        nearest = int(np.argmin(np.abs(axis - value)))
        raise ValueError(
            f"fixed {name}={value!r} is not in the loaded MAR grid; "
            f"nearest available value is {float(axis[nearest])!r}."
        )
    return np.asarray([matches[0]], dtype=np.int64)


def fit_mar_grid(
    I_nA: ArrayLike,
    database: MARDatabase,
    *,
    n_channels: int,
    tau_sum_bounds: tuple[float, float] = (0.0, np.inf),
    sigmaG_G0: ArrayLike | float | None = None,
    voltage_bounds_mV: tuple[float, float] | None = None,
    T_K: float | None = None,
    Delta_meV: float | None = None,
    Delta_mev: float | None = None,
    gamma_meV: float | None = None,
    sigmaV_mV: float | None = None,
    batch_size: int = 50_000,
    backend: FitBackend = "auto",
    progress: bool = True,
) -> MARGridFitResult:
    """Search the full MAR, voltage-noise, and pincode grid.

    ``I_nA`` must already be mapped onto ``database.grid.V_mV``.  The fitted
    quantity is ``I_nA / (V_mV * G0_muS)``.  Non-finite current or uncertainty
    entries and zero voltage are ignored directly; no interpolation is
    performed.  All combinations with replacement of ``n_channels`` are
    tested, optionally restricted by total transmission.

    Voltage broadening is already precalculated along the grid's
    ``sigmaV_mV`` axis.  All pincodes are scored in batches.  Pass any of
    ``T_K``, ``Delta_meV`` (or ``Delta_mev``), ``gamma_meV``, and
    ``sigmaV_mV`` to restrict that parameter to an already-loaded grid value.
    The database is neither reloaded nor recalculated.  ``backend="auto"``
    uses JAX when it is installed and falls back to memory-bounded NumPy.
    The JAX scorer keeps the large model and residual arrays inside the
    compiled calculation and transfers only each batch's minimum back to
    Python.
    """
    if n_channels < 1:
        raise ValueError("n_channels must be at least one.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one.")
    if backend not in ("auto", "jax", "numpy"):
        raise ValueError("backend must be 'auto', 'jax', or 'numpy'.")
    grid = database.grid
    if database.I_nA.shape != grid.current_shape:
        raise ValueError("database current shape does not match its grid.")
    if not np.all(np.isfinite(database.I_nA)):
        raise ValueError("database contains missing or non-finite IVs.")
    if Delta_meV is not None and Delta_mev is not None:
        raise ValueError("pass only one of Delta_meV and Delta_mev.")
    fixed_delta_meV = Delta_meV if Delta_meV is not None else Delta_mev

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
        raise ValueError(
            "fewer than two finite current samples lie in the fit window."
        )
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
    parameter_indices = (
        _selected_grid_indices(grid.T_K, T_K, "T_K"),
        _selected_grid_indices(grid.Delta_meV, fixed_delta_meV, "Delta_meV"),
        _selected_grid_indices(grid.gamma_meV, gamma_meV, "gamma_meV"),
        _selected_grid_indices(grid.sigmaV_mV, sigmaV_mV, "sigmaV_mV"),
    )
    total_global = int(
        np.prod([indices.size for indices in parameter_indices])
    )
    global_indices = product(*parameter_indices)
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
            backend=backend,
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
    """Return conductance-constrained pincodes using branch-and-bound.

    Channel permutations are removed by generating nondecreasing transmission
    indices.  A partial pincode is pruned when its smallest possible
    completion exceeds the upper conductance bound or its largest possible
    completion cannot reach the lower bound.
    """
    tau_grid = _parameter_axis(tau, "tau", 4)
    if n_channels < 1:
        raise ValueError("n_channels must be at least one.")
    lower, upper = map(float, tau_sum_bounds)
    if lower > upper:
        raise ValueError("tau_sum_bounds must satisfy lower <= upper.")
    total = comb(tau_grid.size + n_channels - 1, n_channels)
    progress_bar = tqdm(
        total=total,
        desc="Pincodes",
        unit="candidate",
        disable=not progress,
    )
    accepted: list[tuple[int, ...]] = []
    prefix: list[int] = []
    pending_progress = 0

    def completion_count(start: int, remaining: int) -> int:
        if remaining == 0:
            return 1
        return comb(tau_grid.size - start + remaining - 1, remaining)

    def advance(count: int) -> None:
        nonlocal pending_progress
        pending_progress += count
        if pending_progress >= 10_000:
            progress_bar.update(pending_progress)
            pending_progress = 0

    def search(start: int, remaining: int, transmission: float) -> None:
        possibilities = completion_count(start, remaining)
        tolerance = 1.0e-12

        if transmission > upper + tolerance:
            advance(possibilities)
            return
        if remaining == 0:
            if lower - tolerance <= transmission <= upper + tolerance:
                accepted.append(tuple(prefix))
            advance(1)
            return

        minimum = transmission + remaining * tau_grid[start]
        maximum = transmission + remaining * tau_grid[-1]
        if minimum > upper + tolerance or maximum < lower - tolerance:
            advance(possibilities)
            return

        for index in range(start, tau_grid.size):
            prefix.append(index)
            search(
                index,
                remaining - 1,
                transmission + float(tau_grid[index]),
            )
            prefix.pop()

    search(start=0, remaining=n_channels, transmission=0.0)
    if pending_progress:
        progress_bar.update(pending_progress)
    progress_bar.close()
    if not accepted:
        return np.empty((0, n_channels), dtype=np.int32)
    return np.asarray(accepted, dtype=np.int32)


def _extend_current_bank(
    V_mV: FloatArray,
    currents_nA: FloatArray,
    V_support_mV: FloatArray,
    extrapolation_points: int,
) -> FloatArray:
    """Interpolate internally and linearly extrapolate both IV edges."""
    count = min(int(extrapolation_points), V_mV.size)
    if count < 2:
        raise ValueError("at least two voltage points are required.")
    extended = np.stack(
        [np.interp(V_support_mV, V_mV, current) for current in currents_nA]
    )

    left_x = V_mV[:count]
    left_centered = left_x - np.mean(left_x)
    left_slopes = (
        currents_nA[:, :count] @ left_centered / np.sum(left_centered**2)
    )
    left = V_support_mV < V_mV[0]
    extended[:, left] = currents_nA[:, [0]] + left_slopes[:, None] * (
        V_support_mV[left] - V_mV[0]
    )

    right_x = V_mV[-count:]
    right_centered = right_x - np.mean(right_x)
    right_slopes = (
        currents_nA[:, -count:] @ right_centered / np.sum(right_centered**2)
    )
    right = V_support_mV > V_mV[-1]
    extended[:, right] = currents_nA[:, [-1]] + right_slopes[:, None] * (
        V_support_mV[right] - V_mV[-1]
    )
    return extended


def _evaluate_mar_with_voltage_noise(
    V_mV: FloatArray,
    *,
    tau: FloatArray,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    sigmaV_mV: float,
    extrapolation_points: int = 25,
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

    I_nA = get_Imar_nA(
        V_mV=V_mV,
        tau=tau,
        T_K=T_K,
        Delta_meV=Delta_meV,
        gamma_meV=gamma_meV,
        caching=True,
    )
    V_support_mV = make_bias_support_grid(V_mV, sigmaV_mV)
    I_support_nA = _extend_current_bank(
        V_mV,
        np.asarray(I_nA, dtype=np.float64)[None, :],
        V_support_mV,
        extrapolation_points,
    )[0]
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
        [
            apply_voltage_noise(V, current, sigma, order=32)
            for current in currents
        ]
    )


def _best_pincode(
    bank: FloatArray,
    pincodes: IntArray,
    target: FloatArray,
    sigma: FloatArray,
    batch_size: int,
    *,
    backend: FitBackend = "auto",
    progress_bar=None,
) -> tuple[float, int, FloatArray]:
    selected_backend = _select_fit_backend(backend)
    if selected_backend == "jax":
        return _best_pincode_jax(
            bank,
            pincodes,
            target,
            sigma,
            batch_size,
            progress_bar,
        )
    return _best_pincode_numpy(
        bank,
        pincodes,
        target,
        sigma,
        batch_size,
        progress_bar,
    )


def _select_fit_backend(backend: FitBackend) -> Literal["jax", "numpy"]:
    """Resolve the requested scoring backend without requiring JAX."""
    if backend == "numpy":
        return "numpy"
    try:
        import jax  # noqa: F401
    except ImportError:
        if backend == "jax":
            raise ImportError(
                "backend='jax' requires JAX; install it or use backend='numpy'."
            ) from None
        return "numpy"
    return "jax"


def _best_pincode_numpy(
    bank: FloatArray,
    pincodes: IntArray,
    target: FloatArray,
    sigma: FloatArray,
    batch_size: int,
    progress_bar=None,
) -> tuple[float, int, FloatArray]:
    """Find the best pincode without a ``batch x channel x voltage`` array."""
    best_chi2 = np.inf
    best_index = -1
    best_current: FloatArray | None = None
    for start in range(0, pincodes.shape[0], batch_size):
        stop = min(start + batch_size, pincodes.shape[0])
        batch = pincodes[start:stop]
        models = np.zeros((batch.shape[0], bank.shape[1]), dtype=np.float64)
        for channel in range(batch.shape[1]):
            models += bank[batch[:, channel]]
        models -= target[None, :]
        models /= sigma[None, :]
        chi2 = np.einsum("ij,ij->i", models, models)
        local = int(np.argmin(chi2))
        if chi2[local] < best_chi2:
            best_chi2 = float(chi2[local])
            best_index = start + local
            best_current = np.sum(bank[pincodes[best_index]], axis=0)
        if progress_bar is not None:
            progress_bar.update(stop - start)
    if best_current is None:
        raise RuntimeError("pincode search received no candidates.")
    return best_chi2, best_index, best_current


def _best_pincode_jax(
    bank: FloatArray,
    pincodes: IntArray,
    target: FloatArray,
    sigma: FloatArray,
    batch_size: int,
    progress_bar=None,
) -> tuple[float, int, FloatArray]:
    """Find the best pincode with a fused, memory-bounded JAX scorer."""
    import jax
    import jax.numpy as jnp

    # Use float64 when enabled, otherwise consistently score in float32.  This
    # avoids changing the process-wide JAX configuration from a library helper.
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    bank_device = jax.device_put(np.asarray(bank, dtype=np.dtype(dtype)))
    target_device = jax.device_put(np.asarray(target, dtype=np.dtype(dtype)))
    sigma_device = jax.device_put(np.asarray(sigma, dtype=np.dtype(dtype)))

    batch_minimum = _jax_batch_minimum()

    best_chi2 = np.inf
    best_index = -1
    for start in range(0, pincodes.shape[0], batch_size):
        stop = min(start + batch_size, pincodes.shape[0])
        batch = np.asarray(pincodes[start:stop], dtype=np.int32)
        chi2_device, local_device = batch_minimum(
            bank_device,
            target_device,
            sigma_device,
            jax.device_put(batch),
        )
        chi2 = float(chi2_device)
        local = int(local_device)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_index = start + local
        if progress_bar is not None:
            progress_bar.update(stop - start)

    if best_index < 0:
        raise RuntimeError("pincode search received no candidates.")
    best_current = np.sum(bank[pincodes[best_index]], axis=0)
    return best_chi2, best_index, best_current


@lru_cache(maxsize=1)
def _jax_batch_minimum():
    """Build one compiled scorer reused across all global grid points."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def score(bank, target, sigma, indices):
        models = jnp.sum(bank[indices], axis=1)
        residuals = (models - target[None, :]) / sigma[None, :]
        chi2 = jnp.sum(residuals * residuals, axis=1)
        index = jnp.argmin(chi2)
        return chi2[index], index

    return score


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
            "sigmaG_G0 must be scalar or have the same shape as "
            "database.grid.V_mV."
        )
    return sigma


def _axis(
    values: ArrayLike, name: str, *, uniform: bool = False
) -> FloatArray:
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
