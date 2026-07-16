"""Local steepest-descent fitting for atomic-contact MAR traces.

Unlike :mod:`mar_fit`, this module never constructs the complete set of
channel pincodes.  It walks on the discrete MAR grid, evaluates the immediate
neighbours of the current solution, and accepts the move with the largest
decrease in chi squared.  Optional independent restarts make the local search
less sensitive to its starting point.

The module is self-contained: its grid, database, preparation, and result
types do not depend on ``mar_fit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from superconductivity.models.basics.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)
from superconductivity.models.mar import get_Imar_nA
from superconductivity.utilities.constants import G0_muS
from tqdm.auto import tqdm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MARGrid:
    """Axes of a single-channel MAR current database."""

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
        """Shape of the current bank."""
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
    """Voltage-broadened single-channel MAR currents and their grid."""

    grid: MARGrid
    I_nA: FloatArray
    curves_requested: int = 0
    loading_time_s: float = 0.0


@dataclass(frozen=True)
class MARGradientFitResult:
    """Result of a local, discrete steepest-descent MAR fit."""

    tau: FloatArray
    T_K: float
    Delta_meV: float
    gamma_meV: float
    sigmaV_mV: float
    parameter_values: dict[str, float]
    chi2: float
    reduced_chi2: float
    V_mV: FloatArray
    I_exp_nA: FloatArray
    Ifit_nA: FloatArray
    residual_nA: FloatArray
    parameter_index: tuple[int, int, int, int]
    candidates_tested: int
    iterations: int
    restarts: int
    converged: bool
    fitting_time_s: float


def prepare_mar_trace(
    V_mV: ArrayLike,
    I_nA: ArrayLike,
    Vnan_mV: float = 0.03,
) -> NDArray[np.bool_]:
    """Return a mask excluding non-finite data and the zero-bias region."""
    voltage = np.asarray(V_mV, dtype=np.float64)
    current = np.asarray(I_nA, dtype=np.float64)
    if voltage.shape != current.shape:
        raise ValueError("V_mV and I_nA must have the same shape.")
    return np.isfinite(voltage) & np.isfinite(current) & (np.abs(voltage) > Vnan_mV)


def estimate_GN_bounds(
    V_mV: ArrayLike,
    I_nA: ArrayLike,
    VGN_mV: float = 0.5,
    n_sections: int = 4,
    confidence: float = 2.0,
) -> tuple[float, tuple[float, float]]:
    """Estimate normal conductance and robust bounds from IV plateaus."""
    voltage = np.asarray(V_mV, dtype=np.float64)
    current = np.asarray(I_nA, dtype=np.float64)
    if voltage.shape != current.shape:
        raise ValueError("V_mV and I_nA must have the same shape.")
    finite = np.isfinite(voltage) & np.isfinite(current)
    slopes = []
    for polarity in (-1, 1):
        indices = np.flatnonzero(finite & (polarity * voltage >= VGN_mV))
        for section in np.array_split(indices, n_sections):
            if section.size >= 2:
                slope_uS, _ = np.polyfit(
                    voltage[section],
                    current[section],
                    deg=1,
                )
                slopes.append(slope_uS / G0_muS)
    if len(slopes) < 2:
        raise ValueError("Not enough plateau sections to estimate bounds.")
    values = np.asarray(slopes)
    conductance = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - conductance)))
    margin = max(confidence * spread, 0.02 * conductance, 0.02)
    return conductance, (max(0.0, conductance - margin), conductance + margin)


def prepare_mar_database(
    grid: MARGrid,
    *,
    extrapolation_points: int = 25,
    progress: bool = True,
) -> MARDatabase:
    """Calculate and broaden the single-channel curves needed by ``grid``.

    Native MAR caching remains enabled, so curves already present in the
    package cache are reused.  This function creates no second disk cache.
    """
    if extrapolation_points < 2:
        raise ValueError("extrapolation_points must be at least two.")
    base_shape = grid.current_shape[:-2] + (grid.V_mV.size,)
    base = np.empty(base_shape, dtype=np.float64)
    parameter_indices = list(np.ndindex(base_shape[:-1]))
    start = perf_counter()
    iterator = tqdm(
        parameter_indices,
        desc="MAR curves",
        unit="curve",
        disable=not progress,
    )
    for i_tau, i_T, i_delta, i_gamma in iterator:
        base[i_tau, i_T, i_delta, i_gamma] = get_Imar_nA(
            V_mV=grid.V_mV,
            tau=float(grid.tau[i_tau]),
            T_K=float(grid.T_K[i_T]),
            Delta_meV=float(grid.Delta_meV[i_delta]),
            gamma_meV=float(grid.gamma_meV[i_gamma]),
            caching=True,
        )

    output = np.empty(grid.current_shape, dtype=np.float64)
    support = make_bias_support_grid(
        grid.V_mV,
        float(np.max(grid.sigmaV_mV)),
    )
    global_indices = list(
        np.ndindex(
            grid.T_K.size,
            grid.Delta_meV.size,
            grid.gamma_meV.size,
        )
    )
    iterator = tqdm(
        global_indices,
        desc="Voltage noise",
        unit="bank",
        disable=not progress,
    )
    for i_T, i_delta, i_gamma in iterator:
        bank = base[:, i_T, i_delta, i_gamma]
        extended = _extend_current_bank(
            grid.V_mV,
            bank,
            support,
            extrapolation_points,
        )
        for i_sigma, sigma in enumerate(grid.sigmaV_mV):
            if sigma == 0.0:
                output[:, i_T, i_delta, i_gamma, i_sigma] = bank
                continue
            for i_tau, current in enumerate(extended):
                broadened = apply_voltage_noise(
                    support,
                    current,
                    float(sigma),
                    order=32,
                )
                output[i_tau, i_T, i_delta, i_gamma, i_sigma] = np.interp(
                    grid.V_mV,
                    support,
                    broadened,
                )
    return MARDatabase(
        grid=grid,
        I_nA=output,
        curves_requested=len(parameter_indices),
        loading_time_s=perf_counter() - start,
    )


def fit_mar_gradient(
    I_nA: ArrayLike,
    database: MARDatabase,
    *,
    settings: Mapping[str, tuple[float, float, float, bool]],
    tau_sum_bounds: tuple[float, float] = (0.0, np.inf),
    weights: ArrayLike | float | None = None,
    voltage_bounds_mV: tuple[float, float] | None = None,
    restarts: int = 4,
    max_iterations: int = 500,
    random_seed: int | None = 0,
    progress: bool = True,
) -> MARGradientFitResult:
    """Fit an MAR trace by discrete steepest descent on the loaded grid.

    ``settings`` uses the same ``(guess, lower, upper, fixed)`` tuples as the
    BCS fit helpers.  It must define ``tau_1`` through ``tau_9``, ``T_K``,
    ``Delta_meV``, ``gamma_meV``, and ``sigmaV_mV``.  Disable an unused channel
    with ``(0.0, 0.0, 0.0, True)``.  Every value and bound is mapped onto the
    corresponding loaded grid axis.

    ``weights`` are arbitrary nonnegative point weights in the current-space
    objective ``sum(weights * (I_model_nA - I_exp_nA)**2)``.  A scalar is
    broadcast, zero-weight samples are excluded, and ``None`` gives equal
    weighting.

    At every iteration, each non-fixed parameter is moved by one allowed grid
    point in either direction.  The valid move giving the largest chi-squared
    reduction is accepted.  The first run uses the supplied guesses;
    subsequent runs use random values inside the supplied bounds.

    Notes
    -----
    This is a local optimizer.  It usually evaluates far fewer candidates than
    an exhaustive pincode search, but it does not guarantee the global minimum.
    Increasing ``restarts`` trades additional work for greater robustness.
    """
    start = perf_counter()
    tau_names = tuple(f"tau_{index}" for index in range(1, 10))
    global_names = ("T_K", "Delta_meV", "gamma_meV", "sigmaV_mV")
    missing = [
        name for name in tau_names + global_names if name not in settings
    ]
    if missing:
        raise KeyError(f"Missing fit settings for {', '.join(missing)}.")
    n_channels = len(tau_names)
    if restarts < 0:
        raise ValueError("restarts must be nonnegative.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one.")
    grid = database.grid
    currents = np.asarray(database.I_nA, dtype=np.float64)
    if currents.shape != grid.current_shape:
        raise ValueError("database current shape does not match its grid.")
    if not np.all(np.isfinite(currents)):
        raise ValueError("database contains missing or non-finite IVs.")

    data = np.asarray(I_nA, dtype=np.float64)
    if data.shape != grid.V_mV.shape:
        raise ValueError("I_nA must have the same shape as grid.V_mV.")
    fit_weights = _fit_weights(weights, data.shape)
    fit_mask = (
        np.isfinite(data) & np.isfinite(fit_weights) & (fit_weights > 0.0)
    )
    if voltage_bounds_mV is not None:
        low, high = map(float, voltage_bounds_mV)
        if low >= high:
            raise ValueError("voltage_bounds_mV must satisfy low < high.")
        fit_mask &= (grid.V_mV >= low) & (grid.V_mV <= high)
    if np.count_nonzero(fit_mask) < 2:
        raise ValueError("fewer than two samples are available for fitting.")

    lower, upper = map(float, tau_sum_bounds)
    if lower > upper:
        raise ValueError("tau_sum_bounds must satisfy lower <= upper.")
    tau_selected_and_start = tuple(
        _settings_indices(grid.tau, settings[name], name) for name in tau_names
    )
    tau_selected = tuple(item[0] for item in tau_selected_and_start)
    tau_start = tuple(item[1] for item in tau_selected_and_start)
    global_selected_and_start = tuple(
        _settings_indices(axis, settings[name], name)
        for axis, name in zip(
            (grid.T_K, grid.Delta_meV, grid.gamma_meV, grid.sigmaV_mV),
            global_names,
            strict=True,
        )
    )
    global_selected = tuple(item[0] for item in global_selected_and_start)
    global_start = tuple(item[1] for item in global_selected_and_start)
    minimum_sum = sum(float(grid.tau[indices[0]]) for indices in tau_selected)
    maximum_sum = sum(float(grid.tau[indices[-1]]) for indices in tau_selected)
    if maximum_sum < lower or minimum_sum > upper:
        raise ValueError("tau_sum_bounds conflict with the channel bounds.")
    target = data[fit_mask]
    weights_fit = fit_weights[fit_mask]
    score_cache: dict[tuple[int, ...], float] = {}

    def score(state: tuple[int, ...]) -> float:
        cached = score_cache.get(state)
        if cached is not None:
            return cached
        tau_indices = np.asarray(state[:n_channels], dtype=np.intp)
        i_T, i_delta, i_gamma, i_sigma = state[n_channels:]
        model_I = np.sum(
            currents[
                tau_indices,
                i_T,
                i_delta,
                i_gamma,
                i_sigma,
            ][:, fit_mask],
            axis=0,
        )
        residual = model_I - target
        value = float(np.sum(weights_fit * residual**2))
        score_cache[state] = value
        return value

    rng = np.random.default_rng(random_seed)
    projected = _project_tau_indices(
        tau_start, grid.tau, tau_selected, lower, upper
    )
    if projected is None:
        raise ValueError(
            "the tau guesses cannot be moved into tau_sum_bounds."
        )
    starts = [(projected, global_start)]
    for _ in range(restarts):
        random_tau = tuple(
            int(rng.choice(indices)) for indices in tau_selected
        )
        random_tau = _project_tau_indices(
            random_tau, grid.tau, tau_selected, lower, upper
        )
        if random_tau is None:
            continue
        random_global = tuple(int(rng.choice(indices)) for indices in global_selected)
        starts.append((random_tau, random_global))

    best_state: tuple[int, ...] | None = None
    best_chi2 = np.inf
    total_iterations = 0
    all_converged = True
    run_progress = tqdm(
        starts,
        desc="MAR gradient fit",
        unit="start",
        disable=not progress,
    )
    for tau_start, global_start in run_progress:
        state = tuple(tau_start) + global_start
        current_chi2 = score(state)
        converged = False
        for _ in range(max_iterations):
            total_iterations += 1
            neighbours = _neighbours(
                state,
                n_channels,
                grid,
                tau_selected,
                global_selected,
                lower,
                upper,
            )
            if not neighbours:
                converged = True
                break
            values = np.asarray([score(candidate) for candidate in neighbours])
            index = int(np.argmin(values))
            if values[index] >= current_chi2:
                converged = True
                break
            state = neighbours[index]
            current_chi2 = float(values[index])
        all_converged &= converged
        if current_chi2 < best_chi2:
            best_chi2 = current_chi2
            best_state = state
            run_progress.set_postfix(best_chi2=f"{best_chi2:.4g}")

    if best_state is None:
        raise RuntimeError("gradient search produced no result.")
    tau_indices = np.asarray(best_state[:n_channels], dtype=np.intp)
    i_T, i_delta, i_gamma, i_sigma = best_state[n_channels:]
    fit_current = np.sum(
        currents[tau_indices, i_T, i_delta, i_gamma, i_sigma],
        axis=0,
    )
    residual = np.full_like(data, np.nan)
    residual[fit_mask] = fit_current[fit_mask] - data[fit_mask]
    parameter_values = {
        name: float(value)
        for name, value in zip(
            tau_names,
            grid.tau[tau_indices],
            strict=True,
        )
    }
    parameter_values.update(
        {
            "T_K": float(grid.T_K[i_T]),
            "Delta_meV": float(grid.Delta_meV[i_delta]),
            "gamma_meV": float(grid.gamma_meV[i_gamma]),
            "sigmaV_mV": float(grid.sigmaV_mV[i_sigma]),
        }
    )
    return MARGradientFitResult(
        tau=np.sort(grid.tau[tau_indices])[::-1],
        T_K=float(grid.T_K[i_T]),
        Delta_meV=float(grid.Delta_meV[i_delta]),
        gamma_meV=float(grid.gamma_meV[i_gamma]),
        sigmaV_mV=float(grid.sigmaV_mV[i_sigma]),
        parameter_values=parameter_values,
        chi2=best_chi2,
        reduced_chi2=best_chi2 / np.count_nonzero(fit_mask),
        V_mV=grid.V_mV.copy(),
        I_exp_nA=data.copy(),
        Ifit_nA=fit_current,
        residual_nA=residual,
        parameter_index=(i_T, i_delta, i_gamma, i_sigma),
        candidates_tested=len(score_cache),
        iterations=total_iterations,
        restarts=len(starts) - 1,
        converged=all_converged,
        fitting_time_s=perf_counter() - start,
    )


def _neighbours(
    state: tuple[int, ...],
    n_channels: int,
    grid: MARGrid,
    tau_selected: tuple[NDArray[np.int64], ...],
    global_selected: tuple[NDArray[np.int64], ...],
    lower: float,
    upper: float,
) -> list[tuple[int, ...]]:
    """Return unique, valid, one-grid-step neighbours of ``state``."""
    result: set[tuple[int, ...]] = set()
    tau_state = state[:n_channels]
    globals_state = state[n_channels:]
    for channel in range(n_channels):
        indices = tau_selected[channel]
        if indices.size == 1:
            continue
        position = int(np.flatnonzero(indices == tau_state[channel])[0])
        for direction in (-1, 1):
            new_position = position + direction
            if not 0 <= new_position < indices.size:
                continue
            changed = list(tau_state)
            changed[channel] = int(indices[new_position])
            tau_sum = float(np.sum(grid.tau[changed]))
            if lower <= tau_sum <= upper:
                result.add(tuple(changed) + globals_state)
    # A narrow conductance interval can forbid every single-channel step.
    # Exchange moves preserve the total approximately while redistributing
    # transmission between two channels.
    for first in range(n_channels):
        for second in range(first + 1, n_channels):
            first_indices = tau_selected[first]
            second_indices = tau_selected[second]
            if first_indices.size == 1 or second_indices.size == 1:
                continue
            first_position = int(np.flatnonzero(first_indices == tau_state[first])[0])
            second_position = int(
                np.flatnonzero(second_indices == tau_state[second])[0]
            )
            for direction in (-1, 1):
                new_first = first_position + direction
                new_second = second_position - direction
                if not (
                    0 <= new_first < first_indices.size
                    and 0 <= new_second < second_indices.size
                ):
                    continue
                changed = list(tau_state)
                changed[first] = int(first_indices[new_first])
                changed[second] = int(second_indices[new_second])
                tau_sum = float(np.sum(grid.tau[changed]))
                if lower <= tau_sum <= upper:
                    result.add(tuple(changed) + globals_state)
    for dimension, indices in enumerate(global_selected):
        if indices.size == 1:
            continue
        current = globals_state[dimension]
        position = int(np.flatnonzero(indices == current)[0])
        for direction in (-1, 1):
            new_position = position + direction
            if 0 <= new_position < indices.size:
                changed = list(globals_state)
                changed[dimension] = int(indices[new_position])
                result.add(tau_state + tuple(changed))
    result.discard(state)
    return list(result)


def _project_tau_indices(
    indices: tuple[int, ...],
    tau: FloatArray,
    selected: tuple[NDArray[np.int64], ...],
    lower: float,
    upper: float,
) -> tuple[int, ...] | None:
    """Greedily move guesses into the total-transmission interval."""
    state = tuple(indices)
    visited = {state}
    for _ in range(sum(axis.size for axis in selected) + 1):
        total = float(np.sum(tau[list(state)]))
        if lower <= total <= upper:
            return state
        direction = 1 if total < lower else -1
        candidates = []
        for channel, axis in enumerate(selected):
            position = int(np.flatnonzero(axis == state[channel])[0])
            new_position = position + direction
            if not 0 <= new_position < axis.size:
                continue
            changed = list(state)
            changed[channel] = int(axis[new_position])
            candidate = tuple(changed)
            if candidate not in visited:
                candidates.append(candidate)
        if not candidates:
            return None
        target = lower if direction > 0 else upper
        state = min(
            candidates,
            key=lambda candidate: abs(float(np.sum(tau[list(candidate)])) - target),
        )
        visited.add(state)
    return None


def _settings_indices(
    axis: FloatArray,
    setting: tuple[float, float, float, bool],
    name: str,
) -> tuple[NDArray[np.int64], int]:
    """Return allowed grid indices and the guess index for one parameter."""
    if len(setting) != 4:
        raise ValueError(f"settings[{name!r}] must be (guess, lower, upper, fixed).")
    guess, lower, upper = map(float, setting[:3])
    fixed = bool(setting[3])
    if not np.all(np.isfinite([guess, lower, upper])):
        raise ValueError(f"settings[{name!r}] must contain finite values.")
    if lower > upper:
        raise ValueError(f"settings[{name!r}] must satisfy lower <= upper.")
    if not lower <= guess <= upper:
        raise ValueError(f"the guess for {name} must lie inside its bounds.")
    allowed = np.flatnonzero((axis >= lower) & (axis <= upper))
    if allowed.size == 0:
        raise ValueError(f"the bounds for {name} contain no value on the loaded grid.")
    nearest = int(allowed[np.argmin(np.abs(axis[allowed] - guess))])
    if fixed:
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(guess)) * 8.0
        if not np.isclose(axis[nearest], guess, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"fixed {name}={guess!r} is not on the loaded grid; nearest "
                f"allowed value is {float(axis[nearest])!r}."
            )
        allowed = np.asarray([nearest], dtype=np.int64)
    return np.asarray(allowed, dtype=np.int64), nearest


def _extend_current_bank(
    V_mV: FloatArray,
    currents: FloatArray,
    support: FloatArray,
    extrapolation_points: int,
) -> FloatArray:
    count = min(extrapolation_points, V_mV.size)
    if count < 2:
        raise ValueError("at least two voltage points are required.")
    extended = np.stack([np.interp(support, V_mV, row) for row in currents])
    for edge, voltage, outside in (
        (slice(0, count), V_mV[0], support < V_mV[0]),
        (slice(-count, None), V_mV[-1], support > V_mV[-1]),
    ):
        x = V_mV[edge]
        centered = x - np.mean(x)
        slopes = currents[:, edge] @ centered / np.sum(centered**2)
        boundary = currents[:, 0] if voltage == V_mV[0] else currents[:, -1]
        extended[:, outside] = boundary[:, None] + slopes[:, None] * (
            support[outside] - voltage
        )
    return extended


def _fit_weights(
    weights: ArrayLike | float | None,
    shape: tuple[int, ...],
) -> FloatArray:
    if weights is None:
        return np.ones(shape, dtype=np.float64)
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=np.float64)
    if array.shape != shape:
        raise ValueError("weights must be scalar or match grid.V_mV.")
    if np.any(~np.isfinite(array)):
        raise ValueError("weights must be finite.")
    if np.any(array < 0.0):
        raise ValueError("weights must be nonnegative.")
    if not np.any(array > 0.0):
        raise ValueError("at least one weight must be positive.")
    return array


def _axis(values: ArrayLike, name: str) -> FloatArray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values.")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return axis


def _parameter_axis(values: ArrayLike, name: str, decimals: int) -> FloatArray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values.")
    return np.unique(np.round(axis, decimals=decimals))


__all__ = [
    "MARDatabase",
    "MARGradientFitResult",
    "MARGrid",
    "estimate_GN_bounds",
    "fit_mar_gradient",
    "prepare_mar_database",
    "prepare_mar_trace",
]
