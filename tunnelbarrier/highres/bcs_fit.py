from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jv

jax.config.update("jax_enable_x64", True)

G0_MUS = 77.48091729863648
KB_MEV_K = 0.08617333262145178
H_PVS = 0.004135667696923859
ENERGY_MEV = np.linspace(-4.0, 4.0, 4001, dtype=np.float64)
NOISE_MODEL = "bcs_conv_noise"
N_MAX = 100
NOISE_OVERSAMPLE = 64
NOISE_PADDING_SIGMA = 6.0


@dataclass
class Parameter:
    name: str
    guess: float
    lower: float
    upper: float
    fixed: bool
    value: float | None = None
    error: float | None = None


def get_delta_t_meV(Delta_meV: float, T_K: float) -> float:
    """Return the weak-coupling BCS gap at temperature ``T_K``.

    Parameters
    ----------
    Delta_meV:
        Zero-temperature gap in meV.
    T_K:
        Temperature in kelvin.

    Returns
    -------
    float
        Thermal gap in meV.
    """
    delta = float(Delta_meV)
    temperature = float(T_K)
    if delta < 0.0:
        raise ValueError("Delta_meV must be non-negative.")
    if temperature < 0.0:
        raise ValueError("T_K must be non-negative.")
    if temperature == 0.0:
        return delta

    Tc_K = delta / (1.764 * KB_MEV_K)
    if temperature >= Tc_K:
        return 0.0
    return float(delta * np.tanh(1.74 * np.sqrt(Tc_K / temperature - 1.0)))


def get_fermi(E_meV: np.ndarray, T_K: float) -> np.ndarray:
    """Evaluate the Fermi occupation on an energy grid."""
    temperature = float(T_K)
    if temperature < 0.0:
        raise ValueError("T_K must be non-negative.")

    energy = np.asarray(E_meV, dtype=np.float64)
    if temperature == 0.0:
        return np.where(energy < 0.0, 1.0, 0.0)

    exponent = np.clip(energy / (KB_MEV_K * temperature), -100.0, 100.0)
    return 1.0 / (np.exp(exponent) + 1.0)


def get_dos(E_meV: np.ndarray, Delta_meV: float, gamma_meV: float) -> np.ndarray:
    """Return the Dynes-broadened BCS density of states."""
    delta = float(Delta_meV)
    gamma = float(gamma_meV)
    if delta < 0.0:
        raise ValueError("Delta_meV must be non-negative.")
    if gamma < 0.0:
        raise ValueError("gamma_meV must be non-negative.")
    if delta == 0.0:
        return np.ones_like(E_meV, dtype=np.float64)

    energy = np.asarray(E_meV, dtype=np.complex128) + 1j * gamma
    denominator = np.sqrt(energy * energy - delta * delta)
    dos = np.abs(np.real(energy / denominator), dtype=np.float64)
    dos[np.isnan(dos)] = 0.0
    return np.clip(dos, 0.0, 1e8)


@jax.jit
def _get_delta_t_meV_jax(Delta_meV: jnp.ndarray, T_K: jnp.ndarray) -> jnp.ndarray:
    Tc_K = Delta_meV / (1.764 * KB_MEV_K)
    safe_T_K = jnp.where(T_K == 0.0, 1.0, T_K)
    thermal_delta = Delta_meV * jnp.tanh(
        1.74 * jnp.sqrt(jnp.maximum(Tc_K / safe_T_K - 1.0, 0.0))
    )
    return jnp.where(
        T_K < 0.0,
        jnp.full_like(Delta_meV, jnp.nan),
        jnp.where(
            T_K == 0.0,
            Delta_meV,
            jnp.where(T_K < Tc_K, thermal_delta, jnp.zeros_like(Delta_meV)),
        ),
    )


@jax.jit
def _get_fermi_jax(E_meV: jnp.ndarray, T_K: jnp.ndarray) -> jnp.ndarray:
    safe_T_K = jnp.where(T_K == 0.0, 1.0, T_K)
    exponent = jnp.clip(E_meV / (KB_MEV_K * safe_T_K), -100.0, 100.0)
    thermal_f = 1.0 / (jnp.exp(exponent) + 1.0)
    return jnp.where(
        T_K < 0.0,
        jnp.full_like(E_meV, jnp.nan),
        jnp.where(T_K == 0.0, jnp.where(E_meV < 0.0, 1.0, 0.0), thermal_f),
    )


@jax.jit
def _get_dos_jax(
    E_meV: jnp.ndarray,
    Delta_meV: jnp.ndarray,
    gamma_meV: jnp.ndarray,
) -> jnp.ndarray:
    E_complex = E_meV + 1j * gamma_meV
    dos = E_complex / jnp.sqrt(E_complex**2 - Delta_meV**2)
    dos = jnp.abs(jnp.real(dos))
    dos = jnp.nan_to_num(dos, nan=0.0, posinf=100.0, neginf=0.0)
    clipped = jnp.clip(dos, 0.0, 100.0)
    return jnp.where(Delta_meV == 0.0, jnp.ones_like(E_meV), clipped)


@jax.jit
def _convolution_spectrum(
    E_meV: jnp.ndarray,
    T_K: jnp.ndarray,
    Delta_meV: jnp.ndarray,
    gamma_meV: jnp.ndarray,
) -> jnp.ndarray:
    DeltaT_meV = _get_delta_t_meV_jax(Delta_meV, T_K)
    dos = _get_dos_jax(E_meV, DeltaT_meV, gamma_meV)
    fermi = _get_fermi_jax(E_meV, T_K)
    occupied = dos * fermi
    empty = dos * (1.0 - fermi)
    dE_meV = E_meV[1] - E_meV[0]
    forward = jnp.correlate(empty, occupied, mode="full") * dE_meV
    backward = jnp.correlate(occupied, empty, mode="full") * dE_meV
    return forward - backward


@jax.jit
def _interpolate_convolution(
    V_mV: jnp.ndarray,
    E_meV: jnp.ndarray,
    I_mV: jnp.ndarray,
) -> jnp.ndarray:
    dE_meV = E_meV[1] - E_meV[0]
    Egrid_meV = (
        jnp.arange(
            -(E_meV.size - 1),
            E_meV.size,
            dtype=jnp.float64,
        )
        * dE_meV
    )
    result = jnp.interp(V_mV, Egrid_meV, I_mV)
    return jnp.where(jnp.isfinite(result), result, V_mV)


def kernel(
    V_mV: np.ndarray,
    GN_G0: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    A_mV: float,
    nu_GHz: float,
    sigmaV_mV: float,
) -> np.ndarray:
    """Evaluate the local BCS convolution + PAT + voltage-noise model in nA."""
    V_requested = _validate_voltage_axis(V_mV)
    sigma = _validate_nonnegative_scalar(sigmaV_mV, "sigmaV_mV")
    amplitude = _validate_nonnegative_scalar(A_mV, "A_mV")
    V_evaluate = (
        _make_bias_support_grid(V_requested, sigma) if sigma > 0.0 else V_requested
    )
    current = _base_kernel(
        V_evaluate,
        T_K=float(T_K),
        Delta_meV=float(Delta_meV),
        gamma_meV=float(gamma_meV),
    )
    current = np.asarray(current * (float(GN_G0) * G0_MUS), dtype=np.float64)

    if amplitude != 0.0:
        current = _pat_kernel(
            V_evaluate,
            current,
            amplitude,
            nu_GHz=_validate_positive_scalar(nu_GHz, "nu_GHz"),
            n_max=N_MAX,
        )

    if sigma > 0.0:
        current = np.interp(
            V_requested,
            V_evaluate,
            _apply_voltage_noise(
                V_evaluate,
                current,
                sigma,
                NOISE_OVERSAMPLE,
            ),
        )
    else:
        current = _apply_voltage_noise(
            V_evaluate,
            current,
            sigma,
            NOISE_OVERSAMPLE,
        )
    return np.asarray(current, dtype=np.float64)


def fit_bcs_conv_noise(
    Vbias_mV: np.ndarray,
    Iexp_nA: np.ndarray,
    settings: Mapping[str, tuple[float, float, float, bool]],
    *,
    weights: np.ndarray | None = None,
    maxfev: int = 2_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one IV trace with the local BCS convolution + noise model."""
    from scipy.optimize import curve_fit

    V_mV = _validate_voltage_axis(Vbias_mV)
    I_nA = _validate_current_axis(Iexp_nA, V_mV)
    parameter_list = _make_parameters(settings)
    guess, lower, upper, fixed = _parameters_to_arrays(parameter_list)
    free = ~fixed
    sigma, mask = _weights_to_sigma(weights, length=I_nA.size)

    def fixed_function(V_axis: np.ndarray, *free_values: float) -> np.ndarray:
        full = guess.copy()
        full[free] = np.asarray(free_values, dtype=np.float64)
        return kernel(V_axis, *full)

    if np.any(free):
        popt, pcov = curve_fit(
            f=fixed_function,
            xdata=V_mV[mask],
            ydata=I_nA[mask],
            p0=guess[free],
            sigma=None if sigma is None else sigma[mask],
            absolute_sigma=False,
            bounds=(lower[free], upper[free]),
            maxfev=maxfev,
        )
        popt_free = np.asarray(popt, dtype=np.float64)
        cov_free = np.asarray(pcov, dtype=np.float64)
        perr_free = np.sqrt(np.diag(cov_free))
    else:
        popt_free = np.empty((0,), dtype=np.float64)
        perr_free = np.empty((0,), dtype=np.float64)

    popt_full = guess.copy()
    popt_full[free] = popt_free
    perr_full = np.zeros_like(guess)
    perr_full[free] = perr_free

    I_ini_nA = kernel(V_mV, *guess)
    I_fit_nA = kernel(V_mV, *popt_full)

    for index, parameter in enumerate(parameter_list):
        parameter.value = float(popt_full[index])
        parameter.error = float(perr_full[index])

    n_free = int(np.count_nonzero(free))
    row = {
        "model": NOISE_MODEL,
        "ok": True,
        "error": "",
        **_fit_quality(I_nA, I_fit_nA, n_free=n_free),
    }
    for parameter in parameter_list:
        row[parameter.name] = parameter.value
        row[f"u_{parameter.name}"] = parameter.error

    solution = {
        "V_mV": V_mV,
        "I_exp_nA": I_nA,
        "I_ini_nA": I_ini_nA,
        "I_fit_nA": I_fit_nA,
        "params": tuple(parameter_list),
        "weights": None if weights is None else np.asarray(weights, dtype=np.float64),
        "maxfev": maxfev,
    }
    return row, solution


def plot_bcs_conv_noise_fit(row: Mapping[str, Any], solution: Mapping[str, Any]):
    """Plot a ``fit_bcs_conv_noise`` result, residual, and conductance."""
    import matplotlib.pyplot as plt

    V_mV = np.asarray(solution["V_mV"], dtype=float)
    I_exp_nA = np.asarray(solution["I_exp_nA"], dtype=float)
    I_fit_nA = np.asarray(solution["I_fit_nA"], dtype=float)
    dG_exp_G0 = np.gradient(I_exp_nA, V_mV) / G0_MUS
    dG_fit_G0 = np.gradient(I_fit_nA, V_mV) / G0_MUS
    residual_nA = I_exp_nA - I_fit_nA
    fig, (ax_fit, ax_res, ax_cond) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(5.5, 5.8),
        gridspec_kw={"height_ratios": [3, 1, 2]},
    )
    ax_fit.plot(V_mV, I_exp_nA, "k.", ms=2, label="exp")
    ax_fit.plot(
        V_mV,
        I_fit_nA,
        lw=1.4,
        label=f"{NOISE_MODEL}: RMSE={row['rmse_nA']:.3g} nA",
    )
    ax_res.plot(V_mV, residual_nA, lw=1.0)
    ax_res.axhline(0.0, color="0.4", lw=0.8)
    ax_cond.plot(V_mV, dG_exp_G0, "k.", ms=2, label="exp")
    ax_cond.plot(V_mV, dG_fit_G0, lw=1.4, label="fit")

    ax_fit.set_ylabel(r"$I$ (nA)")
    ax_res.set_ylabel(r"$\Delta I$ (nA)")
    ax_cond.set_xlabel(r"$V_\mathrm{bias}$ (mV)")
    ax_cond.set_ylabel(r"$dI/dV$ ($G_0$)")
    ax_fit.legend(fontsize=8)
    fig.tight_layout()
    return fig


def print_bcs_comparison_table(
    table: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] = ("model", "ok", "rmse_nA", "rss", "error"),
) -> None:
    """Print a compact fixed-width comparison table without pandas."""
    rows = [{column: row.get(column, "") for column in columns} for row in table]
    widths = {
        column: max(
            len(column),
            *(len(_format_table_value(row[column])) for row in rows),
        )
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            "  ".join(
                _format_table_value(row[column]).ljust(widths[column])
                for column in columns
            )
        )


def _base_kernel(
    V_mV: np.ndarray,
    *,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
) -> np.ndarray:
    DeltaT_meV = get_delta_t_meV(Delta_meV, T_K)
    if DeltaT_meV == 0.0:
        return np.asarray(V_mV, dtype=np.float64)

    spectrum = np.asarray(
        _convolution_spectrum(
            jnp.asarray(ENERGY_MEV, dtype=jnp.float64),
            jnp.asarray(T_K, dtype=jnp.float64),
            jnp.asarray(Delta_meV, dtype=jnp.float64),
            jnp.asarray(gamma_meV, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    return np.asarray(
        _interpolate_convolution(
            jnp.asarray(V_mV, dtype=jnp.float64),
            jnp.asarray(ENERGY_MEV, dtype=jnp.float64),
            jnp.asarray(spectrum, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )


def _pat_kernel(
    V_mV: np.ndarray,
    I_nA: np.ndarray,
    A_mV: float,
    *,
    nu_GHz: float,
    n_max: int = N_MAX,
) -> np.ndarray:
    V = _validate_voltage_axis(V_mV)
    I = _validate_current_axis(I_nA, V)
    amplitude = _validate_nonnegative_scalar(A_mV, "A_mV")
    frequency = _validate_positive_scalar(nu_GHz, "nu_GHz")
    n_max_int = int(n_max)
    if n_max_int < 0:
        raise ValueError("n_max must be >= 0.")
    if amplitude == 0.0:
        return I.copy()

    photon_mV = frequency * H_PVS
    sidebands = np.arange(-n_max_int, n_max_int + 1, dtype=np.int32)
    shifts_mV = sidebands.astype(np.float64) * photon_mV
    weights = jv(sidebands, amplitude / photon_mV)
    weights = weights * weights
    slope = float(np.mean(_endpoint_slopes(V, I)))

    output = np.zeros_like(I, dtype=np.float64)
    for shift_mV, weight in zip(shifts_mV, weights):
        output += float(weight) * _interp_with_linear_extrapolation(
            V - shift_mV,
            V,
            I,
            slope,
        )
    return np.asarray(output, dtype=np.float64)


def _interp_with_linear_extrapolation(
    x_eval: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
) -> np.ndarray:
    out = np.interp(x_eval, x, y)
    out = np.where(x_eval < x[0], y[0] + slope * (x_eval - x[0]), out)
    out = np.where(x_eval > x[-1], y[-1] + slope * (x_eval - x[-1]), out)
    return np.asarray(out, dtype=np.float64)


def _endpoint_slopes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = 0.1,
) -> tuple[float, float]:
    count = int(np.ceil(x.shape[0] * frac))
    return (
        float(np.polyfit(x[:count], y[:count], deg=1)[0]),
        float(np.polyfit(x[-count:], y[-count:], deg=1)[0]),
    )


def _make_parameters(
    settings: Mapping[str, tuple[float, float, float, bool]],
) -> list[Parameter]:
    names = (
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "A_mV",
        "nu_GHz",
        "sigmaV_mV",
    )
    missing = [name for name in names if name not in settings]
    if missing:
        raise KeyError(f"Missing fit settings for {', '.join(missing)}.")
    return [
        Parameter(
            name=name,
            guess=float(settings[name][0]),
            lower=float(settings[name][1]),
            upper=float(settings[name][2]),
            fixed=bool(settings[name][3]),
        )
        for name in names
    ]


def _parameters_to_arrays(
    parameters: Sequence[Parameter],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    guess = np.array([parameter.guess for parameter in parameters], dtype=np.float64)
    lower = np.array([parameter.lower for parameter in parameters], dtype=np.float64)
    upper = np.array([parameter.upper for parameter in parameters], dtype=np.float64)
    fixed = np.array([parameter.fixed for parameter in parameters], dtype=bool)
    return guess, lower, upper, fixed


def _weights_to_sigma(
    weights: np.ndarray | None,
    *,
    length: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    if weights is None:
        return None, np.ones(length, dtype=bool)

    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError("weights must have the same shape as Iexp_nA.")
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must be finite.")
    if np.any(array < 0.0):
        raise ValueError("weights must be non-negative.")

    mask = array > 0.0
    if not np.any(mask):
        raise ValueError("At least one weight must be positive.")

    sigma = np.full(length, np.nan, dtype=np.float64)
    sigma[mask] = 1.0 / np.sqrt(array[mask])
    return sigma, mask


def _fit_quality(
    I_exp_nA: np.ndarray,
    I_fit_nA: np.ndarray,
    *,
    n_free: int,
) -> dict[str, float]:
    residual_nA = np.asarray(I_exp_nA - I_fit_nA, dtype=float)
    finite = np.isfinite(residual_nA)
    n_points = int(np.count_nonzero(finite))
    rss = float(np.sum(residual_nA[finite] ** 2))
    rmse = float(np.sqrt(rss / n_points))
    denom = max(n_points - n_free, 1)
    reduced_chi2 = float(rss / denom)

    if rss <= 0.0:
        aic = float("-inf")
        bic = float("-inf")
    else:
        aic = float(n_points * np.log(rss / n_points) + 2 * n_free)
        bic = float(n_points * np.log(rss / n_points) + n_free * np.log(n_points))

    return {
        "n_points": float(n_points),
        "n_free": float(n_free),
        "rss": rss,
        "rmse_nA": rmse,
        "reduced_chi2": reduced_chi2,
        "aic": aic,
        "bic": bic,
    }


def _make_bias_support_grid(
    V_mV: np.ndarray,
    sigmaV_mV: float,
    *,
    padding_sigma: float = NOISE_PADDING_SIGMA,
) -> np.ndarray:
    V = _validate_voltage_axis(V_mV)
    sigma = _validate_nonnegative_scalar(sigmaV_mV, "sigmaV_mV")
    if sigma == 0.0:
        return V.copy()

    step = float(np.median(np.diff(V)))
    pad = max(float(padding_sigma) * sigma, 2.0 * step)
    count = int(np.ceil(pad / step))
    start = float(V[0]) - count * step
    stop = float(V[-1]) + count * step
    size = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, size, dtype=np.float64)


def _apply_voltage_noise(
    V_mV: np.ndarray,
    I_nA: np.ndarray,
    sigmaV_mV: float,
    order: int,
) -> np.ndarray:
    V_support = _validate_voltage_axis(V_mV)
    I_support = _validate_current_axis(I_nA, V_support)
    sigma = _validate_nonnegative_scalar(sigmaV_mV, "sigmaV_mV")
    if sigma == 0.0:
        return I_support.copy()

    order_int = int(order)
    if order_int < 2:
        raise ValueError("order must be >= 2.")

    step = float(np.median(np.diff(V_support)))
    if not np.allclose(np.diff(V_support), step, rtol=1e-8, atol=1e-12):
        return _apply_voltage_noise_general(V_support, I_support, sigma)

    sigma_bins = sigma / step
    radius = max(int(np.ceil(NOISE_PADDING_SIGMA * sigma_bins)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    noise_kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    noise_kernel /= np.sum(noise_kernel)
    I_padded = np.pad(I_support, radius, mode="edge")
    return np.convolve(I_padded, noise_kernel, mode="valid")


def _apply_voltage_noise_general(
    V_mV: np.ndarray,
    I_nA: np.ndarray,
    sigmaV_mV: float,
) -> np.ndarray:
    edges = np.empty(V_mV.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (V_mV[:-1] + V_mV[1:])
    edges[0] = V_mV[0] - 0.5 * (V_mV[1] - V_mV[0])
    edges[-1] = V_mV[-1] + 0.5 * (V_mV[-1] - V_mV[-2])
    widths = np.diff(edges)
    delta = (V_mV[:, None] - V_mV[None, :]) / sigmaV_mV
    weights = np.exp(-0.5 * delta**2) * widths[None, :]
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.asarray(weights @ I_nA, dtype=np.float64)


def _validate_voltage_axis(V_mV: np.ndarray) -> np.ndarray:
    V = np.asarray(V_mV, dtype=np.float64).reshape(-1)
    if V.size < 2:
        raise ValueError("V_mV must contain at least two points.")
    if not np.all(np.isfinite(V)):
        raise ValueError("V_mV must be finite.")
    if np.any(np.diff(V) <= 0.0):
        raise ValueError("V_mV must be strictly increasing.")
    return V


def _validate_current_axis(I_nA: np.ndarray, V_mV: np.ndarray) -> np.ndarray:
    I = np.asarray(I_nA, dtype=np.float64).reshape(-1)
    if I.shape != V_mV.shape:
        raise ValueError("I_nA must have the same shape as V_mV.")
    if not np.all(np.isfinite(I)):
        raise ValueError("I_nA must be finite.")
    return I


def _validate_nonnegative_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return scalar


def _validate_positive_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return scalar


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


__all__ = [
    "ENERGY_MEV",
    "G0_MUS",
    "H_PVS",
    "KB_MEV_K",
    "N_MAX",
    "Parameter",
    "fit_bcs_conv_noise",
    "get_delta_t_meV",
    "get_dos",
    "get_fermi",
    "kernel",
    "plot_bcs_conv_noise_fit",
    "print_bcs_comparison_table",
]
