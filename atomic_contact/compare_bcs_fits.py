from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from superconductivity.api import G0_muS
from superconductivity.optimizers import fit_model
from superconductivity.optimizers.bcs import get_model_spec

DEFAULT_MODELS = (
    "bcs_conv_jax",
    "bcs_conv_noise",
    "bcs_int_jax",
    "bcs_int",
    "bcs_adaptive",
)

NOISE_MODEL = "bcs_conv_noise"


def _parameter_values(parameters) -> list[float]:
    return [parameter.guess for parameter in parameters]


def _make_parameters(
    model: str,
    settings: Mapping[str, tuple[float, float, float, bool]],
):
    spec = get_model_spec(model)
    return [
        replace(
            parameter,
            guess=settings.get(parameter.name, (parameter.guess,))[0],
            lower=settings.get(parameter.name, (None, parameter.lower))[1],
            upper=settings.get(parameter.name, (None, None, parameter.upper))[2],
            fixed=settings.get(parameter.name, (None, None, None, parameter.fixed))[3],
        )
        for parameter in spec.parameters
    ]


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


def compare_bcs_start_values(
    Vbias_mV: np.ndarray,
    Iexp_nA: np.ndarray,
    settings: Mapping[str, tuple[float, float, float, bool]],
    *,
    models: Sequence[str] = DEFAULT_MODELS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate several BCS models with the guesses from ``settings``.

    This does not fit. It only plugs the starting values into each model, so it
    is useful for checking whether the guesses are sane before optimization.
    """
    Vbias_mV = np.asarray(Vbias_mV, dtype=float)
    Iexp_nA = np.asarray(Iexp_nA, dtype=float)
    table: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}

    for model in models:
        try:
            spec = get_model_spec(model)
            parameters = _make_parameters(model, settings)
            I_start_nA = np.asarray(
                spec.function(Vbias_mV, *_parameter_values(parameters)),
                dtype=float,
            )
        except Exception as error:
            table.append(
                {
                    "model": model,
                    "ok": False,
                    "error": str(error),
                    "rss": np.inf,
                    "rmse_nA": np.inf,
                }
            )
            continue

        row = {
            "model": model,
            "ok": True,
            "error": "",
            **_fit_quality(Iexp_nA, I_start_nA, n_free=0),
        }
        for parameter in parameters:
            row[parameter.name] = parameter.guess

        table.append(row)
        curves[model] = {
            "V_mV": Vbias_mV,
            "I_exp_nA": Iexp_nA,
            "I_start_nA": I_start_nA,
            "params": tuple(parameters),
        }

    table.sort(key=lambda row: row.get("rmse_nA", np.inf))
    return table, curves


def compare_bcs_fits(
    Vbias_mV: np.ndarray,
    Iexp_nA: np.ndarray,
    settings: Mapping[str, tuple[float, float, float, bool]],
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    weights: np.ndarray | None = None,
    maxfev: int = 2_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit several BCS models to one IV trace and rank them.

    Parameters
    ----------
    Vbias_mV, Iexp_nA:
        One-dimensional voltage and current arrays.
    settings:
        Mapping from parameter name to ``(guess, lower, upper, fixed)``.
        Parameters missing from the mapping keep the model defaults.
    models:
        Optimizer model keys from ``superconductivity.optimizers.bcs``.
    weights:
        Optional point weights passed to ``fit_model``.
    maxfev:
        Maximum number of function evaluations for each fit.

    Returns
    -------
    table, solutions:
        ``table`` is sorted by increasing BIC. ``solutions`` maps each
        successful model key to the full ``fit_model`` solution.
    """
    Vbias_mV = np.asarray(Vbias_mV, dtype=float)
    Iexp_nA = np.asarray(Iexp_nA, dtype=float)
    table: list[dict[str, Any]] = []
    solutions: dict[str, Any] = {}

    for model in models:
        try:
            parameters = _make_parameters(model, settings)
            solution = fit_model(
                Vbias_mV,
                Iexp_nA,
                model=model,
                parameters=parameters,
                weights=weights,
                maxfev=maxfev,
            )
        except Exception as error:
            table.append(
                {
                    "model": model,
                    "ok": False,
                    "error": str(error),
                    "bic": np.inf,
                }
            )
            continue

        n_free = sum(not parameter.fixed for parameter in solution["params"])
        row = {
            "model": model,
            "ok": True,
            "error": "",
            **_fit_quality(
                Iexp_nA,
                solution["I_fit_nA"],
                n_free=n_free,
            ),
        }
        for parameter in solution["params"]:
            row[parameter.name] = parameter.value
            row[f"u_{parameter.name}"] = parameter.error

        table.append(row)
        solutions[model] = solution

    table.sort(key=lambda row: row.get("bic", np.inf))
    return table, solutions


def fit_bcs_conv_noise(
    Vbias_mV: np.ndarray,
    Iexp_nA: np.ndarray,
    settings: Mapping[str, tuple[float, float, float, bool]],
    *,
    weights: np.ndarray | None = None,
    maxfev: int = 2_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one IV trace with the BCS convolution + voltage-noise model.

    Parameters
    ----------
    Vbias_mV, Iexp_nA:
        One-dimensional voltage and current arrays.
    settings:
        Mapping from parameter name to ``(guess, lower, upper, fixed)``.
        The entries for ``GN_G0``, ``T_K``, ``Delta_meV``, ``gamma_meV``, and
        ``sigmaV_mV`` are used by this model.
    weights:
        Optional point weights passed to ``fit_model``.
    maxfev:
        Maximum number of function evaluations.

    Returns
    -------
    row, solution:
        ``row`` contains fit-quality metrics and parameter values. ``solution``
        is the full ``fit_model`` result.
    """
    Vbias_mV = np.asarray(Vbias_mV, dtype=float)
    Iexp_nA = np.asarray(Iexp_nA, dtype=float)
    parameters = _make_parameters(NOISE_MODEL, settings)

    solution = fit_model(
        Vbias_mV,
        Iexp_nA,
        model=NOISE_MODEL,
        parameters=parameters,
        weights=weights,
        maxfev=maxfev,
    )

    n_free = sum(not parameter.fixed for parameter in solution["params"])
    row = {
        "model": NOISE_MODEL,
        "ok": True,
        "error": "",
        **_fit_quality(
            Iexp_nA,
            solution["I_fit_nA"],
            n_free=n_free,
        ),
    }
    for parameter in solution["params"]:
        row[parameter.name] = parameter.value
        row[f"u_{parameter.name}"] = parameter.error

    return row, solution


def plot_bcs_conv_noise_fit(row: Mapping[str, Any], solution: Mapping[str, Any]):
    """Plot a ``fit_bcs_conv_noise`` result, residual, and conductance."""
    import matplotlib.pyplot as plt

    V_mV = np.asarray(solution["V_mV"], dtype=float)
    I_exp_nA = np.asarray(solution["I_exp_nA"], dtype=float)
    I_fit_nA = np.asarray(solution["I_fit_nA"], dtype=float)
    dG_exp_G0 = np.gradient(I_exp_nA, V_mV) / float(G0_muS)
    dG_fit_G0 = np.gradient(I_fit_nA, V_mV) / float(G0_muS)
    residual_nA = solution["I_exp_nA"] - solution["I_fit_nA"]
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


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def plot_bcs_start_comparison(
    table: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Any],
    *,
    top: int | None = None,
):
    """Plot starting-value model curves and residuals."""
    import matplotlib.pyplot as plt

    successful = [row for row in table if row.get("ok")]
    if top is not None:
        successful = successful[:top]
    if not successful:
        raise ValueError("No successful starting-value curves to plot.")

    first = curves[successful[0]["model"]]
    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.5, 4.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_fit.plot(first["V_mV"], first["I_exp_nA"], "k.", ms=2, label="exp")

    for row in successful:
        model = row["model"]
        curve = curves[model]
        residual_nA = curve["I_exp_nA"] - curve["I_start_nA"]
        label = f"{model}: RMSE={row['rmse_nA']:.3g} nA"
        ax_fit.plot(curve["V_mV"], curve["I_start_nA"], lw=1.4, label=label)
        ax_res.plot(curve["V_mV"], residual_nA, lw=1.0, label=model)

    ax_fit.set_ylabel(r"$I$ (nA)")
    ax_res.set_xlabel(r"$V_\mathrm{bias}$ (mV)")
    ax_res.set_ylabel(r"$\Delta I$ (nA)")
    ax_res.axhline(0.0, color="0.4", lw=0.8)
    ax_fit.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_bcs_fit_comparison(
    table: Sequence[Mapping[str, Any]],
    solutions: Mapping[str, Any],
    *,
    top: int = 3,
):
    """Plot the best fits and residuals from ``compare_bcs_fits``."""
    import matplotlib.pyplot as plt

    successful = [row for row in table if row.get("ok")]
    best_rows = successful[:top]
    if not best_rows:
        raise ValueError("No successful fits to plot.")

    first = solutions[best_rows[0]["model"]]
    fig, (ax_fit, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.5, 4.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_fit.plot(first["V_mV"], first["I_exp_nA"], "k.", ms=2, label="exp")

    for row in best_rows:
        model = row["model"]
        solution = solutions[model]
        residual_nA = solution["I_exp_nA"] - solution["I_fit_nA"]
        label = f"{model}: BIC={row['bic']:.1f}, RMSE={row['rmse_nA']:.3g} nA"
        ax_fit.plot(solution["V_mV"], solution["I_fit_nA"], lw=1.4, label=label)
        ax_res.plot(solution["V_mV"], residual_nA, lw=1.0, label=model)

    ax_fit.set_ylabel(r"$I$ (nA)")
    ax_res.set_xlabel(r"$V_\mathrm{bias}$ (mV)")
    ax_res.set_ylabel(r"$\Delta I$ (nA)")
    ax_res.axhline(0.0, color="0.4", lw=0.8)
    ax_fit.legend(fontsize=8)
    fig.tight_layout()
    return fig
