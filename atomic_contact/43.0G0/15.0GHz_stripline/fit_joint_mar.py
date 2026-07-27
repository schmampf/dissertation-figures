"""Jointly fit static MAR to I(V), V(I), dI/dV, and dV/dI."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
import superconductivity.utilities as scutil
from matplotlib.gridspec import GridSpec
from scipy.optimize import differential_evolution, least_squares


HERE = Path(__file__).resolve().parent
TRACE_INDEX = 0
SWITCHING_CURRENT_NA = 404.0
TARGET_CONDUCTANCE_G0 = 29.0
DELTA_MEV = 0.1895


def load_corrected_trace():
    cache = scutil.load_cache("cache", path=HERE)
    trace = cache.traces[TRACE_INDEX]
    current = np.asarray(trace["I_nA"], dtype=float)
    measured_voltage = np.asarray(trace["V_mV"], dtype=float)
    correction = np.load(HERE / "Rs_data.npz")
    resistance = float(correction["Rs_Ohm"])
    offset = float(correction["Voff_mV"])
    voltage = measured_voltage - offset - resistance * current * 1e-6
    return current, voltage


def sample_raw_trace(current, voltage, voltage_positive, current_positive):
    current_up, voltage_up = sc.upsample(current, voltage, factor=10)

    current_plus = sc.bin_y_over_x(current_up, voltage_up, voltage_positive)
    current_minus = sc.bin_y_over_x(
        current_up,
        voltage_up,
        -voltage_positive[::-1],
    )[::-1]
    current_odd = 0.5 * (current_plus - current_minus)

    voltage_plus = sc.bin_y_over_x(voltage_up, current_up, current_positive)
    voltage_minus = sc.bin_y_over_x(
        voltage_up,
        current_up,
        -current_positive[::-1],
    )[::-1]
    voltage_odd = 0.5 * (voltage_plus - voltage_minus)
    return current_odd, voltage_odd


def interpolate_axis(values, grid, coordinate):
    upper = int(np.searchsorted(grid, coordinate, side="right"))
    upper = int(np.clip(upper, 1, grid.size - 1))
    lower = upper - 1
    fraction = (coordinate - grid[lower]) / (grid[upper] - grid[lower])
    return (1.0 - fraction) * values[lower] + fraction * values[upper]


current_raw, voltage_raw = load_corrected_trace()
voltage_positive = np.linspace(0.0, 0.52, 521)
current_positive = np.linspace(0.0, 1680.0, 421)
current_exp, voltage_exp = sample_raw_trace(
    current_raw,
    voltage_raw,
    voltage_positive,
    current_positive,
)

conductance_exp = (
    np.gradient(current_exp, voltage_positive) / float(sc.G0_muS)
)
resistance_exp = (
    np.gradient(voltage_exp, current_positive) * float(sc.G0_muS)
)

# Precompute a bank over transmission. Channel multiplicities remain linear.
tau_grid = np.linspace(0.01, 1.0, 100)
mar_bank = np.empty(
    (tau_grid.size, voltage_positive.size),
    dtype=float,
)
_, resolved = sc.get_Imar_nA(
    V_mV=voltage_positive,
    tau=tau_grid,
    tau_resolved=True,
    Delta_meV=DELTA_MEV,
    gamma_meV=1e-6,
    T_K=0.0,
    show_progress=False,
)
if resolved.shape[0] != tau_grid.size:
    resolved = resolved.T
mar_bank[:] = resolved


def single_channel_curve(transmission):
    return interpolate_axis(mar_bank, tau_grid, transmission)


def evaluate_model(parameters):
    counts = np.asarray(parameters[:3], dtype=float)
    transmissions = np.asarray(parameters[3:6], dtype=float)
    current_model = np.sum(
        [
            count * single_channel_curve(transmission)
            for count, transmission in zip(counts, transmissions, strict=True)
        ],
        axis=0,
    )
    # Numerical noise can make an otherwise monotonic MAR curve locally turn
    # by a tiny amount, which destabilizes inversion. Preserve only its
    # physically relevant monotonic envelope.
    current_monotonic = np.maximum.accumulate(current_model)
    voltage_model = np.interp(
        current_positive,
        current_monotonic,
        voltage_positive,
        left=0.0,
        right=np.nan,
    )
    voltage_model[current_positive <= SWITCHING_CURRENT_NA] = 0.0
    conductance_model = (
        np.gradient(current_model, voltage_positive) / float(sc.G0_muS)
    )
    resistance_model = (
        np.gradient(voltage_model, current_positive) * float(sc.G0_muS)
    )
    normal_conductance = float(np.dot(counts, transmissions))
    return (
        current_model,
        voltage_model,
        conductance_model,
        resistance_model,
        normal_conductance,
    )


mask_i = (
    np.isfinite(current_exp)
    & (voltage_positive >= 0.035)
    & (voltage_positive <= 0.50)
)
mask_v = (
    np.isfinite(voltage_exp)
    & (current_positive >= 420.0)
    & (current_positive <= 1640.0)
)
mask_g = (
    np.isfinite(conductance_exp)
    & (voltage_positive >= 0.055)
    & (voltage_positive <= 0.48)
)
mask_r = (
    np.isfinite(resistance_exp)
    & (current_positive >= 440.0)
    & (current_positive <= 1550.0)
)


def residual_vector(parameters):
    (
        current_model,
        voltage_model,
        conductance_model,
        resistance_model,
        normal_conductance,
    ) = evaluate_model(parameters)
    if np.any(~np.isfinite(voltage_model[mask_v])):
        return np.full(
            np.count_nonzero(mask_i)
            + np.count_nonzero(mask_v)
            + np.count_nonzero(mask_g)
            + np.count_nonzero(mask_r)
            + 1,
            1e3,
        )

    # Divide every block by sqrt(N) so the four representations receive the
    # intended weights independently of their bin count.
    current_residual = (
        (current_model[mask_i] - current_exp[mask_i])
        / 6.0
        / np.sqrt(np.count_nonzero(mask_i))
    )
    voltage_residual = (
        (voltage_model[mask_v] - voltage_exp[mask_v])
        / 0.003
        / np.sqrt(np.count_nonzero(mask_v))
    )
    conductance_residual = (
        0.45
        * (conductance_model[mask_g] - conductance_exp[mask_g])
        / 12.0
        / np.sqrt(np.count_nonzero(mask_g))
    )
    resistance_residual = (
        0.45
        * (resistance_model[mask_r] - resistance_exp[mask_r])
        / 0.025
        / np.sqrt(np.count_nonzero(mask_r))
    )
    conductance_constraint = np.array(
        [(normal_conductance - TARGET_CONDUCTANCE_G0) / 0.35]
    )
    return np.concatenate(
        (
            current_residual,
            voltage_residual,
            conductance_residual,
            resistance_residual,
            conductance_constraint,
        )
    )


def scalar_objective(parameters):
    residual = residual_vector(parameters)
    return float(np.dot(residual, residual))


# Ordered, partly overlapping tau intervals avoid label permutations while
# still allowing the optimizer to collapse back to a two-group solution.
bounds = [
    (0.0, 60.0),
    (0.0, 60.0),
    (0.0, 40.0),
    (0.10, 0.60),
    (0.45, 0.90),
    (0.82, 1.00),
]
global_result = differential_evolution(
    scalar_objective,
    bounds=bounds,
    seed=4,
    maxiter=120,
    popsize=10,
    polish=False,
    updating="immediate",
    workers=1,
)
lower = np.array([item[0] for item in bounds])
upper = np.array([item[1] for item in bounds])
local_result = least_squares(
    residual_vector,
    x0=global_result.x,
    bounds=(lower, upper),
    max_nfev=4000,
    xtol=1e-11,
    ftol=1e-11,
    gtol=1e-11,
)

parameters = local_result.x
(
    current_fit,
    voltage_fit,
    conductance_fit,
    resistance_fit,
    normal_conductance_fit,
) = evaluate_model(parameters)

counts_fit = parameters[:3]
transmissions_fit = parameters[3:6]
delta_fit = DELTA_MEV

rmse_i = np.sqrt(np.mean((current_fit[mask_i] - current_exp[mask_i]) ** 2))
rmse_v = np.sqrt(np.mean((voltage_fit[mask_v] - voltage_exp[mask_v]) ** 2))
rmse_g = np.sqrt(
    np.mean((conductance_fit[mask_g] - conductance_exp[mask_g]) ** 2)
)
rmse_r = np.sqrt(
    np.mean((resistance_fit[mask_r] - resistance_exp[mask_r]) ** 2)
)
mask_n3 = (
    np.isfinite(voltage_exp)
    & (current_positive >= 610.0)
    & (current_positive <= 730.0)
)
rmse_n3 = np.sqrt(
    np.mean((voltage_fit[mask_n3] - voltage_exp[mask_n3]) ** 2)
)

print(f"counts = {counts_fit}")
print(f"tau = {transmissions_fit}")
print(f"Delta = {delta_fit:.8g} meV")
print(f"GN = {normal_conductance_fit:.8g} G0")
print(f"I(V) RMSE = {rmse_i:.6g} nA")
print(f"V(I) RMSE = {1e3 * rmse_v:.6g} uV")
print(f"dI/dV RMSE = {rmse_g:.6g} G0")
print(f"dV/dI RMSE = {rmse_r:.6g} R0")
print(f"n=3 local V(I) RMSE = {1e3 * rmse_n3:.6g} uV")
print(f"objective = {scalar_objective(parameters):.8g}")

# Mirror the jointly fitted positive branch for a full aligned comparison.
voltage_full = np.concatenate((-voltage_positive[:0:-1], voltage_positive))
current_exp_full = np.concatenate((-current_exp[:0:-1], current_exp))
current_fit_full = np.concatenate((-current_fit[:0:-1], current_fit))
current_axis_full = np.concatenate((-current_positive[:0:-1], current_positive))
voltage_exp_full = np.concatenate((-voltage_exp[:0:-1], voltage_exp))
voltage_fit_full = np.concatenate((-voltage_fit[:0:-1], voltage_fit))
conductance_exp_full = np.concatenate(
    (conductance_exp[:0:-1], conductance_exp)
)
conductance_fit_full = np.concatenate(
    (conductance_fit[:0:-1], conductance_fit)
)
resistance_exp_full = np.concatenate((resistance_exp[:0:-1], resistance_exp))
resistance_fit_full = np.concatenate((resistance_fit[:0:-1], resistance_fit))

fig = plt.figure(figsize=(10.5, 8.0), layout="constrained")
grid = GridSpec(
    2,
    2,
    figure=fig,
    width_ratios=(4.0, 1.45),
    height_ratios=(4.0, 1.45),
)
ax_iv = fig.add_subplot(grid[0, 0])
ax_didv = fig.add_subplot(grid[1, 0], sharex=ax_iv)
ax_dvdi = fig.add_subplot(grid[0, 1], sharey=ax_iv)
ax_empty = fig.add_subplot(grid[1, 1])
ax_empty.axis("off")

stride = max(1, current_raw.size // 12_000)
ax_iv.plot(
    voltage_raw[::stride],
    current_raw[::stride],
    ".",
    color="black",
    markersize=1.1,
    alpha=0.45,
    label="corrected raw data",
    rasterized=True,
)
ax_iv.plot(
    voltage_full,
    current_exp_full,
    color="C0",
    linewidth=1.1,
    label=r"data: sampled $I(V)$",
)
ax_iv.plot(
    voltage_exp_full,
    current_axis_full,
    color="C1",
    linewidth=1.1,
    label=r"data: sampled $V(I)$",
)
ax_iv.plot(
    voltage_full,
    current_fit_full,
    color="C2",
    linestyle="--",
    linewidth=1.4,
    label=r"joint MAR: $I(V)$",
)
ax_iv.plot(
    voltage_fit_full,
    current_axis_full,
    color="C2",
    linewidth=1.5,
    label=r"joint MAR: $V(I)$",
)
ax_iv.set_ylabel(r"$I$ (nA)")
ax_iv.tick_params(labelbottom=False)
ax_iv.legend(frameon=False, fontsize="small", ncols=2)

ax_didv.plot(
    voltage_full,
    conductance_exp_full,
    color="C0",
    linewidth=1.15,
    label="data",
)
ax_didv.plot(
    voltage_full,
    conductance_fit_full,
    color="C2",
    linewidth=1.2,
    label="joint MAR fit",
)
ax_didv.set_xlabel(r"$V_J$ (mV)")
ax_didv.set_ylabel(r"$dI/dV$ ($G_0$)")
ax_didv.set_yscale("symlog", linthresh=10.0)
ax_didv.legend(frameon=False, fontsize="small")

ax_dvdi.plot(
    resistance_exp_full,
    current_axis_full,
    color="C1",
    linewidth=1.15,
    label="data",
)
ax_dvdi.plot(
    resistance_fit_full,
    current_axis_full,
    color="C2",
    linewidth=1.2,
    label="joint MAR fit",
)
ax_dvdi.set_xlabel(r"$dV/dI$ ($R_0$)")
ax_dvdi.tick_params(labelleft=False)
ax_dvdi.legend(frameon=False, fontsize="x-small")

for axis in (ax_iv, ax_didv, ax_dvdi):
    axis.grid(alpha=0.15)

output = HERE / "joint_mar_fit"
fig.savefig(output.with_suffix(".png"), dpi=250)
fig.savefig(output.with_suffix(".pdf"))
np.savez_compressed(
    HERE / "joint_mar_fit.npz",
    counts=counts_fit,
    tau=transmissions_fit,
    Delta_meV=delta_fit,
    GN_G0=normal_conductance_fit,
    voltage_positive_mV=voltage_positive,
    current_fit_nA=current_fit,
    current_positive_nA=current_positive,
    voltage_fit_mV=voltage_fit,
    conductance_fit_G0=conductance_fit,
    resistance_fit_R0=resistance_fit,
)
plt.show()
