"""Test a distributional MAR + overdamped phase-dynamics model.

The MAR current is linear in the number of channels at each transmission.
This script therefore obtains the best non-negative transmission distribution
before fitting the Josephson contribution in the current-biased representation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
import superconductivity.utilities as scutil
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares, lsq_linear


HERE = Path(__file__).resolve().parent
DELTA_MEV = 0.1895
V_EXCLUDE_MV = 0.035
TRACE_INDEX = 0


def load_corrected_trace():
    """Return the raw current and series-resistance-corrected voltage."""
    cache = scutil.load_cache("cache", path=HERE)
    trace = cache.traces[TRACE_INDEX]
    current = np.asarray(trace["I_nA"], dtype=float)
    measured_voltage = np.asarray(trace["V_mV"], dtype=float)

    correction = np.load(HERE / "Rs_data.npz")
    resistance = float(correction["Rs_Ohm"])
    offset = float(correction["Voff_mV"])
    junction_voltage = measured_voltage - offset - resistance * current * 1e-6
    return current, junction_voltage, resistance, offset


def symmetrized_running_branch(current, voltage, voltage_grid):
    """Bin raw data by voltage and return the odd part of the IV curve."""
    positive = sc.bin_y_over_x(current, voltage, voltage_grid)
    negative = sc.bin_y_over_x(current, voltage, -voltage_grid[::-1])[::-1]
    odd_current = 0.5 * (positive - negative)
    return odd_current, positive, negative


def fit_mar_distribution(voltage, current, current_bank, tau):
    """Fit non-negative channel weights with smoothness regularization."""
    finite = np.isfinite(current) & (voltage >= V_EXCLUDE_MV)
    design = current_bank[:, finite].T
    target = current[finite]

    # Adjacent transmission bins should describe a distribution, rather than
    # compensating one another as unrelated fit parameters.
    second_difference = np.diff(np.eye(tau.size), n=2, axis=0)
    smoothness = 0.8

    # The archived high-bias conductance is close to 29 G0. Keep this as a
    # soft constraint so the subgap fit cannot discard most of the contact.
    target_conductance = 29.0
    conductance_weight = 100.0
    augmented_design = np.vstack(
        (
            design,
            smoothness * second_difference,
            conductance_weight * tau[None, :],
        )
    )
    augmented_target = np.concatenate(
        (
            target,
            np.zeros(second_difference.shape[0]),
            [conductance_weight * target_conductance],
        )
    )
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
        max_iter=2000,
        verbose=0,
    )
    weights = result.x
    model = weights @ current_bank
    return weights, model, finite


def inverse_odd_iv(current_positive, voltage_positive):
    """Construct a monotonic odd inverse V(I) for phase averaging."""
    current = np.concatenate((-current_positive[:0:-1], current_positive))
    voltage = np.concatenate((-voltage_positive[:0:-1], voltage_positive))
    order = np.argsort(current)
    current = current[order]
    voltage = voltage[order]
    current, unique = np.unique(current, return_index=True)
    return current, voltage[unique]


def phase_averaged_voltage(
    bias_current,
    critical_current,
    inverse_current,
    inverse_voltage,
    n_phase=2048,
):
    """Return the nonlinear overdamped RSJ voltage for sinusoidal CPR."""
    bias_current = np.asarray(bias_current, dtype=float)
    phase = (np.arange(n_phase) + 0.5) * (2 * np.pi / n_phase)
    supercurrent = critical_current * np.sin(phase)
    result = np.zeros_like(bias_current)
    running = np.abs(bias_current) > critical_current
    if not np.any(running):
        return result

    dissipative_current = bias_current[running, None] - supercurrent[None, :]
    instantaneous_voltage = np.interp(
        dissipative_current,
        inverse_current,
        inverse_voltage,
        left=np.nan,
        right=np.nan,
    )
    valid = np.all(np.isfinite(instantaneous_voltage), axis=1)
    running_voltage = np.full(np.count_nonzero(running), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        running_voltage[valid] = 1.0 / np.mean(
            1.0 / instantaneous_voltage[valid],
            axis=1,
        )
    result[running] = running_voltage
    return result


current_raw, voltage_raw, resistance, offset = load_corrected_trace()

voltage_positive = np.linspace(0.0, 0.55, 551)
current_odd, current_positive, current_negative = symmetrized_running_branch(
    current_raw,
    voltage_raw,
    voltage_positive,
)

tau = np.linspace(0.01, 1.0, 100)
_, current_bank = sc.get_Imar_nA(
    V_mV=voltage_positive,
    tau=tau,
    tau_resolved=True,
    Delta_meV=DELTA_MEV,
    gamma_meV=1e-6,
    T_K=0.0,
    show_progress=True,
)
if current_bank.shape[0] != tau.size:
    current_bank = current_bank.T

weights, current_mar, fit_mask = fit_mar_distribution(
    voltage_positive,
    current_odd,
    current_bank,
    tau,
)


def interpolate_tau_bank(transmission):
    """Interpolate the cached single-channel MAR bank in transmission."""
    return np.asarray(
        [
            np.interp(transmission, tau, current_bank[:, index])
            for index in range(voltage_positive.size)
        ]
    )


def grouped_residual(parameters):
    n_low, tau_low, n_high, tau_high = parameters
    grouped = (
        n_low * interpolate_tau_bank(tau_low)
        + n_high * interpolate_tau_bank(tau_high)
    )
    branch_residual = grouped[fit_mask] - current_odd[fit_mask]
    conductance_residual = 100.0 * (
        n_low * tau_low + n_high * tau_high - 29.0
    )
    return np.concatenate((branch_residual, [conductance_residual]))


grouped_result = least_squares(
    grouped_residual,
    x0=[25.0, 0.53, 16.0, 0.97],
    bounds=([1.0, 0.20, 1.0, 0.80], [80.0, 0.79, 40.0, 1.0]),
)
n_low, tau_low, n_high, tau_high = grouped_result.x
current_grouped = (
    n_low * interpolate_tau_bank(tau_low)
    + n_high * interpolate_tau_bank(tau_high)
)

inverse_current, inverse_voltage = inverse_odd_iv(
    current_mar,
    voltage_positive,
)
grouped_inverse_current, grouped_inverse_voltage = inverse_odd_iv(
    current_grouped,
    voltage_positive,
)

# Compare in the directly measured current-biased representation. Bin only to
# reduce the 46k raw samples; no interpolation is performed across current.
current_grid = np.linspace(-1680.0, 1680.0, 841)
voltage_by_current = sc.bin_y_over_x(voltage_raw, current_raw, current_grid)
valid_current = np.isfinite(voltage_by_current)
comparison_mask = valid_current & (
    (np.abs(current_grid) < 240.0) | (np.abs(current_grid) > 600.0)
)


def residual(parameter):
    critical_current = float(parameter[0])
    model = phase_averaged_voltage(
        current_grid,
        critical_current,
        inverse_current,
        inverse_voltage,
        n_phase=1024,
    )
    if np.any(~np.isfinite(model[comparison_mask])):
        return np.full(np.count_nonzero(comparison_mask), 1.0)
    return model[comparison_mask] - voltage_by_current[comparison_mask]


critical_result = least_squares(
    residual,
    x0=[400.0],
    bounds=([250.0], [550.0]),
    xtol=1e-10,
    ftol=1e-10,
)
critical_current = float(critical_result.x[0])
voltage_rsj = phase_averaged_voltage(
    current_grid,
    critical_current,
    inverse_current,
    inverse_voltage,
)

# In the hysteretic limit, the phase remains trapped on the zero-voltage
# branch and switches directly to the dissipative MAR branch. Estimate the
# threshold from the largest current still observed within 5 uV of zero.
zero_branch = np.isfinite(voltage_by_current) & (np.abs(voltage_by_current) < 0.005)
switching_current = float(np.max(np.abs(current_grid[zero_branch])))
voltage_hysteretic = np.interp(
    current_grid,
    inverse_current,
    inverse_voltage,
    left=np.nan,
    right=np.nan,
)
voltage_hysteretic[np.abs(current_grid) <= switching_current] = 0.0
voltage_grouped = np.interp(
    current_grid,
    grouped_inverse_current,
    grouped_inverse_voltage,
    left=np.nan,
    right=np.nan,
)
voltage_grouped[np.abs(current_grid) <= switching_current] = 0.0

mar_residual = current_mar[fit_mask] - current_odd[fit_mask]
rsj_valid = valid_current & np.isfinite(voltage_rsj)
rsj_residual = voltage_rsj[rsj_valid] - voltage_by_current[rsj_valid]
hysteretic_valid = valid_current & np.isfinite(voltage_hysteretic)
hysteretic_residual = (
    voltage_hysteretic[hysteretic_valid]
    - voltage_by_current[hysteretic_valid]
)
grouped_valid = valid_current & np.isfinite(voltage_grouped)
grouped_voltage_residual = (
    voltage_grouped[grouped_valid] - voltage_by_current[grouped_valid]
)


def mar_threshold_correction(current, parameters):
    """Localized odd voltage shifts at current-biased MAR transitions."""
    current = np.asarray(current, dtype=float)
    magnitude = np.abs(current)
    correction = np.zeros_like(current)
    for center, width, amplitude in np.reshape(parameters, (-1, 3)):
        coordinate = (magnitude - center) / width
        correction += amplitude * coordinate * np.exp(-0.5 * coordinate**2)
    return np.sign(current) * correction


# The raw trace has symmetric sharp transitions near the n=4, 3, and 2 MAR
# thresholds. Fit only localized branch-selection corrections; the two-group
# MAR background and zero-voltage threshold stay fixed.
transition_mask = grouped_valid & (np.abs(current_grid) > switching_current)


def transition_residual(parameters):
    corrected = voltage_grouped + mar_threshold_correction(
        current_grid,
        parameters,
    )
    return corrected[transition_mask] - voltage_by_current[transition_mask]


transition_result = least_squares(
    transition_residual,
    x0=[536.0, 35.0, 0.006, 672.0, 45.0, 0.007, 884.0, 55.0, 0.006],
    bounds=(
        [515.0, 5.0, -0.02, 645.0, 5.0, -0.02, 865.0, 5.0, -0.02],
        [555.0, 60.0, 0.02, 695.0, 60.0, 0.02, 905.0, 60.0, 0.02],
    ),
)
transition_parameters = transition_result.x
voltage_threshold_model = voltage_grouped + mar_threshold_correction(
    current_grid,
    transition_parameters,
)
voltage_threshold_model[np.abs(current_grid) <= switching_current] = 0.0
threshold_valid = valid_current & np.isfinite(voltage_threshold_model)
threshold_residual = (
    voltage_threshold_model[threshold_valid]
    - voltage_by_current[threshold_valid]
)

print(f"series resistance = {resistance:.6g} ohm")
print(f"voltage offset = {offset:.6g} mV")
print(f"Delta = {DELTA_MEV:.6g} meV")
print(f"sum channel weights = {np.sum(weights):.6g}")
print(f"sum tau * weight = {np.dot(tau, weights):.6g} G0")
print(f"effective Ic = {critical_current:.6g} nA")
print(f"observed switching threshold = {switching_current:.6g} nA")
print(f"MAR branch RMSE = {np.sqrt(np.mean(mar_residual**2)):.6g} nA")
print(f"RSJ voltage RMSE = {1e3 * np.sqrt(np.mean(rsj_residual**2)):.6g} uV")
print(
    "hysteretic MAR voltage RMSE = "
    f"{1e3 * np.sqrt(np.mean(hysteretic_residual**2)):.6g} uV"
)
print(
    "two-group model: "
    f"N1={n_low:.4g} at tau1={tau_low:.4g}, "
    f"N2={n_high:.4g} at tau2={tau_high:.4g}, "
    f"G={n_low * tau_low + n_high * tau_high:.4g} G0"
)
print(
    "two-group hysteretic voltage RMSE = "
    f"{1e3 * np.sqrt(np.mean(grouped_voltage_residual**2)):.6g} uV"
)
print(
    "MAR-threshold switching RMSE = "
    f"{1e3 * np.sqrt(np.mean(threshold_residual**2)):.6g} uV"
)
near_n3 = valid_current & (np.abs(np.abs(current_grid) - 670.0) < 60.0)
print(
    "n=3 local RMSE: "
    f"{1e3 * np.sqrt(np.mean((voltage_grouped[near_n3] - voltage_by_current[near_n3])**2)):.4g} "
    "-> "
    f"{1e3 * np.sqrt(np.mean((voltage_threshold_model[near_n3] - voltage_by_current[near_n3])**2)):.4g} uV"
)
for order, (center, width, amplitude) in zip(
    (4, 3, 2),
    np.reshape(transition_parameters, (-1, 3)),
    strict=True,
):
    print(
        f"n={order} transition: I={center:.3f} nA, "
        f"width={width:.3f} nA, amplitude={1e3 * amplitude:.3f} uV"
    )
low = tau < 0.8
print(
    "low-transmission component: "
    f"N={np.sum(weights[low]):.4g}, G={np.dot(tau[low], weights[low]):.4g} G0"
)
print(
    "high-transmission component: "
    f"N={np.sum(weights[~low]):.4g}, G={np.dot(tau[~low], weights[~low]):.4g} G0"
)

fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), layout="constrained")
ax_iv, ax_mar, ax_tau, ax_residual = axes.flat

stride = max(1, current_raw.size // 12_000)
ax_iv.plot(
    current_raw[::stride],
    voltage_raw[::stride],
    ".",
    color="black",
    markersize=1.2,
    alpha=0.55,
    label="corrected raw data",
    rasterized=True,
)
ax_iv.plot(
    current_grid,
    voltage_rsj,
    color="C3",
    linewidth=1.6,
    label="distributional MAR + overdamped phase",
)
ax_iv.plot(
    current_grid,
    voltage_hysteretic,
    color="C0",
    linewidth=1.6,
    label="phase-locked / hysteretic MAR model",
)
ax_iv.plot(
    current_grid,
    voltage_grouped,
    color="C4",
    linestyle="--",
    linewidth=1.3,
    label="two-group hysteretic MAR",
)
ax_iv.plot(
    current_grid,
    voltage_threshold_model,
    color="C2",
    linestyle=":",
    linewidth=1.7,
    label="MAR-threshold switching model",
)
ax_iv.set_xlabel(r"$I$ (nA)")
ax_iv.set_ylabel(r"$V_J$ (mV)")
ax_iv.legend(frameon=False)

ax_mar.plot(
    voltage_positive,
    current_odd,
    color="black",
    linewidth=1.3,
    label="symmetrized measured branches",
)
ax_mar.plot(
    voltage_positive,
    current_mar,
    color="C0",
    linewidth=1.5,
    label="best distributional MAR",
)
ax_mar.plot(
    voltage_positive,
    current_grouped,
    color="C4",
    linestyle="--",
    linewidth=1.3,
    label=r"$\nabla_V$ two-group $I(V)$",
)
ax_mar.axvspan(0.0, V_EXCLUDE_MV, color="0.9", zorder=0)
ax_mar.set_xlabel(r"$|V_J|$ (mV)")
ax_mar.set_ylabel(r"odd $I$ (nA)")
ax_mar.legend(frameon=False)

ax_tau.plot(tau, weights, color="C2", linewidth=1.5)
ax_tau.fill_between(tau, weights, color="C2", alpha=0.2)
ax_tau.set_xlabel(r"transmission $\tau$")
ax_tau.set_ylabel("effective channel density")

ax_residual.plot(
    current_grid[rsj_valid],
    1e3 * rsj_residual,
    color="C3",
    linewidth=1.0,
    alpha=0.65,
    label="overdamped phase",
)
ax_residual.plot(
    current_grid[hysteretic_valid],
    1e3 * hysteretic_residual,
    color="C0",
    linewidth=1.0,
    label="hysteretic MAR",
)
ax_residual.plot(
    current_grid[grouped_valid],
    1e3 * grouped_voltage_residual,
    color="C4",
    linestyle="--",
    linewidth=1.0,
    label="two-group hysteretic MAR",
)
ax_residual.plot(
    current_grid[threshold_valid],
    1e3 * threshold_residual,
    color="C2",
    linestyle=":",
    linewidth=1.2,
    label="MAR-threshold switching",
)
ax_residual.axhline(0.0, color="0.7", linewidth=0.8)
ax_residual.set_xlabel(r"$I$ (nA)")
ax_residual.set_ylabel(r"$V_{model}-V_{exp}$ ($\mu$V)")
ax_residual.legend(frameon=False)

for axis in axes.flat:
    axis.grid(alpha=0.15)

output = HERE / "unirradiated_model_test"
fig.savefig(output.with_suffix(".png"), dpi=250)
fig.savefig(output.with_suffix(".pdf"))

fig_zoom, (ax_zoom, ax_zoom_residual) = plt.subplots(
    2,
    1,
    figsize=(8.0, 6.5),
    sharex=True,
    height_ratios=(3.0, 1.0),
    layout="constrained",
)
positive_zoom = (
    np.isfinite(voltage_by_current)
    & (current_grid >= 420.0)
    & (current_grid <= 980.0)
)
ax_zoom.plot(
    current_grid[positive_zoom],
    voltage_by_current[positive_zoom],
    ".",
    color="black",
    markersize=3.0,
    label="corrected data",
)
ax_zoom.plot(
    current_grid[positive_zoom],
    voltage_grouped[positive_zoom],
    color="C4",
    linestyle="--",
    linewidth=1.4,
    label="two-group hysteretic MAR",
)
ax_zoom.plot(
    current_grid[positive_zoom],
    voltage_threshold_model[positive_zoom],
    color="C2",
    linewidth=1.7,
    label="MAR-threshold switching",
)
for order, color in zip((4, 3, 2), ("C0", "C1", "C3"), strict=True):
    threshold = 2 * DELTA_MEV / order
    ax_zoom.axhline(
        threshold,
        color=color,
        linestyle=":",
        linewidth=0.9,
        alpha=0.8,
        label=rf"$2\Delta/{order}e$",
    )
ax_zoom.set_ylabel(r"$V_J$ (mV)")
ax_zoom.legend(frameon=False, ncols=2, fontsize="small")

ax_zoom_residual.plot(
    current_grid[positive_zoom],
    1e3
    * (
        voltage_grouped[positive_zoom]
        - voltage_by_current[positive_zoom]
    ),
    color="C4",
    linestyle="--",
    linewidth=1.1,
    label="without branch switching",
)
ax_zoom_residual.plot(
    current_grid[positive_zoom],
    1e3
    * (
        voltage_threshold_model[positive_zoom]
        - voltage_by_current[positive_zoom]
    ),
    color="C2",
    linewidth=1.2,
    label="with branch switching",
)
ax_zoom_residual.axhline(0.0, color="0.7", linewidth=0.8)
ax_zoom_residual.set_xlabel(r"$I$ (nA)")
ax_zoom_residual.set_ylabel(r"residual ($\mu$V)")
ax_zoom_residual.legend(frameon=False, fontsize="small")
for axis in (ax_zoom, ax_zoom_residual):
    axis.grid(alpha=0.15)

zoom_output = HERE / "unirradiated_model_jumps"
fig_zoom.savefig(zoom_output.with_suffix(".png"), dpi=250)
fig_zoom.savefig(zoom_output.with_suffix(".pdf"))

# Aligned IV/differential view. Match the evaluation workflow used elsewhere
# in this repository: upsample the parametric curve in acquisition order, bin
# it independently over voltage and current, and only then differentiate the
# two sampled representations.
voltage_derivative_grid = np.linspace(-0.48, 0.48, 1201)
current_derivative_grid = np.linspace(-1680.0, 1680.0, 1201)


def sample_both_directions(current, voltage, *, factor):
    """Upsample a parametric IV and return sampled I(V) and V(I)."""
    current_up, voltage_up = sc.upsample(
        np.asarray(current, dtype=float),
        np.asarray(voltage, dtype=float),
        factor=factor,
    )
    current_over_voltage = sc.bin_y_over_x(
        current_up,
        voltage_up,
        voltage_derivative_grid,
    )
    voltage_over_current = sc.bin_y_over_x(
        voltage_up,
        current_up,
        current_derivative_grid,
    )
    return current_over_voltage, voltage_over_current


current_data_over_voltage, voltage_data_over_current = sample_both_directions(
    current_raw,
    voltage_raw,
    factor=10,
)
current_grouped_over_voltage, voltage_grouped_over_current = (
    sample_both_directions(
        current_grid[grouped_valid],
        voltage_grouped[grouped_valid],
        factor=100,
    )
)
current_threshold_over_voltage, voltage_threshold_over_current = (
    sample_both_directions(
        current_grid[threshold_valid],
        voltage_threshold_model[threshold_valid],
        factor=100,
    )
)

didv_data = (
    np.gradient(current_data_over_voltage, voltage_derivative_grid)
    / float(sc.G0_muS)
)
didv_grouped = (
    np.gradient(current_grouped_over_voltage, voltage_derivative_grid)
    / float(sc.G0_muS)
)
didv_threshold = (
    np.gradient(current_threshold_over_voltage, voltage_derivative_grid)
    / float(sc.G0_muS)
)
dvdi_data = (
    np.gradient(voltage_data_over_current, current_derivative_grid)
    * float(sc.G0_muS)
)
dvdi_grouped = (
    np.gradient(voltage_grouped_over_current, current_derivative_grid)
    * float(sc.G0_muS)
)
dvdi_threshold = (
    np.gradient(voltage_threshold_over_current, current_derivative_grid)
    * float(sc.G0_muS)
)

fig_differential = plt.figure(figsize=(10.5, 8.0), layout="constrained")
differential_grid = GridSpec(
    2,
    2,
    figure=fig_differential,
    width_ratios=(4.0, 1.45),
    height_ratios=(4.0, 1.45),
)
ax_diff_iv = fig_differential.add_subplot(differential_grid[0, 0])
ax_diff_didv = fig_differential.add_subplot(
    differential_grid[1, 0],
    sharex=ax_diff_iv,
)
ax_diff_dvdi = fig_differential.add_subplot(
    differential_grid[0, 1],
    sharey=ax_diff_iv,
)
ax_diff_empty = fig_differential.add_subplot(differential_grid[1, 1])
ax_diff_empty.axis("off")

raw_stride = max(1, current_raw.size // 12_000)
ax_diff_iv.plot(
    voltage_raw[::raw_stride],
    current_raw[::raw_stride],
    ".",
    color="black",
    markersize=1.2,
    alpha=0.55,
    label="corrected data",
    rasterized=True,
)
ax_diff_iv.plot(
    voltage_derivative_grid,
    current_data_over_voltage,
    color="C0",
    linewidth=1.15,
    label=r"data: sampled $I(V)$",
)
ax_diff_iv.plot(
    voltage_data_over_current,
    current_derivative_grid,
    color="C1",
    linewidth=1.15,
    label=r"data: sampled $V(I)$",
)
ax_diff_iv.plot(
    voltage_derivative_grid,
    current_grouped_over_voltage,
    color="C4",
    linestyle="--",
    linewidth=1.3,
    label=r"two-group MAR: sampled $I(V)$",
)
ax_diff_iv.plot(
    voltage_derivative_grid,
    current_threshold_over_voltage,
    color="C2",
    linestyle="--",
    linewidth=1.3,
    label=r"threshold model: sampled $I(V)$",
)
ax_diff_iv.plot(
    voltage_threshold_over_current,
    current_derivative_grid,
    color="C2",
    linewidth=1.6,
    label=r"threshold model: sampled $V(I)$",
)
ax_diff_iv.set_ylabel(r"$I$ (nA)")
ax_diff_iv.tick_params(labelbottom=False)
ax_diff_iv.legend(frameon=False, loc="upper left", fontsize="small", ncols=2)

ax_diff_didv.plot(
    voltage_derivative_grid,
    didv_data,
    color="C0",
    linewidth=1.25,
    label=r"$\nabla_V$ data sampled $I(V)$",
)
ax_diff_didv.plot(
    voltage_derivative_grid,
    didv_grouped,
    color="C4",
    linestyle="--",
    linewidth=1.1,
    label="two-group MAR",
)
ax_diff_didv.plot(
    voltage_derivative_grid,
    didv_threshold,
    color="C2",
    linewidth=1.2,
    label=r"$\nabla_V$ threshold $I(V)$",
)
ax_diff_didv.set_xlabel(r"$V_J$ (mV)")
ax_diff_didv.set_ylabel(r"$dI/dV$ ($G_0$)")
ax_diff_didv.set_yscale("symlog", linthresh=10.0, linscale=1.0)
ax_diff_didv.legend(frameon=False, loc="upper right", fontsize="small")

ax_diff_dvdi.plot(
    dvdi_data,
    current_derivative_grid,
    color="C1",
    linewidth=1.25,
    label=r"$\nabla_I$ data sampled $V(I)$",
)
ax_diff_dvdi.plot(
    dvdi_grouped,
    current_derivative_grid,
    color="C4",
    linestyle="--",
    linewidth=1.1,
    label=r"$\nabla_I$ two-group $V(I)$",
)
ax_diff_dvdi.plot(
    dvdi_threshold,
    current_derivative_grid,
    color="C2",
    linewidth=1.2,
    label=r"$\nabla_I$ threshold $V(I)$",
)
ax_diff_dvdi.set_xlabel(r"$dV/dI$ ($R_0$)")
ax_diff_dvdi.tick_params(labelleft=False)
ax_diff_dvdi.legend(frameon=False, loc="upper right", fontsize="x-small")

for axis in (ax_diff_iv, ax_diff_didv, ax_diff_dvdi):
    axis.grid(alpha=0.15)

differential_output = HERE / "unirradiated_model_differentials"
fig_differential.savefig(differential_output.with_suffix(".png"), dpi=250)
fig_differential.savefig(differential_output.with_suffix(".pdf"))
plt.show()
