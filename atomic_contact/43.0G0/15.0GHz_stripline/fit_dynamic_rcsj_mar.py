"""Fit a stateful nonlinear-RCSJ model with MAR damping to the raw sweep."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
import superconductivity.utilities as scutil
from matplotlib.gridspec import GridSpec


jax.config.update("jax_enable_x64", False)

HERE = Path(__file__).resolve().parent
DELTA_MEV = 0.1895
TRACE_INDEX = 0

# Parsimonious two-group MAR result from try_unirradiated_model.py.
N_LOW = 25.07
TAU_LOW = 0.5270
N_HIGH = 15.48
TAU_HIGH = 0.9694


def load_corrected_trace():
    cache = scutil.load_cache("cache", path=HERE)
    trace = cache.traces[TRACE_INDEX]
    current = np.asarray(trace["I_nA"], dtype=float)
    measured_voltage = np.asarray(trace["V_mV"], dtype=float)
    time = np.asarray(trace["t_s"], dtype=float)

    correction = np.load(HERE / "Rs_data.npz")
    resistance = float(correction["Rs_Ohm"])
    offset = float(correction["Voff_mV"])
    junction_voltage = measured_voltage - offset - resistance * current * 1e-6
    return current, junction_voltage, time


def make_mar_curve(voltage):
    _, resolved = sc.get_Imar_nA(
        V_mV=voltage,
        tau=np.array([TAU_LOW, TAU_HIGH]),
        tau_resolved=True,
        Delta_meV=DELTA_MEV,
        gamma_meV=1e-6,
        T_K=0.0,
        show_progress=False,
    )
    if resolved.shape[0] != 2:
        resolved = resolved.T
    return N_LOW * resolved[0] + N_HIGH * resolved[1]


def make_normalized_cpr(n_harmonics=12):
    phase = (np.arange(8192) + 0.5) * (2 * np.pi / 8192)
    cpr = (
        N_LOW
        * sc.get_cpr_abs_nA(
            phase,
            tau=TAU_LOW,
            Delta_meV=DELTA_MEV,
        )
        + N_HIGH
        * sc.get_cpr_abs_nA(
            phase,
            tau=TAU_HIGH,
            Delta_meV=DELTA_MEV,
        )
    )
    coefficients = np.array(
        [
            np.mean(cpr * np.sin(order * phase)) * 2
            for order in range(1, n_harmonics + 1)
        ]
    )
    reconstructed = np.sum(
        coefficients[:, None]
        * np.sin(np.arange(1, n_harmonics + 1)[:, None] * phase),
        axis=0,
    )
    coefficients /= np.max(np.abs(reconstructed))
    return coefficients


current_raw, voltage_raw, time_raw = load_corrected_trace()

# Bin the directly measured down sweep on an increasing current grid. The
# simulation traverses the reverse order and carries its final state forward.
current_grid = np.linspace(-1680.0, 1680.0, 281)
voltage_exp = sc.bin_y_over_x(voltage_raw, current_raw, current_grid)
finite_exp = np.isfinite(voltage_exp)

voltage_lut = np.linspace(-0.65, 0.65, 2601)
current_mar_lut = make_mar_curve(voltage_lut)
cpr_coefficients = make_normalized_cpr()

voltage_lut_jax = jnp.asarray(voltage_lut, dtype=jnp.float32)
current_mar_jax = jnp.asarray(current_mar_lut, dtype=jnp.float32)
cpr_coefficients_jax = jnp.asarray(cpr_coefficients, dtype=jnp.float32)
harmonics_jax = jnp.arange(
    1,
    cpr_coefficients.size + 1,
    dtype=jnp.float32,
)
current_down_jax = jnp.asarray(current_grid[::-1], dtype=jnp.float32)

# Unit conversion for dphi/dt = 2 pi V / (h/e), with V in mV and t in ps.
H_PVS = np.float32(float(sc.h_pVs))
TWO_PI = np.float32(2 * np.pi)
DT_PS = np.float32(0.015)
N_HOLD = 14_000
BURN_INDEX = 2_000


@jax.jit
def simulate_down_sweep(log10_c_pf, critical_current_na):
    """Return average V(I) for one descending quasistatic current sweep."""
    capacitance_pf = jnp.power(10.0, log10_c_pf)
    dv_factor = DT_PS * 1e-6 / capacitance_pf

    initial_voltage = jnp.interp(
        current_down_jax[0],
        current_mar_jax,
        voltage_lut_jax,
    )
    initial_state = (
        jnp.asarray(0.0, dtype=jnp.float32),
        initial_voltage,
    )

    def bias_step(state, bias_current):
        def time_step(index, carry):
            phase, voltage, voltage_sum = carry
            supercurrent = critical_current_na * jnp.sum(
                cpr_coefficients_jax * jnp.sin(harmonics_jax * phase)
            )
            mar_current = jnp.interp(
                voltage,
                voltage_lut_jax,
                current_mar_jax,
            )
            voltage_next = voltage + dv_factor * (
                bias_current - supercurrent - mar_current
            )
            voltage_next = jnp.clip(voltage_next, -0.65, 0.65)
            phase_next = phase + voltage_next * TWO_PI * DT_PS / H_PVS
            phase_next = jnp.mod(phase_next + np.pi, TWO_PI) - np.pi
            voltage_sum_next = voltage_sum + jnp.where(
                index >= BURN_INDEX,
                voltage_next,
                0.0,
            )
            return phase_next, voltage_next, voltage_sum_next

        phase, voltage = state
        phase, voltage, voltage_sum = jax.lax.fori_loop(
            0,
            N_HOLD,
            time_step,
            (phase, voltage, jnp.asarray(0.0, dtype=jnp.float32)),
        )
        average_voltage = voltage_sum / (N_HOLD - BURN_INDEX)
        return (phase, voltage), average_voltage

    _, voltage_down = jax.lax.scan(
        bias_step,
        initial_state,
        current_down_jax,
    )
    return voltage_down[::-1]


simulate_batch = jax.jit(
    jax.vmap(
        simulate_down_sweep,
        in_axes=(0, 0),
    )
)


def score_grid(capacitance_pf, critical_current_na):
    capacitance = np.asarray(capacitance_pf, dtype=np.float32)
    critical_current = np.asarray(critical_current_na, dtype=np.float32)
    voltage_models = np.asarray(
        simulate_batch(
            jnp.asarray(np.log10(capacitance)),
            jnp.asarray(critical_current),
        )
    )
    residual = voltage_models[:, finite_exp] - voltage_exp[None, finite_exp]
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    return rmse, voltage_models


# The discontinuous objective is better mapped on a compact physical grid than
# handed to a gradient optimizer. Refine around the best grid point once.
capacitance_axis_pf = np.geomspace(0.001, 0.03, 9)
critical_current_axis_na = np.arange(360.0, 461.0, 20.0)
capacitance_grid_pf, critical_current_grid_na = np.meshgrid(
    capacitance_axis_pf,
    critical_current_axis_na,
    indexing="ij",
)
coarse_rmse, _ = score_grid(
    capacitance_grid_pf.ravel(),
    critical_current_grid_na.ravel(),
)
coarse_index = int(np.argmin(coarse_rmse))
coarse_capacitance_pf = float(capacitance_grid_pf.ravel()[coarse_index])
coarse_critical_current_na = float(
    critical_current_grid_na.ravel()[coarse_index]
)
capacitance_refined_pf = np.geomspace(
    coarse_capacitance_pf / 1.5,
    coarse_capacitance_pf * 1.5,
    7,
)
critical_current_refined_na = np.arange(
    coarse_critical_current_na - 15.0,
    coarse_critical_current_na + 15.1,
    5.0,
)
capacitance_refined_grid_pf, critical_current_refined_grid_na = np.meshgrid(
    capacitance_refined_pf,
    critical_current_refined_na,
    indexing="ij",
)
refined_rmse, refined_models = score_grid(
    capacitance_refined_grid_pf.ravel(),
    critical_current_refined_grid_na.ravel(),
)
best_index = int(np.argmin(refined_rmse))
rmse_mV = float(refined_rmse[best_index])
best_capacitance_pf = float(
    capacitance_refined_grid_pf.ravel()[best_index]
)
best_critical_current_na = float(
    critical_current_refined_grid_na.ravel()[best_index]
)
voltage_dynamic = refined_models[best_index]

normal_resistance_ohm = 1e6 / (
    (N_LOW * TAU_LOW + N_HIGH * TAU_HIGH) * float(sc.G0_muS)
)
beta_c = (
    4
    * np.pi
    * best_critical_current_na
    * 1e-9
    * normal_resistance_ohm**2
    * best_capacitance_pf
    * 1e-12
    / (float(sc.h_pVs) * 1e-12)
)

print(f"C = {1e3 * best_capacitance_pf:.6g} fF")
print(f"Ic = {best_critical_current_na:.6g} nA")
print(f"beta_c = {beta_c:.6g}")
print(f"dynamic voltage RMSE = {1e3 * rmse_mV:.6g} uV")

# Sample data and model in both directions using the established repository
# upsample -> bin_y_over_x workflow.
voltage_bins = np.linspace(-0.48, 0.48, 1201)
current_bins = np.linspace(-1680.0, 1680.0, 1201)


def sample_both_directions(current, voltage, factor):
    current_up, voltage_up = sc.upsample(current, voltage, factor=factor)
    current_over_voltage = sc.bin_y_over_x(
        current_up,
        voltage_up,
        voltage_bins,
    )
    voltage_over_current = sc.bin_y_over_x(
        voltage_up,
        current_up,
        current_bins,
    )
    return current_over_voltage, voltage_over_current


current_data_over_voltage, voltage_data_over_current = sample_both_directions(
    current_raw,
    voltage_raw,
    factor=10,
)
current_dynamic_over_voltage, voltage_dynamic_over_current = (
    sample_both_directions(
        current_grid[finite_exp],
        voltage_dynamic[finite_exp],
        factor=100,
    )
)

didv_data = (
    np.gradient(current_data_over_voltage, voltage_bins) / float(sc.G0_muS)
)
didv_dynamic = (
    np.gradient(current_dynamic_over_voltage, voltage_bins) / float(sc.G0_muS)
)
dvdi_data = (
    np.gradient(voltage_data_over_current, current_bins) * float(sc.G0_muS)
)
dvdi_dynamic = (
    np.gradient(voltage_dynamic_over_current, current_bins) * float(sc.G0_muS)
)

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
    markersize=1.2,
    alpha=0.5,
    label="corrected raw data",
    rasterized=True,
)
ax_iv.plot(
    voltage_bins,
    current_data_over_voltage,
    color="C0",
    linewidth=1.1,
    label=r"data: sampled $I(V)$",
)
ax_iv.plot(
    voltage_data_over_current,
    current_bins,
    color="C1",
    linewidth=1.1,
    label=r"data: sampled $V(I)$",
)
ax_iv.plot(
    voltage_bins,
    current_dynamic_over_voltage,
    color="C3",
    linestyle="--",
    linewidth=1.4,
    label=r"dynamic RCSJ-MAR: sampled $I(V)$",
)
ax_iv.plot(
    voltage_dynamic_over_current,
    current_bins,
    color="C3",
    linewidth=1.5,
    label=r"dynamic RCSJ-MAR: sampled $V(I)$",
)
ax_iv.set_ylabel(r"$I$ (nA)")
ax_iv.tick_params(labelbottom=False)
ax_iv.legend(frameon=False, fontsize="small", ncols=2)

ax_didv.plot(
    voltage_bins,
    didv_data,
    color="C0",
    linewidth=1.2,
    label=r"$\nabla_V$ data $I(V)$",
)
ax_didv.plot(
    voltage_bins,
    didv_dynamic,
    color="C3",
    linewidth=1.2,
    label=r"$\nabla_V$ dynamic $I(V)$",
)
ax_didv.set_xlabel(r"$V_J$ (mV)")
ax_didv.set_ylabel(r"$dI/dV$ ($G_0$)")
ax_didv.set_yscale("symlog", linthresh=10.0)
ax_didv.legend(frameon=False, fontsize="small")

ax_dvdi.plot(
    dvdi_data,
    current_bins,
    color="C1",
    linewidth=1.2,
    label=r"$\nabla_I$ data $V(I)$",
)
ax_dvdi.plot(
    dvdi_dynamic,
    current_bins,
    color="C3",
    linewidth=1.2,
    label=r"$\nabla_I$ dynamic $V(I)$",
)
ax_dvdi.set_xlabel(r"$dV/dI$ ($R_0$)")
ax_dvdi.tick_params(labelleft=False)
ax_dvdi.legend(frameon=False, fontsize="x-small")

for axis in (ax_iv, ax_didv, ax_dvdi):
    axis.grid(alpha=0.15)

output = HERE / "dynamic_rcsj_mar_fit"
fig.savefig(output.with_suffix(".png"), dpi=250)
fig.savefig(output.with_suffix(".pdf"))
plt.show()
