# init
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity.api as sc
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from nptdms import TdmsFile
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.special import logsumexp
from scipy.constants import Boltzmann as k_B
from scipy.constants import elementary_charge as e
from superconductivity.api import G0_muS, NDArray64
from superconductivity.style.thesislayout import save_figure
from superconductivity.utilities.functions.fill_nans import fill
from superconductivity.utilities.functions.upsampling import upsample
from tqdm import tqdm

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "atomic_contact")
)
from apply_sigmaV import apply_sigmaV

# get data single-iv ssn set

# file = "/Users/oliver/Documents/measurement data/SSET/22 02b Scheer2/65_6/SCMap.tdms"
file = "/Users/oliver/Documents/measurement data/SSET/22 02b Scheer2/65_6/NCMap.tdms"
# file = "/Users/oliver/Documents/measurement data/SSET/22 03 Scheer2/36/SC_map.tdms"

nu_downsample_Hz = 137
R_TB_OHM = 32.381e3
# Correct the reversed experimental gate wiring once, at readout level.
FIT_GATE_MAX_MV = 1.5
sweep_up = (75_000, 125_000)
sweep_down = (24000, 70000)

keys = []
values = []

with TdmsFile.open(file) as f:
    mkeys = f.groups()
    for mkey in mkeys:
        name = mkey.name
        if name not in ["ZeroOffsetAdjust", "Thermometry", "Mapped"]:
            keys.append(name)
            value = name.split("G")[1].split("V")[0]
            if value[-1] == "m":
                value = float(value[:-1]) * 1e-3
            elif value[-1] == "u":
                value = float(value[:-1]) * 1e-6
            else:
                value = 0
            value = np.round(value, 6)
            values.append(value)

with TdmsFile.open(file) as f:
    amp = (
        f.properties["s_ampl_sample_effective_gain"],
        f.properties["s_ampl_reference_effective_gain"],
    )
    rref = f.properties["s_ampl_resistor"]
    nu_source_Hz = f.properties["s_source_f"]
    dt_sample_s = f[keys[0]].properties["s_dt"]
    nu_sample_Hz = 1 / dt_sample_s

Sample_up_V = []
Reference_up_V = []
Sample_down_V = []
Reference_down_V = []

dt_downsample_s = 1 / nu_downsample_Hz

with TdmsFile.open(file) as f:
    for i, key in enumerate(tqdm(keys)):
        if True:
            sample_V = f[key]["Sample"].as_dataframe()
            reference_V = f[key]["Reference"].as_dataframe()
            sample_V = sample_V.values
            reference_V = reference_V.values
            sweep = sweep_up
            sample_V = sample_V[sweep[0] : sweep[1], 0]
            reference_V = reference_V[sweep[0] : sweep[1], 0]
            time_s = np.arange(len(sample_V)) * dt_sample_s
            nu_time_s = np.arange(0, np.max(time_s), dt_downsample_s)
            nu_sample_V = sc.bin_y_over_x((sample_V), (time_s), nu_time_s)
            nu_reference_V = sc.bin_y_over_x((reference_V), (time_s), nu_time_s)

            Sample_up_V.append(nu_sample_V)
            Reference_up_V.append(nu_reference_V)

            sample_V = f[key]["Sample"].as_dataframe()
            reference_V = f[key]["Reference"].as_dataframe()
            sample_V = sample_V.values
            reference_V = reference_V.values

            sweep = sweep_down
            sample_V = sample_V[sweep[0] : sweep[1], 0]
            reference_V = reference_V[sweep[0] : sweep[1], 0]
            time_s = np.arange(len(sample_V)) * dt_sample_s
            nu_time_s = np.arange(0, np.max(time_s), dt_downsample_s)
            nu_sample_V = sc.bin_y_over_x((sample_V), (time_s), nu_time_s)
            nu_reference_V = sc.bin_y_over_x((reference_V), (time_s), nu_time_s)

            Sample_down_V.append(nu_sample_V)
            Reference_down_V.append(nu_reference_V)

# get highres data
Vbias_mV = np.linspace(-0.75, 0.75, 601)
Ibias_nA = np.linspace(-11.25, 11.25, 601)
Vgate_mV = -np.array(values, dtype=np.float64) * 1e3

Iup_nA = np.zeros((Vgate_mV.shape[0], Vbias_mV.shape[0]))
Vup_mV = np.zeros((Vgate_mV.shape[0], Ibias_nA.shape[0]))
Idown_nA = np.zeros((Vgate_mV.shape[0], Vbias_mV.shape[0]))
Vdown_mV = np.zeros((Vgate_mV.shape[0], Ibias_nA.shape[0]))


for i, key in enumerate(tqdm(keys)):
    vup_mV = Sample_up_V[i] * 1e3 / amp[0]
    iup_nA = Reference_up_V[i] * 1e9 / (amp[1] * rref)
    vdown_mV = Sample_down_V[i] * 1e3 / amp[0]
    idown_nA = Reference_down_V[i] * 1e9 / (amp[1] * rref)

    valid = np.logical_and(np.isfinite(vup_mV), np.isfinite(iup_nA))

    iup_nA, vup_mV = upsample(iup_nA[valid]), upsample(vup_mV[valid])
    # iup_nA, vup_mV = (iup_nA[valid]), (vup_mV[valid])
    Iup_nA[i, :] = sc.bin_y_over_x(iup_nA, vup_mV, Vbias_mV)
    Vup_mV[i, :] = sc.bin_y_over_x(vup_mV, iup_nA, Ibias_nA)

    valid = np.logical_and(np.isfinite(vdown_mV), np.isfinite(idown_nA))
    idown_nA, vdown_mV = upsample(idown_nA[valid]), upsample(vdown_mV[valid])
    # idown_nA, vdown_mV = (idown_nA[valid]), (vdown_mV[valid])
    Idown_nA[i, :] = sc.bin_y_over_x(idown_nA, vdown_mV, Vbias_mV)
    Vdown_mV[i, :] = sc.bin_y_over_x(vdown_mV, idown_nA, Ibias_nA)

# Light smoothing after all traces have been placed on common physical axes.
# The last axis is bias voltage for I(V) and bias current for V(I).
Iup_nA = savgol_filter(Iup_nA, 11, 1, axis=-1)
Vup_mV = savgol_filter(Vup_mV, 11, 1, axis=-1)

# Discard the two bias endpoints. Savitzky-Golay filtering and numerical
# differentiation necessarily use one-sided boundary information there.
Vbias_mV = Vbias_mV[1:-1]
Iup_nA = Iup_nA[:, 1:-1]
Idown_nA = Idown_nA[:, 1:-1]


# Orthodox normal-state SET model and whole-map fit


@dataclass
class OrthodoxFit:
    """Parameters of the normal-state orthodox SET model."""

    R_TB_Ohm: float
    R_BJ_Ohm: float
    C_TB_F: float
    C_BJ_F: float
    C_gate_F: float
    n_gate_offset: float
    V_bias_offset_V: float
    I_offset_A: float
    sigma_V_V: float
    temperature_K: float
    cost: float


def _orthodox_rate(delta_energy_J, resistance_Ohm, temperature_K):
    """Return a sequential-tunnelling rate for a free-energy change.

    Parameters
    ----------
    delta_energy_J : ndarray
        Final minus initial free energy.
    resistance_Ohm : float
        Normal-state tunnel resistance.
    temperature_K : float
        Electron temperature.
    """
    x = np.asarray(delta_energy_J) / (k_B * temperature_K)
    # x/expm1(x) is regular at zero. Clipping avoids overflow without
    # changing any rate on the scale relevant to the stationary solution.
    x_clip = np.clip(x, -700.0, 700.0)
    ratio = np.where(
        np.abs(x_clip) < 1e-7,
        1.0 - x_clip / 2.0 + x_clip**2 / 12.0,
        x_clip / np.expm1(x_clip),
    )
    return k_B * temperature_K * ratio / (e**2 * resistance_Ohm)


def orthodox_current(
    V_bias_V,
    V_gate_V,
    R_TB_Ohm,
    R_BJ_Ohm,
    C_TB_F,
    C_BJ_F,
    C_gate_F,
    temperature_K,
    n_gate_offset=0.0,
    charge_states=12,
):
    """Calculate current through a normal SET using the master equation.

    The tunnel-barrier (TB) electrode is biased and the break-junction (BJ)
    electrode is grounded. Positive current is conventional current flowing
    from TB to BJ. Cotunnelling and environmental/Dynes broadening are not
    included.
    """
    V_bias_V, V_gate_V = np.broadcast_arrays(V_bias_V, V_gate_V)
    shape = V_bias_V.shape
    vb = V_bias_V.ravel()[:, None]
    vg = V_gate_V.ravel()[:, None]
    capacitance_F = C_TB_F + C_BJ_F + C_gate_F

    # Centre the finite charge-state window on the locally preferred charge.
    induced_charge = C_TB_F * vb + C_gate_F * vg + e * n_gate_offset
    centre = np.rint(induced_charge / e).astype(np.int64)
    offsets = np.arange(-charge_states, charge_states + 1)[None, :]
    n = centre + offsets

    def energy(number):
        return (-e * number + induced_charge) ** 2 / (2.0 * capacitance_F)

    energy_n = energy(n)
    add_energy = energy(n + 1) - energy_n
    remove_energy = energy(n - 1) - energy_n

    # '+' adds one excess electron to the island. The voltage-source work is
    # +eV when an electron leaves an electrode and -eV in the reverse event.
    tb_add = _orthodox_rate(
        add_energy + e * vb, R_TB_Ohm, temperature_K
    )
    tb_remove = _orthodox_rate(
        remove_energy - e * vb, R_TB_Ohm, temperature_K
    )
    bj_add = _orthodox_rate(add_energy, R_BJ_Ohm, temperature_K)
    bj_remove = _orthodox_rate(remove_energy, R_BJ_Ohm, temperature_K)

    birth = tb_add + bj_add
    death = tb_remove + bj_remove
    # A one-dimensional birth-death chain obeys detailed balance between
    # adjacent charge states even though each edge contains two reservoirs.
    tiny_rate = np.finfo(np.float64).tiny
    log_ratio = np.log(np.maximum(birth[:, :-1], tiny_rate)) - np.log(
        np.maximum(death[:, 1:], tiny_rate)
    )
    log_probability = np.concatenate(
        [np.zeros((vb.size, 1)), np.cumsum(log_ratio, axis=1)], axis=1
    )
    probability = np.exp(
        log_probability - logsumexp(log_probability, axis=1)[:, None]
    )
    current_A = e * np.sum(probability * (tb_remove - tb_add), axis=1)
    return current_A.reshape(shape)


def fitted_orthodox_current_nA(
    V_bias_mV,
    V_gate_mV,
    R_TB_Ohm=32.381e3,
    R_BJ_Ohm=24.152773306935982e3,
    C_TB_F=0.9573997639379382e-15,
    C_BJ_F=0.21377330962213133e-15,
    C_gate_F=37.32024371158558e-18,
    temperature_K=0.13255022905037123,
    n_gate_offset=0.21298823101042144,
    V_bias_offset_mV=-1.915359845039769e-5,
    I_offset_nA=-0.12531615250123791,
):
    """Evaluate the preferred fitted orthodox SET model.

    Parameters
    ----------
    V_bias_mV, V_gate_mV : array_like
        Bias and gate voltages in mV. Inputs follow NumPy broadcasting.
    R_TB_Ohm, R_BJ_Ohm : float, optional
        Tunnel-barrier and break-junction resistances in ohms.
    C_TB_F, C_BJ_F, C_gate_F : float, optional
        Junction and gate capacitances in farads.
    temperature_K : float, optional
        Electron temperature in kelvin.
    n_gate_offset : float, optional
        Dimensionless offset charge in units of the elementary charge.
    V_bias_offset_mV : float, optional
        Bias-voltage offset in mV.
    I_offset_nA : float, optional
        Current offset in nA.
    Returns
    -------
    numpy.ndarray
        Model current in nA. The defaults are from the up-sweep fit using
        ``V_gate < 1.5 mV``, fixed ``R_TB = 32.381 kOhm``, zero bias noise,
        and free electron temperature.
    """
    current_A = orthodox_current(
        (np.asarray(V_bias_mV) - V_bias_offset_mV) * 1e-3,
        np.asarray(V_gate_mV) * 1e-3,
        R_TB_Ohm,
        R_BJ_Ohm,
        C_TB_F,
        C_BJ_F,
        C_gate_F,
        temperature_K,
        n_gate_offset=n_gate_offset,
    )
    return current_A * 1e9 + I_offset_nA


def fitted_orthodox_map_nA():
    """Return the available experimental map and preferred fitted map.

    Returns
    -------
    Vbias_mV : numpy.ndarray
        One-dimensional retained bias-voltage axis in mV, excluding the two
        boundary-affected endpoints.
    Iexp_nA : numpy.ndarray
        Smoothed experimental up-sweep current in nA with shape
        ``(Vgate_mV.size, Vbias_mV.size)``.
    Ifit_nA : numpy.ndarray
        Preferred orthodox-model current in nA with the same shape. The rows
        correspond to the module-level ``Vgate_mV`` values.
    """
    bias_grid_mV, gate_grid_mV = np.meshgrid(Vbias_mV, Vgate_mV)
    fitted_nA = fitted_orthodox_current_nA(
        bias_grid_mV,
        gate_grid_mV,
    )
    return Vbias_mV.copy(), Iup_nA.copy(), fitted_nA


def fit_orthodox_map(
    V_bias_mV,
    V_gate_mV,
    current_nA,
    temperature_K=0.0966,
    gate_stride=2,
    bias_stride=4,
    fixed_R_TB_Ohm=None,
    fit_temperature=False,
    fit_bias_noise=False,
    gate_max_mV=None,
):
    """Robustly fit one sweep over the complete IV map."""
    gate_indices = np.arange(0, len(V_gate_mV), gate_stride)
    if gate_max_mV is not None:
        gate_indices = gate_indices[V_gate_mV[gate_indices] < gate_max_mV]
    bias_indices = np.arange(0, len(V_bias_mV), bias_stride)
    fit_bias_mV = V_bias_mV[bias_indices]
    fit_gate_mV = V_gate_mV[gate_indices]
    vb_grid_mV, vg_grid_mV = np.meshgrid(fit_bias_mV, fit_gate_mV)
    observed_nA = current_nA[np.ix_(gate_indices, bias_indices)]
    valid = np.isfinite(observed_nA)

    # Logarithms enforce positivity for the five physical parameters.
    initial = np.array(
        [32.381e3, 24.18e3, 0.947e-15, 0.229e-15, 0.0343e-15]
    )
    x0 = np.r_[
        np.log(initial),
        0.179,
        0.0,
        -0.127e-9,
        np.log(0.136),
        np.log(0.01),
    ]
    lower = np.r_[
        np.log([1e3, 1e3, 0.02e-15, 0.02e-15, 0.001e-15]),
        -1.0,
        -0.2e-3,
        -2e-9,
        np.log(0.015),
        np.log(1e-4),
    ]
    upper = np.r_[
        np.log([10e6, 10e6, 10e-15, 10e-15, 2e-15]),
        1.0,
        0.2e-3,
        2e-9,
        np.log(1.0),
        np.log(0.2),
    ]
    if fixed_R_TB_Ohm is not None:
        # least_squares requires strict bounds. This interval fixes R_TB to
        # much better precision than either the map or the external value.
        x0[0] = np.log(fixed_R_TB_Ohm)
        lower[0] = np.log(fixed_R_TB_Ohm * (1.0 - 1e-10))
        upper[0] = np.log(fixed_R_TB_Ohm * (1.0 + 1e-10))
    if not fit_temperature:
        lower[8] = np.log(temperature_K * (1.0 - 1e-10))
        upper[8] = np.log(temperature_K * (1.0 + 1e-10))
    else:
        # The electron temperature cannot meaningfully be inferred below the
        # recorded sample-thermometer temperature; bias noise and temperature
        # are otherwise strongly degenerate there.
        lower[8] = np.log(temperature_K)
    if not fit_bias_noise:
        lower[9] = np.log(1e-4 * (1.0 - 1e-10))
        upper[9] = np.log(1e-4 * (1.0 + 1e-10))
        x0[9] = np.log(1e-4)

    def residual(parameters):
        R_TB, R_BJ, C_TB, C_BJ, C_gate = np.exp(parameters[:5])
        electron_temperature_K = np.exp(parameters[8])
        sigma_V_mV = np.exp(parameters[9])
        theory_nA = 1e9 * orthodox_current(
            (vb_grid_mV * 1e-3) - parameters[6],
            vg_grid_mV * 1e-3,
            R_TB,
            R_BJ,
            C_TB,
            C_BJ,
            C_gate,
            electron_temperature_K,
            n_gate_offset=parameters[5],
        )
        if fit_bias_noise:
            theory_nA = apply_sigmaV(
                fit_bias_mV, theory_nA, sigma_V_mV, axis=-1
            )
        theory_nA = theory_nA + parameters[7] * 1e9
        # A robust outer loss suppresses the locally rearranged end of the map.
        # Work in nA numerically; otherwise the default gradient tolerance can
        # incorrectly accept the initial guess simply because SI currents are
        # of order 1e-9.
        return theory_nA[valid] - observed_nA[valid]

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale="jac",
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=500,
        verbose=1,
    )
    values = np.exp(result.x[:5])
    fit = OrthodoxFit(
        *values,
        n_gate_offset=result.x[5],
        V_bias_offset_V=result.x[6],
        I_offset_A=result.x[7],
        sigma_V_V=np.exp(result.x[9]) * 1e-3 if fit_bias_noise else 0.0,
        temperature_K=np.exp(result.x[8]),
        cost=result.cost,
    )
    return fit, result


def save_fit_diagnostics(
    fit, measured_nA, output_stem="nset_orthodox_fit"
):
    """Save fitted values and a measured/model/residual map figure."""
    output_stem = Path(output_stem)
    output_stem.with_suffix(".json").write_text(
        json.dumps(asdict(fit), indent=2) + "\n"
    )
    vb_grid, vg_grid = np.meshgrid(Vbias_mV, Vgate_mV)
    model_nA = 1e9 * (
        orthodox_current(
            (vb_grid * 1e-3) - fit.V_bias_offset_V,
            vg_grid * 1e-3,
            fit.R_TB_Ohm,
            fit.R_BJ_Ohm,
            fit.C_TB_F,
            fit.C_BJ_F,
            fit.C_gate_F,
            fit.temperature_K,
            fit.n_gate_offset,
        )
        + fit.I_offset_A
    )
    if fit.sigma_V_V > 0.0:
        model_nA = apply_sigmaV(
            Vbias_mV, model_nA, fit.sigma_V_V * 1e3, axis=-1
        )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    extent = [Vbias_mV[0], Vbias_mV[-1], Vgate_mV[0], Vgate_mV[-1]]
    vmax = np.nanpercentile(np.abs(measured_nA), 99)
    images = [measured_nA, model_nA, measured_nA - model_nA]
    titles = ["measurement", "orthodox fit", "residual"]
    limits = [vmax, vmax, np.nanpercentile(np.abs(images[2]), 99)]
    for axis, image_data, title, limit in zip(axes, images, titles, limits):
        image = axis.imshow(
            image_data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        axis.set_xlabel(r"$V_\mathrm{bias}$ (mV)")
        fig.colorbar(image, ax=axis, label="current (nA)")
    axes[0].set_ylabel(r"$V_\mathrm{gate}$ (mV)")
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def save_didv_diagnostics(
    fit, measured_nA, output_stem="nset_orthodox_fit_didv"
):
    """Save measured and fitted differential-conductance maps."""
    output_stem = Path(output_stem)
    vb_grid, vg_grid = np.meshgrid(Vbias_mV, Vgate_mV)
    model_nA = 1e9 * (
        orthodox_current(
            (vb_grid * 1e-3) - fit.V_bias_offset_V,
            vg_grid * 1e-3,
            fit.R_TB_Ohm,
            fit.R_BJ_Ohm,
            fit.C_TB_F,
            fit.C_BJ_F,
            fit.C_gate_F,
            fit.temperature_K,
            fit.n_gate_offset,
        )
        + fit.I_offset_A
    )
    if fit.sigma_V_V > 0.0:
        model_nA = apply_sigmaV(
            Vbias_mV, model_nA, fit.sigma_V_V * 1e3, axis=-1
        )
    delta_bias_mV = Vbias_mV[1] - Vbias_mV[0]

    # nA/mV is numerically equal to microSiemens. No smoothing is applied.
    didv_measured_uS = np.gradient(measured_nA, delta_bias_mV, axis=-1)
    didv_model_uS = np.gradient(model_nA, delta_bias_mV, axis=-1)
    residual_uS = didv_measured_uS - didv_model_uS

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    extent = [Vbias_mV[0], Vbias_mV[-1], Vgate_mV[0], Vgate_mV[-1]]
    conductance_limit = np.nanpercentile(
        didv_measured_uS, 99
    )
    residual_limit = np.nanpercentile(np.abs(residual_uS), 99)
    panels = [didv_measured_uS, didv_model_uS, residual_uS]
    titles = ["measurement up", "orthodox fit", "residual"]
    for index, (axis, panel, title) in enumerate(zip(axes, panels, titles)):
        if index < len(panels) - 1:
            image = axis.imshow(
                panel,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                vmin=0.0,
                vmax=conductance_limit,
            )
        else:
            image = axis.imshow(
                panel,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="coolwarm",
                vmin=-residual_limit,
                vmax=residual_limit,
            )
        axis.set_title(title)
        axis.set_xlabel(r"$V_\mathrm{bias}$ (mV)")
        fig.colorbar(image, ax=axis, label=r"$dI/dV$ ($\mu$S)")
    axes[0].set_ylabel(r"$V_\mathrm{gate}$ (mV)")
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def save_linecut_diagnostics(
    fit,
    measured_nA,
    gate_targets_mV=(0.0, -3.0),
    output_stem="nset_orthodox_fit_linecuts",
):
    """Save measured/model IV and differential-conductance line cuts."""
    output_stem = Path(output_stem)
    gate_indices = [
        int(np.argmin(np.abs(Vgate_mV - target)))
        for target in gate_targets_mV
    ]
    delta_bias_mV = Vbias_mV[1] - Vbias_mV[0]
    fig, axes = plt.subplots(
        2,
        len(gate_indices),
        figsize=(8, 6),
        sharex=True,
        constrained_layout=True,
    )
    for column, gate_index in enumerate(gate_indices):
        gate_mV = Vgate_mV[gate_index]
        measured_cut_nA = measured_nA[gate_index]
        model_cut_nA = 1e9 * (
            orthodox_current(
                (Vbias_mV * 1e-3) - fit.V_bias_offset_V,
                gate_mV * 1e-3,
                fit.R_TB_Ohm,
                fit.R_BJ_Ohm,
                fit.C_TB_F,
                fit.C_BJ_F,
                fit.C_gate_F,
                fit.temperature_K,
                fit.n_gate_offset,
            )
            + fit.I_offset_A
        )
        if fit.sigma_V_V > 0.0:
            model_cut_nA = apply_sigmaV(
                Vbias_mV,
                model_cut_nA,
                fit.sigma_V_V * 1e3,
                axis=-1,
            )
        measured_didv_uS = np.gradient(
            measured_cut_nA, delta_bias_mV
        )
        model_didv_uS = np.gradient(model_cut_nA, delta_bias_mV)

        axes[0, column].plot(
            Vbias_mV, measured_cut_nA, label="measurement", linewidth=1
        )
        axes[0, column].plot(
            Vbias_mV, model_cut_nA, label="orthodox fit", linewidth=2
        )
        axes[0, column].set_title(
            rf"$V_\mathrm{{gate}}={gate_mV:.1f}$ mV"
        )
        axes[1, column].plot(
            Vbias_mV, measured_didv_uS, linewidth=0.8
        )
        axes[1, column].plot(Vbias_mV, model_didv_uS, linewidth=2)
        axes[1, column].set_xlabel(r"$V_\mathrm{bias}$ (mV)")
    axes[0, 0].set_ylabel("current (nA)")
    axes[1, 0].set_ylabel(r"$dI/dV$ ($\mu$S)")
    axes[0, 0].legend()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__" and "--fit" in sys.argv:
    fixed_R_TB_Ohm = None if "--free-rtb" in sys.argv else R_TB_OHM
    if "--fixed-rtb" in sys.argv:
        index = sys.argv.index("--fixed-rtb")
        fixed_R_TB_Ohm = float(sys.argv[index + 1]) * 1e3
    fitted, optimizer_result = fit_orthodox_map(
        Vbias_mV,
        Vgate_mV,
        Iup_nA,
        fixed_R_TB_Ohm=fixed_R_TB_Ohm,
        fit_temperature="--fit-temperature" in sys.argv,
        fit_bias_noise="--fit-bias-noise" in sys.argv,
        gate_max_mV=FIT_GATE_MAX_MV,
    )
    print(json.dumps(asdict(fitted), indent=2))
    output_stem = (
        "nset_orthodox_fit_fixed_rtb_free_temperature_bias_noise"
        if fixed_R_TB_Ohm is not None
        and "--fit-temperature" in sys.argv
        and "--fit-bias-noise" in sys.argv
        else (
            "nset_orthodox_fit_fixed_rtb_free_temperature"
            if fixed_R_TB_Ohm is not None and "--fit-temperature" in sys.argv
            else (
                "nset_orthodox_fit_fixed_rtb_bias_noise"
                if fixed_R_TB_Ohm is not None
                and "--fit-bias-noise" in sys.argv
                else (
                    "nset_orthodox_fit_fixed_rtb"
                    if fixed_R_TB_Ohm is not None
                    else "nset_orthodox_fit"
                )
            )
        )
    )
    save_fit_diagnostics(fitted, Iup_nA, output_stem)
    save_didv_diagnostics(fitted, Iup_nA, output_stem + "_didv")
    save_linecut_diagnostics(
        fitted, Iup_nA, output_stem=output_stem + "_linecuts"
    )
