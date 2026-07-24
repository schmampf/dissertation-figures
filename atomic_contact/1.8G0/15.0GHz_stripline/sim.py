# 1.8G0, 15.0GHz, stripline
import sys
from pathlib import Path

import numpy as np
import superconductivity as sc
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from tqdm.auto import tqdm

atomic_contact_dir = Path(__file__).resolve().parents[2]
if str(atomic_contact_dir) not in sys.path:
    sys.path.insert(0, str(atomic_contact_dir))

from apply_sigmaV import apply_sigmaV
from superconductivity.utilities.functions.upsampling import upsample

path = Path(__file__).resolve().parent

data_eva = np.load(path / "eva.npz")
Vexp_mV = upsample(data_eva["Vbias_mV"])
Iexp_nA = upsample(data_eva["Iexp_nA"])
dIexp = upsample(data_eva["dGexp_G0"])

# data_mar = np.load(f"{path}/mar_data.npz")
# tau = data_mar["tau"]
# Delta_meV = data_mar["Delta_meV"]
# sigmaV_mV = data_mar["sigmaV_mV"]
# gamma_meV = data_mar["gamma_meV"]
# T_K = data_mar["T_K"]

# nu_GHz = data_eta["nu_GHz"]
# data_eta = np.load(f"{path}/eta_data.npz")
# Abias_mV = data_eta["Abias_mV"]

tau = 0.514, 0.402, 0.326, 0.295, 0.226
tau = tuple(transmission for transmission in tau if transmission > 0.0)
Delta_meV = 0.1895
sigmaV_mV = 0.016
sigmaV_map_mV = 0.008
gamma_meV = 1e-6
T_K = 0.0
nu_GHz = 15.0

# ``eta_calibrated`` is retained for the nominal vertical axis.  The measured
# dI/dV map is best reproduced by a smaller amplitude at the junction; this
# can represent attenuation between the calibration point and the contact.
eta = 0.0062212394165747035
Aout_mV = data_eva["Aout_mV"]
Abias_mV = eta * Aout_mV
Abias = eta * Aout_mV / (nu_GHz * sc.h_pVs)

Vbias = np.linspace(-6, 6, 1201)
Vbias_mV = Vbias * Delta_meV

Iexp_nA = sc.bin_y_over_x(Iexp_nA, Vexp_mV, Vbias_mV)
dIexp = sc.bin_y_over_x(dIexp, Vexp_mV, Vbias_mV)

Imar_nA, Ifcs_nA = sc.get_Imar_nA(
    V_mV=Vbias_mV,
    tau=tau,
    T_K=T_K,
    Delta_meV=Delta_meV,
    gamma_meV=gamma_meV,
    charge_resolved=True,
    show_progress=True,
)

# Keep the charge-resolved output only for comparison with the original PAMAR
# implementation.  The current weight-based model still uses the total curve.
pat_model = "threshold_resolved"

# Phenomenological zero-bias branch fitted in sc.py.  The microwave response
# is treated as photon-assisted 2e transport, giving J_n^2(2 A / hf) weights
# and voltage shifts n hf / (2e).
branch_current_nA = 0.98443224
branch_width_mV = 0.00331609
branch_decay_mV = 0.07540756
branch_bessel_exp = 2


def broadened_sc_current(V_mV, weight):
    """Return the zero-bias branch localized by its independent weight."""
    return branch_current_nA * np.tanh(V_mV / branch_width_mV) * weight


def get_threshold_currents(
    I_nA,
    V,
    m_max=6,
    background_width=0.18,
    use_smooth_background=False,
):
    """Split a DC MAR curve into smooth background and SGS-order currents."""
    conductance = np.gradient(I_nA, V)
    step = float(np.mean(np.diff(V)))
    if use_smooth_background:
        background = gaussian_filter1d(
            conductance,
            sigma=background_width / step,
            mode="nearest",
        )
        feature = conductance - background
    else:
        background = np.zeros_like(conductance)
        feature = conductance

    orders = np.arange(1, m_max + 1)
    # Construct weights down to the voltage-grid resolution.  Only orders up
    # to m_max are propagated; all higher orders are grouped as unattributed.
    weight_m_max = max(m_max + 1, int(np.ceil(2.0 / step - 1e-12)))
    construction_orders = np.arange(1, weight_m_max + 2)
    construction_centers = 2.0 / construction_orders
    centers = construction_centers[:weight_m_max]
    neighbor_spacings = construction_centers[:-1] - construction_centers[1:]
    lower_widths = neighbor_spacings[:weight_m_max].copy()
    higher_widths = np.empty_like(centers)
    higher_widths[0] = neighbor_spacings[0]
    higher_widths[1:] = neighbor_spacings[: weight_m_max - 1]
    lower_widths *= 0.45
    higher_widths *= 0.45
    distance = np.abs(V)[None, :] - centers[:, None]
    widths = np.where(
        distance < 0.0,
        lower_widths[:, None],
        higher_widths[:, None],
    )
    weights = np.exp(-0.5 * (distance / widths) ** 2)

    # Complete the partition below the final grid-resolved threshold with one
    # smooth central overflow basis.
    transition = 0.25 * neighbor_spacings[weight_m_max - 1]
    exponent = (np.abs(V) - centers[-1]) / transition
    higher_order_weight = 1.0 / (1.0 + np.exp(np.clip(exponent, -700, 700)))
    sc_weight = 1.0 / (1.0 + (V / (branch_decay_mV / Delta_meV)) ** 2)
    if use_smooth_background:
        normalization = np.sum(weights, axis=0) + higher_order_weight
        weights /= normalization[None, :]
        higher_order_weight /= normalization
    else:
        # Normalize the resolved orders, ghost, and central overflow together.
        normalization = np.sum(weights, axis=0) + higher_order_weight
        weights /= normalization[None, :]
        higher_order_weight /= normalization

    unattributed_weight = np.sum(weights[m_max:], axis=0) + higher_order_weight
    weights = weights[:m_max]

    feature_conductances = feature[None, :] * weights
    feature_currents = []
    zero_index = int(np.argmin(np.abs(V)))
    for feature_conductance in feature_conductances:
        current = cumulative_trapezoid(feature_conductance, V, initial=0.0)
        current -= current[zero_index]
        feature_currents.append(current)
    feature_currents = np.asarray(feature_currents)
    if use_smooth_background:
        background_current = I_nA - np.sum(feature_currents, axis=0)
    else:
        background_current = np.zeros_like(I_nA)
    return (
        orders,
        background_current,
        feature_currents,
        background,
        feature_conductances,
        weights,
        unattributed_weight,
        sc_weight,
    )


(
    threshold_orders,
    Ibackground_nA,
    Ithreshold_nA,
    Gbackground_nA,
    Gthreshold_nA,
    threshold_weights,
    unattributed_weight,
    sc_weight,
) = get_threshold_currents(Imar_nA, Vbias)
pat_progress = tqdm(
    total=threshold_orders.size + 1,
    desc="PAT components",
    unit="kernel",
)
Ipamar0_nA = np.zeros((Abias_mV.size, Vbias_mV.size))
for order, current_nA in zip(threshold_orders, Ithreshold_nA):
    Ipamar0_nA += sc.pat_kernel(
        V_mV=Vbias_mV,
        I_=current_nA,
        A_mV=Abias_mV,
        nu_GHz=nu_GHz,
        m=int(order),
        n_max=1_000,
    )
    pat_progress.update()

Ipamar_mar_nA = Ipamar0_nA
Isc0_nA = broadened_sc_current(Vbias_mV, sc_weight)
Ipa_sc_nA = sc.pat_kernel(
    V_mV=Vbias_mV,
    I_=Isc0_nA,
    A_mV=Abias_mV,
    nu_GHz=nu_GHz,
    m=2,
    exp=branch_bessel_exp,
    n_max=1_000,
)
pat_progress.update()
pat_progress.close()

# The fitted voltage noise is part of the experimental resolution.  Broaden
# the simulated current before taking derivatives; broadening a derivative
# afterwards is substantially more sensitive to grid-scale structure.
Ipamar_mar_nA = apply_sigmaV(
    V_mV=Vbias_mV,
    I_nA=Ipamar_mar_nA,
    sigmaV_mV=sigmaV_map_mV,
    axis=-1,
)
# The fitted branch width already contains the static experimental rounding,
# so applying sigmaV_map a second time to this component would double count it.
Ipamar0_nA = Ipamar_mar_nA + Ipa_sc_nA

# Original PAMAR construction: propagate the microscopic charge-resolved
# currents I_m with their corresponding effective-charge PAT kernels.
legacy_orders = np.arange(1, Ifcs_nA.shape[1] + 1)
Ipamar_legacy_nA = np.zeros((Abias_mV.size, Vbias_mV.size))
legacy_progress = tqdm(
    total=legacy_orders.size,
    desc="legacy charge-resolved PAT",
    unit="kernel",
)
for order, current_nA in zip(legacy_orders, Ifcs_nA.T):
    Ipamar_legacy_nA += sc.pat_kernel(
        V_mV=Vbias_mV,
        I_=current_nA,
        A_mV=Abias_mV,
        nu_GHz=nu_GHz,
        m=int(order),
        n_max=1_000,
    )
    legacy_progress.update()
legacy_progress.close()
Ipamar_legacy_nA = apply_sigmaV(
    V_mV=Vbias_mV,
    I_nA=Ipamar_legacy_nA,
    sigmaV_mV=sigmaV_map_mV,
    axis=-1,
)

# Compare like with like: dimensionless current I/(Delta G0) as a function of
# dimensionless voltage V/Delta.  Keep the nA arrays above for plotting in
# physical units.
current_scale_nA = Delta_meV * sc.G0_muS
Iexp = Iexp_nA / current_scale_nA
Ipamar0 = Ipamar0_nA / (Delta_meV * sc.G0_muS)
Ipamar_mar = Ipamar_mar_nA / current_scale_nA
Ipa_sc = Ipa_sc_nA / current_scale_nA
Ipamar_legacy = Ipamar_legacy_nA / current_scale_nA
Vbias = Vbias_mV / Delta_meV

# d[I/(Delta G0)]/d[V/Delta] = (dI/dV)/G0.  Use the directly
# measured lock-in conductance rather than differentiating the reconstructed
# experimental current, which needlessly amplifies its small integration and
# interpolation errors.
ddIexp = np.gradient(dIexp, Vbias, axis=-1)
dIpamar0 = np.gradient(Ipamar0, Vbias, axis=-1)
dIpamar_mar = np.gradient(Ipamar_mar, Vbias, axis=-1)
dIpa_sc = np.gradient(Ipa_sc, Vbias, axis=-1)
dIpamar_legacy = np.gradient(Ipamar_legacy, Vbias, axis=-1)
ddIpamar0 = np.gradient(dIpamar0, Vbias, axis=-1)

# I/V is undefined exactly at zero bias.  Use the differential conductance
# there (the limiting value for an odd, locally linear IV curve) rather than
# creating an infinity that contaminates neighbouring gradient samples.
zero_bias = np.flatnonzero(np.isclose(Vbias, 0.0, atol=1e-14, rtol=0.0))
Iexp_for_static = Iexp.copy()
Ipamar_for_static = Ipamar0.copy()
Ipamar_mar_for_static = Ipamar_mar.copy()
Ipa_sc_for_static = Ipa_sc.copy()
Ipamar_legacy_for_static = Ipamar_legacy.copy()
if zero_bias.size:
    Iexp_for_static -= Iexp[..., zero_bias]
    Ipamar_for_static -= Ipamar0[..., zero_bias]
    Ipamar_mar_for_static -= Ipamar_mar[..., zero_bias]
    Ipa_sc_for_static -= Ipa_sc[..., zero_bias]
    Ipamar_legacy_for_static -= Ipamar_legacy[..., zero_bias]
Gpamar0 = np.divide(
    Ipamar_for_static,
    Vbias,
    out=np.full_like(Ipamar0, np.nan),
    where=Vbias != 0.0,
)
Gexp = np.divide(
    Iexp_for_static,
    Vbias,
    out=np.full_like(Iexp, np.nan),
    where=Vbias != 0.0,
)
Gpamar_mar = np.divide(
    Ipamar_mar_for_static,
    Vbias,
    out=np.full_like(Ipamar_mar, np.nan),
    where=Vbias != 0.0,
)
Gsc = np.divide(
    Ipa_sc_for_static,
    Vbias,
    out=np.full_like(Ipa_sc, np.nan),
    where=Vbias != 0.0,
)
Gpamar_legacy = np.divide(
    Ipamar_legacy_for_static,
    Vbias,
    out=np.full_like(Ipamar_legacy, np.nan),
    where=Vbias != 0.0,
)
if zero_bias.size:
    Gpamar0[..., zero_bias] = dIpamar0[..., zero_bias]
    Gexp[..., zero_bias] = dIexp[..., zero_bias]
    Gpamar_mar[..., zero_bias] = dIpamar_mar[..., zero_bias]
    Gsc[..., zero_bias] = dIpa_sc[..., zero_bias]
    Gpamar_legacy[..., zero_bias] = dIpamar_legacy[..., zero_bias]

dGpamar0 = np.gradient(Gpamar0, Vbias, axis=-1)
dGexp = np.gradient(Gexp, Vbias, axis=-1)

# dGpamar_G0 = sc.bin_y_over_x(upsample(dGpamar_G0), upsample(Vbias0_mV), Vexp0_mV)

# dGpamar_G0 = apply_sigmaV(V_mV=Vbias0_mV, I_nA=dGpamar_G0, sigmaV_mV=sigmaV_mV, axis=-1)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plot_limits = (-3.0, 3.0)
color_limits = (0.0, 6.0)

fig_maps, axes_maps = plt.subplots(
    1,
    3,
    num=1,
    figsize=(16.0, 4.5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
conductance_maps = (
    (dIpamar0, "weighted PAMAR + SC"),
    (dIexp, "experiment"),
    (dIpamar_legacy, "charge-resolved PAMAR"),
)
for axis, (conductance_map, title) in zip(axes_maps, conductance_maps):
    mesh_maps = axis.pcolormesh(
        Vbias,
        Abias,
        conductance_map,
        shading="auto",
        vmin=color_limits[0],
        vmax=color_limits[1],
    )
    axis.set(
        title=title,
        xlabel=r"$eV/\Delta$",
        xlim=plot_limits,
    )
axes_maps[0].set_ylabel(r"nominal $A/hf$")
fig_maps.colorbar(
    mesh_maps,
    ax=axes_maps,
    label=r"$dI/dV\,/\,G_0$",
)

fig_static_maps, axes_static_maps = plt.subplots(
    1,
    3,
    num=2,
    figsize=(16.0, 4.5),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
static_maps = (
    (Gpamar0, "weighted PAMAR + SC"),
    (Gexp, "experiment"),
    (Gpamar_legacy, "charge-resolved PAMAR"),
)
for axis, (static_map, title) in zip(axes_static_maps, static_maps):
    mesh_static_maps = axis.pcolormesh(
        Vbias,
        Abias,
        static_map,
        shading="auto",
        vmin=color_limits[0],
        vmax=color_limits[1],
    )
    axis.set(
        title=title,
        xlabel=r"$eV/\Delta$",
        xlim=plot_limits,
    )
axes_static_maps[0].set_ylabel(r"nominal $A/hf$")
fig_static_maps.colorbar(
    mesh_static_maps,
    ax=axes_static_maps,
    label=r"$(I/V)\,/\,G_0$",
)

# Infer normalized equilibrium P(E) shapes from the phenomenological static
# ICPT branch.  The unknown E_J**2 prefactor cancels upon normalization.
energy_ueV = 2_000.0 * Vbias_mV
abs_energy_ueV = np.abs(energy_ueV)
abs_voltage_mV = abs_energy_ueV / 2_000.0
Isc_abs_nA = (
    branch_current_nA
    * np.tanh(abs_voltage_mV / branch_width_mV)
    / (1.0 + (abs_voltage_mV / branch_decay_mV) ** 2)
)
kB_ueV_K = 86.17333262145
temperatures_mK = np.arange(0.0, 601.0, 100.0)

fig_pe, ax_pe = plt.subplots(
    num=5,
    figsize=(7.0, 4.5),
    constrained_layout=True,
)
pe_colors = plt.cm.plasma(np.linspace(0.05, 0.9, temperatures_mK.size))
pe_curves = []
for temperature_mK, color in zip(temperatures_mK, pe_colors):
    if temperature_mK == 0.0:
        pe = np.where(energy_ueV > 0.0, Isc_abs_nA, 0.0)
    else:
        kT_ueV = kB_ueV_K * temperature_mK * 1e-3
        denominator = -np.expm1(-abs_energy_ueV / kT_ueV)
        pe_positive = np.divide(
            Isc_abs_nA,
            denominator,
            out=np.zeros_like(Isc_abs_nA),
            where=denominator != 0.0,
        )
        zero_energy = np.isclose(abs_energy_ueV, 0.0)
        pe_positive[zero_energy] = (
            branch_current_nA * kT_ueV / (2_000.0 * branch_width_mV)
        )
        pe = np.where(
            energy_ueV >= 0.0,
            pe_positive,
            pe_positive * np.exp(-abs_energy_ueV / kT_ueV),
        )

    normalization = np.trapezoid(pe, energy_ueV)
    pe /= normalization
    pe_curves.append(pe.copy())
    ax_pe.plot(
        energy_ueV,
        pe,
        color=color,
        label=rf"{temperature_mK:.0f} mK",
    )

ax_pe.set(
    title=r"$P(E)$ inferred from the phenomenological ICPT branch",
    xlabel=r"$E\;(\mu\mathrm{eV})$",
    ylabel=r"$P(E)\;(\mu\mathrm{eV})^{-1}$",
    xlim=(-500.0, 500.0),
    ylim=(0.0, None),
)
ax_pe.grid(alpha=0.2)
ax_pe.legend(ncols=2, fontsize="small")


def effective_impedance_from_pe(energy, pe):
    """Return a regularized effective Re[Z_t(f)] inferred from P(E)."""
    elementary_charge_C = 1.602176634e-19
    planck_Js = 6.62607015e-34
    resistance_quantum_ohm = planck_Js / (2.0 * elementary_charge_C) ** 2
    energy_J = energy * 1e-6 * elementary_charge_C
    energy_step_J = float(np.mean(np.diff(energy_J)))
    energy_step_ueV = float(np.mean(np.diff(energy)))

    characteristic = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pe))) * energy_step_ueV
    time_s = np.fft.fftshift(np.fft.fftfreq(energy.size, d=energy_step_J)) * planck_Js
    positive_time = time_s >= 0.0
    time_s = time_s[positive_time]
    characteristic = characteristic[positive_time]

    # Stop before the Fourier transform becomes too small for a stable phase.
    reliable = np.abs(characteristic) > 1e-4
    unreliable = np.flatnonzero(~reliable)
    stop = unreliable[0] if unreliable.size else time_s.size
    stop = min(time_s.size, max(stop, 16))
    time_s = time_s[:stop]
    phase = np.unwrap(np.angle(characteristic[:stop]))
    phase -= phase[0]
    phase_derivative = np.gradient(phase, time_s)

    # Taper only the final 20% to suppress finite-window ringing.
    taper = np.ones_like(time_s)
    taper_start = int(0.8 * time_s.size)
    taper_coordinate = np.linspace(0.0, 1.0, time_s.size - taper_start)
    taper[taper_start:] = 0.5 * (1.0 + np.cos(np.pi * taper_coordinate))

    frequency_Hz = np.linspace(0.0, 100e9, 401)
    cosine = np.cos(2.0 * np.pi * time_s[:, None] * frequency_Hz[None, :])
    impedance_ohm = (
        -resistance_quantum_ohm
        / np.pi
        * np.trapezoid(
            phase_derivative[:, None] * taper[:, None] * cosine,
            time_s,
            axis=0,
        )
    )
    return frequency_Hz, impedance_ohm


fig_impedance, ax_impedance = plt.subplots(
    num=8,
    figsize=(7.0, 4.5),
    constrained_layout=True,
)
impedance_curves_ohm = []
impedance_lines = []
for temperature_mK, color, pe in zip(
    temperatures_mK,
    pe_colors,
    pe_curves,
):
    frequency_Hz, impedance_ohm = effective_impedance_from_pe(
        energy_ueV,
        pe,
    )
    impedance_curves_ohm.append(impedance_ohm.copy())
    (impedance_line,) = ax_impedance.plot(
        frequency_Hz[1:] * 1e-9,
        impedance_ohm[1:] * 1e-3,
        color=color,
        label=rf"{temperature_mK:.0f} mK",
    )
    impedance_lines.append(impedance_line)

normal_resistance_kohm = 1_000.0 / (sc.G0_muS * np.sum(tau))
normal_resistance_ohm = normal_resistance_kohm * 1_000.0


def parallel_rc_impedance_magnitude(frequency, resistance_ohm, capacitance_F):
    """Return |Z| for a parallel RC environment."""
    return resistance_ohm / np.sqrt(
        1.0 + (2.0 * np.pi * frequency * resistance_ohm * capacitance_F) ** 2
    )


zero_temperature_impedance = impedance_curves_ohm[0]
effective_resistance_ohm = float(zero_temperature_impedance[0])
fit_mask = (frequency_Hz >= 50e9) & (zero_temperature_impedance > 0.0)


def parallel_rc_log_residual(log_capacitance):
    capacitance_F = np.exp(log_capacitance[0])
    model = parallel_rc_impedance_magnitude(
        frequency_Hz[fit_mask],
        effective_resistance_ohm,
        capacitance_F,
    )
    return np.log(model) - np.log(zero_temperature_impedance[fit_mask])


rc_fit = least_squares(
    parallel_rc_log_residual,
    x0=np.log([3.5e-15]),
    bounds=(np.log([1e-18]), np.log([1e-12])),
)
effective_capacitance_F = float(np.exp(rc_fit.x[0]))
cutoff_frequency_Hz = 1.0 / (
    2.0 * np.pi * effective_resistance_ohm * effective_capacitance_F
)
ax_impedance.scatter(
    frequency_Hz[fit_mask] * 1e-9,
    zero_temperature_impedance[fit_mask] * 1e-3,
    s=14,
    facecolors="none",
    edgecolors=pe_colors[0],
    linewidths=0.7,
    label="0 mK points used for fit",
)
ax_impedance.axhline(
    effective_resistance_ohm * 1e-3,
    color=pe_colors[0],
    ls=":",
    lw=1.1,
    label=rf"$Z_{{0\,\mathrm{{mK}}}}(0)={effective_resistance_ohm * 1e-3:.2f}$ k$\Omega$",
)
ax_impedance.plot(
    frequency_Hz[1:] * 1e-9,
    parallel_rc_impedance_magnitude(
        frequency_Hz[1:],
        effective_resistance_ohm,
        effective_capacitance_F,
    )
    * 1e-3,
    color=pe_colors[0],
    ls="--",
    lw=1.5,
    label=(
        rf"0 mK fit: $R={effective_resistance_ohm * 1e-3:.2f}$ k$\Omega$, "
        rf"$C={effective_capacitance_F * 1e15:.2f}$ fF"
    ),
)
print(
    "T=0 two-limit parallel-RC magnitude estimate: "
    f"R={effective_resistance_ohm * 1e-3:.3f} kOhm, "
    f"C={effective_capacitance_F * 1e15:.3f} fF, "
    f"f_c={cutoff_frequency_Hz * 1e-9:.3f} GHz"
)
ax_impedance.axhline(
    normal_resistance_kohm,
    color="k",
    ls="--",
    lw=1.2,
    label=rf"$R_N={normal_resistance_kohm:.2f}\,\mathrm{{k}}\Omega$",
)
ax_impedance.set(
    title=r"effective impedance with a two-limit 0-mK $|Z_{R\parallel C}|$ fit",
    xlabel=r"$f\;(\mathrm{GHz})$",
    ylabel=r"effective $Z\;(\mathrm{k}\Omega)$",
    xlim=(frequency_Hz[1] * 1e-9, 100.0),
)
ax_impedance.set_xscale("log")
ax_impedance.set_yscale("log")
ax_impedance.grid(alpha=0.2)
ax_impedance.legend(ncols=2, fontsize="small")


def pe_from_parallel_rc_zero_temperature(resistance_ohm, capacitance_F):
    """Return a zero-temperature P(E) for a parallel RC environment."""
    elementary_charge_C = 1.602176634e-19
    planck_Js = 6.62607015e-34
    hbar_Js = planck_Js / (2.0 * np.pi)
    pair_resistance_quantum_ohm = planck_Js / (2.0 * elementary_charge_C) ** 2

    sample_count = 2**14
    time_step_s = 0.5e-12
    angular_frequency = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            sample_count,
            d=time_step_s,
        )
    )
    angular_step = 2.0 * np.pi / (sample_count * time_step_s)
    positive_frequency = angular_frequency > 0.0
    real_impedance = np.zeros_like(angular_frequency)
    real_impedance[positive_frequency] = resistance_ohm / (
        1.0
        + (angular_frequency[positive_frequency] * resistance_ohm * capacitance_F) ** 2
    )

    # At T=0, J(t)=2 int_0^inf dω Re[Z]/(R_Q ω)
    # * (exp(-iωt)-1).  Evaluate the integral on the FFT grid.
    phase_spectrum = np.zeros(sample_count, dtype=complex)
    phase_spectrum[positive_frequency] = (
        2.0
        * real_impedance[positive_frequency]
        / pair_resistance_quantum_ohm
        / angular_frequency[positive_frequency]
        * angular_step
    )
    phase_correlation = np.fft.fft(phase_spectrum)
    phase_correlation -= phase_correlation[0]

    characteristic = np.exp(phase_correlation)
    pe_per_J = (
        np.fft.ifft(characteristic)
        * sample_count
        * time_step_s
        / (2.0 * np.pi * hbar_Js)
    )
    energy_J = hbar_Js * angular_frequency
    energy_ueV = energy_J / elementary_charge_C * 1e6
    energy_ueV = np.fft.fftshift(energy_ueV)
    pe_per_ueV = np.fft.fftshift(pe_per_J.real) * (elementary_charge_C * 1e-6)
    pe_per_ueV = np.maximum(pe_per_ueV, 0.0)
    pe_per_ueV /= np.trapezoid(pe_per_ueV, energy_ueV)
    return energy_ueV, pe_per_ueV


pair_energy_ueV = 2_000.0 * Vbias_mV
Isc_exp_nA = Iexp_nA[0] - Imar_nA
Isc_exp_nA = 0.5 * (Isc_exp_nA - Isc_exp_nA[::-1])
icpt_fit_window = np.abs(Vbias_mV) <= 0.040
icpt_fit_point_count = int(np.count_nonzero(icpt_fit_window))


def icpt_shape_from_parallel_rc(resistance_ohm, capacitance_F):
    """Return P(2eV)-P(-2eV) on the experimental voltage grid."""
    pe_energy_ueV, pe_per_ueV = pe_from_parallel_rc_zero_temperature(
        resistance_ohm,
        capacitance_F,
    )
    pe_forward = np.interp(
        pair_energy_ueV,
        pe_energy_ueV,
        pe_per_ueV,
        left=0.0,
        right=0.0,
    )
    pe_backward = np.interp(
        -pair_energy_ueV,
        pe_energy_ueV,
        pe_per_ueV,
        left=0.0,
        right=0.0,
    )
    return pe_forward - pe_backward, pe_energy_ueV, pe_per_ueV


initial_icpt_shape, _, _ = icpt_shape_from_parallel_rc(
    effective_resistance_ohm,
    effective_capacitance_F,
)
initial_icpt_amplitude = np.dot(
    initial_icpt_shape[icpt_fit_window],
    Isc_exp_nA[icpt_fit_window],
) / np.dot(
    initial_icpt_shape[icpt_fit_window],
    initial_icpt_shape[icpt_fit_window],
)


def icpt_current_residual(log_parameters):
    resistance_ohm, capacitance_F, amplitude_nA_ueV = np.exp(log_parameters)
    shape, _, _ = icpt_shape_from_parallel_rc(
        resistance_ohm,
        capacitance_F,
    )
    return amplitude_nA_ueV * shape[icpt_fit_window] - Isc_exp_nA[icpt_fit_window]


icpt_lower_bounds = np.log([10.0, 1e-18, 1e-6])
icpt_upper_bounds = np.log([1e6, 1e-12, 1e6])
icpt_initial_guess = np.log(
    [
        effective_resistance_ohm,
        effective_capacitance_F,
        max(initial_icpt_amplitude, 1e-6),
    ]
)

# R, C, and the E_J**2 amplitude can be strongly correlated over the narrow
# voltage window.  Keep the informed start, then probe the full allowed
# parameter volume with reproducible log-uniform random restarts.
icpt_random_restarts = 30
icpt_rng = np.random.default_rng(20260724)
icpt_starting_points = [icpt_initial_guess]
icpt_starting_points.extend(
    icpt_rng.uniform(
        icpt_lower_bounds,
        icpt_upper_bounds,
        size=(icpt_random_restarts, 3),
    )
)
icpt_fits = []
for start_index, starting_point in enumerate(
    tqdm(icpt_starting_points, desc="direct ICPT fit restarts")
):
    try:
        candidate_fit = least_squares(
            icpt_current_residual,
            x0=starting_point,
            bounds=(icpt_lower_bounds, icpt_upper_bounds),
            max_nfev=100,
        )
    except (FloatingPointError, ValueError):
        continue
    if np.isfinite(candidate_fit.cost):
        icpt_fits.append((candidate_fit.cost, start_index, candidate_fit))

if not icpt_fits:
    raise RuntimeError("All direct ICPT fit restarts failed.")

_, icpt_best_start_index, icpt_fit = min(
    icpt_fits,
    key=lambda fit_result: fit_result[0],
)
(
    icpt_resistance_ohm,
    icpt_capacitance_F,
    icpt_amplitude_nA_ueV,
) = np.exp(icpt_fit.x)
icpt_shape_per_ueV, pe_rc_energy_ueV, pe_rc_per_ueV = icpt_shape_from_parallel_rc(
    icpt_resistance_ohm,
    icpt_capacitance_F,
)
Isc_rc_nA = icpt_amplitude_nA_ueV * icpt_shape_per_ueV
icpt_rms_nA = np.sqrt(
    np.mean((Isc_rc_nA[icpt_fit_window] - Isc_exp_nA[icpt_fit_window]) ** 2)
)
elementary_charge_C = 1.602176634e-19
hbar_Js = 1.054571817e-34
icpt_prefactor_AJ = icpt_amplitude_nA_ueV * 1e-9 * 1e-6 * elementary_charge_C
josephson_energy_J = np.sqrt(
    icpt_prefactor_AJ * hbar_Js / (np.pi * elementary_charge_C)
)
inferred_critical_current_A = 2.0 * elementary_charge_C * josephson_energy_J / hbar_Js
negative_pe_weight = np.trapezoid(
    pe_rc_per_ueV[pe_rc_energy_ueV < 0.0],
    pe_rc_energy_ueV[pe_rc_energy_ueV < 0.0],
)
print(
    "direct ICPT I(V) fit: "
    f"best start={icpt_best_start_index}/"
    f"{icpt_random_restarts}, "
    f"successful starts={len(icpt_fits)}/"
    f"{len(icpt_starting_points)}, "
    f"R={icpt_resistance_ohm * 1e-3:.6g} kOhm, "
    f"C={icpt_capacitance_F * 1e15:.6g} fF, "
    f"amplitude={icpt_amplitude_nA_ueV:.6g} nA ueV, "
    f"RMS={icpt_rms_nA:.6g} nA, "
    f"I_c={inferred_critical_current_A * 1e9:.6g} nA, "
    f"negative-P(E) weight={negative_pe_weight:.3e}"
)
print(f"ICPT fit window: {icpt_fit_point_count} points at " r"|V| <= 0.040 mV")

Isc_exp = Isc_exp_nA / current_scale_nA
Isc_rc = Isc_rc_nA / current_scale_nA
Isc_phenomenological = Isc0_nA / current_scale_nA
dIsc_exp = np.gradient(Isc_exp, Vbias)
dIsc_rc = np.gradient(Isc_rc, Vbias)
dIsc_phenomenological = np.gradient(Isc_phenomenological, Vbias)

fig_icpt_forward, (ax_icpt_static, ax_icpt_differential) = plt.subplots(
    2,
    1,
    num=9,
    figsize=(7.0, 6.0),
    sharex=True,
    constrained_layout=True,
)
ax_icpt_static.plot(
    Vbias,
    Isc_exp_nA,
    color="tab:blue",
    lw=1.3,
    label="MAR-subtracted experiment",
)
ax_icpt_static.plot(
    Vbias[icpt_fit_window],
    Isc_exp_nA[icpt_fit_window],
    linestyle="none",
    marker="o",
    ms=3.5,
    color="tab:blue",
    label=rf"fit samples ($N={icpt_fit_point_count}$)",
)
ax_icpt_static.plot(
    Vbias,
    Isc_rc_nA,
    color="tab:orange",
    lw=1.5,
    label=r"$P(E)$ from fitted $R\parallel C$",
)
ax_icpt_static.plot(
    Vbias,
    Isc0_nA,
    color="tab:green",
    lw=1.5,
    ls="--",
    label=(r"phenomenological: " r"$I_0\tanh(V/V_w)/[1+(V/V_c)^2]$"),
)
ax_icpt_differential.plot(
    Vbias,
    dIsc_exp,
    color="tab:blue",
    lw=1.3,
)
ax_icpt_differential.plot(
    Vbias,
    dIsc_rc,
    color="tab:orange",
    lw=1.5,
)
ax_icpt_differential.plot(
    Vbias,
    dIsc_phenomenological,
    color="tab:green",
    lw=1.5,
    ls="--",
)
ax_icpt_static.set(
    title=(
        rf"direct ICPT fit: $R={icpt_resistance_ohm * 1e-3:.2f}$ k$\Omega$, "
        rf"$C={icpt_capacitance_F * 1e15:.2f}$ fF, "
        rf"$I_c={inferred_critical_current_A * 1e9:.2f}$ nA"
    ),
    ylabel=r"$I_\mathrm{SC}\;(\mathrm{nA})$",
    xlim=(-0.5, 0.5),
)
ax_icpt_differential.set(
    xlabel=r"$eV/\Delta$",
    ylabel=r"$dI_\mathrm{SC}/dV\,/\,G_0$",
)
for axis in (ax_icpt_static, ax_icpt_differential):
    axis.axhline(0.0, color="0.7", lw=0.7)
    axis.grid(alpha=0.2)
ax_icpt_static.legend(fontsize="small")

if pat_model == "threshold_resolved":
    fig_components, (ax_static, ax_conductance, ax_weight) = plt.subplots(
        3,
        1,
        num=3,
        figsize=(7.0, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    component_colors = plt.cm.viridis(np.linspace(0.08, 0.92, threshold_orders.size))
    Ithreshold = Ithreshold_nA / current_scale_nA
    dIthreshold = Gthreshold_nA / current_scale_nA
    Ibackground = Ibackground_nA / current_scale_nA
    dIbackground = Gbackground_nA / current_scale_nA
    Isc = Isc0_nA / current_scale_nA
    dIsc = np.gradient(Isc, Vbias)
    wsc = sc_weight
    Itotal = (np.sum(Ithreshold_nA, axis=0) + Isc0_nA) / current_scale_nA
    dItotal = np.gradient(Itotal, Vbias)

    def current_over_voltage(current, differential):
        static = np.divide(
            current,
            Vbias,
            out=np.full_like(current, np.nan),
            where=Vbias != 0.0,
        )
        static[..., zero_bias] = differential[..., zero_bias]
        return static

    for order, current, conductance, color in zip(
        threshold_orders,
        Ithreshold,
        dIthreshold,
        component_colors,
    ):
        label = rf"$m={order}$"
        ax_static.plot(
            Vbias,
            current_over_voltage(current, conductance),
            color=color,
            label=label,
        )
        ax_conductance.plot(Vbias, conductance, color=color, label=label)
        ax_weight.plot(
            Vbias,
            threshold_weights[order - 1],
            color=color,
            label=label,
        )

        threshold = 2.0 / order
        for sign in (-1.0, 1.0):
            for axis in (ax_conductance, ax_weight):
                axis.axvline(
                    sign * threshold,
                    color=color,
                    lw=0.7,
                    ls=":",
                    alpha=0.55,
                )

    sc_color = "tab:red"
    ax_static.plot(
        Vbias,
        current_over_voltage(Isc, dIsc),
        color=sc_color,
        lw=1.5,
        label="SC",
    )
    ax_conductance.plot(
        Vbias,
        dIsc,
        color=sc_color,
        lw=1.5,
        label="SC",
    )
    ax_weight.plot(
        Vbias,
        wsc,
        color=sc_color,
        lw=1.5,
        label=r"$w_\mathrm{SC}$",
    )
    ax_weight.plot(
        Vbias,
        unattributed_weight,
        color="0.4",
        ls="--",
        lw=1.5,
        label=rf"not attributed ($m>{threshold_orders[-1]}$)",
    )

    ax_static.plot(
        Vbias,
        Gexp[0],
        color="tab:blue",
        ls=":",
        lw=1.8,
        label="experiment, $A=0$",
    )
    ax_conductance.plot(
        Vbias,
        dIexp[0],
        color="tab:blue",
        ls=":",
        lw=1.8,
        label="experiment, $A=0$",
    )

    ax_static.plot(
        Vbias,
        current_over_voltage(Ibackground, dIbackground),
        color="0.5",
        ls="--",
        label="background",
    )
    ax_static.plot(
        Vbias,
        current_over_voltage(Itotal, dItotal),
        color="k",
        label="total",
    )
    ax_conductance.plot(
        Vbias,
        dIbackground,
        color="0.5",
        ls="--",
        label="background",
    )
    ax_conductance.plot(
        Vbias,
        dItotal,
        color="k",
        label="total",
    )

    ax_static.set_ylabel(r"$(I_m/V)\,/\,G_0$")
    ax_conductance.set_ylabel(r"$dI_m/dV\,/\,G_0$")
    ax_weight.set_ylabel(r"$w_m$")
    ax_weight.set_xlabel(r"$eV/\Delta$")
    ax_weight.set_ylim(-0.02, 1.02)
    ax_static.set_xlim(*plot_limits)
    ax_static.axhline(0.0, color="0.7", lw=0.6)
    ax_conductance.axhline(0.0, color="0.7", lw=0.6)
    ax_static.legend(ncols=4, fontsize="small")
    ax_weight.legend(ncols=4, fontsize="small")
    ax_static.set_title("Threshold-resolved MAR components")

fig_joined, ax_joined = plt.subplots(
    num=4,
    figsize=(7.0, 4.5),
    constrained_layout=True,
)
joined_map = np.where(Vbias[None, :] < 0.0, dIexp, dIpamar0)
mesh_joined = ax_joined.pcolormesh(
    Vbias,
    Abias,
    joined_map,
    shading="auto",
    vmin=color_limits[0],
    vmax=color_limits[1],
)
ax_joined.axvline(0.0, color="0.75", lw=0.8)
ax_joined.text(
    0.04,
    0.94,
    "experiment",
    color="white",
    transform=ax_joined.transAxes,
    ha="left",
    va="top",
)
ax_joined.text(
    0.96,
    0.94,
    "simulation",
    color="white",
    transform=ax_joined.transAxes,
    ha="right",
    va="top",
)
ax_joined.set(
    xlabel=r"$eV/\Delta$",
    ylabel=r"nominal $A/hf$",
    xlim=plot_limits,
)
fig_joined.colorbar(mesh_joined, ax=ax_joined, label=r"$dI/dV\,/\,G_0$")

visible_bias = (Vbias >= plot_limits[0]) & (Vbias <= plot_limits[1])
differences = (
    (Gexp - Gpamar0, r"$I/V$ difference", r"$\Delta(I/V)\,/\,G_0$"),
    (
        dIexp - dIpamar0,
        r"$dI/dV$ difference",
        r"$\Delta(dI/dV)\,/\,G_0$",
    ),
)
for figure_number, (difference, title, colorbar_label) in enumerate(
    differences,
    start=6,
):
    difference_limit = np.nanpercentile(
        np.abs(difference[:, visible_bias]),
        95.0,
    )
    fig_difference, ax_difference = plt.subplots(
        num=figure_number,
        figsize=(7.0, 4.5),
        constrained_layout=True,
    )
    mesh_difference = ax_difference.pcolormesh(
        Vbias,
        Abias,
        difference,
        shading="auto",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    ax_difference.set(
        title="experiment - (PAMAR + SC): " + title,
        xlabel=r"$eV/\Delta$",
        ylabel=r"nominal $A/hf$",
        xlim=plot_limits,
    )
    fig_difference.colorbar(
        mesh_difference,
        ax=ax_difference,
        label=colorbar_label,
        extend="both",
    )

shapiro_orders = np.arange(1, 5)
photon_energy_meV = nu_GHz * sc.h_pVs
shapiro_biases = shapiro_orders * photon_energy_meV / (2.0 * Delta_meV)
cut_positions = np.concatenate(([0.0], shapiro_biases))
cut_labels = [r"$V=0$"] + [rf"$V_{{{order}}}$" for order in shapiro_orders]


def conductance_cut(conductance_map, bias):
    """Interpolate one amplitude-dependent conductance cut."""
    return np.asarray(
        [
            np.interp(bias, Vbias, conductance_trace)
            for conductance_trace in conductance_map
        ]
    )


cut_components = (
    ("tab:blue", r"experiment", Gexp, dIexp),
    ("tab:red", r"PAMAR+SC", Gpamar0, dIpamar0),
)
# Temporarily disable the fixed-voltage cuts over microwave amplitude.
plot_voltage_cuts = True
component_handles = [
    Line2D([], [], color=color, lw=1.5, label=label)
    for color, label, _, _ in cut_components
]
sign_handles = [
    Line2D([], [], color="0.25", lw=1.3, ls="-", label=r"$+V_n$"),
    Line2D([], [], color="0.25", lw=1.3, ls="--", label=r"$-V_n$"),
]

voltage_cuts = zip(cut_positions, cut_labels) if plot_voltage_cuts else ()
for column, (bias, label) in enumerate(voltage_cuts):
    fig_conductance_cut, axes_conductance_cut = plt.subplots(
        2,
        1,
        num=30 + column,
        figsize=(6.0, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    ax_static_cut, ax_differential_cut = axes_conductance_cut
    signs = (1.0,) if bias == 0.0 else (1.0, -1.0)
    for color, _, static_map, differential_map in cut_components:
        for sign in signs:
            linestyle = "-" if sign > 0.0 else "--"
            ax_static_cut.plot(
                Abias,
                conductance_cut(static_map, sign * bias),
                color=color,
                ls=linestyle,
                lw=1.3,
            )
            ax_differential_cut.plot(
                Abias,
                conductance_cut(differential_map, sign * bias),
                color=color,
                ls=linestyle,
                lw=1.3,
            )
    ax_static_cut.set_title(label + rf", $eV_n/\Delta={bias:.3f}$")
    ax_static_cut.set_ylabel(r"$(I/V)/G_0$")
    ax_differential_cut.set_ylabel(r"$(dI/dV)/G_0$")
    ax_differential_cut.set_xlabel(r"nominal $A/hf$")
    for ax_cut in (ax_static_cut, ax_differential_cut):
        ax_cut.axhline(0.0, color="0.7", lw=0.7)
        ax_cut.grid(alpha=0.2)
    legend_handles = component_handles.copy()
    if bias != 0.0:
        legend_handles += sign_handles
    ax_static_cut.legend(
        handles=legend_handles,
        ncols=2,
        fontsize="small",
    )

# Show every measured amplitude without changing the scale between pages.
# Each amplitude gets its own figure.  The two panels use the same map row and
# share both axes, making their conductance scales directly comparable.
plot_amplitude_cuts = True
bias_mask = (Vbias >= plot_limits[0]) & (Vbias <= plot_limits[1])
conductance_limits = (
    np.nanmin(
        (
            Gexp[:, bias_mask],
            Gpamar0[:, bias_mask],
            dIexp[:, bias_mask],
            dIpamar0[:, bias_mask],
        )
    ),
    np.nanmax(
        (
            Gexp[:, bias_mask],
            Gpamar0[:, bias_mask],
            dIexp[:, bias_mask],
            dIpamar0[:, bias_mask],
        )
    ),
)

amplitude_cuts = Abias if plot_amplitude_cuts else ()
for index, amplitude in enumerate(amplitude_cuts):
    fig_cuts, axes_cuts = plt.subplots(
        2,
        1,
        num=8 + index,
        figsize=(6.0, 6.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax_static_cut, ax_differential_cut = axes_cuts
    ax_static_cut.plot(
        Vbias,
        Gexp[index],
        color="tab:blue",
        lw=1.2,
        label="experiment",
    )
    ax_static_cut.plot(
        Vbias,
        Gpamar0[index],
        color="tab:orange",
        lw=1.5,
        label="simulation",
    )
    ax_differential_cut.plot(
        Vbias,
        dIexp[index],
        color="tab:blue",
        lw=1.2,
    )
    ax_differential_cut.plot(
        Vbias,
        dIpamar0[index],
        color="tab:orange",
        lw=1.5,
    )

    ax_static_cut.set_title(rf"$A/hf={amplitude:.2f}$")
    ax_static_cut.set_ylabel(r"$(I/V)\,/\,G_0$")
    ax_differential_cut.set_ylabel(r"$dI/dV\,/\,G_0$")
    ax_differential_cut.set_xlabel(r"$eV/\Delta$")
    ax_static_cut.set_xlim(*plot_limits)
    ax_static_cut.set_ylim(*conductance_limits)
    ax_static_cut.legend(fontsize="small")
    ax_static_cut.grid(alpha=0.2)
    ax_differential_cut.grid(alpha=0.2)

plt.show()
