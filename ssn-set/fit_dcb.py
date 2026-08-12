# init
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity.api as sc
from nptdms import TdmsFile
from scipy.optimize import least_squares
from scipy.signal import fftconvolve, savgol_filter
from superconductivity.utilities.functions.upsampling import upsample

Textwidth: float = 4.25279  # in
Textheight: float = 6.85173  # in
# get data single-iv dcb
file = "/Users/oliver/Documents/measurement data/SSET/22 03 Scheer2/36/NC_map.tdms"
key = "2022-03-30 17:05:31 G4.00mV"
nu_sample_Hz, amp, rref = 5000.0, (2000.0, 500.0), 102000.0
dt_sample_s = 0.00019999999999999998
nu_downsample_Hz = 437.0
dt_downsample_s = 1 / nu_downsample_Hz

sweep = (25_000, 75_000)

Vbias0_mV = np.linspace(-0.7, 0.7, 3501)

with TdmsFile.open(file) as f:
    sample_V = f[key]["Sample"].as_dataframe()
    reference_V = f[key]["Reference"].as_dataframe()
    sample_V = sample_V.values
    reference_V = reference_V.values

sample_V = sample_V[sweep[0] : sweep[1], 0]
reference_V = reference_V[sweep[0] : sweep[1], 0]
time_s = np.arange(len(sample_V)) * dt_sample_s

nu_time_s = np.arange(0, np.max(time_s), dt_downsample_s)
nu_sample_V = sc.bin_y_over_x(sample_V, time_s, nu_time_s)
nu_reference_V = sc.bin_y_over_x(reference_V, time_s, nu_time_s)

v_mV = nu_sample_V * 1e3 / amp[0]
i_nA = nu_reference_V * 1e9 / (amp[1] * rref)
valid = np.logical_and(np.isfinite(v_mV), np.isfinite(i_nA))
i_nA, v_mV = upsample(i_nA[valid]), upsample(v_mV[valid])
Iexp0_nA = sc.bin_y_over_x(i_nA, v_mV, Vbias0_mV) + 5

Vbias0_mV += 0.032
Iexp0_nA -= 0.26


# %% finite-temperature DCB fit
# A normal tunnel junction in an electromagnetic environment has
#
# I(V) = G_T/e * integral dE E/(1-exp(-E/kT))
#                    * [P(eV-E) - P(-eV-E)].
#
# Here P(E) is calculated for the real part of a parallel-RC impedance.  The
# environmental temperature is fitted as a nuisance parameter; keeping it in
# the model is important because otherwise thermal rounding is incorrectly
# absorbed into C_eff.
atomic_contact_dir = Path(__file__).resolve().parents[1] / "atomic_contact"
if str(atomic_contact_dir) not in sys.path:
    sys.path.insert(0, str(atomic_contact_dir))

from sc_fit import _parallel_rc_pe

K_B_UEV_PER_K = 86.17333262145
PE_SAMPLE_COUNT = 2**13
PE_TIME_STEP_S = 0.5e-12


def _thermal_rate_ueV(energy_ueV: np.ndarray, temperature_K: float) -> np.ndarray:
    """Return E/[1-exp(-E/kT)] with a stable zero-energy limit."""
    kT_ueV = K_B_UEV_PER_K * temperature_K
    argument = energy_ueV / kT_ueV
    result = np.empty_like(energy_ueV)
    positive = argument > 50.0
    negative = argument < -50.0
    regular = ~(positive | negative) & (np.abs(argument) > 1e-7)
    result[positive] = energy_ueV[positive]
    result[negative] = 0.0
    result[regular] = energy_ueV[regular] / (-np.expm1(-argument[regular]))
    result[~(positive | negative | regular)] = kT_ueV
    return result


def dcb_voltage_mV(
    voltage_mV: np.ndarray,
    Reff_kOhm: float,
    Ceff_fF: float,
    Teff_K: float,
) -> np.ndarray:
    """Return the DCB-broadened voltage entering a normal tunnel current."""
    energy_ueV, pe_per_ueV = _parallel_rc_pe(
        Reff_kOhm * 1e3,
        Ceff_fF * 1e-15,
        Teff_K,
        PE_SAMPLE_COUNT,
        PE_TIME_STEP_S,
    )
    energy_step_ueV = energy_ueV[1] - energy_ueV[0]
    rate_ueV = _thermal_rate_ueV(energy_ueV, Teff_K)
    convolution = fftconvolve(rate_ueV, pe_per_ueV, mode="full")
    convolution *= energy_step_ueV
    convolution_energy_ueV = (
        energy_ueV[0] + energy_ueV[0] + np.arange(convolution.size) * energy_step_ueV
    )
    voltage_energy_ueV = 1e3 * np.asarray(voltage_mV)
    forward = np.interp(voltage_energy_ueV, convolution_energy_ueV, convolution)
    backward = np.interp(-voltage_energy_ueV, convolution_energy_ueV, convolution)
    return (forward - backward) / 1e3


def simulate_dcb_nin_current_nA(
    V_mV: np.ndarray,
    R_eff_kOhm: float = 0.44579,
    C_eff_fF: float = 4.361,
    T_eff_mK: float = 77.37,
    G_T_uS: float = 29.046,
    V0_uV: float = -0.3266,
    I0_nA: float = -0.08457,
) -> np.ndarray:
    """Simulate the fitted normal-junction DCB current.

    Parameters
    ----------
    V_mV:
        Junction voltage in millivolts.
    R_eff_kOhm:
        Effective parallel environmental resistance in kilohms.
    C_eff_fF:
        Effective parallel environmental capacitance in femtofarads.
    T_eff_mK:
        Effective temperature in millikelvins.
    G_T_uS:
        Bare normal-state tunnel conductance in microsiemens.
    V0_uV:
        Voltage-zero offset in microvolts.
    I0_nA:
        Current-zero offset in nanoamperes.

    Returns
    -------
    numpy.ndarray
        DCB current in nanoamperes, with the shape of ``V_mV``.
    """
    effective_voltage_mV = np.asarray(V_mV, dtype=np.float64) - V0_uV * 1e-3
    return (
        G_T_uS
        * dcb_voltage_mV(
            effective_voltage_mV,
            R_eff_kOhm,
            C_eff_fF,
            T_eff_mK * 1e-3,
        )
        + I0_nA
    )


def _fit_model(parameters: np.ndarray, voltage_mV: np.ndarray) -> np.ndarray:
    """Evaluate the DCB model using log parameters for positive quantities."""
    log_R, log_C, log_T, conductance_uS, voltage_offset_mV, current_offset_nA = (
        parameters
    )
    effective_voltage_mV = voltage_mV - voltage_offset_mV
    return (
        conductance_uS
        * dcb_voltage_mV(
            effective_voltage_mV,
            np.exp(log_R),
            np.exp(log_C),
            np.exp(log_T),
        )
        + current_offset_nA
    )


fit_mask = np.isfinite(Vbias0_mV) & np.isfinite(Iexp0_nA) & (np.abs(Vbias0_mV) <= 0.65)
Vfit_mV = Vbias0_mV[fit_mask]
Ifit_nA = Iexp0_nA[fit_mask]

# Downsampling prevents the densely sampled central part from merely making
# each residual evaluation expensive; the final curve is evaluated on every
# measured voltage below.
fit_stride = 4
Vopt_mV = Vfit_mV[::fit_stride]
Iopt_nA = Ifit_nA[::fit_stride]

lower = np.array([np.log(0.05), np.log(0.02), np.log(0.010), 1.0, -0.03, -2.0])
upper = np.array([np.log(200.0), np.log(100.0), np.log(1.0), 100.0, 0.03, 2.0])

# R and C are correlated through the RC cutoff, so use deliberately separated
# starts and retain the best robust-loss solution instead of trusting one
# initial guess.
fit_results = []
for initial_R_kOhm, initial_C_fF in (
    (0.2, 2.0),
    (0.5, 5.0),
    (2.0, 1.0),
    (10.0, 1.0),
    (10.0, 10.0),
    (50.0, 0.2),
):
    initial = np.array(
        [
            np.log(initial_R_kOhm),
            np.log(initial_C_fF),
            np.log(0.080),
            27.0,
            0.0,
            0.0,
        ]
    )
    fit_results.append(
        least_squares(
            lambda parameters: _fit_model(parameters, Vopt_mV) - Iopt_nA,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.03,
            x_scale="jac",
            max_nfev=500,
        )
    )
fit_result = min(fit_results, key=lambda result: result.cost)

(
    log_Reff_kOhm,
    log_Ceff_fF,
    log_Teff_K,
    Gfit_uS,
    Voffset_mV,
    Ioffset_nA,
) = fit_result.x
Reff_kOhm = float(np.exp(log_Reff_kOhm))
Ceff_fF = float(np.exp(log_Ceff_fF))
Teff_K = float(np.exp(log_Teff_K))
Ifit_model_nA = _fit_model(fit_result.x, Vbias0_mV)
fit_rmse_nA = float(
    np.sqrt(np.mean((_fit_model(fit_result.x, Vfit_mV) - Ifit_nA) ** 2))
)

# Approximate local 1-sigma covariance.  The Jacobian returned by SciPy is
# already adjusted for the robust loss.  These uncertainties describe the
# statistical precision within the single-RC model, not model systematics.
degrees_of_freedom = max(1, fit_result.fun.size - fit_result.x.size)
residual_variance_nA2 = float(fit_result.fun @ fit_result.fun) / degrees_of_freedom
parameter_covariance = residual_variance_nA2 * np.linalg.pinv(
    fit_result.jac.T @ fit_result.jac
)
parameter_std = np.sqrt(np.maximum(np.diag(parameter_covariance), 0.0))

Reff_std_kOhm = Reff_kOhm * parameter_std[0]
Ceff_std_fF = Ceff_fF * parameter_std[1]
Teff_std_K = Teff_K * parameter_std[2]
Gfit_std_uS = float(parameter_std[3])
Voffset_std_mV = float(parameter_std[4])
Ioffset_std_nA = float(parameter_std[5])

G_T_G0 = Gfit_uS / sc.G0_muS
G_T_G0_std = Gfit_std_uS / sc.G0_muS
R_T_kOhm = 1e3 / Gfit_uS
R_T_std_kOhm = R_T_kOhm * Gfit_std_uS / Gfit_uS

print(
    "DCB fit (approximate 1-sigma statistical uncertainties):\n"
    f"R_eff_kOhm = {Reff_kOhm:.6g} +/- {Reff_std_kOhm:.2g}\n"
    f"C_eff_fF   = {Ceff_fF:.6g} +/- {Ceff_std_fF:.2g}\n"
    f"T_eff_mK   = {1e3 * Teff_K:.6g} +/- {1e3 * Teff_std_K:.2g}\n"
    f"G_T_uS     = {Gfit_uS:.6g} +/- {Gfit_std_uS:.2g}\n"
    f"G_T_G0     = {G_T_G0:.6g} +/- {G_T_G0_std:.2g}\n"
    f"R_T_kOhm   = {R_T_kOhm:.6g} +/- {R_T_std_kOhm:.2g}\n"
    f"V0_uV      = {1e3 * Voffset_mV:.6g} +/- "
    f"{1e3 * Voffset_std_mV:.2g}\n"
    f"I0_nA      = {Ioffset_nA:.6g} +/- {Ioffset_std_nA:.2g}\n"
    f"RMSE_nA    = {fit_rmse_nA:.6g}"
)

voltage_step_mV = float(np.median(np.diff(Vbias0_mV)))
dIdV_exp_uS = savgol_filter(
    Iexp0_nA,
    window_length=101,
    polyorder=3,
    deriv=1,
    delta=voltage_step_mV,
)
dIdV_fit_uS = np.gradient(Ifit_model_nA, Vbias0_mV)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(Textwidth, 0.95 * Textwidth),
    sharex=True,
    constrained_layout=True,
)
axes[0].plot(Vbias0_mV, Iexp0_nA, color="0.55", lw=0.8, label="data")
axes[0].plot(
    Vbias0_mV,
    Ifit_model_nA,
    color="C3",
    lw=1.2,
    label=r"$R\parallel C$ DCB fit",
)
axes[0].set(ylabel=r"$I$ (nA)")
axes[0].legend(frameon=False)

axes[1].plot(Vbias0_mV, dIdV_exp_uS, color="0.55", lw=0.8)
axes[1].plot(Vbias0_mV, dIdV_fit_uS, color="C3", lw=1.2)
axes[1].set(xlabel=r"$V$ (mV)", ylabel=r"$dI/dV$ ($\mu$S)")
for ax in axes:
    ax.set_xlim(Vfit_mV.min(), Vfit_mV.max())

fit_figure_path = Path(__file__).with_name("fit_dcb.png")
fig.savefig(fit_figure_path, dpi=300)
plt.show()
