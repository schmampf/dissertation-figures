# init
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity.api as sc
from nptdms import TdmsFile
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from superconductivity.models.bcs.backend import E0_meV
from superconductivity.models.bcs.backend.np import convolution_np
from superconductivity.utilities.functions.upsampling import upsample

# get data single-iv tb
file = "/Users/oliver/Documents/measurement data/SSET/22 02b Scheer2/36_9 - unbroken/SCGateMap.tdms"
key = "2022-02-17 20:45:50 G3.60mV"
dt_sample_s = 0.00019999999999999998
nu_downsample_Hz = 43.0
dt_downsample_s = 1 / nu_downsample_Hz

sweep = (25_000, 75_000)

Vbias0_mV = np.linspace(-0.7, 0.7, 3501)

with TdmsFile.open(file) as f:
    amp = (
        f.properties["s_ampl_sample_effective_gain"],
        f.properties["s_ampl_reference_effective_gain"],
    )
    rref = f.properties["s_ampl_resistor"]
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
Iexp0_nA = sc.bin_y_over_x(i_nA, v_mV, Vbias0_mV)


# %% SIN model and fit
# Start with thermal and Dynes rounding only. Set this to True to test an
# additional Gaussian voltage-noise width; it is deliberately disabled in the
# default fit because T, gamma, and sigma_V are strongly correlated.
FIT_SIGMA_V = False


def simulate_sin_current_nA(
    V_mV: np.ndarray,
    G_T_uS: float = 30.88155,
    T_K: float = 0.0681028,
    Delta_meV: float = 0.196106,
    gamma_meV: float = 0.000861382,
    V0_uV: float = -4.55651,
    I0_nA: float = -0.1081845,
    sigmaV_uV: float = 0.0,
) -> np.ndarray:
    """Simulate a Dynes-broadened SIN tunnel-junction current.

    Parameters
    ----------
    V_mV:
        Junction voltage in millivolts. The grid must be increasing.
    G_T_uS:
        Normal-state tunnel conductance in microsiemens.
    T_K:
        Common electron temperature of the normal and superconducting leads.
    Delta_meV:
        Zero-temperature superconducting gap in millielectronvolts.
    gamma_meV:
        Dynes broadening in millielectronvolts.
    V0_uV, I0_nA:
        Voltage and current offsets in microvolts and nanoamperes.
    sigmaV_uV:
        Optional RMS Gaussian voltage noise in microvolts.

    Returns
    -------
    numpy.ndarray
        SIN current in nanoamperes.
    """
    voltage_mV = np.asarray(V_mV, dtype=np.float64)
    if voltage_mV.ndim != 1 or np.any(np.diff(voltage_mV) <= 0.0):
        raise ValueError("V_mV must be a strictly increasing 1D array.")

    effective_voltage_mV = voltage_mV - V0_uV * 1e-3
    current_per_conductance_mV = convolution_np(
        effective_voltage_mV,
        E0_meV,
        T1_K=T_K,
        T2_K=T_K,
        Delta1_meV=0.0,
        Delta2_meV=Delta_meV,
        gamma1_meV=0.0,
        gamma2_meV=gamma_meV,
    )
    current_nA = G_T_uS * current_per_conductance_mV

    if sigmaV_uV > 0.0:
        voltage_step_uV = 1e3 * float(np.median(np.diff(voltage_mV)))
        current_nA = gaussian_filter1d(
            current_nA,
            sigma=sigmaV_uV / voltage_step_uV,
            mode="nearest",
        )
    return np.asarray(current_nA + I0_nA, dtype=np.float64)


def _sin_fit_model(parameters: np.ndarray, voltage_mV: np.ndarray) -> np.ndarray:
    """Evaluate the SIN model with log coordinates for positive parameters."""
    log_G, log_T, Delta_meV, log_gamma, V0_uV, I0_nA = parameters[:6]
    sigmaV_uV = float(np.exp(parameters[6])) if FIT_SIGMA_V else 0.0
    return simulate_sin_current_nA(
        voltage_mV,
        G_T_uS=np.exp(log_G),
        T_K=np.exp(log_T),
        Delta_meV=Delta_meV,
        gamma_meV=np.exp(log_gamma),
        V0_uV=V0_uV,
        I0_nA=I0_nA,
        sigmaV_uV=sigmaV_uV,
    )


fit_mask = np.isfinite(Vbias0_mV) & np.isfinite(Iexp0_nA) & (np.abs(Vbias0_mV) <= 0.65)
Vfit_mV = Vbias0_mV[fit_mask]
Ifit_nA = Iexp0_nA[fit_mask]

# A uniform subsample is sufficient for the smooth nonlinear optimization;
# the reported RMSE and plotted model use every measured point.
fit_stride = 4
Vopt_mV = Vfit_mV[::fit_stride]
Iopt_nA = Ifit_nA[::fit_stride]

initial = np.array(
    [
        np.log(30.88155),
        np.log(0.0681028),
        0.196106,
        np.log(0.000861382),
        -4.55651,
        -0.1081845,
    ]
)
lower = np.array(
    [
        np.log(1.0),
        np.log(0.010),
        0.10,
        np.log(1e-5),
        -30.0,
        -5.0,
    ]
)
upper = np.array(
    [
        np.log(200.0),
        np.log(1.0),
        0.30,
        np.log(0.05),
        30.0,
        5.0,
    ]
)
if FIT_SIGMA_V:
    initial = np.append(initial, np.log(2.0))
    lower = np.append(lower, np.log(0.05))
    upper = np.append(upper, np.log(50.0))

fit_result = least_squares(
    lambda parameters: _sin_fit_model(parameters, Vopt_mV) - Iopt_nA,
    initial,
    bounds=(lower, upper),
    loss="soft_l1",
    f_scale=0.025,
    x_scale="jac",
    max_nfev=500,
)

G_T_uS = float(np.exp(fit_result.x[0]))
T_K = float(np.exp(fit_result.x[1]))
Delta_meV = float(fit_result.x[2])
gamma_meV = float(np.exp(fit_result.x[3]))
V0_uV = float(fit_result.x[4])
I0_nA = float(fit_result.x[5])
sigmaV_uV = float(np.exp(fit_result.x[6])) if FIT_SIGMA_V else 0.0
Imodel_nA = _sin_fit_model(fit_result.x, Vfit_mV)
fit_rmse_nA = float(np.sqrt(np.mean((Imodel_nA - Ifit_nA) ** 2)))

degrees_of_freedom = max(1, fit_result.fun.size - fit_result.x.size)
residual_variance_nA2 = float(fit_result.fun @ fit_result.fun) / degrees_of_freedom
covariance = residual_variance_nA2 * np.linalg.pinv(fit_result.jac.T @ fit_result.jac)
parameter_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))

G_T_std_uS = G_T_uS * parameter_std[0]
T_std_K = T_K * parameter_std[1]
Delta_std_meV = parameter_std[2]
gamma_std_meV = gamma_meV * parameter_std[3]
V0_std_uV = parameter_std[4]
I0_std_nA = parameter_std[5]
G_T_G0 = G_T_uS / sc.G0_muS
G_T_G0_std = G_T_std_uS / sc.G0_muS
R_T_kOhm = 1e3 / G_T_uS
R_T_std_kOhm = R_T_kOhm * G_T_std_uS / G_T_uS

print(
    "SIN fit (approximate 1-sigma statistical uncertainties):\n"
    f"G_T_uS     = {G_T_uS:.6g} +/- {G_T_std_uS:.2g}\n"
    f"G_T_G0     = {G_T_G0:.6g} +/- {G_T_G0_std:.2g}\n"
    f"R_T_kOhm   = {R_T_kOhm:.6g} +/- {R_T_std_kOhm:.2g}\n"
    f"T_mK       = {1e3 * T_K:.6g} +/- {1e3 * T_std_K:.2g}\n"
    f"Delta_ueV  = {1e3 * Delta_meV:.6g} +/- {1e3 * Delta_std_meV:.2g}\n"
    f"gamma_ueV  = {1e3 * gamma_meV:.6g} +/- {1e3 * gamma_std_meV:.2g}\n"
    f"sigmaV_uV  = {sigmaV_uV:.6g}"
    + (f" +/- {sigmaV_uV * parameter_std[6]:.2g}\n" if FIT_SIGMA_V else " (fixed)\n")
    + f"V0_uV      = {V0_uV:.6g} +/- {V0_std_uV:.2g}\n"
    f"I0_nA      = {I0_nA:.6g} +/- {I0_std_nA:.2g}\n"
    f"RMSE_nA    = {fit_rmse_nA:.6g}"
)

voltage_step_mV = float(np.median(np.diff(Vfit_mV)))
dIdV_exp_uS = savgol_filter(
    Ifit_nA,
    window_length=101,
    polyorder=3,
    deriv=1,
    delta=voltage_step_mV,
)
dIdV_model_uS = np.gradient(Imodel_nA, Vfit_mV)

fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.5), sharex=True)
axes[0].plot(Vfit_mV, Ifit_nA, color="0.55", lw=0.8, label="data")
axes[0].plot(Vfit_mV, Imodel_nA, color="C3", lw=1.2, label="SIN fit")
axes[0].set_ylabel(r"$I$ (nA)")
axes[0].legend(frameon=False)
axes[1].plot(Vfit_mV, dIdV_exp_uS, color="0.55", lw=0.8)
axes[1].plot(Vfit_mV, dIdV_model_uS, color="C3", lw=1.2)
axes[1].set(xlabel=r"$V$ (mV)", ylabel=r"$dI/dV$ ($\mu$S)")
fig.tight_layout()
fit_figure_path = Path(__file__).with_name("fit_sin.png")
fig.savefig(fit_figure_path, dpi=300)
plt.show()
