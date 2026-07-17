import numpy as np
from mar_fit import fit_mar

# data
data = np.load("breaking/eva.npz")
V_mV = data["Vbias_mV"]
Iexp_nA = data["Iup_nA"]
x_arbu = data["x_arbu"]
settings = {
    "tau_1": (0.01, 0.00, 1.00, False),
    "tau_2": (0.01, 0.00, 1.00, False),
    "tau_3": (0.01, 0.00, 1.00, False),
    "tau_4": (0.01, 0.00, 1.00, False),
    "tau_5": (0.01, 0.00, 1.00, False),
    "tau_6": (0.01, 0.00, 1.00, False),
    "tau_7": (0.01, 0.00, 1.00, False),
    "tau_8": (0.00, 0.00, 1.00, False),
    "tau_9": (0.00, 0.00, 1.00, False),
    "tau_A": (0.00, 0.00, 1.00, True),
    "tau_B": (0.00, 0.00, 1.00, True),
    "tau_C": (0.00, 0.00, 1.00, True),
    "T_K": (0.00, 0.00, 1.21, True),
    "Delta_meV": (0.1885, 0.187, 0.191, False),
    "gamma_meV": (1e-6, 1e-6, 1e-2, True),
    "sigmaV_mV": (0.026, 0.01, 0.04, False),
}
restarts = 10
Vnan_mV = 0.04
tau_sum_bounds = (0.0, 5.0)

weights = np.divide(1, V_mV, out=np.full_like(V_mV, np.nan), where=V_mV != 0.0)
mask = np.abs(V_mV) <= Vnan_mV
I_nA = np.copy(Iexp_nA)
I_nA[:, mask] = np.nan
results = fit_mar(
    I_nA,
    "grid.pkl",
    settings=settings,
    tau_sum_bounds=tau_sum_bounds,
    restarts=restarts,
    progress=True,
)
Ifit_nA = np.full_like(I_nA, np.nan)
tau = np.full((I_nA.shape[0], 12), np.nan)
T_K = np.full((I_nA.shape[0]), np.nan)
Delta_meV = np.full((I_nA.shape[0]), np.nan)
gamma_meV = np.full((I_nA.shape[0]), np.nan)
sigmaV_mV = np.full((I_nA.shape[0]), np.nan)

for i, result in enumerate(results):
    Ifit_nA[i, :] = result.Ifit_nA
    tau[i, :] = result.tau
    T_K[i] = result.T_K
    Delta_meV[i] = result.Delta_meV
    gamma_meV[i] = result.gamma_meV
    sigmaV_mV[i] = result.sigmaV_mV

GN_G0 = np.nansum(tau, axis=-1)
np.savez(
    "breaking/fit_data.npz",
    V_mV=V_mV,
    x_arbu=x_arbu,
    Iexp_nA=I_nA,
    Ifit_nA=Ifit_nA,
    tau=tau,
    GN_G0=GN_G0,
    T_K=T_K,
    Delta_meV=Delta_meV,
    gamma_meV=gamma_meV,
    sigmaV_mV=sigmaV_mV,
    results=results,
    mask=mask,
    weights=weights,
    settings=settings,
    tau_sum_bounds=tau_sum_bounds,
)
