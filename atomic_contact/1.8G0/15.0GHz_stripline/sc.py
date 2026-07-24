"""Compare an unirradiated atomic-contact trace with its MAR model."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
from scipy.optimize import least_squares

atomic_contact_dir = Path(__file__).resolve().parents[2]
if str(atomic_contact_dir) not in sys.path:
    sys.path.insert(0, str(atomic_contact_dir))

from apply_sigmaV import apply_sigmaV

path = Path(__file__).resolve().parent
data = np.load(path / "eva.npz")

# The first trace is unirradiated: Aout_mV[0] == 0.
Vbias_mV = data["Vbias_mV"]
Iexp_nA = data["Iexp_nA"][0, :]

# Static MAR parameters used in sim.py.
tau = 0.514, 0.402, 0.326, 0.295, 0.226
Delta_meV = 0.1895
sigmaV_mV = 0.008
gamma_meV = 1e-6
T_K = 0.0

Imar_nA = sc.get_Imar_nA(
    V_mV=Vbias_mV,
    tau=tau,
    T_K=T_K,
    Delta_meV=Delta_meV,
    gamma_meV=gamma_meV,
    show_progress=True,
)
Imar_nA = apply_sigmaV(
    V_mV=Vbias_mV,
    I_nA=Imar_nA,
    sigmaV_mV=sigmaV_mV,
)

# Reduced variables: v = eV/Delta and i = I/(Delta G0).
Vbias = Vbias_mV / Delta_meV
current_scale_nA = Delta_meV * sc.G0_muS
Iexp = Iexp_nA / current_scale_nA
Imar = Imar_nA / current_scale_nA

# d i / d v = (dI/dV)/G0 and d v / d i = (dV/dI)/(1/G0).
dIexp_dV = np.gradient(Iexp, Vbias)
dImar_dV = np.gradient(Imar, Vbias)
dV_dIexp = np.divide(
    1.0,
    dIexp_dV,
    out=np.full_like(dIexp_dV, np.nan),
    where=dIexp_dV != 0.0,
)
dV_dImar = np.divide(
    1.0,
    dImar_dV,
    out=np.full_like(dImar_dV, np.nan),
    where=dImar_dV != 0.0,
)

# Phenomenological noise-broadened branch fit to the odd MAR residual.
Ires_nA = Iexp_nA - Imar_nA
Ires_odd_nA = 0.5 * (Ires_nA - Ires_nA[::-1])
sc_fit_limit_mV = 0.040
sc_fit_mask = np.abs(Vbias_mV) <= sc_fit_limit_mV


def broadened_sc_current(
    V_mV,
    branch_current_nA,
    width_mV,
    decay_mV,
):
    """Return a localized phenomenological supercurrent branch."""
    return (
        branch_current_nA
        * np.tanh(V_mV / width_mV)
        / (1.0 + (V_mV / decay_mV) ** 2)
    )


sc_fit = least_squares(
    lambda parameters: (
        broadened_sc_current(Vbias_mV, *parameters)[sc_fit_mask]
        - Ires_odd_nA[sc_fit_mask]
    ),
    x0=(1.0, 0.003, 0.08),
    bounds=((0.0, 0.0001, 0.005), (10.0, 0.1, 1.0)),
    loss="soft_l1",
    f_scale=0.03,
)
branch_current_nA, branch_width_mV, branch_decay_mV = sc_fit.x
Isc_nA = broadened_sc_current(
    Vbias_mV,
    branch_current_nA,
    branch_width_mV,
    branch_decay_mV,
)
sc_rms_nA = np.sqrt(
    np.mean((Isc_nA[sc_fit_mask] - Ires_odd_nA[sc_fit_mask]) ** 2)
)
Isc = Isc_nA / current_scale_nA
dIsc_dV = np.gradient(Isc, Vbias)
dV_dIsc = np.divide(
    1.0,
    dIsc_dV,
    out=np.full_like(dIsc_dV, np.nan),
    where=dIsc_dV != 0.0,
)
Ifit = (Imar_nA + Isc_nA) / current_scale_nA
dIfit_dV = np.gradient(Ifit, Vbias)
dV_dIfit = np.divide(
    1.0,
    dIfit_dV,
    out=np.full_like(dIfit_dV, np.nan),
    where=dIfit_dV != 0.0,
)


def main():
    """Plot current, differential conductance, and differential resistance."""
    fig = plt.figure(
        figsize=(10.0, 8.0),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(3.0, 2.0),
        height_ratios=(3.0, 2.0),
    )
    ax_iv = fig.add_subplot(grid[0, 0])
    ax_didv = fig.add_subplot(grid[1, 0], sharex=ax_iv)
    ax_dvdi = fig.add_subplot(grid[0, 1], sharey=ax_iv)

    ax_iv.plot(
        Vbias,
        Iexp,
        color="tab:blue",
        lw=1.2,
        label="experiment",
    )
    ax_iv.plot(
        Vbias,
        Imar,
        color="tab:orange",
        lw=1.4,
        label="MAR",
    )
    ax_iv.plot(
        Vbias,
        Ifit,
        color="tab:red",
        lw=1.4,
        label="MAR + broadened branch",
    )
    ax_iv.plot(
        Vbias,
        Isc,
        color="tab:green",
        ls="--",
        lw=1.2,
        label="broadened branch",
    )
    ax_iv.set(
        title="Unirradiated atomic contact",
        xlabel=r"$eV/\Delta$",
        ylabel=r"$I/(\Delta G_0)$",
    )

    ax_didv.plot(
        Vbias,
        dIexp_dV,
        color="tab:blue",
        lw=1.2,
        label="experiment",
    )
    ax_didv.plot(
        Vbias,
        dImar_dV,
        color="tab:orange",
        lw=1.4,
        label="MAR",
    )
    ax_didv.plot(
        Vbias,
        dIfit_dV,
        color="tab:red",
        lw=1.4,
        label="MAR + broadened branch",
    )
    ax_didv.plot(
        Vbias,
        dIsc_dV,
        color="tab:green",
        ls="--",
        lw=1.2,
        label="broadened branch",
    )
    ax_didv.set(
        xlabel=r"$eV/\Delta$",
        ylabel=r"$(dI/dV)/G_0$",
    )

    ax_dvdi.plot(
        dV_dIexp,
        Iexp,
        color="tab:blue",
        ls="none",
        marker=".",
        ms=3.0,
        label="experiment",
    )
    ax_dvdi.plot(
        dV_dImar,
        Imar,
        color="tab:orange",
        ls="none",
        marker=".",
        ms=3.0,
        label="MAR",
    )
    ax_dvdi.plot(
        dV_dIfit,
        Ifit,
        color="tab:red",
        ls="none",
        marker=".",
        ms=3.0,
        label="MAR + broadened branch",
    )
    ax_dvdi.plot(
        dV_dIsc,
        Isc,
        color="tab:green",
        ls="none",
        marker=".",
        ms=3.0,
        label="broadened branch",
    )
    ax_dvdi.set(
        xlabel=r"$(dV/dI)/(1/G_0)$",
        ylabel=r"$I/(\Delta G_0)$",
        xlim=(0.0, 4.0),
    )

    for ax in (ax_iv, ax_didv, ax_dvdi):
        ax.axhline(0.0, color="0.75", lw=0.7)
        ax.axvline(0.0, color="0.75", lw=0.7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small")

    ax_iv.tick_params(labelbottom=False)
    ax_dvdi.tick_params(labelleft=False)

    fig_sc, ax_sc = plt.subplots(
        figsize=(7.0, 4.5),
        constrained_layout=True,
    )
    ax_sc.plot(
        Vbias_mV,
        Ires_nA,
        color="0.75",
        lw=1.0,
        label=r"$I_\mathrm{exp}-I_\mathrm{MAR}$",
    )
    ax_sc.plot(
        Vbias_mV,
        Ires_odd_nA,
        color="tab:blue",
        lw=1.3,
        label="odd residual",
    )
    ax_sc.plot(
        Vbias_mV,
        Isc_nA,
        color="tab:red",
        lw=1.5,
        label="broadened-branch fit",
    )
    ax_sc.axvspan(
        -sc_fit_limit_mV,
        sc_fit_limit_mV,
        color="tab:red",
        alpha=0.06,
        label="fit window",
    )
    ax_sc.axhline(0.0, color="0.75", lw=0.7)
    ax_sc.axvline(0.0, color="0.75", lw=0.7)
    ax_sc.set(
        title="MAR-subtracted supercurrent candidate",
        xlabel=r"$V$ (mV)",
        ylabel=r"$I$ (nA)",
        xlim=(-0.08, 0.08),
    )
    ax_sc.grid(alpha=0.2)
    ax_sc.legend(fontsize="small")

    print(
        "Broadened-branch fit: "
        f"Im = {branch_current_nA:.3f} nA, "
        f"Vw = {1e3 * branch_width_mV:.3f} uV, "
        f"Vc = {1e3 * branch_decay_mV:.3f} uV, "
        f"RMS = {sc_rms_nA:.3f} nA"
    )
    plt.show()


if __name__ == "__main__":
    main()
