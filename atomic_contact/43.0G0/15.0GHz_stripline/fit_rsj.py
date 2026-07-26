import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
from scipy.interpolate import RegularGridInterpolator
from superconductivity.utilities.functions.upsampling import upsample

# source /Users/oliver/Documents/cryolab/.venv/bin/activate

tau_grid = np.linspace(0.0, 1.0, 1001)
Vgrid_mV = np.linspace(-0.9, 0.9, 1801)
Igrid_nA = np.linspace(-3400, 3400, 1001)
_, Imar_nA = sc.get_Imar_nA(
    V_mV=Vgrid_mV,
    tau=tau_grid,
    tau_resolved=True,
    Delta_meV=0.1895,
    gamma_meV=1e-6,
    T_K=0.0,
    show_progress=True,
)
Imar_nA = Imar_nA.T
Vmar_mV = sc.bin_y_over_x(
    upsample(Vgrid_mV), upsample(Imar_nA, axis=-1), Igrid_nA, axis=-1
)

Vmar_lookup = RegularGridInterpolator(
    (tau_grid, Igrid_nA),
    Vmar_mV,
    method="linear",
    bounds_error=False,
    fill_value=np.nan,
)


def get_Vmar_mV(I_nA, tau):
    I_nA = np.asarray(I_nA, dtype=float)
    points = np.column_stack(
        (
            np.full(I_nA.size, tau),
            I_nA.ravel(),
        )
    )
    return Vmar_lookup(points).reshape(I_nA.shape)


def get_rsj_Vj_mV(
    Ibias_nA,
    tau,
    alpha,
    get_Vmar_mV,
    get_Isc_nA,
    n_phi=2048,
):
    """Return the nonlinear-RSJ junction voltage Vj(Ibias)."""
    Ibias_nA = np.asarray(Ibias_nA, dtype=float)

    # Midpoints reduce the chance of evaluating exactly at V = 0.
    phi = (np.arange(n_phi, dtype=float) + 0.5) * (2 * np.pi / n_phi)

    Is_nA = alpha * get_Isc_nA(phi, tau)

    # Assuming Imar(0) = 0.
    lower_static_nA = np.min(Is_nA)
    upper_static_nA = np.max(Is_nA)

    Vj_mV = np.zeros_like(Ibias_nA)

    running = (Ibias_nA < lower_static_nA) | (Ibias_nA > upper_static_nA)

    for index in np.flatnonzero(running):
        Iqp_nA = Ibias_nA[index] - Is_nA
        Vinst_mV = get_Vmar_mV(Iqp_nA, tau)

        if np.any(~np.isfinite(Vinst_mV)):
            Vj_mV[index] = np.nan
            continue

        Vj_mV[index] = 1.0 / np.mean(1.0 / Vinst_mV)

    return Vj_mV


path = "43.0G0/15.0GHz_stripline"
data = np.load(f"atomic_contact/{path}/Rs_data.npz")
Rs_Ohm = data["Rs_Ohm"]
Voff_mV = data["Voff_mV"]
data = np.load(f"atomic_contact/{path}/eva.npz")
Ibias_nA = data["Ibias_nA"]
Vexp_mV = data["Vexp_mV"][0, :] - Voff_mV - Ibias_nA * (Rs_Ohm * 1e-6)

plt.plot(
    Igrid_nA,
    get_rsj_Vj_mV(
        Igrid_nA,
        tau=1.0,
        alpha=0.5,
        get_Vmar_mV=get_Vmar_mV,
        get_Isc_nA=sc.get_cpr_abs,
    ),
)
plt.plot(Ibias_nA, Vexp_mV, ".", color="grey")
plt.show()
