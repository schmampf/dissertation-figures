from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc

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


@lru_cache(maxsize=256)
def _get_total_mar_inverse(tau_values):
    """Return the current and voltage axes for a parallel PIN code."""
    transmissions = np.asarray(tau_values, dtype=float)
    tau_indices = np.searchsorted(tau_grid, transmissions)
    tau_indices = np.clip(tau_indices, 1, tau_grid.size - 1)
    use_lower = np.abs(transmissions - tau_grid[tau_indices - 1]) <= np.abs(
        transmissions - tau_grid[tau_indices]
    )
    tau_indices[use_lower] -= 1

    # Imar_nA is the single cached MAR bank with shape (tau, voltage).
    # Parallel channels share voltage, so their currents add directly.
    total_current = np.sum(Imar_nA[tau_indices], axis=0)

    # Numerical MAR curves can contain repeated current values. Sorting and
    # removing duplicates makes the inverse suitable for np.interp.
    order = np.argsort(total_current)
    current = total_current[order]
    voltage = Vgrid_mV[order]
    current, unique = np.unique(current, return_index=True)
    return current, voltage[unique]


def get_Vmar_mV(I_nA, tau):
    """Return MAR voltage for one channel or a parallel PIN code."""
    I_nA = np.asarray(I_nA, dtype=float)
    transmissions = np.asarray(tau, dtype=float)
    if transmissions.ndim > 1 or transmissions.size == 0:
        raise ValueError("tau must be a scalar or a nonempty 1D sequence.")
    if np.any(~np.isfinite(transmissions)) or np.any(
        (transmissions < 0.0) | (transmissions > 1.0)
    ):
        raise ValueError("all transmissions must lie in [0, 1].")

    current, voltage = _get_total_mar_inverse(
        tuple(float(value) for value in transmissions.reshape(-1))
    )
    return np.interp(
        I_nA,
        current,
        voltage,
        left=np.nan,
        right=np.nan,
    )


def get_rsj_Vj_mV(
    Ibias_nA,
    tau,
    alpha,
    get_Vmar_mV,
    get_Isc_nA,
    n_phi=2048,
):
    """Return nonlinear-RSJ junction voltage for a scalar or PIN code."""
    Ibias_nA = np.asarray(Ibias_nA, dtype=float)
    transmissions = np.asarray(tau, dtype=float)
    if transmissions.ndim > 1 or transmissions.size == 0:
        raise ValueError("tau must be a scalar or a nonempty 1D sequence.")
    if np.any(~np.isfinite(transmissions)) or np.any(
        (transmissions < 0.0) | (transmissions > 1.0)
    ):
        raise ValueError("all transmissions must lie in [0, 1].")

    # Midpoints reduce the chance of evaluating exactly at V = 0.
    phi = (np.arange(n_phi, dtype=float) + 0.5) * (2 * np.pi / n_phi)

    if transmissions.ndim == 0:
        Is_nA = alpha * get_Isc_nA(phi, float(transmissions))
    else:
        Is_nA = alpha * np.sum(
            [get_Isc_nA(phi, float(transmission)) for transmission in transmissions],
            axis=0,
        )

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
        tau=(
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ),
        alpha=1.0,
        get_Vmar_mV=get_Vmar_mV,
        get_Isc_nA=sc.get_cpr_abs,
    ),
)
plt.plot(Ibias_nA, Vexp_mV, ".", color="grey")
plt.show()
