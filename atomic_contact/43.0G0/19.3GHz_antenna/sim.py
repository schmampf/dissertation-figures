# 43.0G0, 19.3GHz, antenna
import sys
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity as sc
from matplotlib.colors import Normalize
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar, nnls
from scipy.special import jv
from superconductivity.utilities.functions.upsampling import upsample
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
from sim_pamar import get_weights


def pair_shift_pat_kernel(
    *,
    V_mV,
    I_,
    A_mV,
    nu_GHz,
    m,
    n_max,
    exp=2,
    absolute=False,
    half_offset_branch=False,
):
    """Apply PAT with pair-harmonic coupling and ``n hf / (2 m)`` shifts.

    The shared ``pat_kernel`` uses its harmonic argument in both the voltage
    shift and Bessel argument. Calling it with harmonic ``2 m`` and amplitude
    ``A / m`` gives a shift ``n hf / (2 m)`` and the fixed Bessel argument
    ``2 A / hf`` without modifying the shared implementation. The local
    half-offset branch uses the m=1 Bessel sequence and shifts symmetric
    copies of its voltage ladder by half a fundamental Shapiro spacing.
    """
    if not half_offset_branch:
        return sc.pat_kernel(
            V_mV=V_mV,
            I_=I_,
            A_mV=np.asarray(A_mV, dtype=np.float64) / m,
            nu_GHz=nu_GHz,
            m=2 * m,
            n_max=n_max,
            exp=exp,
            absolute=absolute,
        )

    voltage = np.asarray(V_mV, dtype=np.float64)
    current = np.asarray(I_, dtype=np.float64)
    amplitudes = np.atleast_1d(np.asarray(A_mV, dtype=np.float64))
    photon_energy_mV = sc.h_pVs * nu_GHz
    photon_orders = np.arange(-n_max, n_max + 1)

    bessel = jv(
        photon_orders[:, np.newaxis],
        2 * amplitudes[np.newaxis, :] / photon_energy_mV,
    )
    if absolute:
        bessel = np.abs(bessel)
    bessel_weights = bessel**exp

    fundamental_spacing_mV = photon_energy_mV / 2
    positive_offset_shifts = (
        photon_orders + 0.5
    ) * fundamental_spacing_mV
    negative_offset_shifts = (
        photon_orders - 0.5
    ) * fundamental_spacing_mV

    shifted_current_positive = np.asarray(
        [
            np.interp(
                voltage - shift,
                voltage,
                current,
                left=current[0],
                right=current[-1],
            )
            for shift in positive_offset_shifts
        ]
    )
    shifted_current_negative = np.asarray(
        [
            np.interp(
                voltage - shift,
                voltage,
                current,
                left=current[0],
                right=current[-1],
            )
            for shift in negative_offset_shifts
        ]
    )
    shifted_current = 0.5 * (
        shifted_current_positive + shifted_current_negative
    )
    result = np.einsum("na,nv->av", bessel_weights, shifted_current)
    if np.asarray(A_mV).ndim == 0:
        return result[0]
    return result


nu_GHz = 19.3
GN_G0 = 30.0
Delta_meV = 0.18

path = SCRIPT_DIR
data = np.load(path / "corr.npz")
Aout_mV = data["Aout_mV"]
Icorrbias_nA = data["Ibias_nA"]
Vcorrbias_mV = data["Vbias_mV"]
Icorr_nA = data["Iexp_nA"]
Vcorr_mV = data["Vexp_mV"]
eta = data["eta"]

# tau = 0.5
p = np.arange(1, 20)
# harmonic = tau**n / 2
# # harmonic = np.full((20), 1)
# harmonic[0] = 1
# harmonic[1] = 1

# Nonzero seed weights set the number of pair harmonics. Their values do not
# bias the later linear fit because unit-harmonic responses are recovered.
w_p = np.ones(1, dtype=np.float64)
# w_p = 1 / p

# Use the first four irreducible n/m positions for every available harmonic.
linecut_photon_orders = {}
for harmonic_order in range(1, w_p.size + 1):
    photon_orders = []
    photon_order = 1
    while len(photon_orders) < 4:
        if gcd(photon_order, harmonic_order) == 1:
            photon_orders.append(photon_order)
        photon_order += 1
    linecut_photon_orders[harmonic_order] = tuple(photon_orders)


nmax = 100
mmax = 10
m = np.arange(1, mmax + 1)

# region bias
Abias_mV = Aout_mV * eta
Abias = Abias_mV / (sc.h_pVs * nu_GHz)

Abias1 = np.linspace(0, 3.6, 73)

Abias2 = np.linspace(0, 4.0, 2001)
Abias2_mV = Abias2 * sc.h_pVs * nu_GHz

Vbias = np.linspace(-0.4, 2.4, 481)
Vbias_mV = Vbias * Delta_meV

Vbias0 = np.linspace(-2.4, 2.4, 2401)
Vbias0_mV = Vbias0 * Delta_meV

# endregion

# region exp
Iexp0_nA = sc.bin_y_over_x(upsample(Icorr_nA), upsample(Vcorrbias_mV), Vbias0_mV)
Iexp0 = Iexp0_nA / (Delta_meV * GN_G0 * sc.G0_muS)
dGexp0 = np.gradient(Iexp0, Vbias0, axis=-1)

Iexp_nA = sc.bin_y_over_x(upsample(Icorr_nA), upsample(Vcorrbias_mV), Vbias_mV)
Iexp = Iexp_nA / (Delta_meV * GN_G0 * sc.G0_muS)
dGexp = np.gradient(Iexp, Vbias, axis=-1)
# endregion


# region sc
zeroindex = np.argmin(np.abs(Vbias0))
Isc0 = Iexp0[0, :]

dGsc0 = np.gradient(Isc0, Vbias0)
dGsc0 = np.where(np.abs(Vbias0) < 0.05, dGsc0, np.zeros_like(dGsc0))

dGp0 = np.full((w_p.shape[0], Vbias0.shape[0]), 0.0)
w_p = np.array(w_p, dtype=np.float64) / np.sum(w_p)
for i, w in enumerate(w_p):
    p = 1 + i
    dGp0[i, :] = w * sc.bin_y_over_x(upsample(dGsc0), upsample(Vbias0), Vbias0)
dGp0[np.isnan(dGp0)] = 0

Ip0 = cumulative_trapezoid(dGp0, Vbias0, initial=0.0)
Ip0 = Ip0 - Ip0[:, zeroindex, np.newaxis]
# endregion

# region shapiro
Isp0 = np.full((np.shape(Abias_mV)[0], np.shape(Vbias0_mV)[0]), 0.0)
Isp20 = np.full((np.shape(Abias2_mV)[0], np.shape(Vbias0_mV)[0]), 0.0)
Iicpt0 = np.full((np.shape(Abias_mV)[0], np.shape(Vbias0_mV)[0]), 0.0)
Iicpt20 = np.full((np.shape(Abias2_mV)[0], np.shape(Vbias0_mV)[0]), 0.0)
for i in range(np.shape(Ip0)[0]):
    pair_order = i + 1
    Isp0 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=Ip0[i, :],
        A_mV=Abias_mV,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=1,
        absolute=True,
        n_max=nmax,
    )
    Isp20 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=Ip0[i, :],
        A_mV=Abias2_mV,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=1,
        absolute=True,
        n_max=nmax,
    )
    Iicpt0 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=Ip0[i, :],
        A_mV=Abias_mV,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=2,
        n_max=nmax,
    )
    Iicpt20 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=Ip0[i, :],
        A_mV=Abias2_mV,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=2,
        n_max=nmax,
    )
dGsp0 = np.gradient(Isp0, Vbias0, axis=-1)
dGsp20 = np.gradient(Isp20, Vbias0, axis=-1)
dGicpt0 = np.gradient(Iicpt0, Vbias0, axis=-1)
dGicpt20 = np.gradient(Iicpt20, Vbias0, axis=-1)
# endregion


# region mar
dGmar0 = np.where(np.abs(Vbias0) >= 0.3, dGexp0[0, :], np.zeros_like(dGexp0[0, :]))

weights, unused = get_weights(Vbias=Vbias0, mmax=mmax, return_unused=True)
dGmarm0 = dGmar0[np.newaxis, :] * weights
dGmar0 = dGmar0 * unused
zeroindex = np.argmin(np.abs(Vbias0))

Imarm0 = []
for i, m_i in enumerate(tqdm(m)):
    imarm0 = cumulative_trapezoid(dGmarm0[i, :], Vbias0, initial=0.0)
    imarm0 -= imarm0[zeroindex]
    Imarm0.append(imarm0)
Imarm0 = np.asarray(Imarm0)

Imar0 = cumulative_trapezoid(dGmar0, Vbias0, initial=0.0)
Imar0 -= Imar0[zeroindex]
# endregion

# region pamar

Ipamar0 = np.full((np.shape(Abias_mV)[0], np.shape(Vbias0_mV)[0]), 0.0)
Ipamar20 = np.full((np.shape(Abias2_mV)[0], np.shape(Vbias0_mV)[0]), 0.0)
Ipamar0 += sc.pat_kernel(
    V_mV=Vbias0_mV, I_=Imar0, A_mV=Abias_mV, nu_GHz=nu_GHz, m=mmax + 1, n_max=nmax
)
Ipamar20 += sc.pat_kernel(
    V_mV=Vbias0_mV, I_=Imar0, A_mV=Abias2_mV, nu_GHz=nu_GHz, m=mmax + 1, n_max=nmax
)
for i, m_i in enumerate(tqdm(m)):
    Ipamar0 += sc.pat_kernel(
        V_mV=Vbias0_mV,
        I_=Imarm0[i, :],
        A_mV=Abias_mV,
        nu_GHz=nu_GHz,
        m=m_i,
        n_max=nmax,
    )
    Ipamar20 += sc.pat_kernel(
        V_mV=Vbias0_mV,
        I_=Imarm0[i, :],
        A_mV=Abias2_mV,
        nu_GHz=nu_GHz,
        m=m_i,
        n_max=nmax,
    )
dGpamar0 = np.gradient(Ipamar0, Vbias0, axis=-1)
dGpamar20 = np.gradient(Ipamar20, Vbias0, axis=-1)

# endregion


# region fit the effective p=1 coupling and pair weight
unit_pair_currents = Ip0 / w_p[:, np.newaxis]
fit_pairs = tuple(
    (n_i, m_i)
    for m_i in range(1, w_p.size + 1)
    for n_i in linecut_photon_orders[m_i]
)
fit_indices = np.asarray(
    [
        np.argmin(
            np.abs(
                Vbias0
                - n_i * sc.h_pVs * nu_GHz / (2 * m_i * Delta_meV)
            )
        )
        for n_i, m_i in fit_pairs
    ]
)

# Calculate the unit p=1 ICPT response once on the dense amplitude grid. For
# every ICPT coupling candidate, interpolate only ICPT onto the measured Aout
# grid and analytically refit its non-negative weight. PAMAR stays fixed at the
# saved eta because its background calibration is already established.
unit_icpt20 = pair_shift_pat_kernel(
    V_mV=Vbias0_mV,
    I_=unit_pair_currents[0],
    A_mV=Abias2_mV,
    nu_GHz=nu_GHz,
    m=1,
    exp=2,
    n_max=nmax,
)
unit_dGicpt20 = np.gradient(unit_icpt20, Vbias0, axis=-1)
photon_energy_mV = sc.h_pVs * nu_GHz
saved_eta = float(eta)


def evaluate_eta(eta_candidate):
    """Return RMSE and p=1 weight for an effective ICPT coupling."""
    candidate_amplitude = Aout_mV * eta_candidate / photon_energy_mV
    icpt_at_data = np.column_stack(
        [
            np.interp(candidate_amplitude, Abias2, unit_dGicpt20[:, index])
            for index in fit_indices
        ]
    )
    target = (dGexp0[:, fit_indices] - dGpamar0[:, fit_indices]).ravel()
    matrix = icpt_at_data.ravel()[:, np.newaxis]
    fitted_weight, residual = nnls(matrix, target)
    rmse = residual / np.sqrt(target.size)
    return rmse, fitted_weight[0]


eta_upper = Abias2.max() * photon_energy_mV / Aout_mV.max()
eta_scan = np.linspace(0.5 * saved_eta, eta_upper, 301)
eta_scan_rmse = np.asarray(
    [evaluate_eta(eta_candidate)[0] for eta_candidate in eta_scan]
)
eta_scan_index = np.argmin(eta_scan_rmse)
eta_refine_lower = eta_scan[max(0, eta_scan_index - 1)]
eta_refine_upper = eta_scan[min(eta_scan.size - 1, eta_scan_index + 1)]
eta_result = minimize_scalar(
    lambda eta_candidate: evaluate_eta(eta_candidate)[0],
    bounds=(eta_refine_lower, eta_refine_upper),
    method="bounded",
    options={"xatol": 1e-10},
)
eta_icpt = float(eta_result.x)
fit_rmse, fitted_pair_weight = evaluate_eta(eta_icpt)
w_p = np.asarray([fitted_pair_weight])
icpt_amplitude_scale = eta_icpt / saved_eta

print(f"Saved eta: {saved_eta:.10g}")
print(
    f"Effective ICPT eta: {eta_icpt:.10g} "
    f"({icpt_amplitude_scale:.4f} x saved eta)"
)
print(f"Fitted p=1 ICPT weight: {w_p[0]:.8g}")
print(f"ICPT coupling/weight fit RMSE: {fit_rmse:.4g}")

# Regenerate the two pair models with the fitted absolute weights. The ICPT
# model uses J_n^2; the Shapiro alternative uses |J_n|.
Isp0.fill(0.0)
Isp20.fill(0.0)
Iicpt0.fill(0.0)
Iicpt20.fill(0.0)
for i, (unit_current, weight) in enumerate(zip(unit_pair_currents, w_p)):
    weighted_current = weight * unit_current
    pair_order = i + 1
    Isp0 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=weighted_current,
        A_mV=Abias_mV * icpt_amplitude_scale,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=1,
        absolute=True,
        n_max=nmax,
    )
    Isp20 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=weighted_current,
        A_mV=Abias2_mV * icpt_amplitude_scale,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=1,
        absolute=True,
        n_max=nmax,
    )
    Iicpt0 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=weighted_current,
        A_mV=Abias_mV * icpt_amplitude_scale,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=2,
        n_max=nmax,
    )
    Iicpt20 += pair_shift_pat_kernel(
        V_mV=Vbias0_mV,
        I_=weighted_current,
        A_mV=Abias2_mV * icpt_amplitude_scale,
        nu_GHz=nu_GHz,
        m=pair_order,
        half_offset_branch=pair_order == 2,
        exp=2,
        n_max=nmax,
    )
dGsp0 = np.gradient(Isp0, Vbias0, axis=-1)
dGsp20 = np.gradient(Isp20, Vbias0, axis=-1)
dGicpt0 = np.gradient(Iicpt0, Vbias0, axis=-1)
dGicpt20 = np.gradient(Iicpt20, Vbias0, axis=-1)
# endregion


# region 3D comparison
# The simulated conductance is the sum of the photon-assisted MAR and
# incoherent-pair-tunnelling contributions.
dGsim = dGpamar0 + dGicpt0

fig = plt.figure(figsize=(14, 6), constrained_layout=True)
ax_exp = fig.add_subplot(1, 2, 1, projection="3d")
ax_sim = fig.add_subplot(
    1,
    2,
    2,
    projection="3d",
    sharex=ax_exp,
    sharey=ax_exp,
    sharez=ax_exp,
    shareview=ax_exp,
)

# A common normalization and common axis limits make height and colour
# directly comparable between experiment and simulation.
z_min = min(np.nanmin(dGexp), np.nanmin(dGsim))
z_max = max(np.nanmax(dGexp), np.nanmax(dGsim))
norm = Normalize(vmin=z_min, vmax=z_max)

Vexp_grid, Aexp_grid = np.meshgrid(Vbias, Abias)
Vsim_grid, Asim_grid = np.meshgrid(Vbias0, Abias)

surface_options = {
    "cmap": "viridis",
    "norm": norm,
    "linewidth": 0,
    "antialiased": True,
    # Limit the rendered mesh density to keep interactive rotation smooth.
    "rcount": min(100, Abias.size),
    "ccount": 241,
}
surface_exp = ax_exp.plot_surface(
    Vexp_grid,
    Aexp_grid,
    dGexp,
    **surface_options,
)
ax_sim.plot_surface(
    Vsim_grid,
    Asim_grid,
    dGsim,
    **surface_options,
)

for axis, title in ((ax_exp, r"Experiment: $dG_{\mathrm{exp}}$"),
                    (ax_sim, r"Simulation: $dG_{\mathrm{sim}}$")):
    axis.set(
        title=title,
        xlabel=r"$eV/\Delta$",
        ylabel=r"$A/hf$",
        zlabel=r"$dG/G_N$",
        xlim=(Vbias.min(), Vbias.max()),
        ylim=(Abias.min(), Abias.max()),
        zlim=(z_min, z_max),
    )
    axis.view_init(elev=28, azim=-125)
    axis.set_box_aspect((1.35, 1.0, 0.8))
ax_exp.invert_yaxis()

fig.colorbar(
    surface_exp,
    ax=(ax_exp, ax_sim),
    shrink=0.72,
    pad=0.03,
    label=r"$dG/G_N$",
)
output_path = path / "dG_3d_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved 3D comparison to {output_path}")

# A second rendering clips only the dominant zero-bias peak. Set the upper
# limit from all finite-bias data so that the n != 0 peaks retain their full
# height and color contrast.
zero_bias_half_width = 0.05
finite_bias_exp = np.abs(Vbias) >= zero_bias_half_width
finite_bias_sim = np.abs(Vbias0) >= zero_bias_half_width
contrast_z_max = max(
    np.nanmax(dGexp[:, finite_bias_exp]),
    np.nanmax(dGsim[:, finite_bias_sim]),
)
contrast_norm = Normalize(vmin=z_min, vmax=contrast_z_max, clip=True)

fig_contrast = plt.figure(figsize=(14, 6), constrained_layout=True)
ax_exp_contrast = fig_contrast.add_subplot(1, 2, 1, projection="3d")
ax_sim_contrast = fig_contrast.add_subplot(
    1,
    2,
    2,
    projection="3d",
    sharex=ax_exp_contrast,
    sharey=ax_exp_contrast,
    sharez=ax_exp_contrast,
    shareview=ax_exp_contrast,
)
contrast_surface_options = {
    **surface_options,
    "norm": contrast_norm,
}
surface_exp_contrast = ax_exp_contrast.plot_surface(
    Vexp_grid,
    Aexp_grid,
    np.minimum(dGexp, contrast_z_max),
    **contrast_surface_options,
)
ax_sim_contrast.plot_surface(
    Vsim_grid,
    Asim_grid,
    np.minimum(dGsim, contrast_z_max),
    **contrast_surface_options,
)

for axis, title in (
    (ax_exp_contrast, r"Experiment: $dG_{\mathrm{exp}}$"),
    (ax_sim_contrast, r"Simulation: $dG_{\mathrm{sim}}$"),
):
    axis.set(
        title=title,
        xlabel=r"$eV/\Delta$",
        ylabel=r"$A/hf$",
        zlabel=r"$dG/G_N$",
        xlim=(Vbias.min(), Vbias.max()),
        ylim=(Abias.min(), Abias.max()),
        zlim=(z_min, contrast_z_max),
    )
    axis.view_init(elev=28, azim=-125)
    axis.set_box_aspect((1.35, 1.0, 0.8))
ax_exp_contrast.invert_yaxis()

fig_contrast.colorbar(
    surface_exp_contrast,
    ax=(ax_exp_contrast, ax_sim_contrast),
    shrink=0.72,
    pad=0.03,
    extend="max",
    label=r"$dG/G_N$",
)
contrast_output_path = path / "dG_3d_comparison_aggressive.png"
fig_contrast.savefig(contrast_output_path, dpi=300, bbox_inches="tight")
print(
    "Saved aggressive-contrast 3D comparison to "
    f"{contrast_output_path} (zmax={contrast_z_max:.3g})"
)
# endregion


# region pcolormesh comparison
# Use a tighter shared color range for the maps so that the weaker features
# are not compressed by the sharp zero-bias peak.
mesh_values = np.concatenate((dGexp.ravel(), dGsim.ravel()))
mesh_values = mesh_values[np.isfinite(mesh_values)]
mesh_mean = np.mean(mesh_values)
mesh_std = np.std(mesh_values)
mesh_norm = Normalize(
    vmin=mesh_mean - mesh_std,
    vmax=mesh_mean + mesh_std,
    clip=True,
)

fig_mesh, axes_mesh = plt.subplots(
    1,
    2,
    figsize=(10.5, 5.2),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

mesh_exp = axes_mesh[0].pcolormesh(
    Vbias,
    Abias,
    dGexp,
    shading="auto",
    cmap="viridis",
    norm=mesh_norm,
    rasterized=True,
)
axes_mesh[1].pcolormesh(
    Vbias0,
    Abias,
    dGsim,
    shading="auto",
    cmap="viridis",
    norm=mesh_norm,
    rasterized=True,
)

for axis, title in zip(
    axes_mesh,
    (r"Experiment: $dG_{\mathrm{exp}}$", r"Simulation: $dG_{\mathrm{sim}}$"),
):
    axis.set(
        title=title,
        xlabel=r"$eV/\Delta$",
        xlim=(Vbias.min(), Vbias.max()),
        ylim=(Abias.min(), Abias.max()),
    )
    axis.set_box_aspect(1.0)
axes_mesh[0].set_ylabel(r"$A/hf$")
fig_mesh.colorbar(
    mesh_exp,
    ax=axes_mesh,
    pad=0.02,
    extend="both",
    label=r"$dG/G_N$",
)

mesh_output_path = path / "dG_pcolormesh_comparison.png"
fig_mesh.savefig(mesh_output_path, dpi=300, bbox_inches="tight")
print(f"Saved pcolormesh comparison to {mesh_output_path}")
# endregion


# region amplitude line cuts at n hf / (2 m)
dGmodel_icpt20 = dGpamar20 + dGicpt20
dGmodel_sp20 = dGpamar20 + dGsp20

linecut_pairs = ((0, 1),) + fit_pairs
linecuts_per_harmonic = len(linecut_pairs)

fig_cuts, axes_cuts = plt.subplots(
    w_p.size,
    linecuts_per_harmonic,
    figsize=(3.5 * linecuts_per_harmonic, 3.0 * w_p.size + 0.8),
    sharex=True,
    sharey=False,
    constrained_layout=True,
)
axes_cuts = np.atleast_2d(axes_cuts)
for axis, (n_i, m_i) in zip(axes_cuts.flat, linecut_pairs):
    cut_voltage = n_i * sc.h_pVs * nu_GHz / (2 * m_i * Delta_meV)
    cut_index = np.argmin(np.abs(Vbias0 - cut_voltage))
    sampled_voltage = Vbias0[cut_index]

    axis.plot(
        Abias,
        dGexp0[:, cut_index],
        "o-",
        ms=3.5,
        lw=1.4,
        label="experiment",
    )
    axis.plot(
        Abias2,
        dGmodel_icpt20[:, cut_index],
        lw=1.5,
        label="PAMAR + ICPT",
    )
    axis.plot(
        Abias2,
        dGmodel_sp20[:, cut_index],
        lw=1.5,
        label="PAMAR + Shapiro",
    )
    axis.set(
        title=(
            rf"$n={n_i},\ m={m_i}$"
            "\n"
            rf"$eV/\Delta={sampled_voltage:.3f}$"
        ),
        xlim=(Abias.min(), Abias.max()),
    )
    axis.grid(alpha=0.25)

for axis in axes_cuts[-1, :]:
    axis.set_xlabel(r"$A/hf$")
for axis in axes_cuts[:, 0]:
    axis.set_ylabel(r"$dG/G_N$")

handles, labels = axes_cuts[0, 0].get_legend_handles_labels()
fig_cuts.legend(
    handles,
    labels,
    loc="outside upper center",
    ncols=3,
)
linecut_output_path = path / "dG_amplitude_linecuts.png"
fig_cuts.savefig(linecut_output_path, dpi=300, bbox_inches="tight")
print(f"Saved amplitude line cuts to {linecut_output_path}")
plt.show()
# endregion
