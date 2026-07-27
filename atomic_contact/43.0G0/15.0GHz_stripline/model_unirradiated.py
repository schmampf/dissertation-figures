from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import superconductivity.utilities as scutil
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent


##############################################################################
##############################################################################
### Trace data
##############################################################################
##############################################################################

TRACE_INDEX = 0

# The raw Trace objects live in cache.pkl. The ``traces`` entry currently
# present in eva.npz only contains their field names, while the sampled curves
# and derivatives are stored correctly in the other eva.npz arrays.
cache = scutil.load_cache("cache", path=HERE)
data = np.load(HERE / "eva.npz")

trace = cache.traces[TRACE_INDEX]
It_nA = np.asarray(trace["I_nA"], dtype=float)
Vt_mV = np.asarray(trace["V_mV"], dtype=float)

Vbias_mV = np.asarray(data["Vbias_mV"], dtype=float)
Ibias_nA = np.asarray(data["Ibias_nA"], dtype=float)
Iexp_nA = np.asarray(data["Iexp_nA"][TRACE_INDEX], dtype=float)
Vexp_mV = np.asarray(data["Vexp_mV"][TRACE_INDEX], dtype=float)
dGexp_G0 = np.asarray(data["dGexp_G0"][TRACE_INDEX], dtype=float)
dRexp_R0 = np.asarray(data["dRexp_R0"][TRACE_INDEX], dtype=float)

##############################################################################
##############################################################################
### Modeling starts here
##############################################################################
##############################################################################


##############################################################################
##############################################################################
### Plotting starts here
##############################################################################
##############################################################################

fig = plt.figure(figsize=(10.0, 8.0), layout="constrained")
grid = GridSpec(
    2,
    2,
    figure=fig,
    width_ratios=(4.0, 1.35),
    height_ratios=(4.0, 1.35),
)
ax_iv = fig.add_subplot(grid[0, 0])
ax_didv = fig.add_subplot(grid[1, 0], sharex=ax_iv)
ax_dvdi = fig.add_subplot(grid[0, 1], sharey=ax_iv)
ax_empty = fig.add_subplot(grid[1, 1])
ax_empty.axis("off")

# Plot the acquisition-order samples sparsely enough that individual raw
# points remain visible, then overlay both single-valued resamplings.
raw_stride = max(1, It_nA.size // 12_000)
ax_iv.plot(
    Vt_mV[::raw_stride],
    It_nA[::raw_stride],
    ".",
    color="black",
    markersize=1.2,
    alpha=0.65,
    label=r"raw $I(t),V(t)$",
    rasterized=True,
)
ax_iv.plot(
    Vbias_mV,
    Iexp_nA,
    color="C0",
    linewidth=1.4,
    label=r"sampled $I(V)$",
)
ax_iv.plot(
    Vexp_mV,
    Ibias_nA,
    color="C1",
    linewidth=1.4,
    label=r"sampled $V(I)$",
)
ax_iv.axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
ax_iv.axvline(0.0, color="0.8", linewidth=0.7, zorder=0)
ax_iv.set_ylabel(r"$I$ (nA)")
ax_iv.legend(loc="upper left", frameon=False, markerscale=3)
ax_iv.tick_params(labelbottom=False)

ax_didv.plot(Vbias_mV, dGexp_G0, color="C0", linewidth=1.2)
ax_didv.axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
ax_didv.set_xlabel(r"$V$ (mV)")
ax_didv.set_ylabel(r"$dI/dV$ ($G_0$)")

# Put dV/dI on the horizontal axis so this panel shares the current axis of
# the main IV plot and aligns every current-biased transition vertically.
ax_dvdi.plot(dRexp_R0, Ibias_nA, color="C1", linewidth=1.2)
ax_dvdi.axvline(0.0, color="0.8", linewidth=0.7, zorder=0)
ax_dvdi.set_xlabel(r"$dV/dI$ ($R_0$)")
ax_dvdi.tick_params(labelleft=False)

for axis in (ax_iv, ax_didv, ax_dvdi):
    axis.grid(alpha=0.15)

output_stem = HERE / "unirradiated_iv_derivatives"
fig.savefig(output_stem.with_suffix(".png"), dpi=300)
fig.savefig(output_stem.with_suffix(".pdf"))
plt.show()
