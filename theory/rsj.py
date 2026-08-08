"""Overdamped RSJ model for microwave-driven Shapiro steps.

The junction is a sinusoidal Josephson element in parallel with a constant
resistance. Its equation of motion is

    I_dc + I_ac sin(2 pi f t) = I_c sin(phi) + V / R,
    dphi/dt = 2 e V / hbar.

Currents are expressed in nA, resistance in kOhm, voltage in mV, and drive
frequency in GHz.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import config as _jax_config
from jax import lax
from numpy.typing import ArrayLike, NDArray
from scipy.constants import elementary_charge as e
from scipy.constants import h

_jax_config.update("jax_enable_x64", True)


def sim_V_rsj_mV(
    I_dc_nA: ArrayLike,
    I_ac_nA: ArrayLike,
    I_c_nA: float,
    R_kOhm: float,
    nu_GHz: float,
    n_periods: int = 100,
    n_discard: int = 20,
    n_steps_per_period: int = 200,
) -> NDArray[np.float64]:
    """Return the time-averaged RSJ voltage for DC and AC current sweeps.

    Parameters
    ----------
    I_dc_nA
        One-dimensional DC current-bias axis in nA.
    I_ac_nA
        AC current amplitude in nA. May be a scalar or one-dimensional array.
    I_c_nA
        Critical current in nA.
    R_kOhm
        Linear quasiparticle resistance in kOhm.
    nu_GHz
        Microwave frequency in GHz.
    n_periods
        Total number of microwave periods to simulate.
    n_discard
        Initial periods discarded before averaging.
    n_steps_per_period
        Integration steps per microwave period.

    Returns
    -------
    numpy.ndarray
        Average voltage in mV. The shape is ``(n_ac, n_dc)`` or ``(n_dc,)``
        when ``I_ac_nA`` is scalar. Shapiro steps occur at
        ``V_n = n h f / (2 e)``.
    """
    if I_c_nA < 0.0:
        raise ValueError("I_c_nA must be nonnegative.")
    if R_kOhm <= 0.0:
        raise ValueError("R_kOhm must be positive.")
    if nu_GHz <= 0.0:
        raise ValueError("nu_GHz must be positive.")
    if n_periods <= n_discard:
        raise ValueError("n_periods must be greater than n_discard.")
    if n_discard < 0 or n_steps_per_period < 1:
        raise ValueError("Integration step counts must be nonnegative.")

    ac_is_scalar = np.ndim(I_ac_nA) == 0
    dc = jnp.atleast_1d(jnp.asarray(I_dc_nA, dtype=jnp.float64))
    ac = jnp.atleast_1d(jnp.asarray(I_ac_nA, dtype=jnp.float64))

    dt_s = 1e-9 / (nu_GHz * n_steps_per_period)
    n_steps = n_periods * n_steps_per_period
    n_discard_steps = n_discard * n_steps_per_period
    phase_step_per_mV = 4.0 * np.pi * e / h * dt_s * 1e-3
    drive_phase = 2.0 * np.pi * jnp.arange(n_steps) / n_steps_per_period

    def simulate_amplitude(ac_nA: jnp.ndarray) -> jnp.ndarray:
        def step(carry, drive_phase_i):
            phi, voltage_sum, sample_count, index = carry
            bias_nA = dc + ac_nA * jnp.sin(drive_phase_i)
            qp_current_nA = bias_nA - I_c_nA * jnp.sin(phi)

            # nA * kOhm = microvolt, hence the factor 1e-3 for mV.
            voltage_mV = qp_current_nA * R_kOhm * 1e-3
            phi = phi + phase_step_per_mV * voltage_mV
            phi = jnp.mod(phi + jnp.pi, 2.0 * jnp.pi) - jnp.pi

            include = index >= n_discard_steps
            voltage_sum += jnp.where(include, voltage_mV, 0.0)
            sample_count += include
            return (phi, voltage_sum, sample_count, index + 1), None

        initial = (
            jnp.zeros_like(dc),
            jnp.zeros_like(dc),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
        )
        final, _ = lax.scan(step, initial, drive_phase)
        _, voltage_sum, sample_count, _ = final
        return voltage_sum / sample_count

    voltage_mV = jax.jit(jax.vmap(simulate_amplitude))(ac)
    result = np.asarray(voltage_mV)
    return result[0] if ac_is_scalar else result
