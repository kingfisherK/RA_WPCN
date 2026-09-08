"""Formula-faithful Section-II RA-WPCN system model.

This module implements only the common modeling layer:

* element-wise 3-D HAP postures;
* one closed-form dedicated uplink posture per WD and TDMA slot;
* per-WD closed-form digital MRC;
* actual-power downlink energy covariance;
* energy causality, composite gains, and achievable rates.

It intentionally contains no multi-device posture optimizer, AO routine, or
simulation sweep from Sections IV and V.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray
REFERENCE_BORESIGHT = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class PhysicalPosture:
    """Element-wise mechanical state q_x and posture matrix F_x(q_x)."""

    q: Array
    f: Array


@dataclass(frozen=True)
class SectionIIEvaluation:
    """Evaluation of the Section-II formulas for fixed physical postures."""

    posture_energy: PhysicalPosture
    posture_information: PhysicalPosture | tuple[PhysicalPosture, ...]
    channel_energy: Array
    channel_information: Array
    mrc_information: Array
    harvested_energy: Array
    uplink_power: Array
    gamma: Array
    rates: Array

    @property
    def f_energy(self) -> Array:
        """Direct downlink boresight matrix used by the evaluation."""

        return np.asarray(self.posture_energy.f, dtype=float)

    @property
    def f_information_k(self) -> Array:
        """Dedicated uplink boresights with shape ``(K, 3, N_A)``."""

        if isinstance(self.posture_information, PhysicalPosture):
            return np.repeat(
                np.asarray(self.posture_information.f, dtype=float)[None, :, :],
                self.channel_information.shape[0],
                axis=0,
            )
        return np.stack(
            [np.asarray(posture.f, dtype=float) for posture in self.posture_information],
            axis=0,
        )

    @property
    def common_throughput(self) -> float:
        return float(np.min(self.rates)) if self.rates.size else 0.0


def _real_matrix(value: Array, columns: int, name: str) -> Array:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rows, {columns}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _complex_user_channels(value: Array, name: str) -> Array:
    channels = np.asarray(value, dtype=complex)
    if channels.ndim == 1:
        channels = channels.reshape(1, -1)
    if channels.ndim != 2 or channels.shape[1] == 0:
        raise ValueError(f"{name} must have shape (K, N_A).")
    if not np.all(np.isfinite(channels)):
        raise ValueError(f"{name} must be finite.")
    return channels


def posture_from_angles(zenith: Array, azimuth: Array) -> PhysicalPosture:
    """Construct q_x and F_x for independent element zenith/azimuth angles."""

    zenith = np.asarray(zenith, dtype=float).reshape(-1)
    azimuth = np.asarray(azimuth, dtype=float).reshape(-1)
    if zenith.size == 0 or zenith.shape != azimuth.shape:
        raise ValueError("zenith and azimuth must be nonempty vectors of equal length.")
    if not np.all(np.isfinite(zenith)) or not np.all(np.isfinite(azimuth)):
        raise ValueError("posture angles must be finite.")

    f = np.vstack(
        (
            np.sin(zenith) * np.cos(azimuth),
            np.sin(zenith) * np.sin(azimuth),
            np.cos(zenith),
        )
    )
    q = np.concatenate((zenith, azimuth))
    return PhysicalPosture(q=q, f=f)


def validate_zenith_cap(
    posture: PhysicalPosture,
    alpha_max: float,
    atol: float = 1e-10,
) -> None:
    """Validate ||f_n||=1 and arccos(f_n^T b0) <= alpha_max."""

    f = np.asarray(posture.f, dtype=float)
    if f.ndim != 2 or f.shape[0] != 3:
        raise ValueError("posture.f must have shape (3, N_A).")
    if not 0.0 <= float(alpha_max) <= np.pi / 2.0:
        raise ValueError("alpha_max must lie in [0, pi/2].")
    if not np.allclose(np.linalg.norm(f, axis=0), 1.0, atol=atol):
        raise ValueError("every posture column must have unit norm.")
    tilt = np.arccos(np.clip(REFERENCE_BORESIGHT @ f, -1.0, 1.0))
    if np.any(tilt > float(alpha_max) + atol):
        raise ValueError("posture violates the positive-z spherical cap.")


def validate_spherical_cap_boresights(
    f: Array,
    alpha_max: float,
    *,
    reference_boresight: Array = REFERENCE_BORESIGHT,
    atol: float = 1e-10,
) -> None:
    """Validate direct ``3 x N_A`` boresights on a spherical cap."""

    f = np.asarray(f, dtype=float)
    axis = np.asarray(reference_boresight, dtype=float).reshape(3)
    if f.ndim != 2 or f.shape[0] != 3 or f.shape[1] == 0:
        raise ValueError("f must have shape (3, N_A).")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(axis)):
        raise ValueError("boresights and reference axis must be finite.")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0 or not 0.0 <= float(alpha_max) <= np.pi / 2.0:
        raise ValueError("require a nonzero axis and alpha_max in [0, pi/2].")
    axis = axis / axis_norm
    if not np.allclose(np.linalg.norm(f, axis=0), 1.0, atol=atol, rtol=0.0):
        raise ValueError("every boresight column must have unit norm.")
    if np.any(axis @ f < np.cos(float(alpha_max)) - atol):
        raise ValueError("a boresight violates the spherical cap.")


def element_to_device_geometry(
    element_positions: Array,
    device_positions: Array,
) -> tuple[Array, Array]:
    """Return s_{k,n} and r_{k,n} with shapes (K,N_A,3) and (K,N_A)."""

    elements = _real_matrix(element_positions, 3, "element_positions")
    devices = _real_matrix(device_positions, 3, "device_positions")
    displacement = devices[:, None, :] - elements[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    if np.any(distances <= 0.0):
        raise ValueError("a device must not coincide with an HAP element.")
    directions = displacement / distances[:, :, None]
    return directions, distances


def directional_power_gains(
    posture: PhysicalPosture,
    directions: Array,
    rho: float,
) -> Array:
    """Evaluate g_{k,n} for one physical posture shared by all users."""

    directions = np.asarray(directions, dtype=float)
    f = np.asarray(posture.f, dtype=float)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError("directions must have shape (K, N_A, 3).")
    if f.shape != (3, directions.shape[1]):
        raise ValueError("posture and directions use different element counts.")
    if float(rho) < 0.0:
        raise ValueError("rho must be nonnegative.")

    matches = np.einsum("knd,dn->kn", directions, f)
    peak_gain = 2.0 * (2.0 * float(rho) + 1.0)
    return peak_gain * np.maximum(matches, 0.0) ** (2.0 * float(rho))


def per_slot_directional_power_gains(
    f_information_k: Array,
    directions: Array,
    rho: float,
) -> Array:
    """Evaluate the new uplink gain using one ``F_{I,k}`` per TDMA slot."""

    directions = np.asarray(directions, dtype=float)
    boresights = np.asarray(f_information_k, dtype=float)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError("directions must have shape (K, N_A, 3).")
    expected = (directions.shape[0], 3, directions.shape[1])
    if boresights.shape != expected:
        raise ValueError(f"f_information_k must have shape {expected}.")
    if float(rho) < 0.0:
        raise ValueError("rho must be nonnegative.")
    if not np.allclose(np.linalg.norm(boresights, axis=1), 1.0, atol=1e-10):
        raise ValueError("every F_{I,k} boresight column must have unit norm.")

    matches = np.einsum("knd,kdn->kn", directions, boresights, optimize=True)
    peak_gain = 2.0 * (2.0 * float(rho) + 1.0)
    return peak_gain * np.maximum(matches, 0.0) ** (2.0 * float(rho))


def per_slot_effective_channels(
    propagation_information: Array,
    f_information_k: Array,
    directions: Array,
    rho: float,
) -> Array:
    """Return all dedicated uplink channels after applying ``F_{I,k}``."""

    gains = per_slot_directional_power_gains(
        f_information_k,
        directions,
        rho,
    )
    return effective_channels(propagation_information, gains)


def closed_form_uplink_boresights(
    directions: Array,
    alpha_max: float,
    reference_boresight: Array = REFERENCE_BORESIGHT,
) -> Array:
    """Project every ``s_{k,n}`` onto the new uplink spherical cap.

    The returned tensor contains all dedicated closed-form solutions and has
    shape ``(K, 3, N_A)``.
    """

    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError("directions must have shape (K, N_A, 3).")
    if not 0.0 <= float(alpha_max) <= np.pi / 2.0:
        raise ValueError("alpha_max must lie in [0, pi/2].")
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("directions must be nonzero.")
    targets = directions / norms

    axis = np.asarray(reference_boresight, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("reference_boresight must be nonzero.")
    axis = axis / axis_norm
    alignment = np.clip(np.einsum("knd,d->kn", targets, axis), -1.0, 1.0)
    outside = alignment < np.cos(float(alpha_max))
    projected = targets.copy()
    if np.any(outside):
        tangent = targets - alignment[:, :, None] * axis
        tangent_norm = np.linalg.norm(tangent, axis=2)
        degenerate = outside & (tangent_norm <= 1e-14)
        if np.any(degenerate):
            trial = np.array([1.0, 0.0, 0.0])
            if abs(float(trial @ axis)) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            orthogonal = trial - float(trial @ axis) * axis
            orthogonal /= np.linalg.norm(orthogonal)
            tangent[degenerate] = orthogonal
            tangent_norm[degenerate] = 1.0
        unit_tangent = np.zeros_like(tangent)
        np.divide(
            tangent,
            tangent_norm[:, :, None],
            out=unit_tangent,
            where=tangent_norm[:, :, None] > 0.0,
        )
        boundary = (
            np.cos(float(alpha_max)) * axis
            + np.sin(float(alpha_max)) * unit_tangent
        )
        projected[outside] = boundary[outside]
    return np.transpose(projected, (0, 2, 1))


def effective_channels(propagation: Array, directional_gains: Array) -> Array:
    """Evaluate [h_k^x]_n = sqrt(g_{k,n}^x) * h_tilde_{k,n}^x."""

    propagation = _complex_user_channels(propagation, "propagation")
    gains = np.asarray(directional_gains, dtype=float)
    if gains.shape != propagation.shape:
        raise ValueError("directional_gains and propagation must both have shape (K,N_A).")
    if np.any(gains < 0.0) or not np.all(np.isfinite(gains)):
        raise ValueError("directional gains must be finite and nonnegative.")
    return np.sqrt(gains) * propagation


def energy_covariance_from_beams(beams: Array) -> Array:
    """Return S_E = sum_l v_l v_l^H for beams stored as columns."""

    beams = np.asarray(beams, dtype=complex)
    if beams.ndim == 1:
        beams = beams.reshape(-1, 1)
    if beams.ndim != 2 or beams.shape[0] == 0:
        raise ValueError("beams must have shape (N_A, L_E).")
    return beams @ beams.conj().T


def validate_energy_covariance(
    covariance: Array,
    max_power: float,
    atol: float = 1e-10,
) -> Array:
    """Validate S_E >= 0 and Tr(S_E) <= P_A; return its Hermitian part."""

    covariance = np.asarray(covariance, dtype=complex)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square.")
    if float(max_power) < 0.0:
        raise ValueError("max_power must be nonnegative.")
    hermitian = 0.5 * (covariance + covariance.conj().T)
    if not np.allclose(covariance, hermitian, atol=atol):
        raise ValueError("covariance must be Hermitian.")
    if np.min(np.linalg.eigvalsh(hermitian)) < -atol:
        raise ValueError("covariance must be positive semidefinite.")
    if float(np.trace(hermitian).real) > float(max_power) + atol:
        raise ValueError("covariance trace exceeds max_power.")
    return hermitian


def received_rf_powers(channels_energy: Array, covariance: Array) -> Array:
    """Return h_k^H S_E h_k for every WD."""

    channels = _complex_user_channels(channels_energy, "channels_energy")
    covariance = np.asarray(covariance, dtype=complex)
    if covariance.shape != (channels.shape[1], channels.shape[1]):
        raise ValueError("covariance dimension does not match the channels.")
    powers = np.einsum(
        "kn,nm,km->k",
        channels.conj(),
        covariance,
        channels,
        optimize=True,
    ).real
    return np.maximum(powers, 0.0)


def harvested_energies(
    channels_energy: Array,
    covariance: Array,
    tau_0: float,
    zeta: float,
    physical_time: float = 1.0,
) -> Array:
    """Return E_k = zeta tau_0 T_phys h_k^H S_E h_k."""

    if tau_0 < 0.0 or not 0.0 < zeta <= 1.0 or physical_time <= 0.0:
        raise ValueError("invalid time or energy-conversion parameter.")
    return (
        float(zeta)
        * float(tau_0)
        * float(physical_time)
        * received_rf_powers(channels_energy, covariance)
    )


def mrc_combiners(channels_information: Array) -> Array:
    """Return U_I=[u_I,1,...,u_I,K] for the shared-posture uplink channels."""

    channels = _complex_user_channels(channels_information, "channels_information")
    norms = np.linalg.norm(channels, axis=1)
    combiners = np.zeros_like(channels)
    np.divide(channels, norms[:, None], out=combiners, where=norms[:, None] > 0.0)
    return combiners.T


def mrc_gains(channels_information: Array) -> Array:
    """Return ||h_k^I||_2^2 after closed-form MRC substitution."""

    channels = _complex_user_channels(channels_information, "channels_information")
    return np.sum(np.abs(channels) ** 2, axis=1).real


def composite_gains(
    channels_energy: Array,
    channels_information: Array,
    covariance: Array,
    zeta: float,
    noise_power: float,
) -> Array:
    """Return gamma_k with actual power already contained in S_E."""

    if not 0.0 < zeta <= 1.0 or noise_power <= 0.0:
        raise ValueError("zeta and noise_power must be positive.")
    downlink = received_rf_powers(channels_energy, covariance)
    uplink = mrc_gains(channels_information)
    if downlink.shape != uplink.shape:
        raise ValueError("downlink and uplink user counts do not match.")
    return float(zeta) * downlink * uplink / float(noise_power)


def optimal_uplink_powers(
    harvested_energy: Array,
    tau_k: Array,
    physical_time: float = 1.0,
) -> Array:
    """Use all harvested energy when tau_k>0; return zero for inactive slots."""

    energy = np.asarray(harvested_energy, dtype=float).reshape(-1)
    tau = np.asarray(tau_k, dtype=float).reshape(-1)
    if energy.shape != tau.shape or np.any(energy < 0.0) or np.any(tau < 0.0):
        raise ValueError("harvested_energy and tau_k must be matched nonnegative vectors.")
    if physical_time <= 0.0:
        raise ValueError("physical_time must be positive.")
    powers = np.zeros_like(energy)
    active = tau > 0.0
    powers[active] = energy[active] / (tau[active] * float(physical_time))
    return powers


def achievable_rates(tau_0: float, tau_k: Array, gamma: Array) -> Array:
    """Evaluate R_k with its continuous extension R_k=0 at tau_k=0."""

    tau = np.asarray(tau_k, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if tau.shape != gamma.shape or tau_0 < 0.0:
        raise ValueError("tau_k and gamma must have the same shape and times must be nonnegative.")
    if np.any(tau < 0.0) or np.any(gamma < 0.0):
        raise ValueError("tau_k and gamma must be nonnegative.")
    rates = np.zeros_like(gamma)
    active = tau > 0.0
    rates[active] = tau[active] * np.log2(
        1.0 + float(tau_0) * gamma[active] / tau[active]
    )
    return rates


def evaluate_section_ii_model(
    *,
    element_positions: Array,
    device_positions: Array,
    propagation_energy: Array,
    propagation_information: Array,
    covariance_energy: Array,
    tau_0: float,
    tau_k: Array,
    zeta: float,
    noise_power: float,
    max_power: float,
    rho: float,
    frame_time: float = 1.0,
    physical_time: float = 1.0,
    zenith_energy: Array | None = None,
    azimuth_energy: Array | None = None,
    zenith_information: Array | None = None,
    azimuth_information: Array | None = None,
    f_energy: Array | None = None,
    f_information_k: Array | None = None,
    alpha_max: float | None = None,
    alpha_max_energy: float | None = None,
    alpha_max_information: float | None = None,
) -> SectionIIEvaluation:
    """Evaluate either the unchanged shared-UL model or dedicated ``F_{I,k}``.

    The direct-boresight path is used by Proposed.  The angle-coordinate path
    remains available so all unchanged comparison and regression scripts keep
    their original configurations.
    """

    directions, _ = element_to_device_geometry(element_positions, device_positions)
    direct_mode = f_energy is not None or f_information_k is not None
    if direct_mode:
        if f_energy is None or f_information_k is None:
            raise ValueError("direct mode requires f_energy and f_information_k.")
        f_energy = np.asarray(f_energy, dtype=float)
        f_information_k = np.asarray(f_information_k, dtype=float)
        expected = (directions.shape[0], 3, directions.shape[1])
        if f_energy.shape != (3, directions.shape[1]):
            raise ValueError("f_energy must have shape (3, N_A).")
        if f_information_k.shape != expected:
            raise ValueError(f"f_information_k must have shape {expected}.")
        if alpha_max_energy is not None:
            validate_spherical_cap_boresights(f_energy, alpha_max_energy)
        if alpha_max_information is not None:
            for boresights in f_information_k:
                validate_spherical_cap_boresights(
                    boresights, alpha_max_information
                )

        def posture_from_boresights(boresights: Array) -> PhysicalPosture:
            zenith = np.arccos(np.clip(boresights[2], -1.0, 1.0))
            azimuth = np.arctan2(boresights[1], boresights[0])
            return PhysicalPosture(
                q=np.concatenate((zenith, azimuth)),
                f=np.asarray(boresights, dtype=float),
            )

        posture_energy = posture_from_boresights(f_energy)
        posture_information = tuple(
            posture_from_boresights(boresights)
            for boresights in f_information_k
        )
        gains_energy = directional_power_gains(posture_energy, directions, rho)
        gains_information = per_slot_directional_power_gains(
            f_information_k, directions, rho
        )
    else:
        if any(
            value is None
            for value in (
                zenith_energy,
                azimuth_energy,
                zenith_information,
                azimuth_information,
            )
        ):
            raise ValueError("angle mode requires all four posture angle arrays.")
        posture_energy = posture_from_angles(zenith_energy, azimuth_energy)
        posture_information = posture_from_angles(
            zenith_information, azimuth_information
        )
        if posture_energy.f.shape != posture_information.f.shape:
            raise ValueError(
                "downlink and uplink postures must use the same HAP elements."
            )
        if alpha_max is not None:
            validate_zenith_cap(posture_energy, alpha_max)
            validate_zenith_cap(posture_information, alpha_max)
        gains_energy = directional_power_gains(posture_energy, directions, rho)
        gains_information = directional_power_gains(
            posture_information, directions, rho
        )
    channel_energy = effective_channels(propagation_energy, gains_energy)
    channel_information = effective_channels(
        propagation_information, gains_information
    )
    covariance = validate_energy_covariance(covariance_energy, max_power)
    tau_k = np.asarray(tau_k, dtype=float).reshape(-1)
    if channel_energy.shape[0] != tau_k.size:
        raise ValueError("tau_k must contain one entry per WD.")
    if frame_time <= 0.0:
        raise ValueError("frame_time must be positive.")
    if float(tau_0) + float(np.sum(tau_k)) > float(frame_time) + 1e-12:
        raise ValueError("time allocation exceeds one frame.")

    harvested = harvested_energies(
        channel_energy,
        covariance,
        tau_0,
        zeta,
        physical_time=physical_time,
    )
    uplink_power = optimal_uplink_powers(
        harvested, tau_k, physical_time=physical_time
    )
    gamma = composite_gains(
        channel_energy,
        channel_information,
        covariance,
        zeta,
        noise_power,
    )
    return SectionIIEvaluation(
        posture_energy=posture_energy,
        posture_information=posture_information,
        channel_energy=channel_energy,
        channel_information=channel_information,
        mrc_information=mrc_combiners(channel_information),
        harvested_energy=harvested,
        uplink_power=uplink_power,
        gamma=gamma,
        rates=achievable_rates(tau_0, tau_k, gamma),
    )


def dbm_to_watt(dbm: float) -> float:
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def db_to_linear(db: float) -> float:
    return 10.0 ** (float(db) / 10.0)
