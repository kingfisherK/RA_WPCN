"""Spherical-cap projected-gradient and legacy angle-box Frank--Wolfe updates.

The active spherical-cap variable is the boresight matrix ``F``.  Legacy
angle-box baselines use
``q = [zenith_1,...,zenith_N,azimuth_1,...,azimuth_N]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PostureChannelState:
    """Effective channels and their element-local angle derivatives."""

    channels: Array
    directional_gains: Array
    derivative_zenith: Array
    derivative_azimuth: Array


@dataclass(frozen=True)
class BoxFrankWolfeResult:
    """Result and complete convergence trace for one mechanical posture block."""

    q: Array
    smooth_objective: float
    true_objective: float
    smooth_objective_history: Array
    true_objective_history: Array
    fw_gap_history: Array
    step_history: Array
    relative_gain_history: Array
    point_change_history: Array
    iterations: int
    converged: bool
    reason: str
    feasible: bool
    report: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BoresightChannelState:
    """Effective channels and direct derivatives with respect to boresights."""

    channels: Array
    directional_gains: Array
    channel_gradients: Array


@dataclass(frozen=True)
class SphericalCapProjectedGradientResult:
    """Convergence trace for one spherical-cap projected-gradient block."""

    f: Array
    smooth_objective: float
    true_objective: float
    smooth_objective_history: Array
    true_objective_history: Array
    projected_gradient_mapping_history: Array
    step_history: Array
    relative_gain_history: Array
    point_change_history: Array
    next_step: float
    iterations: int
    converged: bool
    reason: str
    feasible: bool
    report: dict[str, object] = field(default_factory=dict)


def validate_spherical_cap(
    f: Array,
    alpha_max: float,
    *,
    axis: Array = np.array([0.0, 0.0, 1.0]),
    atol: float = 1e-10,
) -> None:
    """Validate column-wise unit boresights inside a spherical cap."""

    f = np.asarray(f, dtype=float)
    axis = np.asarray(axis, dtype=float).reshape(3)
    if f.ndim != 2 or f.shape[0] != 3 or f.shape[1] == 0:
        raise ValueError("f must have shape (3, N_A).")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(axis)):
        raise ValueError("boresights and cap axis must be finite.")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0 or not 0.0 <= float(alpha_max) <= np.pi / 2.0:
        raise ValueError("require a nonzero cap axis and alpha_max in [0, pi/2].")
    axis = axis / axis_norm
    if not np.allclose(np.linalg.norm(f, axis=0), 1.0, atol=atol, rtol=0.0):
        raise ValueError("every boresight must have unit norm.")
    if np.any(axis @ f < np.cos(float(alpha_max)) - atol):
        raise ValueError("a boresight lies outside the spherical cap.")


def project_spherical_cap(
    x: Array,
    alpha_max: float,
    *,
    axis: Array = np.array([0.0, 0.0, 1.0]),
) -> Array:
    """Apply the manuscript spherical-cap projection column by column."""

    x = np.asarray(x, dtype=float)
    axis = np.asarray(axis, dtype=float).reshape(3)
    if x.ndim != 2 or x.shape[0] != 3 or x.shape[1] == 0:
        raise ValueError("x must have shape (3, N_A).")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(axis)):
        raise ValueError("x and the cap axis must be finite.")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0 or not 0.0 <= float(alpha_max) <= np.pi / 2.0:
        raise ValueError("require a nonzero cap axis and alpha_max in [0, pi/2].")
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("every projected vector must be nonzero.")

    axis = axis / axis_norm
    normalized = x / norms
    alignment = axis @ normalized
    outside = alignment < np.cos(float(alpha_max))
    projected = normalized.copy()
    if np.any(outside):
        tangent = normalized - axis[:, None] * alignment[None, :]
        tangent_norms = np.linalg.norm(tangent, axis=0)
        degenerate = outside & (tangent_norms <= 1e-14)
        if np.any(degenerate):
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(reference @ axis)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            orthogonal = reference - float(reference @ axis) * axis
            orthogonal /= np.linalg.norm(orthogonal)
            tangent[:, degenerate] = orthogonal[:, None]
            tangent_norms[degenerate] = 1.0
        projected[:, outside] = (
            np.cos(float(alpha_max)) * axis[:, None]
            + np.sin(float(alpha_max))
            * tangent[:, outside]
            / tangent_norms[outside][None, :]
        )
    return projected


def boresight_channel_state(
    f: Array,
    directions: Array,
    propagation: Array,
    rho: float,
) -> BoresightChannelState:
    """Evaluate hard cosine channels and direct boresight derivatives.

    For the Rcom model, ``rho=2`` makes the channel amplitude proportional to
    ``[f^T s]_+^2``.  Its first derivative is continuous at the front/back
    boundary, so no auxiliary positive-part smoothing is used.
    """

    f = np.asarray(f, dtype=float)
    directions = np.asarray(directions, dtype=float)
    propagation = np.asarray(propagation, dtype=complex)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError("directions must have shape (K, N_A, 3).")
    if f.shape != (3, directions.shape[1]):
        raise ValueError("f must have shape (3, N_A).")
    if propagation.shape != directions.shape[:2]:
        raise ValueError("propagation must have shape (K, N_A).")
    if float(rho) <= 1.0:
        raise ValueError("direct hard-channel differentiation requires rho > 1.")

    matches = np.einsum("knd,dn->kn", directions, f, optimize=True)
    positive = np.maximum(matches, 0.0)
    peak_gain = 2.0 * (2.0 * float(rho) + 1.0)
    sqrt_peak = np.sqrt(peak_gain)
    amplitudes = sqrt_peak * positive ** float(rho)
    channels = amplitudes * propagation
    amplitude_scale = (
        sqrt_peak
        * float(rho)
        * positive ** (float(rho) - 1.0)
        * (matches > 0.0)
    )
    channel_gradients = (
        propagation[:, :, None]
        * amplitude_scale[:, :, None]
        * directions
    )
    return BoresightChannelState(
        channels=channels,
        directional_gains=peak_gain * positive ** (2.0 * float(rho)),
        channel_gradients=channel_gradients,
    )


def mrc_boresight_gains_and_gradients(
    state: BoresightChannelState,
) -> tuple[Array, Array]:
    """Return MRC gains and direct gradients with shape ``(K,3,N_A)``."""

    channels = np.asarray(state.channels, dtype=complex)
    gains = np.sum(np.abs(channels) ** 2, axis=1).real
    gradients_knd = 2.0 * np.real(
        np.conj(channels)[:, :, None] * state.channel_gradients
    )
    return gains, np.transpose(gradients_knd, (0, 2, 1))


def quadratic_boresight_values_and_gradients(
    state: BoresightChannelState,
    matrix: Array,
) -> tuple[Array, Array]:
    """Return ``h_k^H X h_k`` and gradients with shape ``(K,3,N_A)``."""

    channels = np.asarray(state.channels, dtype=complex)
    matrix = np.asarray(matrix, dtype=complex)
    n_antennas = channels.shape[1]
    if matrix.shape != (n_antennas, n_antennas):
        raise ValueError("matrix dimension does not match the posture channels.")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    matrix_channels = channels @ hermitian.T
    values = np.sum(np.conj(channels) * matrix_channels, axis=1).real
    gradients_knd = 2.0 * np.real(
        np.conj(state.channel_gradients) * matrix_channels[:, :, None]
    )
    return np.maximum(values, 0.0), np.transpose(gradients_knd, (0, 2, 1))


def project_boresight_gradients(f: Array, gradients: Array) -> Array:
    """Project column-wise Euclidean gradients onto sphere tangent spaces."""

    f = np.asarray(f, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    if f.shape != gradients.shape or f.ndim != 2 or f.shape[0] != 3:
        raise ValueError("f and gradients must both have shape (3, N_A).")
    return gradients - np.sum(gradients * f, axis=0, keepdims=True) * f


def normalized_spherical_cap_step(current: Array, oracle: Array, step: float) -> Array:
    """Apply the normalized chord update from the paper."""

    trial = np.asarray(current, dtype=float) + float(step) * (
        np.asarray(oracle, dtype=float) - np.asarray(current, dtype=float)
    )
    norms = np.linalg.norm(trial, axis=0, keepdims=True)
    if np.any(norms <= 0.0):
        raise RuntimeError("normalized spherical-cap step reached a zero vector.")
    return trial / norms


def pga_spherical_cap_block(
    smooth_objective: Callable[[Array], float],
    gradient: Callable[[Array], Array],
    true_objective: Callable[[Array], float],
    initial_f: Array,
    alpha_max: float,
    *,
    max_iter: int = 50,
    tolerance: float = 1e-7,
    armijo_c: float = 1e-4,
    backtracking: float = 0.5,
    min_step: float = 1e-12,
    max_step: float = 1.0,
    initial_step: float = 1.0,
    true_tolerance: float = 1e-13,
) -> SphericalCapProjectedGradientResult:
    """Run the manuscript projected-gradient ascent over spherical caps."""

    if max_iter <= 0 or tolerance <= 0.0:
        raise ValueError("max_iter and tolerance must be positive.")
    if not 0.0 < armijo_c < 0.5 or not 0.0 < backtracking < 1.0:
        raise ValueError("invalid Armijo/backtracking constants.")
    if min_step <= 0.0 or max_step < min_step or initial_step <= 0.0:
        raise ValueError("require 0 < min_step <= max_step and initial_step > 0.")
    current = np.asarray(initial_f, dtype=float).copy()
    validate_spherical_cap(current, alpha_max)
    current_smooth = float(smooth_objective(current))
    current_true = float(true_objective(current))
    if not np.isfinite(current_smooth) or not np.isfinite(current_true):
        raise ValueError("initial orientation objective is not finite.")

    smooth_history = [current_smooth]
    true_history = [current_true]
    mapping_history: list[float] = []
    step_history: list[float] = []
    relative_history: list[float] = []
    point_history: list[float] = []
    converged = False
    reason = "max_iter_reached"
    step_state = min(float(max_step), max(float(min_step), float(initial_step)))

    for _ in range(max_iter):
        raw_gradient = np.asarray(gradient(current), dtype=float)
        if raw_gradient.shape != current.shape or not np.all(np.isfinite(raw_gradient)):
            raise ValueError("gradient must match the current boresights and be finite.")
        tangent_gradient = project_boresight_gradients(current, raw_gradient)
        mapping_trial = project_spherical_cap(
            current + step_state * tangent_gradient,
            alpha_max,
        )
        mapping_norm_sq = float(
            np.sum(((mapping_trial - current) / step_state) ** 2)
        )
        mapping_history.append(mapping_norm_sq)
        if mapping_norm_sq <= tolerance:
            converged = True
            reason = "projected_gradient_mapping_below_tol"
            break

        step = step_state
        accepted = False
        while True:
            step = max(float(min_step), float(backtracking) * step)
            candidate = project_spherical_cap(
                current + step * tangent_gradient,
                alpha_max,
            )
            validate_spherical_cap(candidate, alpha_max)
            candidate_smooth = float(smooth_objective(candidate))
            candidate_true = float(true_objective(candidate))
            armijo_increment = float(
                np.sum(tangent_gradient * (candidate - current))
            )
            smooth_ok = (
                candidate_smooth
                >= current_smooth + float(armijo_c) * armijo_increment
            )
            true_ok = candidate_true >= current_true - true_tolerance * max(
                1.0, abs(current_true)
            )
            if (
                np.isfinite(candidate_smooth)
                and np.isfinite(candidate_true)
                and smooth_ok
                and true_ok
            ):
                accepted = True
                break
            if step <= min_step:
                break
        if not accepted:
            reason = "no_admissible_step"
            break

        point_change = float(np.linalg.norm(candidate - current))
        relative_gain = abs(candidate_true - current_true) / max(1.0, abs(current_true))
        current = candidate
        current_smooth = candidate_smooth
        current_true = candidate_true
        step_history.append(step)
        smooth_history.append(current_smooth)
        true_history.append(current_true)
        relative_history.append(relative_gain)
        point_history.append(point_change)
        step_state = min(float(max_step), step / float(backtracking))

    smooth_array = np.asarray(smooth_history, dtype=float)
    true_array = np.asarray(true_history, dtype=float)
    mapping_array = np.asarray(mapping_history, dtype=float)
    step_array = np.asarray(step_history, dtype=float)
    relative_array = np.asarray(relative_history, dtype=float)
    point_array = np.asarray(point_history, dtype=float)
    finite = bool(
        all(
            np.all(np.isfinite(values))
            for values in (
                smooth_array,
                true_array,
                mapping_array,
                step_array,
                relative_array,
                point_array,
            )
        )
    )
    smooth_monotone = bool(
        smooth_array.size < 2 or np.all(np.diff(smooth_array) >= -1e-10)
    )
    true_monotone = bool(
        true_array.size < 2 or np.all(np.diff(true_array) >= -1e-10)
    )
    feasible = True
    try:
        validate_spherical_cap(current, alpha_max)
    except ValueError:
        feasible = False
    final_raw_gradient = np.asarray(gradient(current), dtype=float)
    final_tangent_gradient = project_boresight_gradients(current, final_raw_gradient)
    final_mapping_trial = project_spherical_cap(
        current + step_state * final_tangent_gradient,
        alpha_max,
    )
    final_mapping = float(
        np.sum(((final_mapping_trial - current) / step_state) ** 2)
    )
    final_relative = float(relative_array[-1]) if relative_array.size else 0.0
    final_change = float(point_array[-1]) if point_array.size else 0.0
    stationary = bool(final_mapping <= tolerance)
    if stationary and reason == "max_iter_reached":
        converged = True
        reason = "projected_gradient_mapping_below_tol"
    block_stable = bool(stationary or reason == "no_admissible_step")
    report = {
        "verified": bool(finite and feasible and smooth_monotone and true_monotone),
        "converged": converged,
        "projected_gradient_stationary": stationary,
        "block_stable": block_stable,
        "reason": reason,
        "iterations": len(step_history),
        "finite": finite,
        "feasible": feasible,
        "monotone_smooth_objective": smooth_monotone,
        "monotone_true_objective": true_monotone,
        "initial_smooth_objective": float(smooth_array[0]),
        "final_smooth_objective": float(smooth_array[-1]),
        "initial_true_objective": float(true_array[0]),
        "final_true_objective": float(true_array[-1]),
        "final_projected_gradient_mapping": final_mapping,
        "final_relative_gain": final_relative,
        "final_point_change": final_change,
        "tolerance": float(tolerance),
    }
    return SphericalCapProjectedGradientResult(
        f=current,
        smooth_objective=current_smooth,
        true_objective=current_true,
        smooth_objective_history=smooth_array,
        true_objective_history=true_array,
        projected_gradient_mapping_history=mapping_array,
        step_history=step_array,
        relative_gain_history=relative_array,
        point_change_history=point_array,
        next_step=step_state,
        iterations=len(step_history),
        converged=converged,
        reason=reason,
        feasible=feasible,
        report=report,
    )


def assert_spherical_cap_pga_result(
    result: SphericalCapProjectedGradientResult,
    *,
    require_converged: bool = False,
) -> dict[str, object]:
    """Raise when a spherical-cap PGA trace is invalid or nonmonotone."""

    required = {
        "finite": result.report.get("finite", False),
        "feasible": result.report.get("feasible", False),
        "monotone smooth objective": result.report.get(
            "monotone_smooth_objective", False
        ),
        "monotone true objective": result.report.get(
            "monotone_true_objective", False
        ),
    }
    if require_converged:
        required["projected-gradient stationarity"] = result.report.get(
            "projected_gradient_stationary", False
        )
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "Spherical-cap projected-gradient verification failed: "
            + ", ".join(failed)
            + f"; report={result.report}"
        )
    return result.report


def split_posture_angles(q: Array) -> tuple[Array, Array]:
    """Split an even-length posture vector into zenith and azimuth arrays."""

    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size == 0 or q.size % 2:
        raise ValueError("q must contain N zenith angles followed by N azimuth angles.")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite.")
    n_antennas = q.size // 2
    return q[:n_antennas], q[n_antennas:]


def validate_angle_box(q: Array, lower: Array, upper: Array, atol: float = 1e-12) -> None:
    """Validate one finite posture inside a nonempty coordinate box."""

    q = np.asarray(q, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if q.shape != lower.shape or q.shape != upper.shape:
        raise ValueError("q, lower, and upper must have identical shapes.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("q and its angle bounds must be finite.")
    if np.any(lower > upper):
        raise ValueError("every lower angle bound must not exceed its upper bound.")
    if np.any(q < lower - atol) or np.any(q > upper + atol):
        raise ValueError("q is outside the mechanical angle box.")


def boresight_and_derivatives(q: Array) -> tuple[Array, Array, Array]:
    """Return boresights and derivatives with shape ``(N_A, 3)``."""

    zenith, azimuth = split_posture_angles(q)
    sin_theta = np.sin(zenith)
    cos_theta = np.cos(zenith)
    sin_phi = np.sin(azimuth)
    cos_phi = np.cos(azimuth)
    boresight = np.column_stack(
        (sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
    )
    derivative_zenith = np.column_stack(
        (cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta)
    )
    derivative_azimuth = np.column_stack(
        (-sin_theta * sin_phi, sin_theta * cos_phi, np.zeros_like(zenith))
    )
    return boresight, derivative_zenith, derivative_azimuth


def smooth_positive_part(values: Array, epsilon: float) -> tuple[Array, Array]:
    """Return chi_epsilon(c) and its derivative."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive for posture-gradient evaluation.")
    values = np.asarray(values, dtype=float)
    radius = np.sqrt(values * values + float(epsilon) ** 2)
    return 0.5 * (values + radius), 0.5 * (1.0 + values / radius)


def directional_gain_and_derivatives(
    q: Array,
    directions: Array,
    rho: float,
    *,
    smoothing: float | None,
) -> tuple[Array, Array, Array]:
    """Evaluate directional gains and same-element zenith/azimuth derivatives.

    ``directions`` has shape ``(K, N_A, 3)``.  Passing ``smoothing=None``
    evaluates the physical hard front-side model.  The hard model returns its
    one-sided derivative away from the front--back boundary and is used only
    for diagnostics; optimization uses a positive smoothing value.
    """

    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError("directions must have shape (K, N_A, 3).")
    if float(rho) < 0.0:
        raise ValueError("rho must be nonnegative.")

    boresight, df_dtheta, df_dphi = boresight_and_derivatives(q)
    if boresight.shape[0] != directions.shape[1]:
        raise ValueError("q and directions use different element counts.")
    matches = np.einsum("knd,nd->kn", directions, boresight, optimize=True)
    dc_dtheta = np.einsum("knd,nd->kn", directions, df_dtheta, optimize=True)
    dc_dphi = np.einsum("knd,nd->kn", directions, df_dphi, optimize=True)

    exponent = 2.0 * float(rho)
    peak_gain = 2.0 * (2.0 * float(rho) + 1.0)
    if exponent == 0.0:
        gains = np.full_like(matches, peak_gain)
        return gains, np.zeros_like(matches), np.zeros_like(matches)

    if smoothing is None:
        positive = np.maximum(matches, 0.0)
        positive_derivative = (matches > 0.0).astype(float)
    else:
        positive, positive_derivative = smooth_positive_part(matches, smoothing)

    gains = peak_gain * positive**exponent
    gain_scale = (
        peak_gain
        * exponent
        * positive ** (exponent - 1.0)
        * positive_derivative
    )
    return gains, gain_scale * dc_dtheta, gain_scale * dc_dphi


def posture_channel_state(
    q: Array,
    directions: Array,
    propagation: Array,
    rho: float,
    *,
    smoothing: float | None,
) -> PostureChannelState:
    """Build effective channels and sparse element-local channel derivatives."""

    propagation = np.asarray(propagation, dtype=complex)
    if propagation.ndim == 1:
        propagation = propagation.reshape(1, -1)
    directions = np.asarray(directions, dtype=float)
    if propagation.shape != directions.shape[:2]:
        raise ValueError("propagation must have shape (K, N_A).")
    gains, dg_dtheta, dg_dphi = directional_gain_and_derivatives(
        q,
        directions,
        rho,
        smoothing=smoothing,
    )
    sqrt_gain = np.sqrt(gains)
    channels = sqrt_gain * propagation
    scale = np.zeros_like(propagation, dtype=complex)
    np.divide(
        propagation,
        2.0 * sqrt_gain,
        out=scale,
        where=sqrt_gain > 0.0,
    )
    return PostureChannelState(
        channels=channels,
        directional_gains=gains,
        derivative_zenith=scale * dg_dtheta,
        derivative_azimuth=scale * dg_dphi,
    )


def mrc_gains_and_gradients(state: PostureChannelState) -> tuple[Array, Array]:
    """Return per-WD MRC gains and their gradients with shape ``(K, 2N_A)``."""

    channels = np.asarray(state.channels, dtype=complex)
    gains = np.sum(np.abs(channels) ** 2, axis=1).real
    grad_theta = 2.0 * np.real(np.conj(state.derivative_zenith) * channels)
    grad_phi = 2.0 * np.real(np.conj(state.derivative_azimuth) * channels)
    return gains, np.concatenate((grad_theta, grad_phi), axis=1)


def quadratic_values_and_gradients(
    state: PostureChannelState,
    matrix: Array,
) -> tuple[Array, Array]:
    """Return ``h_k^H X h_k`` and each angle gradient for Hermitian ``X``."""

    matrix = np.asarray(matrix, dtype=complex)
    channels = np.asarray(state.channels, dtype=complex)
    n_antennas = channels.shape[1]
    if matrix.shape != (n_antennas, n_antennas):
        raise ValueError("matrix dimension does not match the posture channels.")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    matrix_channels = channels @ hermitian.T
    values = np.sum(np.conj(channels) * matrix_channels, axis=1).real
    grad_theta = 2.0 * np.real(np.conj(state.derivative_zenith) * matrix_channels)
    grad_phi = 2.0 * np.real(np.conj(state.derivative_azimuth) * matrix_channels)
    return np.maximum(values, 0.0), np.concatenate((grad_theta, grad_phi), axis=1)


def fw_lmo_box(
    gradient: Array,
    lower: Array,
    upper: Array,
    current: Array,
    *,
    zero_tolerance: float = 1e-14,
) -> tuple[Array, float]:
    """Solve the box linear maximization oracle and return its FW gap."""

    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    current = np.asarray(current, dtype=float).reshape(-1)
    validate_angle_box(current, lower, upper)
    if gradient.shape != current.shape or not np.all(np.isfinite(gradient)):
        raise ValueError("gradient must be a finite vector matching current.")
    vertex = current.copy()
    vertex[gradient > zero_tolerance] = upper[gradient > zero_tolerance]
    vertex[gradient < -zero_tolerance] = lower[gradient < -zero_tolerance]
    gap = float(np.dot(gradient, vertex - current))
    return vertex, max(gap, 0.0)


def _fw_report(
    *,
    smooth_history: Array,
    true_history: Array,
    gap_history: Array,
    step_history: Array,
    relative_history: Array,
    point_history: Array,
    q: Array,
    lower: Array,
    upper: Array,
    converged: bool,
    reason: str,
    tolerance: float,
) -> dict[str, object]:
    finite = bool(
        all(
            np.all(np.isfinite(values))
            for values in (
                smooth_history,
                true_history,
                gap_history,
                step_history,
                relative_history,
                point_history,
            )
        )
    )
    true_monotone = bool(
        true_history.size < 2
        or np.all(np.diff(true_history) >= -1e-10 * np.maximum(1.0, np.abs(true_history[:-1])))
    )
    smooth_monotone = bool(
        smooth_history.size < 2
        or np.all(np.diff(smooth_history) >= -1e-10 * np.maximum(1.0, np.abs(smooth_history[:-1])))
    )
    feasible = bool(np.all(q >= lower - 1e-12) and np.all(q <= upper + 1e-12))
    final_gap = float(gap_history[-1]) if gap_history.size else 0.0
    final_relative = float(relative_history[-1]) if relative_history.size else 0.0
    final_change = float(point_history[-1]) if point_history.size else 0.0
    stationary = bool(
        final_gap <= tolerance
        or (relative_history.size and final_relative <= tolerance)
        or (point_history.size and final_change <= tolerance)
    )
    verified = bool(finite and true_monotone and smooth_monotone and feasible and converged and stationary)
    return {
        "verified": verified,
        "converged": bool(converged),
        "reason": reason,
        "iterations": int(max(true_history.size - 1, 0)),
        "finite": finite,
        "feasible": feasible,
        "monotone_smooth_objective": smooth_monotone,
        "monotone_true_objective": true_monotone,
        "initial_smooth_objective": float(smooth_history[0]),
        "final_smooth_objective": float(smooth_history[-1]),
        "initial_true_objective": float(true_history[0]),
        "final_true_objective": float(true_history[-1]),
        "final_fw_gap": final_gap,
        "final_relative_gain": final_relative,
        "final_point_change": final_change,
        "tolerance": float(tolerance),
    }


def fw_angle_block(
    smooth_objective: Callable[[Array], float],
    gradient: Callable[[Array], Array],
    true_objective: Callable[[Array], float],
    initial_q: Array,
    lower: Array,
    upper: Array,
    *,
    max_iter: int = 50,
    tolerance: float = 1e-7,
    armijo_c: float = 1e-4,
    backtracking: float = 0.5,
    min_step: float = 1e-12,
    true_tolerance: float = 1e-13,
    zero_gradient_tolerance: float = 1e-14,
) -> BoxFrankWolfeResult:
    """Maximize a smooth posture surrogate with a true-objective guard."""

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tolerance <= 0.0 or not 0.0 < armijo_c < 1.0:
        raise ValueError("invalid FW tolerance or Armijo constant.")
    if not 0.0 < backtracking < 1.0 or min_step <= 0.0:
        raise ValueError("invalid backtracking parameters.")

    current = np.asarray(initial_q, dtype=float).reshape(-1).copy()
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    validate_angle_box(current, lower, upper)
    current_smooth = float(smooth_objective(current))
    current_true = float(true_objective(current))
    if not np.isfinite(current_smooth) or not np.isfinite(current_true):
        raise ValueError("initial posture objective is not finite.")

    smooth_history = [current_smooth]
    true_history = [current_true]
    gap_history: list[float] = []
    step_history: list[float] = []
    relative_history: list[float] = []
    point_history: list[float] = []
    converged = False
    reason = "max_iter_reached"

    for _ in range(max_iter):
        grad = np.asarray(gradient(current), dtype=float).reshape(-1)
        vertex, gap = fw_lmo_box(
            grad,
            lower,
            upper,
            current,
            zero_tolerance=zero_gradient_tolerance,
        )
        gap_history.append(gap)
        if gap <= tolerance:
            converged = True
            reason = "fw_gap_below_tol"
            break

        direction = vertex - current
        step = 1.0
        accepted = False
        next_q = current
        next_smooth = current_smooth
        next_true = current_true
        while step >= min_step:
            candidate = current + step * direction
            candidate_smooth = float(smooth_objective(candidate))
            candidate_true = float(true_objective(candidate))
            smooth_ok = candidate_smooth >= current_smooth + armijo_c * step * gap
            true_scale = max(1.0, abs(current_true))
            true_ok = candidate_true >= current_true - true_tolerance * true_scale
            if np.isfinite(candidate_smooth) and np.isfinite(candidate_true) and smooth_ok and true_ok:
                next_q = candidate
                next_smooth = candidate_smooth
                next_true = candidate_true
                accepted = True
                break
            step *= backtracking

        if not accepted:
            reason = "line_search_failed"
            break

        point_change = float(np.linalg.norm(next_q - current))
        relative_gain = abs(next_true - current_true) / max(1.0, abs(current_true))
        current = next_q
        current_smooth = next_smooth
        current_true = next_true
        step_history.append(step)
        smooth_history.append(current_smooth)
        true_history.append(current_true)
        relative_history.append(relative_gain)
        point_history.append(point_change)

        # The paper uses the FW gap as the first-order stopping certificate.
        # A protected point-change exit is also retained because the nonsmooth
        # true-objective guard can block a smooth-LSE direction before its
        # surrogate FW gap vanishes.  This is reported as block stability, not
        # as an FW-stationary point of the unguarded smooth surrogate.
        if point_change <= tolerance and relative_gain <= tolerance:
            converged = True
            reason = "protected_point_change"
            break

    smooth_array = np.asarray(smooth_history, dtype=float)
    true_array = np.asarray(true_history, dtype=float)
    gap_array = np.asarray(gap_history, dtype=float)
    step_array = np.asarray(step_history, dtype=float)
    relative_array = np.asarray(relative_history, dtype=float)
    point_array = np.asarray(point_history, dtype=float)
    report = _fw_report(
        smooth_history=smooth_array,
        true_history=true_array,
        gap_history=gap_array,
        step_history=step_array,
        relative_history=relative_array,
        point_history=point_array,
        q=current,
        lower=lower,
        upper=upper,
        converged=converged,
        reason=reason,
        tolerance=tolerance,
    )
    return BoxFrankWolfeResult(
        q=current,
        smooth_objective=current_smooth,
        true_objective=current_true,
        smooth_objective_history=smooth_array,
        true_objective_history=true_array,
        fw_gap_history=gap_array,
        step_history=step_array,
        relative_gain_history=relative_array,
        point_change_history=point_array,
        iterations=len(step_history),
        converged=converged,
        reason=reason,
        feasible=bool(report["feasible"]),
        report=report,
    )


def assert_angle_fw_result(
    result: BoxFrankWolfeResult,
    *,
    require_converged: bool = True,
) -> dict[str, object]:
    """Raise when a posture FW result is nonfinite, infeasible, or nonmonotone."""

    required = {
        "finite": result.report.get("finite", False),
        "feasible": result.report.get("feasible", False),
        "monotone smooth objective": result.report.get("monotone_smooth_objective", False),
        "monotone true objective": result.report.get("monotone_true_objective", False),
    }
    if require_converged:
        required["verified convergence"] = result.report.get("verified", False)
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "Angle Frank-Wolfe verification failed: "
            + ", ".join(failed)
            + f"; report={result.report}"
        )
    return result.report


@dataclass(frozen=True)
class FrankWolfeConfig:
    max_iter: int = 50
    tolerance: float = 1e-6
    armijo_c: float = 1e-4
    backtracking: float = 0.5
    min_step: float = 1e-12
