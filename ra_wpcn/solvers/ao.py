"""Alternating optimization for mechanical angles and energy beamforming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ra_wpcn.solvers.multiple_device import (
    CommonResourceAllocation,
    FairBoresightUpdate,
    MultiDeviceProblem,
    evaluate_boresight_composite_gains,
    mrc_receive_beamformers,
    optimal_uplink_boresights,
    solve_common_resource,
    update_fair_boresights,
)
from ra_wpcn.models.system_model import element_to_device_geometry


Array = np.ndarray


@dataclass(frozen=True)
class FairAOSolution:
    f_energy: Array
    f_information: Array
    f_information_k: Array
    resource: CommonResourceAllocation
    mrc_beamformers: tuple[Array, ...]
    common_throughput: float
    common_rate_history: Array
    mu_history: Array
    smoothing_error_history: Array
    posture_updates: tuple[FairBoresightUpdate, ...]
    converged: bool
    reason: str
    start_index: int

    @property
    def q_energy(self) -> Array:
        return _boresights_to_angles(self.f_energy)

    @property
    def q_information(self) -> Array:
        return _boresights_to_angles(self.f_information)


def _boresights_to_angles(f: Array) -> Array:
    """Convert final/reference boresights to plotting coordinates only."""

    f = np.asarray(f, dtype=float)
    if f.ndim != 2 or f.shape[0] != 3:
        raise ValueError("f must have shape (3, N_A).")
    zenith = np.arccos(np.clip(f[2], -1.0, 1.0))
    azimuth = np.arctan2(f[1], f[0])
    return np.concatenate((zenith, azimuth))


def free_space_propagation(
    distances: Array,
    *,
    wavelength: float = 0.125,
    reference_gain: float = 1.0,
) -> Array:
    """Build posture-independent free-space complex channel coefficients."""

    distances = np.asarray(distances, dtype=float)
    if np.any(distances <= 0.0) or wavelength <= 0.0 or reference_gain <= 0.0:
        raise ValueError("distances, wavelength, and reference_gain must be positive.")
    amplitude = np.sqrt(reference_gain) * wavelength / (4.0 * np.pi * distances)
    return amplitude * np.exp(-1j * 2.0 * np.pi * distances / wavelength)


def build_free_space_problem(
    *,
    element_positions: Array,
    device_positions: Array,
    zeta: float,
    noise_power: float,
    max_power: float,
    lower_energy: Array | None = None,
    upper_energy: Array | None = None,
    lower_information: Array | None = None,
    upper_information: Array | None = None,
    alpha_max_energy: float | None = None,
    alpha_max_information: float | None = None,
    frame_time: float = 1.0,
    rho: float = 2.0,
    angle_smoothing: float = 1e-3,
    wavelength: float = 0.125,
    uplink_wavelength: float | None = None,
) -> MultiDeviceProblem:
    """Construct a deterministic Section-IV problem from physical geometry."""

    directions, distances = element_to_device_geometry(element_positions, device_positions)
    num_antennas = directions.shape[1]

    def resolve_box(
        lower: Array | None,
        upper: Array | None,
        alpha_max: float | None,
        name: str,
    ) -> tuple[Array, Array, float]:
        if lower is None or upper is None:
            if lower is not None or upper is not None or alpha_max is None:
                raise ValueError(
                    f"{name}: provide both angle-box bounds or alpha_max."
                )
            lower = np.concatenate(
                (
                    np.zeros(num_antennas),
                    np.full(num_antennas, -np.pi),
                )
            )
            upper = np.concatenate(
                (
                    np.full(num_antennas, float(alpha_max)),
                    np.full(num_antennas, np.pi),
                )
            )
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if alpha_max is None:
            alpha_max = float(np.max(upper[:num_antennas]))
        return lower, upper, float(alpha_max)

    lower_energy, upper_energy, alpha_max_energy = resolve_box(
        lower_energy, upper_energy, alpha_max_energy, "energy"
    )
    lower_information, upper_information, alpha_max_information = resolve_box(
        lower_information,
        upper_information,
        alpha_max_information,
        "information",
    )
    propagation_energy = free_space_propagation(distances, wavelength=wavelength)
    propagation_information = free_space_propagation(
        distances,
        wavelength=wavelength if uplink_wavelength is None else uplink_wavelength,
    )
    return MultiDeviceProblem(
        directions=directions,
        propagation_energy=propagation_energy,
        propagation_information=propagation_information,
        lower_energy=lower_energy,
        upper_energy=upper_energy,
        lower_information=lower_information,
        upper_information=upper_information,
        zeta=zeta,
        noise_power=noise_power,
        max_power=max_power,
        frame_time=frame_time,
        rho=rho,
        angle_smoothing=angle_smoothing,
        alpha_max_energy=alpha_max_energy,
        alpha_max_information=alpha_max_information,
    )


def default_posture(problem: MultiDeviceProblem, *, information: bool = False) -> Array:
    """Return a feasible interior posture aimed toward the geometric user center."""

    directions = np.asarray(problem.directions, dtype=float)
    mean_direction = np.mean(directions, axis=0)
    norm = np.linalg.norm(mean_direction, axis=1)
    fallback = np.tile(np.array([0.0, 0.0, 1.0]), (problem.num_antennas, 1))
    valid = norm > 0.0
    fallback[valid] = mean_direction[valid] / norm[valid, None]
    theta = np.arccos(np.clip(fallback[:, 2], -1.0, 1.0))
    phi = np.arctan2(fallback[:, 1], fallback[:, 0])
    q = np.concatenate((theta, phi))
    lower = problem.lower_information if information else problem.lower_energy
    upper = problem.upper_information if information else problem.upper_energy
    # Initialization may be projected; all subsequent FW iterates are
    # projection-free convex combinations inside the box.
    return np.minimum(np.maximum(q, lower), upper)


def default_boresights(
    problem: MultiDeviceProblem,
    *,
    information: bool = False,
) -> Array:
    """Return center-directed unit boresights projected to the spherical cap."""

    mean_direction = np.mean(np.asarray(problem.directions, dtype=float), axis=0)
    norms = np.linalg.norm(mean_direction, axis=1, keepdims=True)
    rows = np.tile(np.array([0.0, 0.0, 1.0]), (problem.num_antennas, 1))
    valid = norms[:, 0] > 0.0
    rows[valid] = mean_direction[valid] / norms[valid]
    alpha_max = (
        problem.cap_half_angle_information
        if information
        else problem.cap_half_angle_energy
    )
    axis = np.array([0.0, 0.0, 1.0])
    cosine = np.cos(alpha_max)
    sine = np.sin(alpha_max)
    alignment = rows @ axis
    outside = alignment < cosine
    if np.any(outside):
        tangent = rows[outside] - alignment[outside, None] * axis
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        degenerate = tangent_norm[:, 0] <= 1e-15
        tangent[degenerate] = np.array([1.0, 0.0, 0.0])
        tangent_norm[degenerate] = 1.0
        rows[outside] = cosine * axis + sine * tangent / tangent_norm
    return rows.T


def random_boresight_starts(
    problem: MultiDeviceProblem,
    count: int,
    *,
    seed: int = 0,
) -> list[tuple[Array, Array]]:
    """Return one center-directed and additional uniform spherical-cap starts."""

    if count <= 0:
        raise ValueError("count must be positive.")
    starts = [
        (
            default_boresights(problem),
            default_boresights(problem, information=True),
        )
    ]
    rng = np.random.default_rng(seed)

    def sample(alpha_max: float) -> Array:
        cos_tilt = rng.uniform(
            np.cos(alpha_max), 1.0, size=problem.num_antennas
        )
        sin_tilt = np.sqrt(np.maximum(1.0 - cos_tilt**2, 0.0))
        azimuth = rng.uniform(-np.pi, np.pi, size=problem.num_antennas)
        return np.vstack(
            (
                sin_tilt * np.cos(azimuth),
                sin_tilt * np.sin(azimuth),
                cos_tilt,
            )
        )

    for _ in range(count - 1):
        starts.append(
            (
                sample(problem.cap_half_angle_energy),
                sample(problem.cap_half_angle_information),
            )
        )
    return starts


def random_posture_starts(
    problem: MultiDeviceProblem,
    count: int,
    *,
    seed: int = 0,
) -> list[tuple[Array, Array]]:
    """Return center-directed plus random feasible posture starts."""

    if count <= 0:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(seed)
    starts = [(default_posture(problem), default_posture(problem, information=True))]
    for _ in range(count - 1):
        q_energy = rng.uniform(problem.lower_energy, problem.upper_energy)
        q_information = rng.uniform(problem.lower_information, problem.upper_information)
        starts.append((q_energy, q_information))
    return starts


def _fair_single_start(
    problem: MultiDeviceProblem,
    f_energy: Array,
    f_information_reference: Array,
    f_information_k: Array,
    *,
    start_index: int,
    max_outer_iter: int,
    pga_max_iter: int,
    ao_tolerance: float,
    pga_tolerance: float,
    pga_initial_step: float,
    pga_backtracking: float,
    pga_armijo_c: float,
    pga_min_step: float,
    pga_max_step: float,
    mu_0: float,
    mu_max: float,
    mu_growth: float,
    lse_tolerance: float,
    stable_rounds_required: int,
    solver: str | None,
) -> FairAOSolution:
    resource = solve_common_resource(
        problem, f_energy, f_information_k, solver=solver
    )
    history = [resource.common_rate]
    mu = float(mu_0)
    mu_history = [mu]
    error_history = [float(np.log(problem.num_users) / mu)]
    updates: list[FairBoresightUpdate] = []
    stable_rounds = 0
    converged = False
    reason = "max_outer_iter_reached"

    for _ in range(max_outer_iter):
        posture_update = update_fair_boresights(
            problem,
            f_energy,
            f_information_k,
            resource.tau_k,
            resource.weighted_covariance,
            mu=mu,
            mu_max=mu_max,
            mu_growth=mu_growth,
            pga_max_iter=pga_max_iter,
            pga_tolerance=pga_tolerance,
            pga_initial_step=pga_initial_step,
            pga_backtracking=pga_backtracking,
            pga_armijo_c=pga_armijo_c,
            pga_min_step=pga_min_step,
            pga_max_step=pga_max_step,
        )
        f_energy = posture_update.f_energy
        resource_new = solve_common_resource(
            problem, f_energy, f_information_k, solver=solver
        )
        if resource_new.common_rate < history[-1] - 5e-6 * max(1.0, abs(history[-1])):
            raise RuntimeError("common-throughput AO objective decreased after an accepted block.")
        relative_gain = abs(resource_new.common_rate - history[-1]) / max(
            1.0, abs(history[-1])
        )
        resource = resource_new
        history.append(resource.common_rate)
        updates.append(posture_update)
        mu = posture_update.mu
        mu_history.append(mu)
        smoothing_error = float(np.log(problem.num_users) / mu)
        error_history.append(smoothing_error)
        block_verified = bool(
            posture_update.downlink_result.report["verified"]
            and posture_update.downlink_result.report["block_stable"]
        )
        if (
            relative_gain <= ao_tolerance
            and block_verified
            and smoothing_error <= lse_tolerance
        ):
            stable_rounds += 1
        else:
            stable_rounds = 0
        if stable_rounds >= stable_rounds_required:
            converged = True
            reason = "ao_pga_and_lse_tolerances"
            break

    _, _, _, _, channels_information = evaluate_boresight_composite_gains(
        problem, f_energy, f_information_k, resource.covariance
    )
    return FairAOSolution(
        f_energy=np.asarray(f_energy, dtype=float),
        f_information=np.asarray(f_information_reference, dtype=float),
        f_information_k=np.asarray(f_information_k, dtype=float),
        resource=resource,
        mrc_beamformers=tuple(mrc_receive_beamformers(channels_information)),
        common_throughput=resource.common_rate,
        common_rate_history=np.asarray(history, dtype=float),
        mu_history=np.asarray(mu_history, dtype=float),
        smoothing_error_history=np.asarray(error_history, dtype=float),
        posture_updates=tuple(updates),
        converged=converged,
        reason=reason,
        start_index=start_index,
    )


def solve_fair_rate_ao(
    problem: MultiDeviceProblem,
    *,
    boresight_starts: list[tuple[Array, Array]] | None = None,
    max_outer_iter: int = 20,
    pga_max_iter: int = 30,
    ao_tolerance: float = 1e-6,
    pga_tolerance: float = 1e-7,
    pga_initial_step: float = 1.0,
    pga_backtracking: float = 0.5,
    pga_armijo_c: float = 1e-4,
    pga_min_step: float = 1e-12,
    pga_max_step: float = 1.0,
    mu_0: float = 5.0,
    mu_max: float = 200.0,
    mu_growth: float = 2.0,
    lse_tolerance: float | None = None,
    stable_rounds_required: int = 2,
    solver: str | None = None,
) -> FairAOSolution:
    """Run Algorithm 2 and retain the start with the largest true common rate."""

    if mu_0 <= 0.0 or mu_max < mu_0 or mu_growth <= 1.0:
        raise ValueError("require 0 < mu_0 <= mu_max and mu_growth > 1.")
    if lse_tolerance is None:
        lse_tolerance = float(np.log(problem.num_users) / mu_max)
    if boresight_starts is None:
        boresight_starts = [
            (
                default_boresights(problem),
                default_boresights(problem, information=True),
            )
        ]
    f_information_k = optimal_uplink_boresights(problem)
    solutions = [
        _fair_single_start(
            problem,
            np.asarray(f_energy, dtype=float),
            np.asarray(f_information, dtype=float),
            f_information_k,
            start_index=index,
            max_outer_iter=max_outer_iter,
            pga_max_iter=pga_max_iter,
            ao_tolerance=ao_tolerance,
            pga_tolerance=pga_tolerance,
            pga_initial_step=pga_initial_step,
            pga_backtracking=pga_backtracking,
            pga_armijo_c=pga_armijo_c,
            pga_min_step=pga_min_step,
            pga_max_step=pga_max_step,
            mu_0=mu_0,
            mu_max=mu_max,
            mu_growth=mu_growth,
            lse_tolerance=lse_tolerance,
            stable_rounds_required=stable_rounds_required,
            solver=solver,
        )
        for index, (f_energy, f_information) in enumerate(boresight_starts)
    ]
    return max(solutions, key=lambda solution: solution.common_throughput)


@dataclass(frozen=True)
class AOConfig:
    max_outer_iter: int = 30
    tolerance: float = 1e-6
    stable_rounds_required: int = 2
    mu_0: float = 20.0
    mu_max: float = 200.0
    mu_growth: float = 2.0

class WPCNOptimizers:
    """Bind one physical multi-device model to its resource and angle solvers."""

    def __init__(self, problem: MultiDeviceProblem):
        self.problem = problem


    def closed_form_uplink_boresights(self):
        return optimal_uplink_boresights(self.problem)


    def optimize_fair_resource(self, f_energy, f_information, **kwargs):
        return solve_common_resource(
            self.problem,
            f_energy,
            f_information,
            **kwargs,
        )

    def optimize_fair_rate_boresights(
        self,
        f_energy,
        f_information,
        tau_k,
        weighted_covariance,
        **kwargs,
    ):
        return update_fair_boresights(
            self.problem,
            f_energy,
            f_information,
            tau_k,
            weighted_covariance,
            **kwargs,
        )


    def solve_fair_rate(self, **kwargs):
        return solve_fair_rate_ao(self.problem, **kwargs)
