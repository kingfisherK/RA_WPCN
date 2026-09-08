"""Formula-faithful Section-IV multi-device resource and posture blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ra_wpcn.solvers.angle_optimization import (
    BoresightChannelState,
    SphericalCapProjectedGradientResult,
    boresight_channel_state,
    pga_spherical_cap_block,
    mrc_gains_and_gradients,
    posture_channel_state,
    quadratic_boresight_values_and_gradients,
    quadratic_values_and_gradients,
)
from ra_wpcn.models.system_model import (
    closed_form_uplink_boresights as project_uplink_boresights,
    mrc_combiners,
    mrc_gains,
    per_slot_effective_channels,
)


Array = np.ndarray


@dataclass(frozen=True)
class MultiDeviceProblem:
    """Posture-independent inputs and physical parameters for Section IV."""

    directions: Array
    propagation_energy: Array
    propagation_information: Array
    lower_energy: Array
    upper_energy: Array
    lower_information: Array
    upper_information: Array
    zeta: float
    noise_power: float
    max_power: float
    frame_time: float = 1.0
    rho: float = 2.0
    angle_smoothing: float = 1e-3
    alpha_max_energy: float | None = None
    alpha_max_information: float | None = None

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions, dtype=float)
        propagation_energy = np.asarray(self.propagation_energy, dtype=complex)
        propagation_information = np.asarray(self.propagation_information, dtype=complex)
        if directions.ndim != 3 or directions.shape[2] != 3:
            raise ValueError("directions must have shape (K, N_A, 3).")
        if propagation_energy.shape != directions.shape[:2]:
            raise ValueError("propagation_energy must have shape (K, N_A).")
        if propagation_information.shape != directions.shape[:2]:
            raise ValueError("propagation_information must have shape (K, N_A).")
        n_angles = 2 * directions.shape[1]
        for name in (
            "lower_energy",
            "upper_energy",
            "lower_information",
            "upper_information",
        ):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.size != n_angles or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain 2*N_A finite angles.")
        if np.any(np.asarray(self.lower_energy) > np.asarray(self.upper_energy)):
            raise ValueError("invalid downlink angle box.")
        if np.any(np.asarray(self.lower_information) > np.asarray(self.upper_information)):
            raise ValueError("invalid uplink angle box.")
        if not 0.0 < float(self.zeta) <= 1.0:
            raise ValueError("zeta must lie in (0, 1].")
        if self.noise_power <= 0.0 or self.max_power <= 0.0 or self.frame_time <= 0.0:
            raise ValueError("noise_power, max_power, and frame_time must be positive.")
        if self.rho < 0.0 or self.angle_smoothing <= 0.0:
            raise ValueError("rho must be nonnegative and angle_smoothing positive.")
        for name in ("alpha_max_energy", "alpha_max_information"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= np.pi / 2.0:
                raise ValueError(f"{name} must lie in [0, pi/2].")

    @property
    def num_users(self) -> int:
        return int(np.asarray(self.directions).shape[0])

    @property
    def num_antennas(self) -> int:
        return int(np.asarray(self.directions).shape[1])

    @property
    def cap_half_angle_energy(self) -> float:
        if self.alpha_max_energy is not None:
            return float(self.alpha_max_energy)
        return float(np.max(np.asarray(self.upper_energy)[: self.num_antennas]))

    @property
    def cap_half_angle_information(self) -> float:
        if self.alpha_max_information is not None:
            return float(self.alpha_max_information)
        return float(np.max(np.asarray(self.upper_information)[: self.num_antennas]))


@dataclass(frozen=True)
class CommonResourceAllocation:
    tau_0: float
    tau_k: Array
    weighted_covariance: Array
    covariance: Array
    energy_beams: tuple[Array, ...]
    rates: Array
    common_rate: float
    solver: str
    status: str
    constraint_residuals: dict[str, float]


@dataclass(frozen=True)
class FairBoresightUpdate:
    f_energy: Array
    f_information: Array
    f_information_k: Array
    downlink_result: SphericalCapProjectedGradientResult
    rates: Array
    true_common_rate: float
    smooth_common_rate: float
    smoothing_error_bound: float
    mu: float


def angle_box(
    num_antennas: int,
    alpha_max: float,
    azimuth_min: float = -np.pi,
    azimuth_max: float = np.pi,
) -> tuple[Array, Array]:
    """Construct the paper's zenith/azimuth box for one WPT or WIT stage."""

    if num_antennas <= 0 or not 0.0 <= alpha_max <= np.pi / 2.0:
        raise ValueError("invalid antenna count or maximum tilt.")
    if azimuth_min > azimuth_max:
        raise ValueError("azimuth_min must not exceed azimuth_max.")
    lower = np.concatenate(
        (np.zeros(num_antennas), np.full(num_antennas, azimuth_min))
    )
    upper = np.concatenate(
        (np.full(num_antennas, alpha_max), np.full(num_antennas, azimuth_max))
    )
    return lower, upper


def optimal_uplink_boresights(problem: MultiDeviceProblem) -> Array:
    """Return all dedicated closed-form uplink postures ``F_{I,k}^*``."""

    return project_uplink_boresights(
        problem.directions,
        problem.cap_half_angle_information,
    )


def user_rates(tau_0: float, tau_k: Array, gamma: Array) -> Array:
    """Evaluate frame-average rates with the continuous value zero at tau_k=0."""

    tau_k = np.asarray(tau_k, dtype=float).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if tau_k.shape != gamma.shape or tau_0 < 0.0:
        raise ValueError("tau_k and gamma must be matched nonnegative inputs.")
    if np.any(tau_k < 0.0) or np.any(gamma < 0.0):
        raise ValueError("tau_k and gamma must be nonnegative.")
    rates = np.zeros_like(gamma)
    active = tau_k > 0.0
    rates[active] = tau_k[active] * np.log2(
        1.0 + float(tau_0) * gamma[active] / tau_k[active]
    )
    return rates


def stable_logsumexp(values: Array) -> float:
    """Numerically stable log-sum-exp for a finite one-dimensional array."""

    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty finite vector.")
    shift = float(np.max(values))
    return float(shift + np.log(np.sum(np.exp(values - shift))))


def smooth_min(values: Array, mu: float) -> float:
    """Return ``-log(sum(exp(-mu*values)))/mu``."""

    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    values = np.asarray(values, dtype=float).reshape(-1)
    return -stable_logsumexp(-float(mu) * values) / float(mu)


def softmin_weights(values: Array, mu: float) -> Array:
    """Stable gradient weights for the log-sum-exp smooth minimum."""

    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    values = np.asarray(values, dtype=float).reshape(-1)
    shifted = -float(mu) * (values - float(np.min(values)))
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def mrc_receive_beamformers(uplink_channels: list[Array] | Array) -> list[Array]:
    """Return one closed-form MRC column for each shared-posture uplink channel."""

    channels = np.asarray(uplink_channels, dtype=complex)
    if channels.ndim == 3 and channels.shape[2] == 1:
        channels = channels[:, :, 0]
    if channels.ndim == 1:
        channels = channels.reshape(1, -1)
    combiners = mrc_combiners(channels)
    return [combiners[:, [k]] for k in range(combiners.shape[1])]


def energy_channel_matrices(channels_energy: Array) -> Array:
    """Return all rank-one matrices ``h_k h_k^H`` with shape ``(K,N,N)``."""

    channels = np.asarray(channels_energy, dtype=complex)
    if channels.ndim != 2:
        raise ValueError("channels_energy must have shape (K, N_A).")
    return np.einsum("kn,km->knm", channels, channels.conj(), optimize=True)


def recover_energy_beams(covariance: Array, tolerance: float = 1e-9) -> tuple[Array, ...]:
    """Recover an exact deterministic multi-beam factorization from a PSD matrix."""

    covariance = np.asarray(covariance, dtype=complex)
    hermitian = 0.5 * (covariance + covariance.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    threshold = max(float(tolerance), float(tolerance) * max(1.0, np.max(eigenvalues)))
    beams = []
    for index in np.where(eigenvalues > threshold)[0]:
        beams.append(np.sqrt(float(eigenvalues[index])) * eigenvectors[:, [index]])
    return tuple(beams)


def _hard_states(
    problem: MultiDeviceProblem,
    q_energy: Array,
    q_information: Array,
):
    energy = posture_channel_state(
        q_energy,
        problem.directions,
        problem.propagation_energy,
        problem.rho,
        smoothing=None,
    )
    information = posture_channel_state(
        q_information,
        problem.directions,
        problem.propagation_information,
        problem.rho,
        smoothing=None,
    )
    return energy, information


def evaluate_composite_gains(
    problem: MultiDeviceProblem,
    q_energy: Array,
    q_information: Array,
    covariance: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return gamma, downlink powers, uplink gains, and both channel arrays."""

    energy_state, information_state = _hard_states(problem, q_energy, q_information)
    downlink, _ = quadratic_values_and_gradients(energy_state, covariance)
    uplink, _ = mrc_gains_and_gradients(information_state)
    gamma = problem.zeta * downlink * uplink / problem.noise_power
    return gamma, downlink, uplink, energy_state.channels, information_state.channels


def weighted_covariance_rates_angles(
    problem: MultiDeviceProblem,
    q_energy: Array,
    q_information: Array,
    tau_k: Array,
    weighted_covariance: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate rates using Z_E=tau_0*S_E, plus its energy and MRC factors."""

    tau_k = np.asarray(tau_k, dtype=float).reshape(-1)
    energy_state, information_state = _hard_states(problem, q_energy, q_information)
    energy_terms, _ = quadratic_values_and_gradients(energy_state, weighted_covariance)
    uplink_gains, _ = mrc_gains_and_gradients(information_state)
    rates = np.zeros_like(tau_k)
    active = tau_k > 0.0
    rho = np.zeros_like(tau_k)
    rho[active] = (
        problem.zeta
        * energy_terms[active]
        * uplink_gains[active]
        / (problem.noise_power * tau_k[active])
    )
    rates[active] = tau_k[active] * np.log2(1.0 + rho[active])
    return rates, energy_terms, uplink_gains


def _hard_boresight_states(
    problem: MultiDeviceProblem,
    f_energy: Array,
    f_information: Array,
) -> tuple[BoresightChannelState, Array]:
    """Evaluate the downlink state and shared/dedicated uplink channels.

    A ``3 x N_A`` uplink matrix is retained for unchanged comparison schemes.
    Proposed passes the new ``K x 3 x N_A`` tensor and therefore uses one
    closed-form posture per TDMA slot.
    """

    energy = boresight_channel_state(
        f_energy,
        problem.directions,
        problem.propagation_energy,
        problem.rho,
    )
    f_information = np.asarray(f_information, dtype=float)
    if f_information.ndim == 2:
        information_channels = boresight_channel_state(
            f_information,
            problem.directions,
            problem.propagation_information,
            problem.rho,
        ).channels
    elif f_information.ndim == 3:
        information_channels = per_slot_effective_channels(
            problem.propagation_information,
            f_information,
            problem.directions,
            problem.rho,
        )
    else:
        raise ValueError(
            "f_information must be shared (3,N_A) or dedicated (K,3,N_A)."
        )
    return energy, information_channels


def evaluate_boresight_composite_gains(
    problem: MultiDeviceProblem,
    f_energy: Array,
    f_information: Array,
    covariance: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return hard-model factors for spherical-cap boresights."""

    energy_state, information_channels = _hard_boresight_states(
        problem, f_energy, f_information
    )
    energy_terms, _ = quadratic_boresight_values_and_gradients(
        energy_state, covariance
    )
    uplink_gains = mrc_gains(information_channels)
    gamma = problem.zeta * energy_terms * uplink_gains / problem.noise_power
    return (
        gamma,
        energy_terms,
        uplink_gains,
        energy_state.channels,
        information_channels,
    )


def weighted_covariance_rates(
    problem: MultiDeviceProblem,
    f_energy: Array,
    f_information: Array,
    tau_k: Array,
    weighted_covariance: Array,
) -> tuple[Array, Array, Array]:
    """Evaluate Rcom factors directly from spherical-cap boresights."""

    tau_k = np.asarray(tau_k, dtype=float).reshape(-1)
    energy_state, information_channels = _hard_boresight_states(
        problem, f_energy, f_information
    )
    energy_terms, _ = quadratic_boresight_values_and_gradients(
        energy_state, weighted_covariance
    )
    uplink_gains = mrc_gains(information_channels)
    rates = np.zeros_like(tau_k)
    active = tau_k > 0.0
    snr = np.zeros_like(tau_k)
    snr[active] = (
        problem.zeta
        * energy_terms[active]
        * uplink_gains[active]
        / (problem.noise_power * tau_k[active])
    )
    rates[active] = tau_k[active] * np.log2(1.0 + snr[active])
    return rates, energy_terms, uplink_gains


def solve_common_resource(
    problem: MultiDeviceProblem,
    f_energy: Array,
    f_information: Array,
    *,
    solver: str | None = None,
    solver_tolerance: float = 1e-7,
) -> CommonResourceAllocation:
    """Solve the fixed-posture PSD/exponential-cone resource block exactly.

    CVXPY is imported locally so the sum-throughput branch has no conic-solver
    dependency.  Install the package listed in ``requirements.txt`` to run the
    fair AO branch.
    """

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - exercised in dependency-free use.
        raise RuntimeError(
            "The fair resource block requires cvxpy; install requirements.txt."
        ) from exc

    energy_state, information_channels = _hard_boresight_states(
        problem, f_energy, f_information
    )
    uplink_gains = mrc_gains(information_channels)
    channel_matrices = energy_channel_matrices(energy_state.channels)
    k_users = problem.num_users
    n_antennas = problem.num_antennas
    tau_0 = cp.Variable(nonneg=True, name="tau_0")
    tau_k = cp.Variable(k_users, nonneg=True, name="tau_k")
    weighted_covariance_fraction = cp.Variable(
        (n_antennas, n_antennas), hermitian=True, name="W_E"
    )
    common_rate = cp.Variable(nonneg=True, name="R_com")
    constraints = [
        tau_0 + cp.sum(tau_k) <= problem.frame_time,
        weighted_covariance_fraction >> 0,
        cp.real(cp.trace(weighted_covariance_fraction)) <= tau_0,
    ]
    for index in range(k_users):
        scaled_channel_matrix = (
            problem.zeta
            * uplink_gains[index]
            * problem.max_power
            / problem.noise_power
            * channel_matrices[index]
        )
        energy_term = cp.real(
            cp.trace(scaled_channel_matrix @ weighted_covariance_fraction)
        )
        constraints.append(
            -cp.rel_entr(tau_k[index], tau_k[index] + energy_term)
            >= common_rate * np.log(2.0)
        )
    optimization = cp.Problem(cp.Maximize(common_rate), constraints)
    installed = set(cp.installed_solvers())
    selected = solver or ("CLARABEL" if "CLARABEL" in installed else "SCS")
    if selected not in installed:
        raise RuntimeError(f"Requested CVXPY solver {selected!r} is not installed.")
    solve_kwargs: dict[str, object] = {"solver": selected, "verbose": False}
    if selected == "SCS":
        solve_kwargs.update({"eps": solver_tolerance, "max_iters": 50_000})
    optimization.solve(**solve_kwargs)
    if optimization.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Fair resource solve failed with status {optimization.status!r}.")

    tau_0_value = max(float(tau_0.value), 0.0)
    tau_k_value = np.maximum(np.asarray(tau_k.value, dtype=float).reshape(-1), 0.0)
    z_raw = problem.max_power * np.asarray(
        weighted_covariance_fraction.value, dtype=complex
    )
    if (
        not np.isfinite(tau_0_value)
        or not np.all(np.isfinite(tau_k_value))
        or not np.all(np.isfinite(z_raw))
    ):
        raise RuntimeError("Fair resource solver returned nonfinite primal values.")
    total_time = tau_0_value + float(np.sum(tau_k_value))
    if total_time > problem.frame_time:
        time_scale = problem.frame_time / total_time
        tau_0_value *= time_scale
        tau_k_value *= time_scale
        z_raw *= time_scale
    z_hermitian = 0.5 * (z_raw + z_raw.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(z_hermitian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    z_value = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    trace_limit = tau_0_value * problem.max_power
    trace_value = float(np.trace(z_value).real)
    if trace_value > trace_limit > 0.0:
        z_value *= trace_limit / trace_value
    covariance = (
        z_value / tau_0_value
        if tau_0_value > solver_tolerance
        else np.zeros_like(z_value)
    )
    beams = recover_energy_beams(covariance)
    rates, _, _ = weighted_covariance_rates(
        problem, f_energy, f_information, tau_k_value, z_value
    )
    common_value = float(np.min(rates)) if rates.size else 0.0
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(z_value)))
    residuals = {
        "time": max(tau_0_value + float(np.sum(tau_k_value)) - problem.frame_time, 0.0),
        "trace": max(float(np.trace(z_value).real) - trace_limit, 0.0),
        "psd": max(-minimum_eigenvalue, 0.0),
        "rate": max(float(common_rate.value) - common_value, 0.0),
    }
    return CommonResourceAllocation(
        tau_0=tau_0_value,
        tau_k=tau_k_value,
        weighted_covariance=z_value,
        covariance=covariance,
        energy_beams=beams,
        rates=rates,
        common_rate=common_value,
        solver=selected,
        status=str(optimization.status),
        constraint_residuals=residuals,
    )


def update_fair_boresights(
    problem: MultiDeviceProblem,
    f_energy: Array,
    f_information: Array,
    tau_k: Array,
    weighted_covariance: Array,
    *,
    mu: float,
    mu_max: float | None = None,
    mu_growth: float = 2.0,
    pga_max_iter: int = 30,
    pga_tolerance: float = 1e-7,
    pga_initial_step: float = 1.0,
    pga_backtracking: float = 0.5,
    pga_armijo_c: float = 1e-4,
    pga_min_step: float = 1e-12,
    pga_max_step: float = 1.0,
) -> FairBoresightUpdate:
    """Update only ``F_E`` using LSE; ``F_{I,k}^*`` remains closed form."""

    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    if mu_max is None:
        mu_max = float(mu)
    if mu_max < mu or mu_growth <= 1.0:
        raise ValueError("require mu_max >= mu and mu_growth > 1.")
    current_mu = float(mu)
    tau_k = np.asarray(tau_k, dtype=float).reshape(-1)
    f_energy = np.asarray(f_energy, dtype=float)
    information_input = np.asarray(f_information, dtype=float)
    f_information_k = (
        information_input
        if information_input.ndim == 3
        else optimal_uplink_boresights(problem)
    )
    _, information_channels = _hard_boresight_states(
        problem, f_energy, f_information_k
    )
    uplink_fixed = mrc_gains(information_channels)

    def rate_from_factors(energy_terms: Array, uplink_gains: Array) -> Array:
        rates = np.zeros_like(tau_k)
        active = tau_k > 0.0
        snr = np.zeros_like(tau_k)
        snr[active] = (
            problem.zeta
            * energy_terms[active]
            * uplink_gains[active]
            / (problem.noise_power * tau_k[active])
        )
        rates[active] = tau_k[active] * np.log2(1.0 + snr[active])
        return rates

    def downlink_data(candidate: Array) -> tuple[Array, Array]:
        state = boresight_channel_state(
            candidate,
            problem.directions,
            problem.propagation_energy,
            problem.rho,
        )
        energy_terms, energy_gradients = quadratic_boresight_values_and_gradients(
            state, weighted_covariance
        )
        rates = rate_from_factors(energy_terms, uplink_fixed)
        active = tau_k > 0.0
        snr = np.zeros_like(tau_k)
        snr[active] = (
            problem.zeta
            * energy_terms[active]
            * uplink_fixed[active]
            / (problem.noise_power * tau_k[active])
        )
        coefficients = np.zeros_like(tau_k)
        coefficients[active] = (
            problem.zeta
            * uplink_fixed[active]
            / (problem.noise_power * np.log(2.0) * (1.0 + snr[active]))
        )
        return rates, coefficients[:, None, None] * energy_gradients

    def downlink_smooth_objective(candidate: Array) -> float:
        rates, _ = downlink_data(candidate)
        return smooth_min(rates, current_mu)

    def downlink_gradient(candidate: Array) -> Array:
        rates, rate_gradients = downlink_data(candidate)
        return np.sum(
            softmin_weights(rates, current_mu)[:, None, None] * rate_gradients,
            axis=0,
        )

    def downlink_true_objective(candidate: Array) -> float:
        rates, _, _ = weighted_covariance_rates(
            problem, candidate, f_information_k, tau_k, weighted_covariance
        )
        return float(np.min(rates))

    current_step = float(pga_initial_step)
    while True:
        downlink_result = pga_spherical_cap_block(
            downlink_smooth_objective,
            downlink_gradient,
            downlink_true_objective,
            f_energy,
            problem.cap_half_angle_energy,
            max_iter=pga_max_iter,
            tolerance=pga_tolerance,
            initial_step=current_step,
            backtracking=pga_backtracking,
            armijo_c=pga_armijo_c,
            min_step=pga_min_step,
            max_step=pga_max_step,
        )
        f_energy = downlink_result.f
        current_step = downlink_result.next_step
        if not downlink_result.converged or current_mu >= float(mu_max):
            break
        current_mu = min(float(mu_growth) * current_mu, float(mu_max))
    final_rates, _, _ = weighted_covariance_rates(
        problem,
        downlink_result.f,
        f_information_k,
        tau_k,
        weighted_covariance,
    )
    if information_input.ndim == 2:
        information_reference = information_input
    else:
        information_reference = np.mean(f_information_k, axis=0)
        norms = np.linalg.norm(information_reference, axis=0, keepdims=True)
        information_reference = np.divide(
            information_reference,
            norms,
            out=np.tile(np.array([[0.0], [0.0], [1.0]]), (1, problem.num_antennas)),
            where=norms > 0.0,
        )
    return FairBoresightUpdate(
        f_energy=downlink_result.f,
        f_information=information_reference,
        f_information_k=f_information_k,
        downlink_result=downlink_result,
        rates=final_rates,
        true_common_rate=float(np.min(final_rates)),
        smooth_common_rate=smooth_min(final_rates, current_mu),
        smoothing_error_bound=float(np.log(problem.num_users) / current_mu),
        mu=current_mu,
    )
