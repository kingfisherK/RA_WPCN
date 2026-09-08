"""Fixed user topologies, experiment settings, and the five Rcom schemes."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Set before NumPy/CVXPY imports in every Windows worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import lambertw

from ra_wpcn.solvers.ao import default_boresights, solve_fair_rate_ao
from ra_wpcn.models.system_model import dbm_to_watt
from ra_wpcn.solvers.angle_optimization import (
    boresight_channel_state, pga_spherical_cap_block,
    mrc_boresight_gains_and_gradients, quadratic_boresight_values_and_gradients,
)
from ra_wpcn.solvers.multiple_device import (
    MultiDeviceProblem, angle_box, energy_channel_matrices,
    evaluate_boresight_composite_gains, solve_common_resource,
    update_fair_boresights, weighted_covariance_rates, smooth_min, softmin_weights,
)

ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_PATH = ROOT / "data" / "user_topologies_100.npy"
TOPOLOGY_SHA256 = "35f702ce29d35473c7baed29ec85c37eb6e5267095600c91c6da93494ba8f1ea"
SCHEME_NAMES = ("Proposed", "ET", "FHAP", "Random BF", "Shared FE/FI")
FHAP_FIXED_TILT_DEG = 20.0

@dataclass(frozen=True)
class SimulationSetup:
    """Section-V defaults retained from the existing manuscript setup."""

    topology_seed: int = 2026
    topology_bank_size: int = 100
    max_users_per_snapshot: int = 10
    min_user_distance_m: float = 2.0
    max_user_distance_m: float = 8.0
    hap_wd_vertical_separation_m: float = 1.0
    default_num_users: int = 4
    default_num_antennas: int = 8
    default_power_dbm: float = 20.0
    default_path_loss_exponent: float = 2.5
    frame_time: float = 1.0
    harvesting_efficiency: float = 0.5
    noise_power_dbm: float = -90.0
    reference_distance_m: float = 1.0
    reference_power_gain: float = 1e-3
    wd_gain_dbi: float = 3.0
    wavelength_m: float = 0.125
    element_spacing_wavelengths: float = 0.5
    max_tilt_deg: float = 75.0
    directional_rho: float = 2.0
    angle_smoothing: float = 1e-3


@dataclass(frozen=True, order=True)
class SimulationCase:
    alpha: float
    power_dbm: float
    num_users: int
    num_antennas: int

    @property
    def label(self) -> str:
        return (
            f"alpha={self.alpha:g},PA={self.power_dbm:g},"
            f"K={self.num_users},NA={self.num_antennas}"
        )


@dataclass(frozen=True)
class Sweep:
    key: str
    values: tuple[float, ...]
    cases: tuple[SimulationCase, ...]
    x_header: str
    x_label: str
    integer_ticks: bool


@lru_cache(maxsize=1)
def _load_topology_bank() -> np.ndarray:
    """Load and verify the immutable 100-snapshot input; never regenerate it."""
    bank = np.load(TOPOLOGY_PATH, allow_pickle=False)
    if bank.shape != (100, 10, 2) or bank.dtype != np.dtype("float64"):
        raise ValueError("The fixed topology bank must have shape (100, 10, 2) and dtype float64.")
    payload = bank.tobytes(order="C")
    if hashlib.sha256(payload).hexdigest() != TOPOLOGY_SHA256:
        raise ValueError("The fixed topology bank fingerprint does not match the original data.")
    # A bytes-backed array cannot be made writable by a caller.
    return np.frombuffer(payload, dtype=np.float64).reshape(100, 10, 2)


def topology_bank(setup: SimulationSetup) -> np.ndarray:
    """Return the same stored polar-coordinate samples for every experiment.

    A sample's norm is the HAP-center link distance used by _build_problem;
    actual horizontal positions account for the HAP/WD height difference.
    """
    settings = (setup.topology_seed, setup.topology_bank_size,
                setup.max_users_per_snapshot, setup.min_user_distance_m,
                setup.max_user_distance_m)
    if settings != (2026, 100, 10, 2.0, 8.0):
        raise ValueError("Topology settings must match the fixed 100-snapshot bank; use --snapshots to select a prefix.")
    return _load_topology_bank()


def _build_problem(
    positions_xy: np.ndarray,
    case: SimulationCase,
    setup: SimulationSetup,
) -> MultiDeviceProblem:
    """Map one prepared 2-D topology into the active Section-II/IV model."""

    n_antennas = case.num_antennas
    spacing = setup.element_spacing_wavelengths * setup.wavelength_m
    x_coordinates = (np.arange(n_antennas) - (n_antennas - 1.0) / 2.0) * spacing
    element_positions = np.column_stack(
        (x_coordinates, np.zeros(n_antennas), np.zeros(n_antennas))
    )
    positions_xy = np.asarray(positions_xy, dtype=float)
    center_distances = np.linalg.norm(positions_xy, axis=1)
    vertical_separation = float(setup.hap_wd_vertical_separation_m)
    if vertical_separation <= 0.0:
        raise ValueError("hap_wd_vertical_separation_m must be positive.")
    if np.any(center_distances <= vertical_separation):
        raise ValueError(
            "every sampled HAP-center--WD distance must exceed the vertical separation."
        )
    # The prepared 2--8 m topology radii remain the three-dimensional link
    # distances.  Project only their horizontal component and place WDs in the
    # positive-z service region, as defined in the manuscript.
    horizontal_scale = np.sqrt(
        center_distances**2 - vertical_separation**2
    ) / center_distances
    device_positions = np.column_stack(
        (positions_xy * horizontal_scale[:, None], np.full(case.num_users, vertical_separation))
    )

    displacement = device_positions[:, None, :] - element_positions[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    directions = displacement / distances[:, :, None]

    # The HAP peak gain is supplied by the cosine pattern as
    # G_max^cos = 2(2 rho + 1); do not multiply a second HAP antenna gain.
    fixed_link_gain = 10.0 ** (setup.wd_gain_dbi / 10.0)
    path_power_gain = (
        setup.reference_power_gain
        * fixed_link_gain
        * (setup.reference_distance_m / distances) ** case.alpha
    )
    phase = np.exp(-1j * 2.0 * np.pi * distances / setup.wavelength_m)
    propagation = np.sqrt(path_power_gain) * phase

    lower, upper = angle_box(
        n_antennas,
        np.deg2rad(setup.max_tilt_deg),
        -np.pi,
        np.pi,
    )
    return MultiDeviceProblem(
        directions=directions,
        propagation_energy=propagation,
        propagation_information=propagation,
        lower_energy=lower,
        upper_energy=upper,
        lower_information=lower,
        upper_information=upper,
        zeta=setup.harvesting_efficiency,
        noise_power=dbm_to_watt(setup.noise_power_dbm),
        max_power=dbm_to_watt(case.power_dbm),
        frame_time=setup.frame_time,
        rho=setup.directional_rho,
        angle_smoothing=setup.angle_smoothing,
        alpha_max_energy=np.deg2rad(setup.max_tilt_deg),
        alpha_max_information=np.deg2rad(setup.max_tilt_deg),
    )


def _fixed_hap_posture(problem: MultiDeviceProblem) -> np.ndarray:
    """The FHAP baseline: every element retains the positive-z reference posture."""

    return np.concatenate(
        (np.zeros(problem.num_antennas), np.zeros(problem.num_antennas))
    )


def _fixed_hap_boresights(problem: MultiDeviceProblem) -> np.ndarray:
    """Positive-z spherical-cap initialization used by the RA optimizers."""

    return np.tile(
        np.array([[0.0], [0.0], [1.0]]), (1, problem.num_antennas)
    )


def _fixed_sector_fhap_boresights(
    problem: MultiDeviceProblem,
    *,
    tilt_deg: float = 20.0,
) -> np.ndarray:
    """Deterministic non-optimized sector fan used only by the FHAP baseline."""

    tilt = np.deg2rad(float(tilt_deg))
    cap_limit = min(
        problem.cap_half_angle_energy,
        problem.cap_half_angle_information,
    )
    if not 0.0 <= tilt <= cap_limit:
        raise ValueError("FHAP fixed tilt must lie inside both spherical caps.")
    azimuth = -np.pi + (
        np.arange(problem.num_antennas) + 0.5
    ) * (2.0 * np.pi / problem.num_antennas)
    return np.vstack(
        (
            np.sin(tilt) * np.cos(azimuth),
            np.sin(tilt) * np.sin(azimuth),
            np.full(problem.num_antennas, np.cos(tilt)),
        )
    )


def _solve_common_with_fallback(
    problem: MultiDeviceProblem,
    f_energy: np.ndarray,
    f_information: np.ndarray,
):
    errors: list[Exception] = []
    for solver in ("CLARABEL", "SCS"):
        try:
            return solve_common_resource(
                problem, f_energy, f_information, solver=solver
            )
        except Exception as exc:  # pragma: no cover - depends on solver status.
            errors.append(exc)
    raise RuntimeError("Both common-resource solvers failed.") from errors[-1]


def _equal_time_common_resource(
    problem: MultiDeviceProblem,
    f_energy: np.ndarray,
    f_information: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Optimize the energy covariance while all K+1 time slots remain equal."""

    import cvxpy as cp

    tau = problem.frame_time / (problem.num_users + 1.0)
    isotropic = np.eye(problem.num_antennas, dtype=complex) * (
        problem.max_power / problem.num_antennas
    )
    _, _, uplink_gains, channels_energy, _ = evaluate_boresight_composite_gains(
        problem, f_energy, f_information, isotropic
    )
    channel_matrices = energy_channel_matrices(channels_energy)
    if np.any(uplink_gains <= 0.0):
        return 0.0, tau * isotropic

    covariance_fraction = cp.Variable(
        (problem.num_antennas, problem.num_antennas), hermitian=True, name="Y_ET"
    )
    minimum_snr = cp.Variable(nonneg=True, name="snr_ET")
    constraints = [
        covariance_fraction >> 0,
        cp.real(cp.trace(covariance_fraction)) <= 1.0,
    ]
    for index in range(problem.num_users):
        energy_term = cp.real(
            cp.trace(channel_matrices[index] @ covariance_fraction)
        )
        coefficient = (
            problem.zeta
            * uplink_gains[index]
            * problem.max_power
            / problem.noise_power
        )
        constraints.append(coefficient * energy_term >= minimum_snr)
    optimization = cp.Problem(cp.Maximize(minimum_snr), constraints)

    last_error: Exception | None = None
    z_raw = None
    for solver in ("CLARABEL", "SCS"):
        kwargs: dict[str, object] = {"solver": solver, "verbose": False}
        if solver == "SCS":
            kwargs.update({"eps": 1e-7, "max_iters": 50_000})
        try:
            optimization.solve(**kwargs)
            if optimization.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                z_raw = (
                    tau
                    * problem.max_power
                    * np.asarray(covariance_fraction.value, dtype=complex)
                )
                break
        except Exception as exc:  # pragma: no cover - solver-specific fallback.
            last_error = exc
    if z_raw is None:
        z_value = tau * isotropic
        tau_k = np.full(problem.num_users, tau)
        rates, _, _ = weighted_covariance_rates(
            problem, f_energy, f_information, tau_k, z_value
        )
        return float(np.min(rates)), z_value

    z_hermitian = 0.5 * (z_raw + z_raw.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(z_hermitian)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    z_value = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    trace_limit = tau * problem.max_power
    trace_value = float(np.trace(z_value).real)
    if trace_value > trace_limit > 0.0:
        z_value *= trace_limit / trace_value
    tau_k = np.full(problem.num_users, tau)
    rates, _, _ = weighted_covariance_rates(
        problem, f_energy, f_information, tau_k, z_value
    )
    return float(np.min(rates)), z_value


def _solve_equal_time_boresight_ao(
    problem: MultiDeviceProblem,
    *,
    max_outer_iter: int = 4,
    pga_max_iter: int = 15,
    f_energy_start: np.ndarray | None = None,
    f_information_start: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """ET baseline: equal slots with spherical-cap orientation optimization."""

    f_energy = (
        default_boresights(problem)
        if f_energy_start is None
        else np.asarray(f_energy_start, dtype=float).copy()
    )
    f_information = (
        default_boresights(problem, information=True)
        if f_information_start is None
        else np.asarray(f_information_start, dtype=float).copy()
    )
    best_rate, weighted_covariance = _equal_time_common_resource(
        problem, f_energy, f_information
    )
    best = (
        best_rate,
        f_energy.copy(),
        f_information.copy(),
        weighted_covariance.copy(),
    )
    tau = problem.frame_time / (problem.num_users + 1.0)
    tau_k = np.full(problem.num_users, tau)
    mu = 20.0
    for _ in range(max_outer_iter):
        update = update_fair_boresights(
            problem,
            f_energy,
            f_information,
            tau_k,
            weighted_covariance,
            mu=mu,
            pga_max_iter=pga_max_iter,
            pga_tolerance=1e-5,
        )
        f_energy = update.f_energy
        f_information = update.f_information
        try:
            rate, weighted_covariance = _equal_time_common_resource(
                problem, f_energy, f_information
            )
        except RuntimeError:
            break
        if rate > best[0]:
            best = (
                rate,
                f_energy.copy(),
                f_information.copy(),
                weighted_covariance.copy(),
            )
        relative_gain = abs(rate - best_rate) / max(1.0, abs(best_rate))
        best_rate = max(best_rate, rate)
        mu = min(2.0 * mu, 80.0)
        if relative_gain <= 1e-4:
            break
    return best


def _solve_proposed_common(
    problem: MultiDeviceProblem,
    boresight_starts: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    max_outer_iter: int = 4,
):
    candidates = []
    errors: list[Exception] = []
    for start in boresight_starts:
        solved = False
        for solver in ("CLARABEL", "SCS"):
            try:
                candidates.append(
                    solve_fair_rate_ao(
                        problem,
                        boresight_starts=[start],
                        max_outer_iter=max_outer_iter,
                        pga_max_iter=15,
                        ao_tolerance=1e-4,
                        pga_tolerance=1e-5,
                        mu_0=20.0,
                        mu_max=80.0,
                        mu_growth=2.0,
                        stable_rounds_required=1,
                        solver=solver,
                    )
                )
                solved = True
                break
            except Exception as exc:  # pragma: no cover - solver fallback.
                errors.append(exc)
        if not solved:
            continue
    if not candidates:
        raise RuntimeError("All proposed common-rate starts failed.") from errors[-1]
    return max(candidates, key=lambda solution: solution.common_throughput)


def build_sweeps(setup: SimulationSetup) -> dict[str, Sweep]:
    power_values = (10.0, 15.0, 20.0, 25.0, 30.0)
    user_values = (2.0, 4.0, 6.0, 8.0, 10.0)
    antenna_values = (4.0, 6.0, 8.0, 10.0, 12.0)

    def case(
        *,
        alpha: float = setup.default_path_loss_exponent,
        power: float = setup.default_power_dbm,
        users: int = setup.default_num_users,
        antennas: int = setup.default_num_antennas,
    ) -> SimulationCase:
        return SimulationCase(alpha, power, users, antennas)

    return {
        "power": Sweep(
            key="power",
            values=power_values,
            cases=tuple(case(power=value) for value in power_values),
            x_header="power_dbm",
            x_label=r"$P_A$ (dBm)",
            integer_ticks=False,
        ),
        "users": Sweep(
            key="users",
            values=user_values,
            cases=tuple(case(users=int(value)) for value in user_values),
            x_header="K",
            x_label=r"$K$",
            integer_ticks=True,
        ),
        "antennas": Sweep(
            key="antennas",
            values=antenna_values,
            cases=tuple(case(antennas=int(value)) for value in antenna_values),
            x_header="N_A",
            x_label=r"$N_A$",
            integer_ticks=True,
        ),
    }


@dataclass(frozen=True)
class StudyConfig:
    random_codebook_size: int = 32
    random_vector_length: int = 12
    shared_sum_outer_iterations: int = 6
    shared_common_outer_iterations: int = 4
    shared_fw_iterations: int = 15


@dataclass(frozen=True)
class FixedBeamCommonAllocation:
    common_rate: float
    tau_0: float
    tau_k: np.ndarray


@dataclass(frozen=True)
class SharedAngleSolution:
    rate: float
    posture: np.ndarray
    rate_history: np.ndarray


def _shared_boresight_start_from_separate(
    problem: MultiDeviceProblem,
    f_energy: np.ndarray,
    f_information: np.ndarray,
) -> np.ndarray:
    """Form a feasible column-wise midpoint for the shared spherical cap."""

    f_energy = np.asarray(f_energy, dtype=float)
    f_information = np.asarray(f_information, dtype=float)
    if f_energy.shape != (3, problem.num_antennas) or f_information.shape != f_energy.shape:
        raise ValueError("separate boresights must both have shape (3, N_A).")
    combined = f_energy + f_information
    norms = np.linalg.norm(combined, axis=0, keepdims=True)
    degenerate = norms[0] <= 1e-15
    combined[:, degenerate] = np.array([[0.0], [0.0], [1.0]])
    norms[:, degenerate] = 1.0
    return combined / norms


def _shared_boresight_common_gradient_data(
    problem: MultiDeviceProblem,
    f: np.ndarray,
    tau_k: np.ndarray,
    weighted_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    energy_state = boresight_channel_state(
        f,
        problem.directions,
        problem.propagation_energy,
        problem.rho,
    )
    information_state = boresight_channel_state(
        f,
        problem.directions,
        problem.propagation_information,
        problem.rho,
    )
    energy, energy_gradients = quadratic_boresight_values_and_gradients(
        energy_state, weighted_covariance
    )
    uplink, uplink_gradients = mrc_boresight_gains_and_gradients(information_state)
    tau_k = np.asarray(tau_k, dtype=float)
    active = tau_k > 0.0
    rho = np.zeros_like(tau_k)
    rho[active] = (
        problem.zeta
        * energy[active]
        * uplink[active]
        / (problem.noise_power * tau_k[active])
    )
    rates = np.zeros_like(tau_k)
    rates[active] = tau_k[active] * np.log2(1.0 + rho[active])
    coefficient = np.zeros_like(tau_k)
    coefficient[active] = (
        problem.zeta
        / (problem.noise_power * np.log(2.0) * (1.0 + rho[active]))
    )
    rate_gradients = coefficient[:, None, None] * (
        uplink[:, None, None] * energy_gradients
        + energy[:, None, None] * uplink_gradients
    )
    return rates, rate_gradients


def _update_shared_common_boresights(
    problem: MultiDeviceProblem,
    f: np.ndarray,
    tau_k: np.ndarray,
    weighted_covariance: np.ndarray,
    *,
    mu: float,
    fw_iterations: int,
) -> np.ndarray:
    alpha_max = min(
        problem.cap_half_angle_energy,
        problem.cap_half_angle_information,
    )

    def smooth_objective(candidate: np.ndarray) -> float:
        rates, _ = _shared_boresight_common_gradient_data(
            problem, candidate, tau_k, weighted_covariance
        )
        return smooth_min(rates, mu)

    def gradient(candidate: np.ndarray) -> np.ndarray:
        rates, rate_gradients = _shared_boresight_common_gradient_data(
            problem, candidate, tau_k, weighted_covariance
        )
        return np.sum(
            softmin_weights(rates, mu)[:, None, None] * rate_gradients,
            axis=0,
        )

    def true_objective(candidate: np.ndarray) -> float:
        rates, _, _ = weighted_covariance_rates(
            problem, candidate, candidate, tau_k, weighted_covariance
        )
        return float(np.min(rates))

    result = pga_spherical_cap_block(
        smooth_objective,
        gradient,
        true_objective,
        f,
        alpha_max,
        max_iter=fw_iterations,
        tolerance=1e-5,
    )
    return np.asarray(result.f, dtype=float)


def solve_shared_boresight_common(
    problem: MultiDeviceProblem,
    f_start: np.ndarray,
    config: StudyConfig,
) -> SharedAngleSolution:
    """Optimize Rcom under the exact equality constraint FE == FI."""

    f = np.asarray(f_start, dtype=float).copy()
    resource = _solve_common_with_fallback(problem, f, f)
    best_rate = float(resource.common_rate)
    rate_history = [best_rate]
    best_f = f.copy()
    mu = 20.0
    for _ in range(config.shared_common_outer_iterations):
        candidate_f = _update_shared_common_boresights(
            problem,
            f,
            resource.tau_k,
            resource.weighted_covariance,
            mu=mu,
            fw_iterations=config.shared_fw_iterations,
        )
        try:
            candidate_resource = _solve_common_with_fallback(
                problem, candidate_f, candidate_f
            )
        except RuntimeError:
            break
        if candidate_resource.common_rate < resource.common_rate - 5e-6 * max(
            1.0, abs(resource.common_rate)
        ):
            break
        relative_gain = abs(
            candidate_resource.common_rate - resource.common_rate
        ) / max(1.0, abs(resource.common_rate))
        f = candidate_f
        resource = candidate_resource
        if resource.common_rate > best_rate:
            best_rate = float(resource.common_rate)
            best_f = f.copy()
        rate_history.append(best_rate)
        mu = min(2.0 * mu, 80.0)
        if relative_gain <= 1e-4:
            break
    return SharedAngleSolution(
        best_rate,
        best_f,
        np.asarray(rate_history, dtype=float),
    )


def _required_uplink_times(
    common_rate: float,
    tau_0: float,
    gamma: np.ndarray,
) -> np.ndarray:
    if common_rate <= 0.0:
        return np.zeros_like(gamma)
    rate_nats = common_rate * np.log(2.0)
    a = np.asarray(gamma, dtype=float) * tau_0
    c = rate_nats / a
    if np.any(c >= 1.0):
        return np.full_like(gamma, np.inf)
    argument = -c * np.exp(-c)
    branch = np.real(lambertw(argument, k=-1))
    exponent = -branch - c
    return rate_nats / exponent


def fixed_beam_common_allocation(
    gamma: np.ndarray,
    frame_time: float,
) -> FixedBeamCommonAllocation:
    """Optimize WPT/uplink times for one prescribed energy covariance."""

    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if gamma.size == 0 or np.any(gamma < 0.0) or not np.all(np.isfinite(gamma)):
        raise ValueError("gamma must contain finite nonnegative fixed-beam coefficients.")
    if np.any(gamma == 0.0):
        return FixedBeamCommonAllocation(0.0, 0.0, np.zeros_like(gamma))

    def allocation_at_tau0(tau_0: float) -> FixedBeamCommonAllocation:
        remaining = frame_time - tau_0
        if tau_0 <= 0.0 or remaining <= 0.0:
            return FixedBeamCommonAllocation(0.0, tau_0, np.zeros_like(gamma))
        a = gamma * tau_0
        upper = float(
            np.min(remaining * np.log2(1.0 + a / remaining))
        )
        low = 0.0
        high = upper
        best_times = np.zeros_like(gamma)
        for _ in range(55):
            middle = 0.5 * (low + high)
            times = _required_uplink_times(middle, tau_0, gamma)
            if float(np.sum(times)) <= remaining:
                low = middle
                best_times = times
            else:
                high = middle
        return FixedBeamCommonAllocation(low, tau_0, best_times)

    lower = frame_time * 1e-6
    upper = frame_time * (1.0 - 1e-6)
    optimized = minimize_scalar(
        lambda value: -allocation_at_tau0(float(value)).common_rate,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": frame_time * 1e-7, "maxiter": 80},
    )
    candidates = (
        allocation_at_tau0(float(optimized.x)),
        allocation_at_tau0(frame_time * 0.25),
        allocation_at_tau0(frame_time * 0.50),
        allocation_at_tau0(frame_time * 0.75),
    )
    return max(candidates, key=lambda item: item.common_rate)


def random_beam_covariances(
    problem: MultiDeviceProblem,
    setup: SimulationSetup,
    snapshot_index: int,
    config: StudyConfig,
) -> tuple[np.ndarray, ...]:
    """Return a common-random-number codebook for all parameter sweeps."""

    if problem.num_antennas > config.random_vector_length:
        raise ValueError("random_vector_length is smaller than N_A.")
    seed_sequence = np.random.SeedSequence(
        [setup.topology_seed, snapshot_index, 9137]
    )
    rng = np.random.default_rng(seed_sequence)
    raw = rng.standard_normal(
        (config.random_codebook_size, config.random_vector_length)
    ) + 1j * rng.standard_normal(
        (config.random_codebook_size, config.random_vector_length)
    )
    covariances = []
    for row in raw:
        beam = row[: problem.num_antennas]
        beam = beam / np.linalg.norm(beam)
        covariances.append(
            problem.max_power * np.outer(beam, beam.conj())
        )
    return tuple(covariances)


def _best_random_common_rate(
    problem: MultiDeviceProblem,
    f_energy: np.ndarray,
    f_information: np.ndarray,
    covariances: tuple[np.ndarray, ...],
) -> float:
    best = 0.0
    for covariance in covariances:
        gamma, _, _, _, _ = evaluate_boresight_composite_gains(
            problem, f_energy, f_information, covariance
        )
        allocation = fixed_beam_common_allocation(gamma, problem.frame_time)
        best = max(best, allocation.common_rate)
    return best


@dataclass(frozen=True)
class CommonSnapshotResult:
    case: SimulationCase
    snapshot_index: int
    proposed: float
    equal_time: float
    fixed_hap: float
    random_bf: float
    shared_posture: float


def _coverage_boresights(
    problem: MultiDeviceProblem,
    *,
    information: bool,
) -> np.ndarray:
    """Return a deterministic fan of spherical-cap boresights."""

    n_antennas = problem.num_antennas
    alpha_max = (
        problem.cap_half_angle_information
        if information
        else problem.cap_half_angle_energy
    )
    phi = -np.pi + (np.arange(n_antennas) + 0.5) * (2.0 * np.pi / n_antennas)
    return np.vstack(
        (
            np.sin(alpha_max) * np.cos(phi),
            np.sin(alpha_max) * np.sin(phi),
            np.full(n_antennas, np.cos(alpha_max)),
        )
    )
