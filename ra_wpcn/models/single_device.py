"""Formula-complete single-device RA-WPCN model for paper Section III."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.special import lambertw
except Exception:  # pragma: no cover - SciPy is optional
    lambertw = None


Array = np.ndarray
REFERENCE_BORESIGHT = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class TimeAllocation:
    """Closed-form WPT/WIT time allocation."""

    tau_0: float
    tau_k: Array
    z: float
    objective: float


@dataclass(frozen=True)
class SingleDeviceSolution:
    """End-to-end solution of the free-space single-device problem."""

    f_energy: Array
    f_information: Array
    v_energy: Array
    u_information: Array
    g_energy: float
    g_information: float
    gamma: float
    time_allocation: TimeAllocation
    covariance_energy: Array
    harvested_energy: float
    uplink_power: float
    residual_energy: Array
    residual_information: Array


@dataclass(frozen=True)
class UPAGainBounds:
    """Inscribed/circumscribed-disk bounds for a rectangular UPA."""

    radius_lower: float
    radius_upper: float
    gain_lower: float
    gain_upper: float


def _as_column(vector: Array) -> Array:
    return np.asarray(vector, dtype=complex).reshape(-1, 1)


def _normalize(vector: Array) -> Array:
    vector = _as_column(vector)
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector.")
    return vector / norm


def _unit_rows(vectors: Array, name: str = "vectors") -> Array:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"{name} must have shape (rows, 3).")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} must be nonzero.")
    return vectors / norms


def _unit_vector(vector: Array, name: str) -> Array:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vector / norm


def _user_geometry(element_positions: Array, user_position: Array) -> tuple[Array, Array]:
    elements = np.asarray(element_positions, dtype=float)
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("element_positions must have shape (N_A, 3).")
    user = np.asarray(user_position, dtype=float).reshape(1, 3)
    displacement = user - elements
    distances = np.linalg.norm(displacement, axis=1)
    if np.any(distances <= 0.0):
        raise ValueError("user_position must not coincide with an antenna element.")
    return displacement / distances[:, None], distances


def project_angle_2d(
    target_angle: float,
    reference_angle: float,
    max_rotation_angle: float,
) -> float:
    """Project a planar angle onto a wrapped feasible interval."""

    delta = (target_angle - reference_angle + np.pi) % (2.0 * np.pi) - np.pi
    clipped = np.clip(delta, -abs(max_rotation_angle), abs(max_rotation_angle))
    return float((reference_angle + clipped + np.pi) % (2.0 * np.pi) - np.pi)


def optimal_pointing_vectors(
    element_positions: Array,
    user_position: Array,
    theta_max: float,
    reference_boresight: Array = REFERENCE_BORESIGHT,
) -> Array:
    """Project each WD direction onto the boresight cone in (III-32).

    ``theta_max`` is the allowed deviation from ``reference_boresight``.  The
    default reference is the paper's positive-z vector e3=[0,0,1]^T.
    """

    if not 0.0 <= float(theta_max) <= np.pi / 2.0:
        raise ValueError("theta_max must lie in [0, pi/2].")
    directions, _ = _user_geometry(element_positions, user_position)
    b0 = _unit_vector(reference_boresight, "reference_boresight")
    cos_alpha = np.clip(directions @ b0, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    result = directions.copy()
    outside = alpha > float(theta_max)

    if np.any(outside):
        d = directions[outside] - cos_alpha[outside, None] * b0[None, :]
        d_norm = np.linalg.norm(d, axis=1)
        # If the target is exactly opposite b0, the nearest cone boundary is
        # azimuth-degenerate.  Choose a deterministic orthogonal direction.
        degenerate = d_norm <= 1e-14
        if np.any(degenerate):
            trial = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(trial, b0)) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            orthogonal = trial - np.dot(trial, b0) * b0
            orthogonal /= np.linalg.norm(orthogonal)
            d[degenerate] = orthogonal
            d_norm[degenerate] = 1.0
        tangent = d / d_norm[:, None]
        result[outside] = (
            np.cos(theta_max) * b0[None, :]
            + np.sin(theta_max) * tangent
        )
    return _unit_rows(result, "optimal pointing vectors")


def optimal_pointing_angles(pointing_vectors: Array) -> tuple[Array, Array]:
    """Return the global zenith and azimuth angles in (III-33)--(III-36)."""

    f = _unit_rows(pointing_vectors, "pointing_vectors")
    zenith = np.arccos(np.clip(f[:, 2], -1.0, 1.0))
    azimuth = np.arctan2(f[:, 1], f[:, 0])
    return zenith, azimuth


def residual_misalignment_angles(
    element_positions: Array,
    user_position: Array,
    pointing_vectors: Array,
) -> Array:
    """Return omega_n=arccos(f_n^T s_n)."""

    directions, _ = _user_geometry(element_positions, user_position)
    f = _unit_rows(pointing_vectors, "pointing_vectors")
    if f.shape != directions.shape:
        raise ValueError("pointing_vectors and element_positions must have equal rows.")
    return np.arccos(np.clip(np.sum(f * directions, axis=1), -1.0, 1.0))


def theoretical_residual_angles(
    element_positions: Array,
    user_position: Array,
    theta_max: float,
    reference_boresight: Array = REFERENCE_BORESIGHT,
) -> Array:
    """Return [alpha_n-theta_max]_+ from (III-37)."""

    directions, _ = _user_geometry(element_positions, user_position)
    b0 = _unit_vector(reference_boresight, "reference_boresight")
    alpha = np.arccos(np.clip(directions @ b0, -1.0, 1.0))
    return np.maximum(alpha - float(theta_max), 0.0)


def cosine_ra_los_channel(
    element_positions: Array,
    user_position: Array,
    pointing_vectors: Array,
    wavelength: float = 1.0,
    aperture_area: float = 1.0,
    boresight_gain: float = 1.0,
    pattern_order: float = 1.0,
) -> Array:
    """Generate the free-space LoS channel used in (III-24)--(III-26)."""

    if wavelength <= 0.0 or aperture_area < 0.0 or boresight_gain < 0.0:
        raise ValueError("invalid wavelength, aperture area, or boresight gain.")
    if pattern_order < 0.0:
        raise ValueError("pattern_order must be nonnegative.")

    directions, distances = _user_geometry(element_positions, user_position)
    pointing = _unit_rows(pointing_vectors, "pointing_vectors")
    if pointing.shape != directions.shape:
        raise ValueError("pointing_vectors and element_positions must have equal rows.")
    cos_eps = np.sum(pointing * directions, axis=1)
    directional = np.maximum(cos_eps, 0.0) ** (2.0 * float(pattern_order))
    amplitude = np.sqrt(
        float(aperture_area)
        * float(boresight_gain)
        * directional
        / (4.0 * np.pi * distances**2)
    )
    phase = np.exp(-1j * 2.0 * np.pi * distances / float(wavelength))
    return (amplitude * phase).reshape(-1, 1)


def single_link_gain_from_residuals(
    distances: Array,
    residual_angles: Array,
    aperture_area: float,
    boresight_gain: float,
    pattern_order: float,
) -> float:
    """Evaluate the closed-form gain in (III-38)."""

    distances = np.asarray(distances, dtype=float).reshape(-1)
    residual = np.asarray(residual_angles, dtype=float).reshape(-1)
    if distances.shape != residual.shape or np.any(distances <= 0.0):
        raise ValueError("distances and residual_angles must be matched valid vectors.")
    coefficient = float(aperture_area) * float(boresight_gain) / (4.0 * np.pi)
    return float(
        coefficient
        * np.sum(np.cos(residual) ** (2.0 * float(pattern_order)) / distances**2)
    )


def mrt_beamformer(channel: Array) -> Array:
    """Unit-norm maximum-ratio transmission beam."""

    return _normalize(channel)


def mrt_covariance(channel: Array, tx_power: float) -> Array:
    """Return S_E^*=P_A v_E^* v_E^{*,H} from (III-10)."""

    if tx_power < 0.0:
        raise ValueError("tx_power must be nonnegative.")
    beam = mrt_beamformer(channel)
    return float(tx_power) * beam @ beam.conj().T


def mrc_beamformer(channel: Array) -> Array:
    """Unit-norm maximum-ratio combining vector."""

    return _normalize(channel)


def effective_gain(channel: Array) -> float:
    """Return ||h||_2^2 after MRT/MRC substitution."""

    channel = _as_column(channel)
    return float(np.vdot(channel, channel).real)


def composite_gain(
    g_energy: float,
    g_information: float,
    zeta: float,
    tx_power: float,
    noise_power: float,
) -> float:
    """Return Gamma=zeta P_A G_E G_I/sigma^2."""

    if g_energy < 0.0 or g_information < 0.0:
        raise ValueError("single-link gains must be nonnegative.")
    if not 0.0 < zeta <= 1.0 or tx_power < 0.0 or noise_power <= 0.0:
        raise ValueError("invalid energy, power, or noise parameter.")
    return float(zeta * tx_power * g_energy * g_information / noise_power)


def solve_z_from_gain(gain: float, tol: float = 1e-12, max_iter: int = 100) -> float:
    """Solve z(ln z-1)=gain-1 for z>=1."""

    gain = float(gain)
    if gain <= 0.0:
        return 1.0
    if lambertw is not None:
        value = np.exp(1.0 + lambertw((gain - 1.0) / np.e, k=0).real)
        if np.isfinite(value) and value >= 1.0:
            return float(value)

    def residual(z: float) -> float:
        return z * np.log(z) - z + 1.0 - gain

    left, right = 1.0, max(2.0, 1.0 + gain)
    while residual(right) < 0.0:
        right *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        if residual(mid) >= 0.0:
            right = mid
        else:
            left = mid
        if right - left <= tol * max(1.0, right):
            break
    return float(0.5 * (left + right))


def optimal_single_device_time(
    gamma: float,
    frame_time: float = 1.0,
) -> TimeAllocation:
    """Evaluate (III-43)--(III-50) for a frame of length ``frame_time``."""

    gamma = float(max(gamma, 0.0))
    frame_time = float(frame_time)
    if frame_time <= 0.0:
        raise ValueError("frame_time must be positive.")
    if gamma <= 0.0:
        return TimeAllocation(0.0, np.array([frame_time]), 1.0, 0.0)

    z = solve_z_from_gain(gamma)
    denominator = gamma + z - 1.0
    tau_0 = frame_time * (z - 1.0) / denominator
    tau_1 = frame_time * gamma / denominator
    objective = tau_1 * np.log2(z)
    return TimeAllocation(
        tau_0=float(tau_0),
        tau_k=np.array([float(tau_1)]),
        z=float(z),
        objective=float(objective),
    )


def optimal_throughput_phi(gamma: float) -> float:
    """Return Phi(Gamma), using the stable time-allocation representation."""

    return optimal_single_device_time(gamma, frame_time=1.0).objective


def solve_single_device_closed_form(
    element_positions: Array,
    user_position: Array,
    theta_max: float,
    zeta: float,
    tx_power: float,
    noise_power: float,
    frame_time: float = 1.0,
    wavelength: float = 1.0,
    aperture_area: float = 1.0,
    boresight_gain: float = 1.0,
    pattern_order: float = 1.0,
    *,
    theta_max_information: float | None = None,
    boresight_gain_information: float | None = None,
    pattern_order_information: float | None = None,
    reference_boresight: Array = REFERENCE_BORESIGHT,
) -> SingleDeviceSolution:
    """Solve the full Section-III chain without a numerical optimizer."""

    theta_i = theta_max if theta_max_information is None else theta_max_information
    gain_i = (
        boresight_gain
        if boresight_gain_information is None
        else boresight_gain_information
    )
    order_i = (
        pattern_order
        if pattern_order_information is None
        else pattern_order_information
    )
    f_energy = optimal_pointing_vectors(
        element_positions,
        user_position,
        theta_max,
        reference_boresight=reference_boresight,
    )
    f_information = optimal_pointing_vectors(
        element_positions,
        user_position,
        theta_i,
        reference_boresight=reference_boresight,
    )
    h_energy = cosine_ra_los_channel(
        element_positions,
        user_position,
        f_energy,
        wavelength=wavelength,
        aperture_area=aperture_area,
        boresight_gain=boresight_gain,
        pattern_order=pattern_order,
    )
    h_information = cosine_ra_los_channel(
        element_positions,
        user_position,
        f_information,
        wavelength=wavelength,
        aperture_area=aperture_area,
        boresight_gain=gain_i,
        pattern_order=order_i,
    )
    g_energy = effective_gain(h_energy)
    g_information = effective_gain(h_information)
    gamma = composite_gain(
        g_energy,
        g_information,
        zeta,
        tx_power,
        noise_power,
    )
    time = optimal_single_device_time(gamma, frame_time=frame_time)
    covariance = mrt_covariance(h_energy, tx_power)
    harvested = (
        float(zeta)
        * time.tau_0
        * float(np.vdot(h_energy, covariance @ h_energy).real)
    )
    tau_1 = float(time.tau_k[0])
    uplink_power = harvested / tau_1 if tau_1 > 0.0 else 0.0
    return SingleDeviceSolution(
        f_energy=f_energy,
        f_information=f_information,
        v_energy=mrt_beamformer(h_energy),
        u_information=mrc_beamformer(h_information),
        g_energy=g_energy,
        g_information=g_information,
        gamma=gamma,
        time_allocation=time,
        covariance_energy=covariance,
        harvested_energy=float(harvested),
        uplink_power=float(uplink_power),
        residual_energy=residual_misalignment_angles(
            element_positions, user_position, f_energy
        ),
        residual_information=residual_misalignment_angles(
            element_positions, user_position, f_information
        ),
    )


def ula_span_angle(num_antennas: int, spacing_over_range: float) -> float:
    """Return Delta_span(N_x)=atan(N_x delta/2)."""

    if int(num_antennas) <= 0 or spacing_over_range <= 0.0:
        raise ValueError("num_antennas and spacing_over_range must be positive.")
    return float(np.arctan(int(num_antennas) * spacing_over_range / 2.0))


def ula_inner_region_limit(theta_max: float, spacing_over_range: float) -> int:
    """Return Nbar=2 floor(tan(theta_max)/delta)+1."""

    if spacing_over_range <= 0.0 or not 0.0 <= theta_max < np.pi / 2.0:
        raise ValueError("require delta>0 and theta_max in [0,pi/2).")
    return int(2 * np.floor(np.tan(theta_max) / spacing_over_range) + 1)


def ula_single_link_gain(
    num_antennas: int,
    spacing_over_range: float,
    aperture_over_spacing_squared: float,
    theta_max: float,
) -> float:
    """Evaluate the piecewise ULA gain in (III-56) for p=1/2."""

    if aperture_over_spacing_squared < 0.0:
        raise ValueError("aperture_over_spacing_squared must be nonnegative.")
    span = ula_span_angle(num_antennas, spacing_over_range)
    threshold = ula_inner_region_limit(theta_max, spacing_over_range)
    if int(num_antennas) <= threshold:
        bracket = span
    else:
        bracket = theta_max + np.sin(span - theta_max)
    return float(
        2.0
        * aperture_over_spacing_squared
        * spacing_over_range
        * bracket
        / np.pi
    )


def ula_infinite_single_link_gain(
    spacing_over_range: float,
    aperture_over_spacing_squared: float,
    theta_max: float,
) -> float:
    """Evaluate the finite ULA gain limit in (III-59)."""

    if spacing_over_range <= 0.0 or aperture_over_spacing_squared < 0.0:
        raise ValueError("invalid normalized spacing or aperture.")
    return float(
        2.0
        * aperture_over_spacing_squared
        * spacing_over_range
        * (theta_max + np.cos(theta_max))
        / np.pi
    )


def upa_inner_region_gain(
    num_x: int,
    num_y: int,
    boresight_gain: float,
    aperture_over_spacing_squared: float,
    spacing_over_range: float,
) -> float:
    """Evaluate the UPA approximation in (III-65)."""

    if min(int(num_x), int(num_y)) <= 0:
        raise ValueError("UPA dimensions must be positive.")
    coefficient = (
        boresight_gain
        * np.pi
        * aperture_over_spacing_squared
        * spacing_over_range**2
        / 64.0
    )
    return float(coefficient * int(num_x) * int(num_y))


def upa_disk_gain(
    radius: float,
    range_to_array: float,
    theta_max: float,
    pattern_order: float,
    boresight_gain: float,
    aperture_over_spacing_squared: float,
    quadrature_points: int = 4097,
) -> float:
    """Numerically evaluate the disk gain function G_x(R) in (III-70)."""

    if radius < 0.0 or range_to_array <= 0.0 or quadrature_points < 3:
        raise ValueError("invalid radius, range, or quadrature size.")
    d = min(float(radius), float(range_to_array) * np.tan(theta_max))
    lower = np.arctan(d / range_to_array) - theta_max
    upper = np.arctan(radius / range_to_array) - theta_max
    integral = 0.0
    if upper > lower:
        omega = np.linspace(lower, upper, int(quadrature_points))
        integrand = (
            np.cos(omega) ** (2.0 * pattern_order)
            * np.tan(omega + theta_max)
        )
        trapezoid = getattr(np, "trapezoid", np.trapz)
        integral = float(trapezoid(integrand, omega))
    bracket = 0.5 * np.log1p((d / range_to_array) ** 2) + integral
    return float(
        0.5
        * boresight_gain
        * aperture_over_spacing_squared
        * bracket
    )


def upa_gain_bounds(
    num_x: int,
    num_y: int,
    spacing: float,
    range_to_array: float,
    theta_max: float,
    pattern_order: float,
    boresight_gain: float,
    aperture_over_spacing_squared: float,
) -> UPAGainBounds:
    """Evaluate (III-67)--(III-71)."""

    if min(int(num_x), int(num_y)) <= 0 or spacing <= 0.0:
        raise ValueError("invalid UPA dimensions or spacing.")
    radius_lower = 0.5 * min(num_x * spacing, num_y * spacing)
    radius_upper = 0.5 * np.hypot(num_x * spacing, num_y * spacing)
    return UPAGainBounds(
        radius_lower=float(radius_lower),
        radius_upper=float(radius_upper),
        gain_lower=upa_disk_gain(
            radius_lower,
            range_to_array,
            theta_max,
            pattern_order,
            boresight_gain,
            aperture_over_spacing_squared,
        ),
        gain_upper=upa_disk_gain(
            radius_upper,
            range_to_array,
            theta_max,
            pattern_order,
            boresight_gain,
            aperture_over_spacing_squared,
        ),
    )


def high_power_asymptotics(
    c_n: float,
    tx_power: float,
) -> dict[str, float]:
    """Evaluate the leading expressions in (III-76)--(III-80)."""

    if c_n <= 0.0 or tx_power <= 0.0:
        raise ValueError("c_n and tx_power must be positive.")
    gamma = c_n * tx_power
    w_a = (
        float(lambertw((gamma - 1.0) / np.e, k=0).real)
        if lambertw is not None
        else float(np.log(max(gamma, np.e)) - np.log(np.log(max(gamma, np.e))))
    )
    tau_0_asymptotic = 1.0 / (w_a + 1.0)
    tau_1_asymptotic = w_a / (w_a + 1.0)
    rate_lambert = (
        float(lambertw(gamma / np.e, k=0).real) / np.log(2.0)
        if lambertw is not None
        else (np.log(gamma) - np.log(np.log(gamma)) - 1.0) / np.log(2.0)
    )
    rate_expansion = (
        np.log2(gamma)
        - np.log2(np.log(gamma))
        - 1.0 / np.log(2.0)
        if gamma > 1.0
        else np.nan
    )
    return {
        "gamma": float(gamma),
        "w_a": float(w_a),
        "tau_0_asymptotic": float(tau_0_asymptotic),
        "tau_1_asymptotic": float(tau_1_asymptotic),
        "rate_lambert": float(rate_lambert),
        "rate_expansion": float(rate_expansion),
    }


def ra_fa_throughput_comparison(
    gamma_ra: float,
    gamma_fa: float,
) -> dict[str, float]:
    """Evaluate (1)--(8) of the RA--FA comparison."""

    if gamma_ra < gamma_fa or gamma_fa < 0.0:
        raise ValueError("require gamma_ra >= gamma_fa >= 0.")
    rate_ra = optimal_throughput_phi(gamma_ra)
    rate_fa = optimal_throughput_phi(gamma_fa)
    return {
        "gamma_ra": float(gamma_ra),
        "gamma_fa": float(gamma_fa),
        "rate_ra": float(rate_ra),
        "rate_fa": float(rate_fa),
        "rate_ratio": float(rate_ra / rate_fa) if rate_fa > 0.0 else np.inf,
    }


def symmetric_ula_ra_fa_factor(theta_max: float) -> float:
    """Return (theta_max+cos(theta_max))^2 from (III-63) and (9)."""

    if not 0.0 <= theta_max <= np.pi / 2.0:
        raise ValueError("theta_max must lie in [0, pi/2].")
    return float((theta_max + np.cos(theta_max)) ** 2)
