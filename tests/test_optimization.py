"""Gradients, resource allocation, and AO integration tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np
import importlib.util

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ra_wpcn.solvers.angle_optimization import (
    assert_spherical_cap_pga_result,
    directional_gain_and_derivatives,
    fw_lmo_box,
    mrc_gains_and_gradients,
    posture_channel_state,
    project_spherical_cap,
    quadratic_boresight_values_and_gradients,
    quadratic_values_and_gradients,
)
from ra_wpcn.solvers.ao import default_posture
from ra_wpcn.solvers.multiple_device import (
    angle_box,
    recover_energy_beams,
    smooth_min,
    softmin_weights,
    update_fair_boresights,
)
from ra_wpcn.solvers.angle_optimization import boresight_channel_state, pga_spherical_cap_block, mrc_boresight_gains_and_gradients
from ra_wpcn.solvers.ao import build_free_space_problem, default_boresights, solve_fair_rate_ao
from ra_wpcn.solvers.multiple_device import optimal_uplink_boresights, solve_common_resource


def build_problem(rho: float = 2.0):
    num_antennas = 3
    elements = np.column_stack(
        (np.linspace(-0.1, 0.1, num_antennas), np.zeros(num_antennas), np.zeros(num_antennas))
    )
    users = np.array([[-1.5, -0.5, 3.0], [1.0, 0.8, 3.5]])
    lower, upper = angle_box(
        num_antennas,
        np.deg2rad(70.0),
        -np.pi / 2.0,
        np.pi / 2.0,
    )
    return build_free_space_problem(
        element_positions=elements,
        device_positions=users,
        lower_energy=lower,
        upper_energy=upper,
        lower_information=lower,
        upper_information=upper,
        zeta=0.6,
        noise_power=1e-9,
        max_power=1.0,
        rho=rho,
        angle_smoothing=1e-3,
    )


class AngleDerivativeTests(unittest.TestCase):
    def test_rcom_direct_channel_requires_rho_greater_than_one(self):
        problem = build_problem()
        with self.assertRaises(ValueError):
            boresight_channel_state(
                default_boresights(problem),
                problem.directions,
                problem.propagation_information,
                1.0,
            )

    def test_cosine_pattern_uses_rho_two_and_peak_gain_ten(self):
        q = np.array([0.0, 0.0])
        directions = np.array(
            [
                [[0.0, 0.0, 1.0]],
                [[0.0, 0.0, -1.0]],
                [[1.0, 0.0, 0.0]],
            ]
        )
        physical, _, _ = directional_gain_and_derivatives(
            q, directions, 2.0, smoothing=None
        )
        np.testing.assert_allclose(
            physical[:, 0], np.array([10.0, 0.0, 0.0]), atol=1e-15
        )

        epsilon_c = 1e-3
        smoothed, _, _ = directional_gain_and_derivatives(
            q, directions, 2.0, smoothing=epsilon_c
        )
        self.assertAlmostEqual(smoothed[2, 0], 10.0 * (epsilon_c / 2.0) ** 4)

    def test_direct_boresight_gradients_match_finite_differences(self):
        problem = build_problem()
        f = default_boresights(problem, information=True)
        state = boresight_channel_state(
            f,
            problem.directions,
            problem.propagation_information,
            problem.rho,
        )
        gains, gain_gradients = mrc_boresight_gains_and_gradients(state)
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(problem.num_antennas, problem.num_antennas))
        matrix = matrix @ matrix.T
        values, value_gradients = quadratic_boresight_values_and_gradients(
            state, matrix
        )
        step = 1e-6
        finite_gain = np.zeros_like(gain_gradients)
        finite_value = np.zeros_like(value_gradients)
        for antenna in range(problem.num_antennas):
            for coordinate in range(3):
                delta = np.zeros_like(f)
                delta[coordinate, antenna] = step
                plus = boresight_channel_state(
                    f + delta,
                    problem.directions,
                    problem.propagation_information,
                    problem.rho,
                )
                minus = boresight_channel_state(
                    f - delta,
                    problem.directions,
                    problem.propagation_information,
                    problem.rho,
                )
                plus_gains, _ = mrc_boresight_gains_and_gradients(plus)
                minus_gains, _ = mrc_boresight_gains_and_gradients(minus)
                plus_values, _ = quadratic_boresight_values_and_gradients(plus, matrix)
                minus_values, _ = quadratic_boresight_values_and_gradients(minus, matrix)
                finite_gain[:, coordinate, antenna] = (
                    plus_gains - minus_gains
                ) / (2.0 * step)
                finite_value[:, coordinate, antenna] = (
                    plus_values - minus_values
                ) / (2.0 * step)
        np.testing.assert_allclose(gain_gradients, finite_gain, rtol=2e-5, atol=1e-10)
        np.testing.assert_allclose(value_gradients, finite_value, rtol=2e-5, atol=1e-10)
        self.assertTrue(np.all(gains > 0.0))
        self.assertTrue(np.all(values >= 0.0))

    def test_spherical_cap_projection_and_boundary_stationarity(self):
        alpha = np.deg2rad(70.0)
        projected = project_spherical_cap(
            np.array([[1.0], [0.0], [-0.2]]),
            alpha,
        )
        np.testing.assert_allclose(np.linalg.norm(projected, axis=0), 1.0)
        np.testing.assert_allclose(projected[2], np.cos(alpha), atol=1e-12)

        current = np.array([[np.sin(alpha)], [0.0], [np.cos(alpha)]])
        target_angle = np.deg2rad(80.0)
        target = np.array(
            [[np.sin(target_angle)], [0.0], [np.cos(target_angle)]]
        )
        result = pga_spherical_cap_block(
            lambda f: float(np.sum(f * target)),
            lambda f: target,
            lambda f: float(np.sum(f * target)),
            current,
            alpha,
            tolerance=1e-12,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertTrue(result.report["projected_gradient_stationary"])

    def test_true_rate_guard_failure_is_not_reported_as_pga_convergence(self):
        initial = np.array([[0.0], [0.0], [1.0]])
        result = pga_spherical_cap_block(
            lambda f: float(f[0, 0]),
            lambda f: np.array([[1.0], [0.0], [0.0]]),
            lambda f: -abs(float(f[0, 0])),
            initial,
            np.deg2rad(70.0),
            max_iter=2,
        )
        self.assertEqual(result.reason, "no_admissible_step")
        self.assertFalse(result.converged)
        self.assertFalse(result.report["projected_gradient_stationary"])
        self.assertTrue(result.report["block_stable"])
        self.assertTrue(result.report["verified"])
        np.testing.assert_array_equal(result.f, initial)
        self.assertEqual(result.step_history.size, 0)

    def test_pga_step_is_the_manuscript_projected_gradient_update(self):
        initial = np.array([[0.0], [0.0], [1.0]])
        gradient = np.array([[1.0], [0.0], [0.0]])
        alpha = np.deg2rad(70.0)
        result = pga_spherical_cap_block(
            lambda f: float(f[0, 0]),
            lambda f: gradient,
            lambda f: float(f[0, 0]),
            initial,
            alpha,
            max_iter=1,
            tolerance=1e-20,
            initial_step=1.0,
            backtracking=0.5,
        )
        expected = project_spherical_cap(initial + 0.5 * gradient, alpha)
        np.testing.assert_allclose(result.f, expected, atol=1e-12)
        np.testing.assert_allclose(result.step_history, [0.5], atol=0.0)

    def test_mrc_and_quadratic_gradients_match_finite_differences(self):
        problem = build_problem()
        q = default_posture(problem, information=True)
        state = posture_channel_state(
            q,
            problem.directions,
            problem.propagation_information,
            problem.rho,
            smoothing=problem.angle_smoothing,
        )
        gains, gain_gradients = mrc_gains_and_gradients(state)
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(problem.num_antennas, problem.num_antennas))
        matrix = matrix @ matrix.T
        values, value_gradients = quadratic_values_and_gradients(state, matrix)

        step = 1e-6
        finite_gain = np.zeros_like(gain_gradients)
        finite_value = np.zeros_like(value_gradients)
        for coordinate in range(q.size):
            delta = np.zeros_like(q)
            delta[coordinate] = step
            plus = posture_channel_state(
                q + delta,
                problem.directions,
                problem.propagation_information,
                problem.rho,
                smoothing=problem.angle_smoothing,
            )
            minus = posture_channel_state(
                q - delta,
                problem.directions,
                problem.propagation_information,
                problem.rho,
                smoothing=problem.angle_smoothing,
            )
            plus_gains, _ = mrc_gains_and_gradients(plus)
            minus_gains, _ = mrc_gains_and_gradients(minus)
            plus_values, _ = quadratic_values_and_gradients(plus, matrix)
            minus_values, _ = quadratic_values_and_gradients(minus, matrix)
            finite_gain[:, coordinate] = (plus_gains - minus_gains) / (2.0 * step)
            finite_value[:, coordinate] = (plus_values - minus_values) / (2.0 * step)

        np.testing.assert_allclose(gain_gradients, finite_gain, rtol=2e-5, atol=1e-10)
        np.testing.assert_allclose(value_gradients, finite_value, rtol=2e-5, atol=1e-10)
        self.assertTrue(np.all(gains > 0.0))
        self.assertTrue(np.all(values >= 0.0))

    def test_box_lmo_dominates_random_feasible_points(self):
        rng = np.random.default_rng(9)
        lower = np.array([-1.0, 0.0, -0.5, 1.0])
        upper = np.array([2.0, 3.0, 0.5, 2.0])
        current = rng.uniform(lower, upper)
        gradient = np.array([0.4, -0.7, 0.0, 1.2])
        vertex, gap = fw_lmo_box(gradient, lower, upper, current)
        candidates = rng.uniform(lower, upper, size=(500, lower.size))
        self.assertTrue(np.all(vertex >= lower) and np.all(vertex <= upper))
        self.assertGreaterEqual(float(gradient @ vertex), float(np.max(candidates @ gradient)) - 1e-12)
        self.assertGreaterEqual(gap, 0.0)

    def test_zero_exponent_removes_posture_dependence(self):
        problem = build_problem(rho=0.0)
        q = default_posture(problem)
        state = posture_channel_state(
            q,
            problem.directions,
            problem.propagation_energy,
            problem.rho,
            smoothing=problem.angle_smoothing,
        )
        _, gradients = mrc_gains_and_gradients(state)
        np.testing.assert_array_equal(gradients, np.zeros_like(gradients))


class FairRateAlgorithmTests(unittest.TestCase):
    def test_lse_bound_and_weights(self):
        values = np.array([0.2, 0.7, 1.1])
        mu = 25.0
        smoothed = smooth_min(values, mu)
        self.assertLessEqual(smoothed, float(np.min(values)) + 1e-15)
        self.assertGreaterEqual(smoothed, float(np.min(values)) - np.log(values.size) / mu - 1e-15)
        weights = softmin_weights(values, mu)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
        self.assertEqual(int(np.argmax(weights)), int(np.argmin(values)))

    def test_common_resource_constraints_and_beam_recovery(self):
        problem = build_problem()
        f_energy = default_boresights(problem)
        f_information = optimal_uplink_boresights(problem)
        resource = solve_common_resource(problem, f_energy, f_information)
        self.assertGreater(resource.common_rate, 0.0)
        self.assertLessEqual(max(resource.constraint_residuals.values()), 2e-6)
        recovered = sum(
            (beam @ beam.conj().T for beam in resource.energy_beams),
            np.zeros_like(resource.covariance),
        )
        np.testing.assert_allclose(recovered, resource.covariance, rtol=1e-6, atol=1e-8)
        self.assertEqual(len(resource.energy_beams), len(recover_energy_beams(resource.covariance)))

    def test_lse_pga_and_fair_ao_preserve_true_common_rate(self):
        problem = build_problem()
        f_energy = default_boresights(problem)
        f_information = optimal_uplink_boresights(problem)
        resource = solve_common_resource(problem, f_energy, f_information)
        update = update_fair_boresights(
            problem,
            f_energy,
            f_information,
            resource.tau_k,
            resource.weighted_covariance,
            mu=20.0,
            pga_max_iter=40,
            pga_tolerance=1e-6,
        )
        assert_spherical_cap_pga_result(update.downlink_result)
        np.testing.assert_allclose(update.f_information_k, f_information)
        self.assertGreaterEqual(
            update.true_common_rate,
            resource.common_rate - 2e-7,
        )

        solution = solve_fair_rate_ao(
            problem,
            max_outer_iter=5,
            pga_max_iter=40,
            ao_tolerance=1e-5,
            pga_tolerance=1e-6,
            mu_0=20.0,
            mu_max=40.0,
            mu_growth=2.0,
            lse_tolerance=np.log(problem.num_users) / 40.0,
            stable_rounds_required=1,
        )
        self.assertTrue(np.all(np.diff(solution.common_rate_history) >= -2e-7))
        self.assertTrue(np.all(np.diff(solution.mu_history) >= 0.0))
        self.assertLessEqual(float(np.max(solution.mu_history)), 40.0)

    def test_lse_mu_continuation_is_capped(self):
        problem = build_problem()
        f_energy = default_boresights(problem)
        f_information = optimal_uplink_boresights(problem)
        resource = solve_common_resource(problem, f_energy, f_information)
        update = update_fair_boresights(
            problem,
            f_energy,
            f_information,
            resource.tau_k,
            resource.weighted_covariance,
            mu=30.0,
            mu_max=40.0,
            mu_growth=2.0,
            pga_tolerance=1.0,
        )
        self.assertEqual(update.mu, 40.0)


def build_fik_problem():
    num_antennas = 3
    elements = np.column_stack(
        (
            np.linspace(-0.12, 0.12, num_antennas),
            np.zeros(num_antennas),
            np.zeros(num_antennas),
        )
    )
    devices = np.array([[-1.3, -0.4, 3.0], [1.1, 0.7, 3.5]])
    problem = build_free_space_problem(
        element_positions=elements,
        device_positions=devices,
        alpha_max_energy=np.deg2rad(65.0),
        alpha_max_information=np.deg2rad(65.0),
        zeta=0.6,
        noise_power=1e-9,
        max_power=1.0,
        rho=2.0,
    )
    return problem, elements, devices


class AOIntegrationTests(unittest.TestCase):

    @unittest.skipUnless(
        importlib.util.find_spec("cvxpy") is not None,
        "cvxpy is required for the fair resource block",
    )
    def test_fair_resource_uses_same_fik_without_changing_constraints(self):
        problem, _, _ = build_fik_problem()
        f_information_k = optimal_uplink_boresights(problem)
        resource = solve_common_resource(
            problem,
            default_boresights(problem),
            f_information_k,
        )
        self.assertLessEqual(
            resource.tau_0 + float(np.sum(resource.tau_k)),
            problem.frame_time + 1e-8,
        )
        self.assertLessEqual(
            float(np.trace(resource.covariance).real),
            problem.max_power + 1e-7,
        )
        self.assertAlmostEqual(resource.common_rate, float(np.min(resource.rates)))

    @unittest.skipUnless(
        importlib.util.find_spec("cvxpy") is not None,
        "cvxpy is required for the fair AO branch",
    )
    def test_fair_ao_keeps_fik_closed_form_and_updates_only_fe(self):
        problem, _, _ = build_fik_problem()
        expected = optimal_uplink_boresights(problem)
        solution = solve_fair_rate_ao(
            problem,
            max_outer_iter=2,
            pga_max_iter=4,
            mu_0=20.0,
            mu_max=40.0,
            stable_rounds_required=1,
        )
        np.testing.assert_allclose(solution.f_information_k, expected, atol=1e-13)
        self.assertTrue(np.all(np.diff(solution.common_rate_history) >= -1e-8))
        for update in solution.posture_updates:
            np.testing.assert_allclose(update.f_information_k, expected, atol=1e-13)
            self.assertFalse(hasattr(update, "uplink_result"))

if __name__ == "__main__":
    unittest.main()
