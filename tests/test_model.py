"""Model and closed-form analytical regression tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ra_wpcn.models.single_device import (
    mrt_covariance,
    optimal_pointing_vectors,
    optimal_single_device_time,
    residual_misalignment_angles,
    solve_single_device_closed_form,
    theoretical_residual_angles,
    ula_infinite_single_link_gain,
    ula_single_link_gain,
)
from ra_wpcn.models.system_model import (
    composite_gains,
    mrc_gains,
    posture_from_angles,
    validate_zenith_cap,
    validate_energy_covariance,
)
from ra_wpcn.solvers.angle_optimization import boresight_channel_state, pga_spherical_cap_block, mrc_boresight_gains_and_gradients
from ra_wpcn.solvers.ao import build_free_space_problem, default_boresights
from ra_wpcn.solvers.multiple_device import optimal_uplink_boresights
from ra_wpcn.models.system_model import (
    REFERENCE_BORESIGHT,
    closed_form_uplink_boresights,
    evaluate_section_ii_model,
    mrc_combiners,
    per_slot_directional_power_gains,
    per_slot_effective_channels,
    received_rf_powers,
)


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


class ReconstructedSectionIITests(unittest.TestCase):
    def test_zero_uplink_channel_has_zero_combiner_and_gain(self):
        channels = np.array([[0.0 + 0.0j, 0.0 + 0.0j], [1.0 + 0.0j, 1.0j]])
        combiners = mrc_combiners(channels)
        np.testing.assert_array_equal(combiners[:, 0], np.zeros(2, dtype=complex))
        np.testing.assert_allclose(np.linalg.norm(combiners[:, 1]), 1.0)
        np.testing.assert_allclose(mrc_gains(channels), np.array([0.0, 2.0]))

    def setUp(self):
        self.elements = np.array(
            [
                [-0.25, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ]
        )
        self.devices = np.array(
            [
                [-0.5, 0.0, 3.0],
                [0.75, 0.5, 4.0],
            ]
        )
        self.propagation_energy = np.array(
            [
                [1.0 + 0.2j, 0.8 - 0.1j],
                [0.6 + 0.4j, 1.1 + 0.3j],
            ]
        )
        self.propagation_information = np.array(
            [
                [0.9 - 0.2j, 1.0 + 0.5j],
                [0.7 + 0.1j, 0.8 - 0.4j],
            ]
        )

    def test_one_shared_uplink_posture_generates_all_per_wd_mrc_vectors(self):
        covariance = np.diag([0.3, 0.2]).astype(complex)
        result = evaluate_section_ii_model(
            element_positions=self.elements,
            device_positions=self.devices,
            zenith_energy=np.array([0.0, 0.0]),
            azimuth_energy=np.zeros(2),
            zenith_information=np.array([0.1, 0.2]),
            azimuth_information=np.array([0.0, 0.4]),
            propagation_energy=self.propagation_energy,
            propagation_information=self.propagation_information,
            covariance_energy=covariance,
            tau_0=0.2,
            tau_k=np.array([0.3, 0.4]),
            zeta=0.5,
            noise_power=1e-3,
            max_power=0.5,
            rho=1.0,
            alpha_max=0.3,
        )

        # q_I has one 2N_A state, not a K-by-2N_A per-user posture.
        self.assertEqual(result.posture_information.q.shape, (4,))
        self.assertEqual(result.posture_information.f.shape, (3, 2))
        self.assertEqual(result.mrc_information.shape, (2, 2))
        np.testing.assert_allclose(
            np.linalg.norm(result.mrc_information, axis=0),
            np.ones(2),
            atol=1e-12,
        )
        for k in range(2):
            achieved = abs(
                np.vdot(
                    result.mrc_information[:, k],
                    result.channel_information[k],
                )
            ) ** 2
            self.assertAlmostEqual(
                achieved,
                float(np.linalg.norm(result.channel_information[k]) ** 2),
                places=12,
            )

    def test_composite_gain_uses_actual_covariance_power_once(self):
        covariance = np.diag([0.3, 0.2]).astype(complex)
        h_e = self.propagation_energy
        h_i = self.propagation_information
        gamma = composite_gains(
            h_e,
            h_i,
            covariance,
            zeta=0.6,
            noise_power=0.2,
        )
        expected_downlink = np.array(
            [np.vdot(h, covariance @ h).real for h in h_e]
        )
        expected = 0.6 * expected_downlink * mrc_gains(h_i) / 0.2
        np.testing.assert_allclose(gamma, expected, rtol=1e-13, atol=1e-13)

        # Scaling S_E scales gamma linearly; there is no external P_A factor.
        np.testing.assert_allclose(
            composite_gains(h_e, h_i, 0.5 * covariance, 0.6, 0.2),
            0.5 * gamma,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_posture_and_covariance_constraints(self):
        posture = posture_from_angles(
            np.array([0.0, 0.2]),
            np.array([0.0, 0.5]),
        )
        validate_zenith_cap(posture, alpha_max=0.25)
        covariance = np.array([[0.2, 0.05j], [-0.05j, 0.3]])
        checked = validate_energy_covariance(covariance, max_power=0.5)
        self.assertAlmostEqual(float(np.trace(checked).real), 0.5)


class ReconstructedSectionIIITests(unittest.TestCase):
    def setUp(self):
        self.elements = np.column_stack(
            (np.linspace(-0.75, 0.75, 4), np.zeros(4), np.zeros(4))
        )
        self.user = np.array([1.5, 0.3, 3.0])

    def test_cone_projection_matches_residual_closed_form(self):
        theta_max = np.deg2rad(20.0)
        pointing = optimal_pointing_vectors(
            self.elements,
            self.user,
            theta_max,
            reference_boresight=REFERENCE_BORESIGHT,
        )
        measured = residual_misalignment_angles(
            self.elements, self.user, pointing
        )
        theoretical = theoretical_residual_angles(
            self.elements,
            self.user,
            theta_max,
            reference_boresight=REFERENCE_BORESIGHT,
        )
        np.testing.assert_allclose(measured, theoretical, atol=2e-8)

    def test_mrt_covariance_and_lambert_time_solution(self):
        channel = np.array([1.0 + 0.5j, 0.2 - 0.1j, -0.3j, 0.8])
        covariance = mrt_covariance(channel, tx_power=2.5)
        self.assertAlmostEqual(float(np.trace(covariance).real), 2.5, places=12)
        self.assertEqual(np.linalg.matrix_rank(covariance, tol=1e-10), 1)

        allocation = optimal_single_device_time(gamma=7.5)
        self.assertAlmostEqual(
            allocation.tau_0 + float(allocation.tau_k[0]), 1.0, places=12
        )
        self.assertAlmostEqual(
            allocation.z * (np.log(allocation.z) - 1.0),
            7.5 - 1.0,
            places=10,
        )

    def test_end_to_end_solution_satisfies_energy_equality(self):
        solution = solve_single_device_closed_form(
            self.elements,
            self.user,
            theta_max=np.deg2rad(45.0),
            zeta=0.55,
            tx_power=0.8,
            noise_power=1e-4,
        )
        tau_1 = float(solution.time_allocation.tau_k[0])
        self.assertGreater(solution.gamma, 0.0)
        self.assertAlmostEqual(
            solution.uplink_power * tau_1,
            solution.harvested_energy,
            places=12,
        )
        self.assertAlmostEqual(
            float(np.trace(solution.covariance_energy).real),
            0.8,
            places=12,
        )

    def test_ula_piecewise_gain_converges_to_finite_limit(self):
        delta = 0.02
        xi = 0.4
        theta_max = np.deg2rad(35.0)
        finite = ula_single_link_gain(1_000_001, delta, xi, theta_max)
        limit = ula_infinite_single_link_gain(delta, xi, theta_max)
        self.assertLess(abs(finite - limit) / limit, 1e-4)


class ClosedFormFIkTests(unittest.TestCase):
    def test_closed_form_is_feasible_distinct_and_elementwise_optimal(self):
        problem, _, _ = build_fik_problem()
        f_information_k = optimal_uplink_boresights(problem)
        self.assertEqual(
            f_information_k.shape,
            (problem.num_users, 3, problem.num_antennas),
        )
        np.testing.assert_allclose(
            np.linalg.norm(f_information_k, axis=1),
            1.0,
            atol=1e-12,
        )
        self.assertTrue(
            np.all(
                np.einsum("d,kdn->kn", REFERENCE_BORESIGHT, f_information_k)
                >= np.cos(problem.cap_half_angle_information) - 1e-12
            )
        )
        self.assertFalse(np.allclose(f_information_k[0], f_information_k[1]))

        optimum = per_slot_directional_power_gains(
            f_information_k,
            problem.directions,
            problem.rho,
        )
        rng = np.random.default_rng(12)
        for _ in range(200):
            cos_tilt = rng.uniform(
                np.cos(problem.cap_half_angle_information),
                1.0,
                size=(problem.num_users, problem.num_antennas),
            )
            sin_tilt = np.sqrt(1.0 - cos_tilt**2)
            azimuth = rng.uniform(
                -np.pi,
                np.pi,
                size=(problem.num_users, problem.num_antennas),
            )
            candidate = np.stack(
                (
                    sin_tilt * np.cos(azimuth),
                    sin_tilt * np.sin(azimuth),
                    cos_tilt,
                ),
                axis=1,
            )
            candidate_gain = per_slot_directional_power_gains(
                candidate,
                problem.directions,
                problem.rho,
            )
            self.assertTrue(np.all(optimum >= candidate_gain - 1e-12))

    def test_shared_posture_shape_is_rejected(self):
        problem, _, _ = build_fik_problem()
        shared = default_boresights(problem)
        with self.assertRaises(ValueError):
            per_slot_directional_power_gains(
                shared,
                problem.directions,
                problem.rho,
            )

    def test_projection_matches_a_target_inside_the_new_cap(self):
        directions = np.array([[[0.1, 0.0, np.sqrt(0.99)]]])
        projected = closed_form_uplink_boresights(directions, np.deg2rad(20.0))
        expected = np.transpose(directions, (0, 2, 1))
        np.testing.assert_allclose(projected, expected, atol=1e-12)

    def test_closed_form_agrees_with_spherical_cap_pga_verification(self):
        problem, _, _ = build_fik_problem()
        closed_form = optimal_uplink_boresights(problem)
        initial = np.tile(REFERENCE_BORESIGHT[:, None], (1, problem.num_antennas))
        for user in range(problem.num_users):
            directions = problem.directions[[user]]
            propagation = problem.propagation_information[[user]]

            def objective(f):
                state = boresight_channel_state(
                    f, directions, propagation, problem.rho
                )
                return float(mrc_boresight_gains_and_gradients(state)[0][0])

            def gradient(f):
                state = boresight_channel_state(
                    f, directions, propagation, problem.rho
                )
                return mrc_boresight_gains_and_gradients(state)[1][0]

            result = pga_spherical_cap_block(
                objective,
                gradient,
                objective,
                initial,
                problem.cap_half_angle_information,
                max_iter=100,
                tolerance=1e-14,
                initial_step=1e3,
                max_step=1e3,
            )
            self.assertAlmostEqual(
                result.true_objective,
                objective(closed_form[user]),
                delta=1e-9 * max(1.0, objective(closed_form[user])),
            )


class BeamformingAndFormulaTests(unittest.TestCase):
    def test_mrc_is_optimal_for_every_dedicated_channel(self):
        problem, _, _ = build_fik_problem()
        f_information_k = optimal_uplink_boresights(problem)
        channels = per_slot_effective_channels(
            problem.propagation_information,
            f_information_k,
            problem.directions,
            problem.rho,
        )
        combiners = mrc_combiners(channels)
        rng = np.random.default_rng(5)
        for index, channel in enumerate(channels):
            mrc_gain = abs(np.vdot(combiners[:, index], channel)) ** 2
            self.assertAlmostEqual(mrc_gain, float(np.vdot(channel, channel).real), places=14)
            for _ in range(100):
                candidate = rng.normal(size=channel.size) + 1j * rng.normal(
                    size=channel.size
                )
                candidate /= np.linalg.norm(candidate)
                self.assertLessEqual(
                    abs(np.vdot(candidate, channel)) ** 2,
                    mrc_gain + 1e-18,
                )

    def test_section_ii_keeps_original_energy_gamma_and_rate_formulas(self):
        problem, elements, devices = build_fik_problem()
        f_information_k = optimal_uplink_boresights(problem)
        f_energy = np.tile(
            REFERENCE_BORESIGHT[:, None], (1, problem.num_antennas)
        )
        covariance = np.eye(problem.num_antennas, dtype=complex) / problem.num_antennas
        tau_0 = 0.3
        tau_k = np.array([0.3, 0.4])
        evaluation = evaluate_section_ii_model(
            element_positions=elements,
            device_positions=devices,
            f_energy=f_energy,
            f_information_k=f_information_k,
            propagation_energy=problem.propagation_energy,
            propagation_information=problem.propagation_information,
            covariance_energy=covariance,
            tau_0=tau_0,
            tau_k=tau_k,
            zeta=problem.zeta,
            noise_power=problem.noise_power,
            max_power=problem.max_power,
            rho=problem.rho,
            alpha_max_energy=problem.cap_half_angle_energy,
            alpha_max_information=problem.cap_half_angle_information,
        )
        downlink = received_rf_powers(evaluation.channel_energy, covariance)
        uplink = np.sum(np.abs(evaluation.channel_information) ** 2, axis=1)
        expected_gamma = problem.zeta * downlink * uplink / problem.noise_power
        expected_rates = tau_k * np.log2(1.0 + tau_0 * expected_gamma / tau_k)
        np.testing.assert_allclose(evaluation.gamma, expected_gamma, rtol=1e-12)
        np.testing.assert_allclose(evaluation.rates, expected_rates, rtol=1e-12)
        np.testing.assert_allclose(evaluation.f_information_k, f_information_k)

if __name__ == "__main__":
    unittest.main()
