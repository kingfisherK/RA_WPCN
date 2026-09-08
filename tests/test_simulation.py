"""Fixed topologies, five-scheme performance, and actual AO convergence."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from unittest.mock import patch
import cvxpy as cp

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ra_wpcn.common.simulation import (
    SimulationCase,
    SimulationSetup,
    _build_problem,
    _fixed_sector_fhap_boresights,
    _fixed_hap_posture,
    build_sweeps,
    topology_bank,
)
from ra_wpcn.solvers.angle_optimization import directional_gain_and_derivatives
from ra_wpcn.common import simulation as sim
from scripts import run as runner
from ra_wpcn.common import plotting
from ra_wpcn.common.simulation import StudyConfig, fixed_beam_common_allocation, random_beam_covariances


class PerformanceCurveSetupTests(unittest.TestCase):
    def test_sector_fhap_is_fixed_feasible_and_tilted(self):
        setup = SimulationSetup()
        case = SimulationCase(
            setup.default_path_loss_exponent,
            setup.default_power_dbm,
            setup.default_num_users,
            setup.default_num_antennas,
        )
        problem = _build_problem(topology_bank(setup)[0, : case.num_users], case, setup)
        boresights = _fixed_sector_fhap_boresights(problem)
        self.assertEqual(boresights.shape, (3, problem.num_antennas))
        np.testing.assert_allclose(np.linalg.norm(boresights, axis=0), 1.0)
        np.testing.assert_allclose(
            boresights[2], np.cos(np.deg2rad(20.0)), atol=1e-14
        )
        self.assertTrue(np.any(np.abs(boresights[:2]) > 0.0))

    def test_topology_bank_matches_original_fingerprint(self):
        bank = topology_bank(SimulationSetup())
        self.assertEqual(bank.shape, (100, 10, 2))
        self.assertEqual(
            hashlib.sha256(bank.tobytes()).hexdigest(),
            "35f702ce29d35473c7baed29ec85c37eb6e5267095600c91c6da93494ba8f1ea",
        )

    def test_three_sweeps_reduce_to_thirteen_unique_cases(self):
        sweeps = build_sweeps(SimulationSetup())
        self.assertEqual(set(sweeps), {"power", "users", "antennas"})
        self.assertTrue(all(len(sweep.values) == 5 for sweep in sweeps.values()))
        unique_cases = {case for sweep in sweeps.values() for case in sweep.cases}
        self.assertEqual(len(unique_cases), 13)

    def test_positive_z_fhap_has_positive_front_side_gain(self):
        setup = SimulationSetup()
        case = SimulationCase(
            setup.default_path_loss_exponent,
            setup.default_power_dbm,
            setup.default_num_users,
            setup.default_num_antennas,
        )
        positions = topology_bank(setup)[0, : case.num_users]
        problem = _build_problem(positions, case, setup)
        gains, _, _ = directional_gain_and_derivatives(
            _fixed_hap_posture(problem),
            problem.directions,
            setup.directional_rho,
            smoothing=None,
        )
        self.assertTrue(np.all(problem.directions[:, :, 2] > 0.0))
        self.assertTrue(np.all(gains > 0.0))


class FixedBeamAllocationTests(unittest.TestCase):
    def test_lambert_allocation_matches_conic_reference(self):
        gamma = np.asarray([2.0, 5.0, 10.0])
        result = fixed_beam_common_allocation(gamma, 1.0)

        tau_0 = cp.Variable(nonneg=True)
        tau_k = cp.Variable(gamma.size, nonneg=True)
        common_rate = cp.Variable(nonneg=True)
        constraints = [tau_0 + cp.sum(tau_k) <= 1.0]
        for index, coefficient in enumerate(gamma):
            constraints.append(
                -cp.rel_entr(
                    tau_k[index],
                    tau_k[index] + coefficient * tau_0,
                )
                >= common_rate * np.log(2.0)
            )
        problem = cp.Problem(cp.Maximize(common_rate), constraints)
        problem.solve(solver="CLARABEL")

        self.assertAlmostEqual(result.common_rate, common_rate.value, places=7)
        self.assertAlmostEqual(
            result.tau_0 + float(np.sum(result.tau_k)), 1.0, places=7
        )


class RandomBeamCodebookTests(unittest.TestCase):
    def test_codebook_is_reproducible_rank_one_and_full_power(self):
        setup = sim.SimulationSetup()
        case = sim.SimulationCase(
            setup.default_path_loss_exponent,
            setup.default_power_dbm,
            setup.default_num_users,
            setup.default_num_antennas,
        )
        positions = sim.topology_bank(setup)[0, : case.num_users]
        problem = sim._build_problem(positions, case, setup)
        config = StudyConfig(random_codebook_size=4)
        first = random_beam_covariances(problem, setup, 0, config)
        second = random_beam_covariances(problem, setup, 0, config)

        self.assertEqual(len(first), 4)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertAlmostEqual(np.trace(left).real, problem.max_power)
            eigenvalues = np.linalg.eigvalsh(left)
            self.assertLessEqual(np.count_nonzero(eigenvalues > 1e-10), 1)


class ExportedFiveSchemeCurveTests(unittest.TestCase):
    def test_proposed_solver_failure_retains_feasible_et_fallback(self):
        setup = sim.SimulationSetup()
        case = sim.SimulationCase(2.5, 20.0, 2, 4)
        config = StudyConfig(random_codebook_size=2)
        with patch.object(sim, "_solve_proposed_common", side_effect=RuntimeError("solver failed")):
            result = runner._simulate_common_task((case, 0, setup, config))
        self.assertTrue(np.isfinite(result.proposed))
        self.assertGreaterEqual(result.proposed, result.equal_time)

    def test_three_rcom_tables_preserve_hierarchy_and_direction(self):
        source_dir = CODE_ROOT / "IEEE_Figures" / "source_data"
        files_and_directions = {
            "common_rate_vs_powerpa.csv": 1.0,
            "common_rate_vs_usersk.csv": -1.0,
            "common_rate_vs_antennasn.csv": 1.0,
        }
        for filename, direction in files_and_directions.items():
            with self.subTest(filename=filename):
                table = np.loadtxt(
                    source_dir / filename, delimiter=",", skiprows=1
                )
                self.assertEqual(table.shape, (5, 6))
                values = table[:, 1:]
                self.assertTrue(np.all(values[:, 0] > values[:, 1]))
                self.assertTrue(np.all(values[:, 1] > values[:, 2]))
                self.assertTrue(np.all(values[:, 0] >= values[:, 3]))
                self.assertTrue(np.all(values[:, 0] >= values[:, 4]))
                self.assertTrue(
                    np.all(direction * np.diff(values, axis=0) >= -1e-8)
                )


class FixedTopologyTests(unittest.TestCase):
    def tearDown(self):
        sim._load_topology_bank.cache_clear()

    def test_stored_bank_is_identical_for_all_antenna_counts_and_read_only(self):
        stored = np.load(sim.TOPOLOGY_PATH, allow_pickle=False)
        for count in (6, 8, 10):
            bank = topology_bank(replace(SimulationSetup(), default_num_antennas=count))
            np.testing.assert_array_equal(bank, stored)
            with self.assertRaises(ValueError):
                bank[0, 0, 0] = 0.0
            with self.assertRaises(ValueError):
                bank.setflags(write=True)

    def test_missing_data_is_reported_without_regeneration(self):
        sim._load_topology_bank.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.npy"
            with patch.object(sim, "TOPOLOGY_PATH", missing):
                with self.assertRaises(FileNotFoundError):
                    topology_bank(SimulationSetup())
                self.assertFalse(missing.exists())

    def test_modified_coordinates_fail_fingerprint_validation(self):
        altered = np.load(sim.TOPOLOGY_PATH, allow_pickle=False)
        altered[0, 0, 0] += 1e-9
        sim._load_topology_bank.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "altered.npy"
            np.save(path, altered, allow_pickle=False)
            with patch.object(sim, "TOPOLOGY_PATH", path):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    topology_bank(SimulationSetup())

    def test_changed_topology_parameters_cannot_silently_select_new_users(self):
        with self.assertRaisesRegex(ValueError, "fixed 100-snapshot bank"):
            topology_bank(replace(SimulationSetup(), topology_seed=2027))


class CommonConvergenceTests(unittest.TestCase):
    def test_preserves_initial_value_and_accepted_updates(self):
        updates = np.array([1.0, 1.5, 1.5 - 1e-7, 2.0])
        history = runner._align_ao_history(updates, 6)
        np.testing.assert_array_equal(history[:4], updates)
        np.testing.assert_array_equal(history[4:], [2.0, 2.0, 2.0])

    def test_rejects_empty_history(self):
        with self.assertRaisesRegex(RuntimeError, "empty convergence history"):
            runner._align_ao_history([], 5)

    def test_rejects_resampling_and_nonmonotone_history(self):
        with self.assertRaisesRegex(ValueError, "iteration budget"):
            runner._align_ao_history([0.1, 0.2, 0.3, 0.4], 2)
        with self.assertRaisesRegex(RuntimeError, "nonmonotone"):
            runner._align_ao_history([1.0, 1.5, 1.4, 2.0], 4)

    def test_convergence_uses_proposed_history_without_baseline_endpoints(self):
        setup = SimulationSetup()
        f = np.tile(np.array([[0.0], [0.0], [1.0]]), (1, setup.default_num_antennas))
        proposed = SimpleNamespace(
            common_rate_history=np.array([0.4, 0.7, 0.8]),
            common_throughput=0.8, converged=True,
        )
        with (
            patch.object(sim, "_solve_equal_time_boresight_ao", return_value=(0.2, f, f, None)),
            patch.object(sim, "_solve_proposed_common", return_value=proposed) as solver,
            patch.object(sim, "solve_shared_boresight_common", side_effect=AssertionError("Shared called")),
            patch.object(sim, "random_beam_covariances", side_effect=AssertionError("Random BF called")),
        ):
            result = runner._simulate_proposed_one((0, setup, 20.0, 4))
        self.assertEqual(solver.call_args.kwargs["max_outer_iter"], 4)
        self.assertEqual(result.iterations_completed, 2)
        np.testing.assert_array_equal(result.history, [0.4, 0.7, 0.8, 0.8, 0.8])

    def test_exported_convergence_preserves_rates_without_performance_csv(self):
        iterations = np.arange(3)
        counts = (6, 8, 10)
        statistics = {
            count: {"mean": np.array([0.1, 0.2, 0.3]) * count,
                    "std": np.zeros(3), "se": np.zeros(3)}
            for count in counts
        }
        metadata = {"num_snapshots_used": 1,
                    "qa": {"all_snapshot_histories_monotone": True}}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch.object(plotting, "_plot_convergence"):
                plotting.export_convergence_results(
                    output_dir=destination, iterations=iterations, statistics=statistics,
                    metadata=metadata, antenna_counts=counts,
                )
            table = np.loadtxt(destination / "source_data/fair_convergence_vs_iteration.csv",
                               delimiter=",", skiprows=1)
            np.testing.assert_array_equal(table[:, 0], iterations)
            for column, count in enumerate(counts, start=1):
                np.testing.assert_allclose(table[:, column], statistics[count]["mean"], rtol=1e-12)
            self.assertFalse((destination / "source_data/common_rate_vs_antennasn.csv").exists())


if __name__ == "__main__":
    unittest.main()
