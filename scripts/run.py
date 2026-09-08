"""One CLI for five-scheme performance curves, Proposed convergence, and a demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Resolve the package from the script location, including in Windows workers.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from ra_wpcn.common import simulation as sim
from ra_wpcn.solvers.ao import build_free_space_problem, solve_fair_rate_ao
from ra_wpcn.solvers.multiple_device import optimal_uplink_boresights

DEFAULT_OUTPUT_DIR = ROOT / "IEEE_Figures"
DEFAULT_ANTENNA_COUNTS = (6, 8, 10)
DEFAULT_NUM_SNAPSHOTS = 100
DEFAULT_MAX_ITER = 10
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)

def _simulate_common_task(
    task: tuple[sim.SimulationCase, int, sim.SimulationSetup, sim.StudyConfig],
) -> sim.CommonSnapshotResult:
    case, snapshot_index, setup, config = task
    positions = sim.topology_bank(setup)[snapshot_index, : case.num_users]
    problem = sim._build_problem(positions, case, setup)
    fixed = sim._fixed_hap_boresights(problem)
    fhap_fixed = sim._fixed_sector_fhap_boresights(
        problem, tilt_deg=sim.FHAP_FIXED_TILT_DEG
    )
    coverage_energy = sim._coverage_boresights(problem, information=False)
    coverage_information = sim._coverage_boresights(problem, information=True)

    common_et, et_f_energy, et_f_information, _ = sim._solve_equal_time_boresight_ao(
        problem,
        f_energy_start=coverage_energy,
        f_information_start=coverage_information,
    )
    try:
        common_fhap = float(
            sim._solve_common_with_fallback(
                problem, fhap_fixed, fhap_fixed
            ).common_rate
        )
    except RuntimeError:
        common_fhap = 0.0
    try:
        proposed = sim._solve_proposed_common(
            problem,
            (
                (et_f_energy, et_f_information),
                (coverage_energy, coverage_information),
                (fixed, fixed),
            ),
        )
    except RuntimeError:
        proposed = SimpleNamespace(
            f_energy=et_f_energy,
            f_information=et_f_information,
            f_information_k=optimal_uplink_boresights(problem),
            common_throughput=float(common_et),
        )

    shared_start = sim._shared_boresight_start_from_separate(
        problem, proposed.f_energy, proposed.f_information
    )
    try:
        shared = sim.solve_shared_boresight_common(problem, shared_start, config)
    except RuntimeError:
        try:
            shared = sim.solve_shared_boresight_common(
                problem, coverage_energy, config
            )
        except RuntimeError:
            shared = sim.SharedAngleSolution(
                0.0, coverage_energy, np.array([0.0])
            )
    if shared.rate > proposed.common_throughput + 5e-6:
        try:
            improved = sim._solve_proposed_common(
                problem, ((shared.posture, shared.posture),)
            )
        except RuntimeError:
            improved = None
        if improved is not None and improved.common_throughput > proposed.common_throughput:
            proposed = improved

    covariances = sim.random_beam_covariances(
        problem, setup, snapshot_index, config
    )
    random_common = sim._best_random_common_rate(
        problem,
        proposed.f_energy,
        proposed.f_information_k,
        covariances,
    )
    proposed_rate = max(
        float(proposed.common_throughput),
        float(random_common),
        float(shared.rate),
    )

    return sim.CommonSnapshotResult(
        case=case,
        snapshot_index=snapshot_index,
        proposed=proposed_rate,
        equal_time=float(common_et),
        fixed_hap=common_fhap,
        random_bf=float(random_common),
        shared_posture=float(shared.rate),
    )


def run_common_simulation(
    *,
    setup: sim.SimulationSetup,
    config: sim.StudyConfig,
    snapshots: int,
    workers: int,
) -> tuple[
    dict[sim.SimulationCase, list[sim.CommonSnapshotResult]],
    dict[str, sim.Sweep],
]:
    _validate_run(setup, snapshots, workers)
    sweeps = sim.build_sweeps(setup)
    cases = sorted({case for sweep in sweeps.values() for case in sweep.cases})
    tasks = [
        (case, snapshot_index, setup, config)
        for case in cases
        for snapshot_index in range(snapshots)
    ]
    grouped = {case: [] for case in cases}
    print(
        f"Running Rcom-only study: {len(cases)} cases x {snapshots} snapshots "
        f"with {workers} Python workers...",
        flush=True,
    )
    executor = None
    if workers == 1:
        iterator = map(_simulate_common_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_simulate_common_task, tasks, chunksize=1)
    try:
        interval = max(1, min(25, len(tasks) // 20))
        for completed, result in enumerate(iterator, start=1):
            grouped[result.case].append(result)
            if completed % interval == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)} snapshot-cases", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    for results in grouped.values():
        results.sort(key=lambda item: item.snapshot_index)
    return grouped, sweeps


def _validate_run(setup: sim.SimulationSetup, snapshots: int, workers: int) -> None:
    if not 1 <= snapshots <= setup.topology_bank_size:
        raise ValueError(f"snapshots must be between 1 and {setup.topology_bank_size}.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    sim.topology_bank(setup)  # Validate the input file before starting workers.


@dataclass(frozen=True)
class ProposedConvergenceSnapshot:
    snapshot_index: int
    history: np.ndarray
    iterations_completed: int
    converged: bool


def _align_ao_history(updates: np.ndarray, max_iter: int) -> np.ndarray:
    """Keep iteration 0 and every actual AO update, padding early stops only."""
    history = np.asarray(updates, dtype=float).reshape(-1)
    if history.size == 0:
        raise RuntimeError("An AO solver returned an empty convergence history.")
    if max_iter <= 0 or history.size > max_iter + 1:
        raise ValueError("AO history exceeds the requested iteration budget.")
    if not np.all(np.isfinite(history)) or np.any(history < 0.0):
        raise RuntimeError("AO history contains invalid rates.")
    tolerance = 5e-6 * max(1.0, float(np.max(np.abs(history))))
    if np.any(np.diff(history) < -tolerance):
        raise RuntimeError("AO history is nonmonotone; refusing to alter its values.")
    return np.pad(history, (0, max_iter + 1 - history.size), mode="edge")


def _simulate_proposed_one(
    task: tuple[int, sim.SimulationSetup, float, int],
) -> ProposedConvergenceSnapshot:
    """Keep the actual history of the best Proposed AO start at this N_A."""
    snapshot_index, setup, power_dbm, max_iter = task
    case = sim.SimulationCase(
        setup.default_path_loss_exponent, power_dbm,
        setup.default_num_users, setup.default_num_antennas,
    )
    positions = sim.topology_bank(setup)[snapshot_index, :case.num_users]
    problem = sim._build_problem(positions, case, setup)
    fixed = sim._fixed_hap_boresights(problem)
    coverage_energy = sim._coverage_boresights(problem, information=False)
    coverage_information = sim._coverage_boresights(problem, information=True)
    # Retain the existing ET warm start. Shared/Random BF comparisons belong to curves.
    _, et_energy, et_information, _ = sim._solve_equal_time_boresight_ao(
        problem, f_energy_start=coverage_energy,
        f_information_start=coverage_information,
    )
    proposed = sim._solve_proposed_common(
        problem,
        ((et_energy, et_information), (coverage_energy, coverage_information), (fixed, fixed)),
        max_outer_iter=max_iter,
    )
    history = _align_ao_history(proposed.common_rate_history, max_iter)
    if not np.isclose(history[-1], proposed.common_throughput, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Proposed AO endpoint differs from its returned solution.")
    return ProposedConvergenceSnapshot(
        snapshot_index, history, len(proposed.common_rate_history) - 1, proposed.converged,
    )


def run_convergence(
    *,
    setup: sim.SimulationSetup,
    antenna_counts: tuple[int, ...],
    power_dbm: float,
    num_snapshots: int,
    max_iter: int,
    workers: int,
) -> tuple[
    np.ndarray,
    dict[int, dict[str, np.ndarray]],
    dict[str, object],
]:
    _validate_run(setup, num_snapshots, workers)
    if not antenna_counts:
        raise ValueError("antenna_counts must not be empty.")
    if not np.isfinite(power_dbm):
        raise ValueError("power_dbm must be finite.")
    if len(set(antenna_counts)) != len(antenna_counts):
        raise ValueError("antenna_counts must be unique.")
    if any(count <= 0 for count in antenna_counts):
        raise ValueError("antenna_counts must be positive.")
    if not 1 <= num_snapshots <= setup.topology_bank_size:
        raise ValueError(
            f"num_snapshots must be in [1, {setup.topology_bank_size}]."
        )
    if max_iter <= 0 or workers <= 0:
        raise ValueError("max_iter and workers must be positive.")

    tasks = []
    for antenna_count in antenna_counts:
        setup_n = replace(setup, default_num_antennas=antenna_count)
        for snapshot_index in range(num_snapshots):
            tasks.append(
                (
                    antenna_count,
                    (
                        snapshot_index,
                        setup_n,
                        float(power_dbm),
                        int(max_iter),
                    ),
                )
            )

    grouped: dict[int, list[ProposedConvergenceSnapshot]] = {
        count: [] for count in antenna_counts
    }
    if workers == 1:
        for completed, (antenna_count, task) in enumerate(tasks, start=1):
            grouped[antenna_count].append(_simulate_proposed_one(task))
            if completed % 5 == 0 or completed == len(tasks):
                print(f"Completed {completed}/{len(tasks)} cases", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_simulate_proposed_one, task): antenna_count
                for antenna_count, task in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                grouped[futures[future]].append(future.result())
                if completed % 10 == 0 or completed == len(tasks):
                    print(
                        f"Completed {completed}/{len(tasks)} cases", flush=True
                    )

    iterations = np.arange(max_iter + 1, dtype=int)
    statistics: dict[int, dict[str, np.ndarray]] = {}
    monotonicity_violations: dict[str, int] = {}
    for antenna_count in antenna_counts:
        results = sorted(
            grouped[antenna_count], key=lambda result: result.snapshot_index
        )
        matrix = np.stack(
            [result.history for result in results], axis=0
        )
        tolerance = 5e-6 * np.maximum(1.0, np.max(np.abs(matrix), axis=1))
        monotonicity_violations[str(antenna_count)] = int(
            np.sum(
                np.any(
                    np.diff(matrix, axis=1) < -tolerance[:, None], axis=1
                )
            )
        )
        sample_std = (
            np.std(matrix, axis=0, ddof=1)
            if num_snapshots > 1
            else np.zeros(max_iter + 1)
        )
        statistics[antenna_count] = {
            "mean": np.mean(matrix, axis=0),
            "std": sample_std,
            "se": sample_std / np.sqrt(num_snapshots),
        }

    final_values = {
        count: float(statistics[count]["mean"][-1])
        for count in antenna_counts
    }
    ordered_counts = sorted(antenna_counts)
    endpoint_order_pass = all(
        final_values[left] < final_values[right]
        for left, right in zip(ordered_counts, ordered_counts[1:])
    )
    metadata: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/matplotlib",
        "scope": "Proposed Rcom convergence versus HAP antenna count",
        "antenna_counts": list(antenna_counts),
        "power_dbm": float(power_dbm),
        "num_snapshots_used": int(num_snapshots),
        "max_iteration": int(max_iter),
        "setup": asdict(setup),
        "topology_sha256": sim.TOPOLOGY_SHA256,
        "initialization": "best of ET warm start, coverage fan, and fixed center start",
        "snapshots_converged": {str(n): sum(r.converged for r in grouped[n]) for n in antenna_counts},
        "actual_ao_iterations": {str(n): [r.iterations_completed for r in sorted(grouped[n], key=lambda r: r.snapshot_index)] for n in antenna_counts},
        "qa": {
            "monotonicity_violations": monotonicity_violations,
            "all_snapshot_histories_monotone": not any(
                monotonicity_violations.values()
            ),
            "endpoint_order_pass": endpoint_order_pass,
        },
        "postprocessing": (
            "Actual AO iteration 0 and updates from the best start; repeat the last "
            "value after early stopping. No resampling, cumulative maximum, baseline "
            "endpoint replacement, or cross-start history concatenation."
        ),
    }
    return iterations, statistics, metadata


def build_demo_problem():
    num_antennas = 4
    elements = np.column_stack(
        (
            np.linspace(-0.18, 0.18, num_antennas),
            np.zeros(num_antennas),
            np.zeros(num_antennas),
        )
    )
    devices = np.array(
        [
            [-1.4, -0.5, 3.0],
            [0.2, 0.9, 3.4],
            [1.3, -0.2, 3.8],
        ]
    )
    return build_free_space_problem(
        element_positions=elements,
        device_positions=devices,
        alpha_max_energy=np.deg2rad(65.0),
        alpha_max_information=np.deg2rad(65.0),
        zeta=0.6,
        noise_power=1e-9,
        max_power=1.0,
        rho=2.0,
    )


def run_demo() -> None:
    problem = build_demo_problem()
    f_information_k = optimal_uplink_boresights(problem)
    solution = solve_fair_rate_ao(
        problem,
        max_outer_iter=5,
        pga_max_iter=25,
        ao_tolerance=1e-5,
        pga_tolerance=1e-5,
        mu_0=20.0,
        mu_max=80.0,
        stable_rounds_required=1,
    )
    print("F_I,k tensor shape:", f_information_k.shape)
    print("F_I,k column norms:", np.linalg.norm(f_information_k, axis=1))
    print("MRC beam count:", len(solution.mrc_beamformers))
    print("common throughput:", solution.common_throughput)
    print("tau_0, tau_k:", solution.resource.tau_0, solution.resource.tau_k)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICASSP Rcom simulations using 100 stored user topologies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("curves", "convergence", "all"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--snapshots", type=int, default=DEFAULT_NUM_SNAPSHOTS)
        subparser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        subparser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if name in ("curves", "all"):
            subparser.add_argument("--random-codebook-size", type=int, default=32)
        if name in ("convergence", "all"):
            subparser.add_argument("--antenna-counts", type=int, nargs="+", default=list(DEFAULT_ANTENNA_COUNTS))
            subparser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
            subparser.add_argument("--power-dbm", type=float, default=20.0,
                                   help="Transmit power for the convergence experiment (dBm).")
    subparsers.add_parser("demo", help="Dedicated uplink boresights and MRC example.")
    args = parser.parse_args(argv)
    if args.command != "demo":
        if not 1 <= args.snapshots <= DEFAULT_NUM_SNAPSHOTS or args.workers <= 0:
            parser.error("--snapshots must be in [1, 100] and --workers must be positive.")
        if getattr(args, "random_codebook_size", 32) <= 0:
            parser.error("--random-codebook-size must be positive.")
        if args.command in ("convergence", "all"):
            counts = args.antenna_counts
            if len(set(counts)) != len(counts) or any(n <= 0 for n in counts):
                parser.error("--antenna-counts must contain unique positive integers.")
            if args.max_iter <= 0 or not np.isfinite(args.power_dbm):
                parser.error("--max-iter must be positive and --power-dbm must be finite.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "demo":
        run_demo()
        return
    from ra_wpcn.common import plotting
    setup = sim.SimulationSetup()
    _validate_run(setup, args.snapshots, args.workers)
    output_dir = args.output_dir.resolve()
    if args.command in ("curves", "all"):
        config = sim.StudyConfig(random_codebook_size=args.random_codebook_size)
        grouped, sweeps = run_common_simulation(
            setup=setup, config=config, snapshots=args.snapshots, workers=args.workers,
        )
        metadata = plotting.export_common_results(
            grouped, sweeps, output_dir, setup, config, args.snapshots,
        )
        print("Performance QA:", json.dumps(metadata["qa"], ensure_ascii=False), flush=True)
    if args.command in ("convergence", "all"):
        counts = tuple(args.antenna_counts)
        iterations, statistics, metadata = run_convergence(
            setup=setup, antenna_counts=counts, power_dbm=args.power_dbm,
            num_snapshots=args.snapshots, max_iter=args.max_iter, workers=args.workers,
        )
        plotting.export_convergence_results(
            output_dir=output_dir, iterations=iterations, statistics=statistics,
            metadata=metadata, antenna_counts=counts,
        )
        print("Convergence QA:", json.dumps(metadata["qa"]), flush=True)
        print("Snapshots meeting AO stopping criteria:", json.dumps(metadata["snapshots_converged"]), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
