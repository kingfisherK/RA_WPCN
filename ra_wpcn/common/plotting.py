"""IEEE figures and source-data exports for the Rcom experiments."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from ra_wpcn.common import simulation as sim
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "IEEE_Figures"

def _atomic_open(path: Path, *, newline: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline=newline,
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    return handle, Path(handle.name)


def _write_wide_csv(
    path: Path,
    iterations: np.ndarray,
    trajectories: dict[str, np.ndarray],
) -> None:
    schemes = list(trajectories)
    handle, temporary = _atomic_open(path, newline="")
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(["iteration", *schemes])
            for row_index, iteration in enumerate(iterations):
                writer.writerow(
                    [
                        int(iteration),
                        *[
                            f"{float(trajectories[scheme][row_index]):.12g}"
                            for scheme in schemes
                        ],
                    ]
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_figure_atomic(fig, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    handle, temporary = _atomic_open(path, newline="")
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _configure_plotting() -> None:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Times",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.2,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.55,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )


def _configure_paper_plotting() -> None:
    """Use a fixed IEEE canvas and 10-point text for paper figures."""

    _configure_plotting()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
        }
    )


def _save_figure_pair(fig, stem: Path) -> None:
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {})):
        _save_figure_atomic(fig, stem.with_suffix(suffix), **kwargs)


FILE_STEMS = {
    ("power", "common"): ("common_rate_vs_powerpa", "extended_common_rate"),
    ("users", "common"): ("common_rate_vs_usersk", "extended_common_rate"),
    ("antennas", "common"): ("common_rate_vs_antennasn", "extended_common_rate"),
}


def _plot_five_scheme_curve(
    x: np.ndarray,
    means: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    integer_ticks: bool,
    stem: Path,
) -> None:
    styles = (
        ("Proposed", "#d62728", "-", "o"),
        ("ET", "#1f77b4", "--", "s"),
        ("FHAP", "#2ca02c", "-.", "^"),
        ("Random BF", "#ff7f0e", (0, (3, 1, 1, 1)), "x"),
        ("Shared FE/FI", "#9467bd", ":", "D"),
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=600)
    for column, (label, color, linestyle, marker) in enumerate(styles):
        marker_kwargs = {}
        if label == "Shared FE/FI":
            marker_kwargs = {
                "markerfacecolor": "white",
                "markeredgewidth": 0.9,
                "markersize": 4.2,
            }
        ax.plot(
            x,
            means[:, column],
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
            linewidth=1.25 if label == "Proposed" else 1.05,
            zorder=5 if label == "Proposed" else 4 if label == "Shared FE/FI" else 3,
            **marker_kwargs,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    if integer_ticks:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_pad = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.margins(y=0.08)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(
        loc="best",
        ncol=1,
        handlelength=2.6,
        labelspacing=0.35,
        columnspacing=0.9,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout(pad=0.8)
    _save_figure_pair(fig, stem)
    plt.close(fig)


def _curve_data(
    grouped: dict[sim.SimulationCase, list[sim.CommonSnapshotResult]],
    sweep: sim.Sweep,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = []
    for case in sweep.cases:
        rows = grouped[case]
        samples.append(
            np.asarray(
                [
                    [
                        row.proposed,
                        row.equal_time,
                        row.fixed_hap,
                        row.random_bf,
                        row.shared_posture,
                    ]
                    for row in rows
                ],
                dtype=float,
            )
        )
    means = np.asarray([np.mean(values, axis=0) for values in samples])
    standard_deviations = np.asarray(
        [
            np.std(values, axis=0, ddof=1)
            if values.shape[0] > 1
            else np.zeros(values.shape[1])
            for values in samples
        ]
    )
    standard_errors = standard_deviations / np.sqrt(
        np.asarray([values.shape[0] for values in samples], dtype=float)[:, None]
    )
    return (
        np.asarray(sweep.values, dtype=float),
        means,
        standard_deviations,
        standard_errors,
    )


def _qa(curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    expected_directions = {
        "power": 1.0,
        "users": -1.0,
        "antennas": 1.0,
    }
    scheme_names = ("Proposed", "ET", "FHAP", "Random BF", "Shared FE/FI")
    trend_checks = {}
    tier_checks = {}
    for key, (_, means, _, _) in curves.items():
        directed_steps = expected_directions[key] * np.diff(means, axis=0)
        trend_checks[key] = {
            scheme: bool(np.all(directed_steps[:, index] > 0.0))
            for index, scheme in enumerate(scheme_names)
        }
        top_lower = np.minimum(means[:, 0], means[:, 4])
        middle_upper = np.maximum(means[:, 1], means[:, 3])
        middle_lower = np.minimum(means[:, 1], means[:, 3])
        tier_checks[key] = {
            "top_above_middle": bool(np.all(top_lower > middle_upper)),
            "middle_above_fhap": bool(np.all(middle_lower > means[:, 2])),
            "fhap_below_every_other_scheme": bool(
                np.all(means[:, 2, None] < means[:, [0, 1, 3, 4]])
            ),
        }
    return {
        "expected_trends_pass": bool(
            all(all(per_scheme.values()) for per_scheme in trend_checks.values())
        ),
        "trend_checks": trend_checks,
        "three_tier_pass": bool(
            all(
                item["top_above_middle"]
                and item["middle_above_fhap"]
                and item["fhap_below_every_other_scheme"]
                for item in tier_checks.values()
            )
        ),
        "tier_checks": tier_checks,
    }


def export_common_results(
    grouped: dict[sim.SimulationCase, list[sim.CommonSnapshotResult]],
    sweeps: dict[str, sim.Sweep],
    output_dir: Path,
    setup: sim.SimulationSetup,
    config: sim.StudyConfig,
    snapshots: int,
) -> dict[str, object]:
    common_dir = output_dir / "extended_common_rate"
    source_dir = output_dir / "source_data"
    common_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    _configure_paper_plotting()

    curves = {key: _curve_data(grouped, sweep) for key, sweep in sweeps.items()}
    statistics_rows = []
    for key, sweep in sweeps.items():
        x, means, standard_deviations, standard_errors = curves[key]
        filename = FILE_STEMS[(key, "common")][0]
        _atomic_write_csv(
            source_dir / f"{filename}.csv",
            [sweep.x_header, "proposed", "et", "fhap", "random_bf", "shared_fe_fi"],
            [
                [f"{value:.12g}", *[f"{item:.12g}" for item in row]]
                for value, row in zip(x, means)
            ],
        )
        for row_index, value_x in enumerate(x):
            for scheme_index, scheme in enumerate(sim.SCHEME_NAMES):
                statistics_rows.append(
                    [
                        key,
                        f"{value_x:.12g}",
                        scheme,
                        f"{means[row_index, scheme_index]:.12g}",
                        f"{standard_deviations[row_index, scheme_index]:.12g}",
                        f"{standard_errors[row_index, scheme_index]:.12g}",
                        snapshots,
                    ]
                )
        _plot_five_scheme_curve(
            x,
            means,
            x_label=sweep.x_label,
            y_label=r"$R_{\mathrm{com}}$ (bps/Hz)",
            integer_ticks=sweep.integer_ticks,
            stem=common_dir / filename,
        )

    _atomic_write_csv(
        source_dir / "rcom_channel_update_statistics.csv",
        ["sweep", "x", "scheme", "mean", "std", "sem", "n"],
        statistics_rows,
    )
    qa = _qa(curves)
    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/matplotlib",
        "scope": "Rcom versus PA, K, and NA under the cosine-pattern channel update",
        "channel_model": {
            "rho": setup.directional_rho,
            "cosine_exponent": 2.0 * setup.directional_rho,
            "peak_directional_gain": 2.0 * (2.0 * setup.directional_rho + 1.0),
            "external_hap_gain_applied": False,
            "wd_gain_dbi": setup.wd_gain_dbi,
            "rcom_positive_part_smoothing": None,
        },
        "setup": asdict(setup),
        "topology_sha256": sim.TOPOLOGY_SHA256,
        "study_config": asdict(config),
        "fhap_baseline": {
            "definition": "fixed non-optimized azimuth-sector fan",
            "tilt_deg": sim.FHAP_FIXED_TILT_DEG,
            "azimuth_spacing": "uniform over [-pi, pi)",
            "used_as_ra_multistart": False,
        },
        "num_snapshots_used": snapshots,
        "statistics": "Monte Carlo means, sample standard deviations, and SEM.",
        "curve_smoothing_or_postprocessing": False,
        "qa": qa,
    }
    return metadata


def _write_statistics(
    path: Path,
    iterations: np.ndarray,
    statistics: dict[int, dict[str, np.ndarray]],
    antenna_counts: tuple[int, ...],
    num_snapshots: int,
) -> None:
    handle, temporary = _atomic_open(path, newline="")
    try:
        with handle:
            writer = csv.writer(handle)
            writer.writerow(["N_A", "iteration", "mean", "std", "se", "n"])
            for antenna_count in antenna_counts:
                values = statistics[antenna_count]
                for index, iteration in enumerate(iterations):
                    writer.writerow(
                        [
                            antenna_count,
                            int(iteration),
                            f"{float(values['mean'][index]):.12g}",
                            f"{float(values['std'][index]):.12g}",
                            f"{float(values['se'][index]):.12g}",
                            num_snapshots,
                        ]
                    )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plot_convergence(
    iterations: np.ndarray,
    trajectories: dict[int, np.ndarray],
    antenna_counts: tuple[int, ...],
    stem: Path,
) -> None:
    _configure_paper_plotting()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    styles = {
        10: ("#d62728", "o", "-"),
        8: ("#1f77b4", "s", "--"),
        6: ("#2ca02c", "^", "-."),
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.75), dpi=600)
    for antenna_count in antenna_counts:
        color, marker, linestyle = styles.get(
            antenna_count, ("#666666", "D", ":")
        )
        ax.plot(
            iterations,
            trajectories[antenna_count],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=rf"$N_A={antenna_count}$",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$R_{\mathrm{com}}$ (bps/Hz)")
    x_min = float(iterations.min())
    x_max = float(iterations.max())
    x_pad = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    ax.legend(
        loc="lower right",
        ncol=1,
        handlelength=2.0,
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="none",
    )
    fig.tight_layout(pad=0.8)
    # Preserve the declared IEEE single-column canvas.  The shared exporter
    # defaults to a tight bounding box, which otherwise shrinks this figure.
    full_canvas = fig.bbox_inches
    _save_figure_atomic(
        fig, stem.with_suffix(".png"), dpi=600, bbox_inches=full_canvas
    )
    _save_figure_atomic(
        fig, stem.with_suffix(".pdf"), bbox_inches=full_canvas
    )
    plt.close(fig)


def export_convergence_results(
    *,
    output_dir: Path,
    iterations: np.ndarray,
    statistics: dict[int, dict[str, np.ndarray]],
    metadata: dict[str, object],
    antenna_counts: tuple[int, ...],
) -> None:
    """Export actual Proposed AO histories independently of performance CSVs."""
    if not metadata["qa"]["all_snapshot_histories_monotone"]:
        raise RuntimeError("A Proposed snapshot history is nonmonotone.")
    source_dir = output_dir.resolve() / "source_data"
    trajectories = {count: statistics[count]["mean"] for count in antenna_counts}
    _write_wide_csv(
        source_dir / "fair_convergence_vs_iteration.csv", iterations,
        {f"N_A={count}": trajectories[count] for count in antenna_counts},
    )
    _write_statistics(
        source_dir / "fair_convergence_statistics.csv", iterations, statistics,
        antenna_counts, metadata["num_snapshots_used"],
    )
    _plot_convergence(
        iterations, trajectories, antenna_counts,
        output_dir.resolve() / "convergence" / "fair_convergence_vs_iteration",
    )
