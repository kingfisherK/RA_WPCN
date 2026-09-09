# RA-WPCN: ROTATABLE ANTENNAS ENABLED WIRELESS POWERED COMMUNICATION NETWORKS: JOINT OPTIMIZATION OF ANTENNA ORIENTATION AND RESOURCE ALLOCATION

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the simulation implementation and a fixed user-topology
dataset for the paper:

> **ROTATABLE ANTENNAS ENABLED WIRELESS POWERED COMMUNICATION NETWORKS: JOINT OPTIMIZATION OF ANTENNA ORIENTATION AND RESOURCE ALLOCATION**  
> Yiqing Li, Xingshuo Mao, Miao Jiang, and Guangchi Zhang

## Overview

RA-WPCN is a joint antenna orientation, energy beamforming, and time allocation
design for common-throughput
maximization in a RA-enabled wireless powered communication
network. Each outer iteration optimizes the downlink antenna orientations,
energy beamforming, and time allocation, while the dedicated uplink orientations
and MRC combiners follow closed-form solutions.

The release also contains ET, FHAP, Random BF, and Shared FE/FI as reference
implementations.

## Methods

| Component | Description | Location |
| --- | --- | --- |
| Proposed | Closed-form dedicated uplink orientations and AO with spherical-cap PGA for downlink orientations | [`ra_wpcn/solvers/ao.py`](ra_wpcn/solvers/ao.py) |
| Resource allocation | Semidefinite and exponential-cone common-throughput optimization of energy covariance and time | [`ra_wpcn/solvers/multiple_device.py`](ra_wpcn/solvers/multiple_device.py) |
| Single-device solution | Closed-form antenna orientations and time allocation, with array-gain analysis | [`ra_wpcn/models/single_device.py`](ra_wpcn/models/single_device.py) |
| Reference schemes | Equal time (ET), fixed 20-degree sector orientations (FHAP), random rank-one beams (Random BF), and shared uplink/downlink orientations (Shared FE/FI) | [`ra_wpcn/common/simulation.py`](ra_wpcn/common/simulation.py) |


## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   └── user_topologies_100.npy   # Fixed input for 100 simulation snapshots
├── ra_wpcn/
│   ├── __init__.py              # Public model and solver API
│   ├── common/
│   │   ├── __init__.py
│   │   ├── simulation.py        # Topology loading, experiment setup, and baselines
│   │   └── plotting.py          # IEEE figures, CSV exports, and statistics
│   ├── models/
│   │   ├── __init__.py
│   │   ├── system_model.py      # Channels, energy harvesting, MRC, and throughput
│   │   └── single_device.py     # Single-device closed-form solution and gain analysis
│   └── solvers/
│       ├── __init__.py
│       ├── angle_optimization.py # Spherical-cap PGA and gradient validation
│       ├── multiple_device.py   # Resource allocation and downlink orientation updates
│       └── ao.py                # Alternating optimization and initialization
├── scripts/
│   ├── __init__.py
│   └── run.py                  # Performance curves, convergence, and demo entry point
├── tests/
│   ├── test_model.py
│   ├── test_optimization.py
│   └── test_simulation.py
├── IEEE_Figures/
│   ├── convergence/            # Convergence figures (PDF/PNG)
│   ├── extended_common_rate/   # Performance figures (PDF/PNG)
│   └── source_data/            # Curve data and statistics (CSV)

```

## Requirements

The release has been validated with Python 3.10.11, NumPy 2.2.6, SciPy 1.15.3,
CVXPY 1.7.5, Matplotlib 3.10.5, and SciencePlots 2.2.1. Install the runtime
dependencies within the version bounds in `requirements.txt` with:

```bash
python -m pip install -r requirements.txt
```

The resource-allocation solver uses CVXPY with CLARABEL or SCS.

## Quick start

Run the dedicated-uplink and MRC example from the repository root
(`codeRESTRUCTURE`):

```bash
python -B scripts/run.py demo
```

The script constructs one deterministic three-user problem and reports the
uplink-orientation tensor, MRC beam count, common throughput, and time allocation.

Run the tests and reproduce the performance and convergence experiments with:

```bash
python -B -m unittest discover -s tests -v
python -B scripts/run.py curves --snapshots 100 --workers 8
python -B scripts/run.py convergence --snapshots 100 --workers 8
```

For a smoke run, use a separate output directory:

```bash
python -B scripts/run.py all --snapshots 1 --workers 2 --output-dir smoke_outputs
```

Use `python -B scripts/run.py all --help` for the experiment options.
The scripts read `data/user_topologies_100.npy` and write PDF/PNG figures and CSV
tables to `IEEE_Figures` by default, with both paths resolved from the repository
location. The bundled convergence files predate the current AO-history recorder;
rerun `convergence` to update them.


## License

This project is released under the [MIT License](LICENSE).


