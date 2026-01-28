# RSSDA: Recursive Small-Step Semi-Decentralized A*

A solver for Semi-Decentralized POMDPs (SDec-POMDPs), generalizing the Dec-POMDP/MPOMDP framework with synchronization triggers that enable agents to share information where possible and/or desired.

## Overview

RSSDA solves multi-agent planning problems where:
- Agents have **partial observability** of the environment
- Communication is **intermittent** and conditioned on states, joint actions, or joint observations, rather than always-on or never-available
- **Synchronization triggers** define when agents can share observations

This approach bridges fully centralized (always communicate) and fully decentralized (never communicate) planning in multiagent systems.

## Installation

```bash
git clone https://github.com/yourusername/RSSDA.git
cd RSSDA
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, numpy, numba, psutil, pandas

## Quick Start

```bash
# Run Box Pushing domain (horizon 5)
python benchmarks/sdec_box.py 5

# Run Tiger domain (horizon 4)
python benchmarks/sdec_tiger.py 4

# Run Fire Fighting domain (horizon 6)
python benchmarks/sdec_fireFight3houses.py 6
```

## Benchmark Domains

| Domain | Description | Agents | States |
|--------|-------------|--------|--------|
| **Box Pushing** | Coordinate to push boxes on a grid | 2 | 100 |
| **Tiger** | Classic door-opening coordination problem | 2 | 2 |
| **Fire Fighting** | Extinguish fires across three houses | 2 | 432 |
| **Mars Rovers** | Coordinate rock sampling and data transmission | 2 | 256 |
| **Maritime MEDEVAC** | Helicopter-ship patient rescue coordination | 2 | 512 |
| **Labyrinth** | Graph search with configurable sync triggers; tractable for approximate solvers | 2 | Variable |

## Configuration

Each benchmark supports different execution modes via `TRIGGER_MODE`:

- `"centralized"` - Agents always synchronized (upper bound)
- `"semi"` - Synchronization at trigger states (default)
- `"decentralized"` - No synchronization (lower bound)
- `"decentralized_RSMAA"` - Baseline RS-MAA* algorithm

Solver parameters can be configured via `RSSDAConfig`, and include:

```python
config = RSSDAConfig(
    algorithm="approximate",    # "exact" or "approximate"
    iter_limit=2000,            # Max policy nodes per stage
    rec_limit=1,                # Recursion depth
    heuristic_type="HYBRID",    # "QMDP", "HYBRID", or "POMDP"
    TI1=True,                   # [approximation] Interleaved planning and search
    TI2=True,                   # [approximation] Pruning
    TI3=True,                   # [approximation] Tail heuristics
    TI4=True,                   # [approximation] Lossy clustering via sliding windows
)
```

## Project Structure

```
RSSDA/
├── RSSDA.py                 # Core SD-POMDP solver
├── requirements.txt
├── baselines/
│   ├── decPOMDP.py          # RS-MAA* baseline (fully decentralized)
│   └── parser_decPOMDP.py
├── benchmarks/
│   ├── sdec_box.py
│   ├── sdec_tiger.py
│   ├── sdec_mars.py
│   ├── sdec_fireFight3houses.py
│   ├── sdec_labyrinth.py
│   ├── maritimemedevac.py
│   └── *.data               # Domain specification files
├── labyrinth_benchmarks/    # Labyrinth domain data files
└── DARPA_SubT_sites_graphics/    # DARPA subterannean challenge graphs
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{alhusseini2025,
  title={A Semi-Decentralized Approach to Multiagent Control},
  author={Al-Husseini, Mahdi and Wray, Kyle H and Kochenderfer, Mykel J},
  booktitle={2026 International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year={2026},
  organization={}
}
```

## License

MIT License - see individual source files for details.
