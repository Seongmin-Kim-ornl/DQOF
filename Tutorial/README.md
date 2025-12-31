# DQOF Tutorial
Distributed Quantum Optimization Framework (DQOF) for large-scale Higher-Order Unconstrained Binary Optimization (HUBO)

---

## Overview
This tutorial demonstrates how to run **DQOF** on large-scale **higher-order binary optimization (HUBO)** problems using a distributed quantum optimization workflow.  
All required scripts, submission files, and example HUBO instances are provided in the `Tutorial/` folder.

The workflow is designed for **HPC environments** and leverages multi-core or multi-node execution via SLURM.

---

## Directory Structure
```text
Tutorial/
├── DQOF.py                 # Core DQOF solver
├── DQOF_run.py             # Driver script for running DQOF
├── AL_DQOF_run.py          # Active Learning (AL) + DQOF workflow
├── submit_DQOF.sl          # SLURM submission script for DQOF
├── submit_AL_DQOF.sl       # SLURM submission script for AL-DQOF
├── HUBOs/                  # Example HUBO problem instances
├── ML/                     # Surrogate models (e.g., Factorization Machine)
├── sim/                    # Numerical simulation tools (e.g., TMM)
└── README.md               # This tutorial
