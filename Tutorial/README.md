# DQOF Tutorial: Distributed Quantum Optimization for Higher-Order Problems

This tutorial provides a minimal, self-contained workflow for running the **Distributed Quantum Optimization Framework (DQOF)** and its **active learning (AL-DQOF)** extension on higher-order unconstrained binary optimization (HUBO) problems.

All required scripts, example HUBO instances, datasets, and SLURM submission files are included in this folder.

---

## Directory Structure

```text
Tutorial/
├── 1_dataset/              # Example datasets for active learning
├── ML/                     # Machine learning models (e.g., 3rd-order FM)
├── sim/                    # Numerical solvers (e.g., TMM)
├── utils/                  # Utility functions
│
├── 40_QUBO.txt              # Example QUBO/HUBO instance
├── 40_k_third.txt           # Example 3rd-order interaction file
│
├── DQOF.py                  # Core DQOF implementation
├── DQOF_run.py              # DQOF execution script (entry point)
├── AL_DQOF_run.py           # Active-learning with DQOF execution script
│
├── submit_DQOF.sl           # SLURM submission script for DQOF
├── submit_AL_DQOF.sl        # SLURM submission script for AL-DQOF
│
├── Check_num_cores.ipynb    # Notebook to estimate required CPU cores
└── README.md                # This file
