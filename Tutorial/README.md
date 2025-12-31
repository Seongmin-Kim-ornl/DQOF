# DQOF Tutorial: Distributed Quantum Optimization for Higher-Order Problems

This tutorial provides a minimal, self-contained workflow for running the **Distributed Quantum Optimization Framework (DQOF)** and its **active learning (AL-DQOF)** extension on higher-order unconstrained binary optimization (HUBO) problems.

All required scripts, example HUBO instances, datasets, and SLURM submission files are included in this folder.

---

## Overview of Main Scripts

### `DQOF.py`
Core implementation of **DQOF** for solving large-scale HUBO problems.
- Decomposes a dense HUBO into multiple sub-HUBOs
- Solves sub-HUBOs in parallel using quantum circuits
- Designed for **multi-core / multi-node HPC execution**

---

### `DQOF_run.py`
Driver script for running a **DQOF optimization**.
- Loads HUBO input files
- Configures decomposition and parallel execution
- Calls `DQOF.py`

---

### `AL_DQOF_run.py`
Driver script for **Active Learning + DQOF**.
- Iteratively integrates:
  - 3rd-order Factorization Machine surrogate (`ML/`)
  - DQOF optimization
  - Simulation feedback (`sim/`)
- Updates training data across AL cycles

---

## Checking Required CPU Cores

Before submitting jobs, use the notebook below to estimate how many CPU cores are required.

### `Check_num_cores.ipynb`
This notebook helps determine:
- Recommended number of CPU cores
- Whether your SLURM resource request is sufficient

**Recommended usage**
1. Open Check_num_cores.ipynb
2. Set problem_size and sub_HUBO_size
3. Verify num_parallel and total cores needed


---

## Running DQOF or AL_DQOF

### Step 1: edit parameters in: `DQOF_run.py` or `AL_DQOF_run.py`
### Step 2: update resource settings in the SLURM scripts: `submit_DQOF.sl` or `submit_AL_DQOF.sl`
### Step 3: sbatch the job:
submit_DQOF.sl or sbatch submit_AL_DQOF.sl



---

## Directory Structure
```text
Tutorial/
├── 1_dataset/               # Example datasets for active learning
├── ML/                      # Machine learning models (e.g., 3rd-order FM)
├── sim/                     # Numerical solvers (e.g., TMM)
├── utils/                   # Utility functions
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
