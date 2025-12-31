# Distributed Quantum Optimization for Dense, Large-Scale Higher-Order Problems

## Introduction
Many real-world problems can be naturally formulated as higher-order optimization tasks involving dense, multi-variable interactions, which are challenging to solve for classical methods. Quantum approximate optimization algorithm offers a promising route, but quantum hardware constraints and a focus on quadratic formulations have limited practical progress. Here, we introduce a distributed quantum optimization framework (DQOF) designed to solve dense, large-scale, higher-order binary optimization (HUBO) problems. DQOF assigns quantum circuits a central computational role for exploring higher-order optimization landscapes, while classical high-performance computing orchestrates large-scale parallelism and coordination. A clustering strategy enables wide quantum circuits without increasing circuit depth, allowing efficient execution on near-term quantum hardware. We demonstrate high-accuracy solutions for HUBOs up to 500 variables, significantly outperforming conventional approaches in both solution quality and scalability. Applied to materials optimization, DQOF enables efficient discovery of high-performance designs. These results establish DQOF as a practical and scalable computational paradigm for large-scale scientific optimization.


## Instructions
This repository contains an implementation of **DQOF** for solving large-scale **higher-order unconstrained binary optimization (HUBO)** problems.

### Usage
- `DQOF.py`  
  Core **DQOF** solver.
  Designed for **multi-core and/or multi-node execution** and intended to be run on **HPC systems**.
 
- `AL-DQOF.py`  
  Active learning (AL) pipeline that iteratively integrates:
  - 3rd-order Factorization Machine surrogate modeling (`ML/FM.py`)
  - Distributed quantum optimization framework via **DQOF**
  - Transfer Matrix Method simulations (`sim/TMM_cal.py`)

### Data
- `Examples/`  
  Example HUBO matrices used in this study.
- `Example_AL_dataset/`  
  Example datasets for running the AL-DQOF workflow.

### Optimization Utilities
- `Clustering/`  
  Clustering-based strategy to improve quantum hardware utilization in DQOF.  
  Includes representative HUBO instances.


## Key Hyperparameters
### DQOF
- `problem_size`  
  Total number of binary variables in the HUBO problem.

- `sub_HUBO_size`  
  Size of each sub-HUBO used in the decomposition.

- `num_sub_HUBOs`  
  Number of sub-HUBOs generated from the original HUBO problem.

- `num_parallel`  
  Number of DQAOA instances, executed in aprallel.

- `num_DQAOA_iters`  
  Number of optimization iterations for the DQOF solver.

> **Note:** The total number of CPU cores or nodes is specified at job submission time
> (e.g., via SLURM) and is not hard-coded in the script.

---
### AL-DQOF (Active Learning + DQOF)
In addition to the DQOF hyperparameters above, the active learning workflow introduces:
- `AL_num_iters`  
  Total number of active learning cycles.
  
- `AL_iters`  
  Active learning iteration counter  
  (typically initialized to `0`).


## Note
Qiskit is under active development. For hardware execution, the latest versions of `qiskit` and `qiskit-ibm-runtime` may be required.


## Citation
`@article{kim2024distributed,
  title={Distributed Quantum Approximate Optimization Algorithm on a Quantum-Centric Supercomputing Architecture},
  author={Kim, Seongmin and Pascuzzi, Vincent R and Xu, Zhihao and Luo, Tengfei and Lee, Eungkyu and Suh, In-Saeng},
  journal={arXiv preprint arXiv:2407.20212},
  year={2024}
}`
