#!/bin/bash
#SBATCH -A <your_project_ID>
#SBATCH -J DQOF
#SBATCH -p batch
#SBATCH -N 9
#SBATCH -t 23:59:00
#SBATCH -o slurm-%j.out

unset SLURM_EXPORT_ENV

cd $SLURM_SUBMIT_DIR
date

module load <necessary_modules>
source <your_env>/bin/activate

srun -n 411 python DQOF_run.py
