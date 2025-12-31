#!/bin/bash
#SBATCH -A <your_project_ID>
#SBATCH -J ALDQOF
#SBATCH -p batch
#SBATCH -N 3
#SBATCH -t 23:59:00
#SBATCH -o slurm-%j.out

unset SLURM_EXPORT_ENV

cd $SLURM_SUBMIT_DIR
date

module load <necessary_modules>
source <your_env>/bin/activate

srun -n 131 python AL_DQOF_run.py
