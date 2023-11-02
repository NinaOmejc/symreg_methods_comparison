#!/bin/bash
#SBATCH --job-name=mlj-d
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-504 # vdp 314-315
#SBATCH --output=./symreg_methods_comparison/slurm/slurm_output_%A_%a.out

JOBINDEX=$((SLURM_ARRAY_TASK_ID))

cd ./symreg_methods_comparison/
singularity exec symreg.sif python3.7 ./src/dso_system_identification.py ${JOBINDEX}

