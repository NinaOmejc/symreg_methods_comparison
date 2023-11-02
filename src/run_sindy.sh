#!/bin/bash
#SBATCH --job-name=mlj-s
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=3GB
#SBATCH --array=0-240 # 156 == vdp no noise
#SBATCH --output=./symreg_methods_comparison/slurm/slurm_output_sindy_%A_%a.out

JOBINDEX=$((SLURM_ARRAY_TASK_ID))

cd ./symreg_methods_comparison/
singularity exec symreg.sif python3.7 ./src/sindy_system_identification.py ${JOBINDEX}

