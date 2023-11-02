#!/bin/bash
#SBATCH --job-name=mlj-p
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-1
#SBATCH --output=./symreg_methods_comparison/slurm/slurm_output_proged_%A_%a.out

echo "this is subjob" $(($1*1000 + $SLURM_ARRAY_TASK_ID))""
date
cd ./symreg_methods_comparison/
singularity exec symreg.sif python3.7 ./src/proged_system_identification_fullobs.py $(($1*1000 + $SLURM_ARRAY_TASK_ID))

echo "completed" 

