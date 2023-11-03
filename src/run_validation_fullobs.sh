#!/bin/bash
#SBATCH --job-name=mlj-val
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=0-999
#SBATCH --output=./symreg_methods_comparison/slurm/slurm_output_%A_%a.out # /dev/null #./symreg_methods_comparison/slurm/slurm_output_%A_%a.out

JOBINDEX=$((0 + SLURM_ARRAY_TASK_ID))
#JOBINDEX=$((1000 + SLURM_ARRAY_TASK_ID))
#JOBINDEX=$((2000 + SLURM_ARRAY_TASK_ID))

echo "this is subjob" $(($JOBINDEX))""
date
cd ./symreg_methods_comparison/
singularity exec symreg.sif python3.7 ./src/common_validation_hpc.py ${JOBINDEX}
date
.
# 0-999
# 1000 + (0-999)
# 2000 + (0-17)