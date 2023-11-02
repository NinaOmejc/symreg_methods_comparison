#!/bin/bash
#SBATCH --job-name=outer
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --array=0-1
#SBATCH --output=/dev/null #./symreg_methods_comparison/slurm/slurm_output_proged_outer_%A_%a.out   # /dev/null

date
echo "this is batch "$SLURM_ARRAY_TASK_ID""

sbatch symreg_methods_comparison/src/revision_run_proged.sh $SLURM_ARRAY_TASK_ID

echo "completed"

