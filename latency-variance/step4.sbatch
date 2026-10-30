#!/bin/bash
# Step 4 as one batch job: smoke test, then the "within" and "across" runs.
# Submit from the repo root on watgpu.cs (the login node):
#   sbatch latency-variance/step4.sbatch
# Needs setup.sh to have been run once (it writes .work-root and the env).
#SBATCH --job-name=step4-latency
#SBATCH --gres=gpu:1
#SBATCH --constraint=ADA6000
#SBATCH --exclude=watgpu108,watgpu208
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%x-%j.out
##SBATCH --mail-user=<you>@uwaterloo.ca
##SBATCH --mail-type=END,FAIL
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
echo "node $(hostname) job $SLURM_JOB_ID cpus $SLURM_CPUS_PER_TASK $(date)"

bash run.sh
N_PROCS=1 N_REQUESTS=50 RUN_TAG="within-$SLURM_JOB_ID" bash latency-variance/run_repeat.sh
N_PROCS=5 N_REQUESTS=20 RUN_TAG="across-$SLURM_JOB_ID" bash latency-variance/run_repeat.sh
echo "STEP4_DONE $(date)"
