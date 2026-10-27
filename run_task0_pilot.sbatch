#!/bin/bash
#SBATCH -J foresight_horizon_t0
#SBATCH -p ShangHAI
#SBATCH -A hexm-shanghai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o slurm_logs/task0_pilot-%j.out
set -euo pipefail
cd /inspurfs/group/hexm/embo/foresight_skeleton
source .venv/bin/activate
export PYTHONPATH="$PWD/horizon_experiment/torch_cu128${PYTHONPATH:+:$PYTHONPATH}"
python horizon_experiment/train_horizon_failure.py --data horizon_experiment/data_full --task 0 --seed 0 --epochs 40 --output horizon_experiment/results/task0_pilot.json
