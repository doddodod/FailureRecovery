#!/bin/bash
#SBATCH -J foresight_horizon_full
#SBATCH -p ShangHAI
#SBATCH -A hexm-shanghai
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -o slurm_logs/full_4gpu-%j.out
set -euo pipefail
cd /inspurfs/group/hexm/embo/foresight_skeleton
source .venv/bin/activate
export PYTHONPATH="$PWD/horizon_experiment/torch_cu128${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p horizon_experiment/results/full
for seed in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$seed python horizon_experiment/train_horizon_failure.py --data horizon_experiment/data_full --seed $seed --output horizon_experiment/results/full/seed${seed}.json &
done
CUDA_VISIBLE_DEVICES=3 python horizon_experiment/train_horizon_failure.py --data horizon_experiment/data_full --seed 0 --shuffle-actions --output horizon_experiment/results/full/shuffled_actions.json &
wait
