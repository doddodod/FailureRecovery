#!/bin/bash
set -euo pipefail
cd /inspurfs/group/hexm/embo/foresight_skeleton
python horizon_experiment/download_manifest.py --manifest horizon_experiment/manifest_task0.txt --output horizon_experiment/data_full --workers 1
echo "TASK0_READY $(date -Is)"
sbatch horizon_experiment/run_task0_pilot.sbatch
python horizon_experiment/download_manifest.py --manifest horizon_experiment/manifest_full.txt --output horizon_experiment/data_full --workers 1
echo "FULL_READY $(date -Is)"
sbatch horizon_experiment/run_full_4gpu.sbatch
