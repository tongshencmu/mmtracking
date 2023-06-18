#!/usr/bin/env bash
#SBATCH --time=48:00:00

set -x

PARTITION='GPU-shared'
JOB_NAME=$1
CONFIG=$2
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    -t "48:00:00" \
    --job-name=${JOB_NAME} \
    --gres=gpu:v100-32:4 \
    --ntasks-per-node=8 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u $(dirname "$0")/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
