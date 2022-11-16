#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port=29444 $(dirname "$0")/train.py --dist $CONFIG ${@:3}
