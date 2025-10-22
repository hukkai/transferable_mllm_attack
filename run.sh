#!/usr/bin/env bash
MASTER_PORT=$((12000 + $RANDOM % 20000))

do_random_pad=1
do_random_jpeg=1
do_random_noise=1
job_id=$1

OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 --master-port=$MASTER_PORT \
	batch_attack.py --input_size=299 --eps=8 --exclude=9 \
	                --model_dtype=float16 \
			--job_id=$job_id
