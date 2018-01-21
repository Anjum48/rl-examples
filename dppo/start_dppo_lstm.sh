#!/bin/bash

PS=1  # parameter servers
N_WORKER=4  # parallel workers
N_AGG=2  # Gradients to aggregate

DATETIME=`date +'%Y%m%d-%H%M%S'`

p=0
while [ $p -lt $PS ]
do
    echo "Starting Parameter Server $p"
    CUDA_VISIBLE_DEVICES='' python3 dppo_lstm.py --timestamp=$DATETIME --job_name="ps" --task_index=$p \
    --workers=$N_WORKER  --agg=$N_AGG --ps=$PS &>/dev/null &
    p=$(( $p + 1 ))
done

w=0
while [ $w -lt $N_WORKER ]
do
    echo "Starting Worker $w"
    CUDA_VISIBLE_DEVICES='' python3 dppo_lstm.py --timestamp=$DATETIME --job_name="worker" --task_index=$w \
    --workers=$N_WORKER  --agg=$N_AGG --ps=$PS &>/dev/null &
    w=$(( $w + 1 ))
done