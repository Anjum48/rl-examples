#!/bin/bash

PS=0  # parameter servers
N_WORKER=16  # parallel workers

DATETIME=`date +'%Y%m%d-%H%M%S'`

p=0
while [ $p -lt $PS ]
do
    echo "Starting Parameter Server $p"
    CUDA_VISIBLE_DEVICES='' python3 DPPO_LSTM.py --timestamp=$DATETIME --job_name="ps" --task_index=$p &>/dev/null &
    p=$(( $p + 1 ))
done

w=0
while [ $w -lt $N_WORKER ]
do
    echo "Starting Worker $w"
    CUDA_VISIBLE_DEVICES='' python3 DPPO_LSTM.py --timestamp=$DATETIME --job_name="worker" --task_index=$w &>/dev/null &
    w=$(( $w + 1 ))
done