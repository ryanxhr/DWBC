2#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

for env in "halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2" "ant-medium-replay-v2"; do
for x in 2 5 10; do
for seed in 0 1; do
for eta in 0.1 0.3 0.5; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --algorithm "DWBC" \
  --env $env \
  --split_x $x \
  --pu_learning True \
  --eta $eta \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done
done
done