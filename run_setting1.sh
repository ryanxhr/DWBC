#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

for env in "halfcheetah-expert-v2" "hopper-expert-v2" "walker2d-expert-v2" "ant-expert-v2"; do
for x in 30 60; do
for seed in 0 1; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --algorithm "DWBC" \
  --env $env \
  --split_x $x \
  --pu_learning False \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done
done