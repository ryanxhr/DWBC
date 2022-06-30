#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

for env in "pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1"; do
for x in 30 60; do
for seed in 2 3; do

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