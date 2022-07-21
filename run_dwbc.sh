#!/bin/bash

# Script to reproduce results

GPU_LIST=(0 1 2 3)

for alpha in 7.5; do
for seed in 0 1 2 3 4; do

# setting 1
for env in "halfcheetah-expert-v2" "hopper-expert-v2" "walker2d-expert-v2" "ant-expert-v2"; do
for x in 30 60 90; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --algorithm "DWBC" \
  --env $env \
  --split_x $x \
  --alpha $alpha \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done

# setting 2
for env in "halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2" "ant-medium-replay-v2"; do
for x in 2 5 10; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --algorithm "DWBC" \
  --env $env \
  --split_x $x \
  --alpha $alpha \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done

# setting 3
for env in "pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1"; do
for x in 30 60 90; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --algorithm "DWBC" \
  --env $env \
  --split_x $x \
  --alpha $alpha \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done

done
done