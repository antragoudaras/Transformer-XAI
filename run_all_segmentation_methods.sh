#!/bin/bash

# entire script fails if a single command fails
set -e

methods=("rollout" "lrp" "transformer_attribution" "lrp_last_layer" "attn_last_layer" "attn_gradcam" "full_lrp")

for method in "${methods[@]}"
do
    echo "Running method: $method"
    sbatch --job-name "${method}" segmentation_initial_methods.job --method "${method}" --imagenet-seg-path "./gtsegs_ijcv.mat" 
done