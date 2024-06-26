#!/bin/bash

# entire script fails if a single command fails
set -e

methods=("rollout" "gradient_rollout_cls_spec" "transformer_attribution" "lrp_last_layer" "attn_last_layer" "attn_gradcam" "full_lrp")
# methods=("attn_gradcam")
# methods=("gradient_rollout_cls_spec")
# methods=("rollout")

for method in "${methods[@]}"
do
    echo "Running method: $method"
    sbatch --job-name "${method}" segmentation_initial_methods_deit.job --method "${method}" --imagenet-seg-path "./gtsegs_ijcv.mat" 
done