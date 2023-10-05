#!/bin/bash

# get sequence number from command line argument 1 - exit if not provided
if [ -z "$1" ]
then
    echo "No sequence number provided"
    exit
fi
seq_no=$1

# cd to the sequence number in the dataset
cd /volume/data/lfodo/core/seq$seq_no
pwd

# define a list of noise values to add for translational noise
# trans_noise_list="0.0 0.01 0.02 0.05 0.1 0.15 0.2"
# rot_noise_list="0.0 0.1 0.5 1 2 5 10"
trans_noise_list="0.1"
rot_noise_list="0.0"

# loop over each time in the sequence
for trans_noise in $trans_noise_list; do
    for rot_noise in $rot_noise_list; do
        filename="transforms_trans${trans_noise}_rot${rot_noise}"

        res_dir="res_trans${trans_noise}_rot${rot_noise}"
        mkdir -p $res_dir

        # echo "Adding translational noise $noise to sequence $seq_no, writing output to $filename"

        # generate the ground truth poses from colmap
        python3 /volume/scripts/colmap2noisynerf.py --images 8/ --aabb_scale 4 --out $filename --rotation_sigma 0 --translation_sigma 0 --gt-kitti ${res_dir}/gt_extrinsics.txt

        # generate the noisy poses and the transforms.json for training - OVERWRITES PREVIOUS FILE
        python3 /volume/scripts/colmap2noisynerf.py --images 8/ --aabb_scale 4 --out $filename --rotation_sigma $rot_noise --translation_sigma $trans_noise
        # train a nerf for this file
        python3 /volume/scripts/noisy_pose_run_nerf.py \
        ${filename}_train.json \
        --n_steps 5000 \
        --screenshot_transforms ${filename}_test.json \
        --screenshot_dir ${res_dir}/poseopt/ \
        --test_transforms ${filename}_test.json \
        --train_extrinsic \
        --extrinsics_export_path ${res_dir}/opt_extrinsics.txt

        python3 /volume/scripts/noisy_pose_run_nerf.py \
        ${filename}_train.json \
        --n_steps 5000 \
        --screenshot_transforms ${filename}_test.json \
        --screenshot_dir ${res_dir}/noopt/ \
        --test_transforms ${filename}_test.json \
        --extrinsics_export_path ${res_dir}/noisy_extrinsics.txt
    done 
done