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

python3 /volume/scripts/colmap2noisynerf.py --images 8/ --aabb_scale 4 --out transforms
python3 /volume/scripts/incremental_nerf.py \
	--sequence $seq_no \
	--initial_images 10 \
	--n_steps 5000 \
	--addition_algorithm temporal

