#!/bin/bash

# check that $1 exists
if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi

echo data/lfodo/core/seq$1
cd data/lfodo/core/seq$1
python3 /volume/scripts/colmap2nerf.py --run_colmap --colmap_camera_model PINHOLE --images 8/ --aabb_scale 4 --overwrite


# # for each directory in the current directory cd to the directory
# # and run colmap feature extraction and matching
# # then cd back to the original directory
# for d in data/lfodo/core/seq{2..32} ; do
#     echo "Processing $d"
#     cd $d
#     python3 /volume/scripts/colmap2nerf.py --run_colmap --colmap_camera_model PINHOLE --images 8/ --aabb_scale 4 --overwrite

#     cd -
# done