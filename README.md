# Towards Online NeRF for mobile robotics

# Building the container:

To build the container initially:

`docker build -t instantngp -f .devcontainer/Dockerfile .`

Once that completes, run `./init-ngp-container` to start the container

# Useful commands and scripts

- main runner script is `do_noisy_nerf.sh` this runs the main noisy_poses stuff
- `evo_traj kitti noisy_extrinsics.txt opt_extrinsics.txt --ref=gt_extrinsics.txt -p` to get a 3d plot of the camera poses

- to generate instant-ngp style `transforms.json` poses FOR STATE OF THE ART POSES  by running COLMAP, run this command from the directory where you would like transforms.json to be generated:  `python3 /volume/scripts/colmap2nerf.py --run_colmap --colmap_camera_model PINHOLE --images 8/ --aabb_scale 2`
- to generate noisy poses from the COLMAP ones, run this command: `python3 /volume/scripts/colmap2noisynerf.py --images 8/ --aabb_scale 4 --out transforms.json --rotation_sigma 0 --translation_sigma 0`
- run `./init-ngp-container` to start the instant-ngp container
- to run instant-ngp with a gui on a training dataset, all that is required is the `transforms.json` file that tells which images to train on and camera parameters i.e. `./instant-ngp path/to/transforms.json`
- to script running instant-ngp without a gui, run: `/volume/scripts/noisy_pose_run_nerf.py path/to/transforms.json --n_steps 25000 --screenshot_transforms path/to/transforms/to/screenshot.json --screenshot_dir output --test_transforms path/to/test/transforms.json --train_extrinsic`

- `rsync -rvnc --delete /home/josh/Documents/uni/2023_2/thesis/online_NERF/data/lfodo/ josh@stockton:/home/josh/Documents/uni/2023_2/thesis/online_NERF/data/lfodo/` will check if there is any different in any file in the trees for the dataset where I'm generating all of my results - will delete results that i have deleted from `fraser` so be careful

## Notes
- if some of the corners/sides of the images are being cut off then change the aabb_scale parameter in the transforms.json