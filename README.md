# Towards Online NeRF

# Building the container:

To build the container initially:

`docker build -t instantngp -f .devcontainer/Dockerfile .`

Once that completes, run `./init-ngp-container` to start the container


# Useful commands and scripts

- to generate instant-ngp style `transforms.json` by running COLMAP, run this command from the directory where you would like transforms.json to be generated:  

`python3 /volume/scripts/colmap2nerf.py --run_colmap --colmap_camera_model PINHOLE --images 8/ --aabb_scale 2`

- to generate TUM style trajectories for ATE and RPE metrics: 

- run `./init-ngp-container` to start the instant-ngp container

- to run instant-ngp with a gui on a training dataset, all that is required is the `transforms.json` file that tells which images to train on and camera parameters i.e. `./instant-ngp path/to/transforms.json`

