# Towards Online NeRF

## Setup

Create virtual environment for the visual odometry in the `vo/monoVO-python/` directory:

```bash
cd vo/monoVO-python/
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# What have I done

- more playing around with visual odometry 
- fixed issue where resolution was very low = i was just downsampling by 10x after an initial investigation so nws there
- download KITTI and install basic visual odom pipeline
- sync KITTI dataset to `fraser`
- figure out odom pipeline on KITTI dataset
- commit working code to github
- update camera parameters
- update paths to LearnLFOdo dataset
- make new branch
- write parsing function for transformation matrices (annotations)
- run odom pipeline on my dataset
  - will need to fix coordinate frame issue

# What should be done next 

- figure out how to plot odom better to ensure im getting good results
  - also fix scaling issue
- install and run instantNGP
- run instantNGP on LearnLFOdo dataset ground truths
- run instantNGP on LearnLFOdo odometry data

- generate results for presentation and report later o 
- figure out plan for next 2 weeks
--- by end of TOMORROW ---