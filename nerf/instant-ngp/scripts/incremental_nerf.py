#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import subprocess
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

class TransformsJSONGenerator():
	def __init__(self, filename) -> None:
		jsontransforms = json.load(open(filename, "r"))
		self.transforms = sorted(jsontransforms["frames"], key=lambda x: x["file_path"])
		# print(json.dumps(jsontransforms, indent=2))
		# print(self.transforms)
		self.empty_header = jsontransforms.copy()
		self.empty_header["frames"] = []
		# print(self.empty_header)
		# print(self.transforms[0])

	def generate_transforms(self, frames, output_filename='sometransforms.json'):
		frames = [frame for frame in frames if frame < len(self.transforms)]
		out = self.empty_header.copy()
		out["frames"] = [self.transforms[i] for i in frames]
		with open(output_filename, "w") as f:
			json.dump(out, f, indent=2)
		# print(json.dumps(out, indent=2))
		# print(json.dumps(self.empty_header, indent=2))
		return output_filename


def parse_args():
	parser = argparse.ArgumentParser(description="Run instant neural graphics primitives incrementally on a sequence")

	parser.add_argument("--sequence", default="", type=str, help="The sequence to load. Can be the sequence's name or a full path to the sequence.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--initial_images", type=int, default=-1, help="Number of initial images to train the first model on")
	## addition algorithm to use, one of ["Adaptive", "Naive", "Temporal", "Distance"]
	parser.add_argument('--addition_algorithm', default="adaptive", type=str, help="The algorithm to use for adding images to the training set", choices=["adaptive", "naive", "temporal", "distance"])

	
	# parser.add_argument("files", nargs="*", help="Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.")

	# parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.")
	# parser.add_argument("--mode", default="", type=str, help=argparse.SUPPRESS) # deprecated
	# parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	# parser.add_argument("--load_snapshot", "--snapshot", default="", help="Load this snapshot before training. recommended extension: .ingp/.msgpack")
	# parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .ingp/.msgpack")

	# parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes, but helps with high PSNR on synthetic scenes.")
	# parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	# parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	# parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	# parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	# parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	# parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	# parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	# parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	# parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	# parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	# parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	# parser.add_argument("--video_render_range", type=int, nargs=2, default=(-1, -1), metavar=("START_FRAME", "END_FRAME"), help="Limit output to frames between START_FRAME and END_FRAME (inclusive)")
	# parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	# parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video (video.mp4) or video frames (video_%%04d.png).")

	# parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	# parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")
	# parser.add_argument("--marching_cubes_density_thresh", default=2.5, type=float, help="Sets the density threshold for marching cubes.")

	# parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	# parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	# parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	# parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	# parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	# parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")
	# parser.add_argument("--vr", action="store_true", help="Render to a VR headset.")

	# parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")


	return parser.parse_args()

def train_nerf(train_path, n_steps, test_path, save_model=False, model_path=""):
	command = f"""python3 /volume/scripts/incremental_run.py
{train_path}
--n_steps {n_steps}
--test_transforms {test_path}
--screenshot_transforms {test_path}
--screenshot_dir incremental/initial/
--save_snapshot incremental/{model_path}"""
# --numerical_results_path init_results.csv"""
	command = command.split()
	print(command)
	# os.system(command)
	subprocess.run(command, stdout=subprocess.PIPE)

def render_pretrained_nerf(model_path, transforms):
	command = f"""python3 /volume/scripts/incremental_run.py
--load_snapshot incremental/{model_path}
--test_transforms {transforms}
"""
	command = command.split()
	print(command)
	result = subprocess.run(command, stdout=subprocess.PIPE)
	lines = result.stdout.decode("utf-8").split("\n")
	# print(lines)
	results_line = list(filter(lambda x: "PSNR" in x, lines))[0]
	stats = json.loads(results_line)
	return stats

if __name__ == "__main__":
	args = parse_args()
	print(os.getcwd())

	# Load the sequence transforms.json and get the list of frames to easily generate new ones
	# keep a subset of frames separate for testing on the final nerfs to give conclusive results
	loader = TransformsJSONGenerator("transforms_train.json")
	# transform_loader.generate_transforms([1,2])

	# train an initial nerf on the first N images of the sequence and save to a specific path, render it on the testing poses and save the psnr results etc
	init_transforms = "transforms_initial.json"
	loader.generate_transforms(range(args.initial_images), init_transforms)

	# keep track of the current nerf model and the images it was trained on
	current_training_poses = list(range(args.initial_images))
	last_added_pose = max(current_training_poses) # keep track of the last pose added to the training set
	training_stats = {}
	total_training_time = 0

	# start a timer
	start_time = time.perf_counter()
	train_nerf(init_transforms, args.n_steps, "transforms_test.json", save_model=True, model_path="current_model.ingp")
	total_training_time += time.perf_counter() - start_time

	next_transforms = loader.generate_transforms(current_training_poses)
	stats = render_pretrained_nerf("current_model.ingp", next_transforms)
	print(f"PSNR: {stats['PSNR']}, SSIM: {stats['SSIM']}, LPIPS: {stats['LPIPS']}")
	training_stats[last_added_pose] = stats

	for idx in range(args.initial_images, len(loader.transforms)-1):
		print(f"Incoming image {idx}, path {loader.transforms[idx]['file_path']}")

		# load the nerf and evaluate on the next M(1) images in the sequence (for averaging purposes)
		next_transforms = loader.generate_transforms(range(idx, idx+2))
		projected_stats = render_pretrained_nerf("current_model.ingp", next_transforms)
		# print(f"PSNR: {stats['PSNR']}, SSIM: {stats['SSIM']}, LPIPS: {stats['LPIPS']}")
		# training_stats[last_added_pose] = stats

		add_image = False
		if args.addition_algorithm == "naive":
			print("Checking using naive algorithm")
			# always add the pose
			add_image = True

		elif args.addition_algorithm == "temporal":
			print("Checking using temporal algorithm")
			
			pass
		elif args.addition_algorithm == "distance":
			print("Checking using distance algorithm")
			pass
		else:
			assert args.addition_algorithm == "adaptive"
			print("Checking using adaptive algorithm")
			# print(training_stats)
			last_trained_pose = training_stats[last_added_pose]
			most_recent_pose = projected_stats
			# print(last_trained_pose)
			print(most_recent_pose)

			# if the PSNR gap between the last trained pose and the most recent pose is greater than some threshold, add the image and retrain the nerf
			# if last_trained_pose["PSNR"] - most_recent_pose["PSNR"] > 15:
			# 	add_image = True
			if most_recent_pose["PSNR"] < 24:
				add_image = True
			# if last_trained_pose["SSIM"] - most_recent_pose["SSIM"] > 0.18:
			# 	add_image = True
			# if last_train_pose["LPIPS"] - most_recent_pose["LPIPS"] < -0.2:
			# 	add_image = True

		if add_image:
			current_training_poses.append(idx)
			last_added_pose = idx
			curr_transforms = loader.generate_transforms(current_training_poses)

			# retrain the nerf 
			# save the nerf to a new model path
			start_time = time.perf_counter()
			train_nerf(curr_transforms, args.n_steps, "transforms_test.json", save_model=True, model_path="current_model.ingp")
			total_training_time += time.perf_counter() - start_time
			stats = render_pretrained_nerf("current_model.ingp", curr_transforms)
			training_stats[last_added_pose] =  stats
			print(f"Adding image to training set:\n\tCurrent training images are {current_training_poses}")
			print(f"\tTotal training time so far is {total_training_time} seconds")
			print(f"\t")







		# check the PSNR/SSIM/LPIPS of the nerf on the next M(1) images to see if we should add this image to the training set
		# alternatively choose whether to add this image based on simply temporal or distance based metrics


		# if we should add this image to the training set, add it and retrain the nerf

	# update the base_nerf to the new model and repeat the process until all frames have been considered

	# keep timing statistics on training time
	# keep track of the number of images in the training set for each trained nerf and which images are included
	# keep note of 





