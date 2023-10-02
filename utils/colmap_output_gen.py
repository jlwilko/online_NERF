import argparse
import numpy as np
import json
import cv2

def main():
    # get base path from first command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", help="path to base directory", default="data/lfodo/core/seq34/")

    # add argument to use default camera parameters or not
    parser.add_argument(
        "-c",
        "--camera",
        action="store_true",
        default=True,
        help="use default camera parameters",
    )
    args = parser.parse_args()
    base_path = args.base_path

    # from vo_output.json file, generate the images.txt and cameras.txt files that colmap would generate
    with open(base_path + "vo_poses.json", "r") as json_file:
        vo_output = json.load(json_file)
    
    with open(base_path + "cameras.txt", "w") as camera_file:
        camera_file.write("# Camera list with one line of data per camera:\n")
        camera_file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        camera_file.write("# Number of cameras: 1\n")

        img = cv2.imread(base_path + "8/" + str(0).zfill(10) + ".png", 0,)

        if args.camera:
            print("doing camera stuff")
            camera_file.write(f"1 PINHOLE {str(img.shape[1])} {str(img.shape[0])} 197.68828 127.033999 197.68828 91.16238\n")

    with open(base_path + "images.txt", "w") as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        images_file.write(f"# Number of images: {str(len(vo_output))}\n")

        for idx, entry in enumerate(vo_output):
            R = entry["AbsolutePose"]["R"]
            # convert R to quaternion
            qw = 0.5 * np.sqrt(1 + R[0][0] + R[1][1] + R[2][2])
            qx = (R[2][1] - R[1][2]) / (4 * qw)
            qy = (R[0][2] - R[2][0]) / (4 * qw)
            qz = (R[1][0] - R[0][1]) / (4 * qw)

            (tx, ty, tz) = entry["AbsolutePose"]["Translation"]
            images_file.write(f"{str(idx)} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {str(idx).zfill(10)}.png\n")
            # need second line for keypoints from COLMAP reconstruction
            images_file.write("0 0 0\n")

    return


if __name__ == "__main__":
    main()

