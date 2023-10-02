import argparse
import numpy as np
import os

def format_images_txt(in_path, out_path):
    # read file
    with open(in_path, "r") as file:
        processed = 0
        with open(out_path, "w") as out_file:
            for idx, line in enumerate(file):
                if line[0] == "#":
                    continue
                # ignore every other line
                if idx % 2 == 1:
                    continue
                
                # split line
                split_line = line.split(" ")
                image_id = split_line[0]
                (qw, qx, qy, qz) = split_line[1:5]
                (tx, ty, tz) = split_line[5:8]
                out_file.write(f"{str(image_id)} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    return

def format_gt_npy(in_path, out_path):
    data = np.load(in_path)
    with open(out_path, "w") as file:
        for idx, entry in enumerate(data):
            qw = 0.5 * np.sqrt(1 + entry[0][0] + entry[1][1] + entry[2][2])
            qx = (entry[2][1] - entry[1][2]) / (4 * qw)
            qy = (entry[0][2] - entry[2][0]) / (4 * qw)
            qz = (entry[1][0] - entry[0][1]) / (4 * qw)
            tx = entry[0][3]
            ty = entry[1][3]
            tz = entry[2][3]
            file.write(f"{str(idx)} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
    return

def main():
    # get path to convert from first command line arg
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to convert")
    parser.add_argument("output", help="path to output dir")
    args = parser.parse_args()
    path = args.path

    format_images_txt(os.path.join(path, "images.txt"), os.path.join(args.output, "traj1.txt"))
    format_gt_npy(os.path.join(path, "poses_gt_first_cam_renorm.npy"), os.path.join(args.output, "traj_gt.txt"))
    print(args.output)
    return

if __name__ == "__main__":
    main()