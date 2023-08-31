import numpy as np
import cv2
import glob

from visual_odometry import PinholeCamera, VisualOdometry

base_path = "/home/josh/Documents/uni/2023_2/thesis/online_NERF/data/LearnLFOdo_Dataset_renormalised/core/seq2/"


cam = PinholeCamera(256, 192, 197.68828, 197.68828, 127.033999, 91.16238)
vo = VisualOdometry(
    cam,
    base_path + "poses_gt_base_cam_renorm.npy",
)

traj = np.zeros((600, 600, 3), dtype=np.uint8)

# count the number of images in the directory with glob
image_count = len(glob.glob1(base_path + "0/", "*.png"))
for img_id in range(image_count):
    img = cv2.imread(
        base_path + "0/" + str(img_id).zfill(10) + ".png",
        0,
    )

    vo.update(img, img_id)

    cur_t = vo.cur_t
    if img_id > 2:
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0.0, 0.0, 0.0
    draw_x, draw_y = int(x*100) + 290, int(z*100) + 90
    true_x, true_y = int(vo.trueX*100) + 290, int(vo.trueY*100) + 90
    print(f"x: {x}, y: {y}, z: {z}")
    print(f"trueX: {vo.trueX}, trueY: {vo.trueY}, trueZ: {vo.trueZ}")

    cv2.circle(
        traj,
        (draw_x, draw_y),
        1,
        (img_id * 255 / image_count, 255 - img_id * 255 / image_count, 0),
        1,
    )
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fcm y=%2fcm z=%2fcm" % (x, y, z)
    cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    cv2.imshow("Road facing camera", img)
    cv2.imshow("Trajectory", traj)
    cv2.waitKey(1)

cv2.imwrite("map.png", traj)
cv2.waitKey(0)
