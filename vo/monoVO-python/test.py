import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from visual_odometry import PinholeCamera, VisualOdometry

base_path = "/home/josh/Documents/uni/2023_2/thesis/online_NERF/data/LearnLFOdo_Dataset_renormalised/core/seq6/"
# base_path = "/home/josh/Documents/uni/2023_2/thesis/online_NERF/data/LearnLFOdo_Dataset_renormalised/core/seq44/"


cam = PinholeCamera(256, 192, 197.68828, 197.68828, 127.033999, 91.16238)
vo = VisualOdometry(
    cam,
    base_path + "poses_gt_base_cam_renorm.npy",
)

traj = np.zeros((600, 600, 3), dtype=np.uint8)


# count the number of images in the directory with glob
image_count = len(glob.glob1(base_path + "0/", "*.png"))

t_path = np.zeros((image_count, 3))
R_path = np.zeros((image_count, 4, 4))

t_gt = np.zeros((image_count, 3))

for img_id in range(image_count):
    img = cv2.imread(
        base_path + "0/" + str(img_id).zfill(10) + ".png",
        0,
    )

    vo.update(img, img_id)

    # R_path[img_id] = vo.cur_R
    t_gt[img_id] = np.array((vo.trueX, vo.trueY, -vo.trueZ))
    cur_t = vo.cur_t.T.flatten() if img_id >= 2 else t_gt[img_id]
    print(cur_t)
    t_path[img_id] = cur_t[[0,2,1]]


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

# plot t_path and t_gt on a 3d plot 
t_path = t_path[2:, :]
t_gt = t_gt[2:, :]
print(t_path)
print(t_gt)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(t_path[:,0], t_path[:,1], t_path[:,2], label="t_path")
ax.plot(t_gt[:,0], t_gt[:,1], t_gt[:,2], label="t_gt")
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

