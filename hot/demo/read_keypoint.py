import numpy as np
import os
print(os.getcwd())
key_point_2d = np.load("./demo/output/crop_test/input_2D/input_keypoints_2d.npz", allow_pickle=True)["reconstruction"]
print(key_point_2d)
key_point_3d = np.load("./demo/output/crop_test/output_3D/output_keypoints_3d.npz", allow_pickle=True)["reconstruction"]
print(key_point_3d)
