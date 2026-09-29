import os

import cv2
import numpy as np
import torch
import transforms3d.euler as euler
from torch.utils.data import ConcatDataset, Dataset

from datasets.common import image_to_tensor, load_pair_index, sample_training_pair

# Camera intrinsics used for training and evaluating the released KITTI model
# (images are resized to 1216 x 384). Keep unchanged when using the released weights.
KITTI_K = np.array([
    [7.070912000000e+02, 0.000000000000e+00, 5.068873000000e+02],
    [0.000000000000e+00, 7.070912000000e+02, 1.901104000000e+02],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]
])
# Transform from the vehicle frame (x forward, y left, z up) to the camera frame.
KITTI_T_CAM_BODY = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])
# Camera frame -> vehicle frame for the ground-truth poses.
_T_CAR_CAM = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])


class KITTISequence(Dataset):
    """One KITTI odometry sequence (left color camera, image_2).

    train: random frame pairs drawn from a pre-computed pair index.
    test:  consecutive pairs of frames sub-sampled with a fixed stride.
    """

    def __init__(self, root_dir, sequence, split, pair_index_path=None, test_stride=1, rot_pair_ratio=0.5):
        assert split in ['train', 'test']
        self.split = split
        self.rot_threshold = int(round(rot_pair_ratio * 10))

        seq_dir = os.path.join(root_dir, 'data_odometry_color/dataset/sequences', sequence)
        image_dir = os.path.join(seq_dir, 'image_2')
        self.images = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        self.timestamps = np.loadtxt(os.path.join(seq_dir, 'times.txt'))

        poses = np.loadtxt(os.path.join(root_dir, 'poses', sequence + '.txt')).reshape(-1, 3, 4)
        T_car_cam_inv = np.linalg.inv(_T_CAR_CAM)
        self.poses_xyzrpy = []
        for pose in poses:
            pose_car = np.dot(np.dot(_T_CAR_CAM, np.vstack((pose, [0, 0, 0, 1]))), T_car_cam_inv)
            x, y, z = pose_car[:3, 3]
            roll, pitch, yaw = self.rot_matrix_to_euler(pose_car[:3, :3])
            self.poses_xyzrpy.append([x, y, z, roll, pitch, yaw])

        if self.split == "train":
            self.nearby_points, self.nearby_points_rot = load_pair_index(pair_index_path)
        else:
            self.images = self.images[::test_stride]
            self.timestamps = self.timestamps[::test_stride]
            self.poses_xyzrpy = self.poses_xyzrpy[::test_stride]

    def __len__(self):
        if self.split == "train":
            return len(self.images)
        return len(self.images) - 1

    def __getitem__(self, idx):
        if self.split == "train":
            idx1, idx2 = sample_training_pair(idx, self.nearby_points, self.nearby_points_rot,
                                              len(self), self.rot_threshold)
        else:
            idx1, idx2 = idx, idx + 1

        images = []
        poses = np.zeros((2, 4, 4), dtype=np.float64)
        timestamp = None
        for k, idx_temp in enumerate([idx1, idx2]):
            image = cv2.resize(cv2.imread(self.images[idx_temp]), (1216, 384), interpolation=cv2.INTER_LINEAR)
            images.append(torch.stack([image_to_tensor(image)]))
            poses[k] = self.planar_pose(self.poses_xyzrpy[idx_temp])
            if timestamp is None:
                timestamp = self.timestamps[idx_temp]

        return torch.stack(images), poses, timestamp

    @staticmethod
    def rot_matrix_to_euler(R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy >= 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return [x, y, z]

    @staticmethod
    def planar_pose(pose):
        """3-DoF (x, y, yaw) pose as a 4x4 matrix."""
        x, y, _, _, _, yaw = pose
        T = np.eye(4)
        T[:3, :3] = euler.euler2mat(yaw, 0, 0, 'szyx')
        T[:3, 3] = [x, y, 0]
        return T


class KITTISequences(Dataset):

    def __init__(self, root_dir, sequences, split, pair_index_paths=None, test_strides=None, rot_pair_ratio=0.5):
        pair_index_paths = pair_index_paths or [None] * len(sequences)
        test_strides = test_strides or {}
        datasets = [KITTISequence(root_dir, seq, split, path, test_strides.get(seq, 1), rot_pair_ratio)
                    for seq, path in zip(sequences, pair_index_paths)]
        self.dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]
