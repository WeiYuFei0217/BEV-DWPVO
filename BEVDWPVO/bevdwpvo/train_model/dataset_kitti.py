import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import struct
import matplotlib.pyplot as plt
import transforms3d.euler as euler

import pickle
from tqdm import tqdm

from PIL import Image
import numpy as np

import random
from config_files import data_kitti_dir


class KITTIDataset(Dataset):
    def __init__(self, root_dir, sequence, phase, dataset_item_range="0.0-4.0", pickle_name="pickle.txt", IS_MONO=True, NO_DEPTH=True):
        self.root_dir = root_dir
        self.sequence = sequence
        self.phase = phase
        self.dataset_item_range = dataset_item_range
        self.pickle_name = pickle_name
        self.IS_MONO = IS_MONO
        self.NO_DEPTH = NO_DEPTH
        self.images = []
        self.poses = []
        self.pose_xyzrpy = []
        self.depth_images = []

        if self.phase == "train":
            self.pickle = pickle.load(open(self.pickle_name, 'rb'))
            self.nearby_points = self.pickle[0]
            self.nearby_points_R = self.pickle[1]

        image_dir = os.path.join(self.root_dir, 'data_odometry_color/dataset/sequences', self.sequence, 'image_2')
        self.image_dir = image_dir
        pose_file = os.path.join(self.root_dir, 'poses', self.sequence+'.txt')
        depth_dir = os.path.join(self.root_dir, 'depth', self.sequence)
        self.depth_dir = depth_dir
        timestamp_file = os.path.join(self.root_dir, 'data_odometry_color/dataset/sequences', self.sequence, 'times.txt')
        self.timestamps = np.loadtxt(timestamp_file)

        print(sequence)

        image_files = sorted(os.listdir(image_dir))
        self.images.extend([os.path.join(image_dir, f) for f in image_files])
        depth_files = sorted(os.listdir(depth_dir))
        self.depth_images.extend([os.path.join(depth_dir, f) for f in depth_files])
        
        poses = np.loadtxt(pose_file).reshape(-1, 3, 4)
        poses_homogeneous = np.array([np.vstack((pose, [0, 0, 0, 1])) for pose in poses])
        self.poses = poses_homogeneous

        transform_matrix = np.array([
            [0, 0, 1, 0], 
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        transform_matrix_inv = np.linalg.inv(transform_matrix)
        poses_transformed = []
        for pose in self.poses:
            pose_transformed = np.dot(np.dot(transform_matrix, pose), transform_matrix_inv)
            poses_transformed.append(pose_transformed)
        self.poses_car = poses_transformed

        poses_xyzrpy_car = []
        for pose in self.poses_car:
            x, y, z = pose[:3, 3]
            R = pose[:3, :3]
            roll, pitch, yaw = self.rot_matrix_to_euler(R)
            poses_xyzrpy_car.append([x, y, z, roll, pitch, yaw])
        self.poses_xyzrpy_car = poses_xyzrpy_car

        if self.phase == "test" and self.sequence == '09' and self.dataset_item_range == "0.0-4.0":
            self.images = self.images[::2]
            self.depth_images = self.depth_images[::2]
            self.timestamps = self.timestamps[::2]
            self.poses = self.poses[::2]
            self.poses_car = self.poses_car[::2]
            self.poses_xyzrpy_car = self.poses_xyzrpy_car[::2]
        if self.phase == "test" and self.sequence == '10' and self.dataset_item_range == "0.0-4.0":
            self.images = self.images[::3]
            self.depth_images = self.depth_images[::3]
            self.timestamps = self.timestamps[::3]
            self.poses = self.poses[::3]
            self.poses_car = self.poses_car[::3]
            self.poses_xyzrpy_car = self.poses_xyzrpy_car[::3]

    def __len__(self):
        if self.phase == "train":
            return len(self.images)
        if self.phase == "test":
            return len(self.images) - 1

    def __getitem__(self, idx):
        all_2_5_images = []
        all_2_5_depth = []
        idx1 = idx
        idx2 = idx + 1

        if self.phase == "train":
            idx1 = idx

            correct_idx2_list = []
            correct_idx2_list_rot = []

            correct_idx2_list = self.nearby_points[idx1]
            correct_idx2_list_rot = self.nearby_points_R[idx1]

            if len(correct_idx2_list_rot) != 0 and len(correct_idx2_list) != 0:
                if random.randint(1, 10) <= 5:
                    idx2 = random.choice(correct_idx2_list_rot)
                else:
                    idx2 = random.choice(correct_idx2_list)
            elif len(correct_idx2_list_rot) != 0:
                idx2 = random.choice(correct_idx2_list_rot)
            elif len(correct_idx2_list) != 0:
                idx2 = random.choice(correct_idx2_list)
            else:
                while len(correct_idx2_list) == 0 and len(correct_idx2_list_rot) == 0:
                    idx1 = random.randint(0, len(self)-1)
                    correct_idx2_list = self.nearby_points[idx1]
                    correct_idx2_list_rot = self.nearby_points_R[idx1]
                    if len(correct_idx2_list_rot) != 0 and len(correct_idx2_list) != 0:
                        if random.randint(1, 10) <= 5:
                            idx2 = random.choice(correct_idx2_list_rot)
                        else:
                            idx2 = random.choice(correct_idx2_list)
                    elif len(correct_idx2_list_rot) != 0:
                        idx2 = random.choice(correct_idx2_list_rot)
                    elif len(correct_idx2_list) != 0:
                        idx2 = random.choice(correct_idx2_list)

        if self.phase == "test":
            idx1 = idx
            idx2 = idx + 1

        poses = np.zeros((2, 4, 4), dtype=np.float64)
        timestamp = None

        for idx_now, idx_temp in enumerate([idx1, idx2]):
            image_path = self.images[idx_temp]
            depth_images_path = self.depth_images[idx_temp]
            input_image = cv2.imread(image_path)
            input_image = cv2.resize(input_image, (1216, 384), interpolation=cv2.INTER_LINEAR)
            tensors = self.images_to_tensor([input_image])
            all_2_5_images.append(torch.stack(tensors))

            poses[idx_now] = self.rpy2gtmatrix(self.poses_xyzrpy_car[idx_temp])

            if timestamp is None:
                timestamp = self.timestamps[idx_temp]

            if self.NO_DEPTH:
                depth_map = np.zeros((384, 1216))
                depth_map = torch.tensor(depth_map, dtype=torch.float)
                all_2_5_depth.append(torch.stack([depth_map]))
            else:
                depth_map = np.load(depth_images_path)
                depth_map = cv2.resize(depth_map, (1216, 384), interpolation=cv2.INTER_LINEAR)
                depth_map = torch.tensor(depth_map, dtype=torch.float)
                all_2_5_depth.append(torch.stack([depth_map]))

        all_2_5_images = torch.stack(all_2_5_images)
        all_2_5_depth = torch.stack(all_2_5_depth)

        if self.phase == "train":
            return all_2_5_images, poses, all_2_5_depth, timestamp, idx1, idx2
        else:
            return all_2_5_images, poses, all_2_5_depth, timestamp

    def rot_matrix_to_euler(self, R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return [x, y, z]

    def rpy2gtmatrix(self, pose):
        x, y, z, roll, pitch, yaw = pose
        pitch = 0
        roll = 0
        z = 0
        R = euler.euler2mat(yaw, pitch, roll, 'szyx')
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def images_to_tensor(self, images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensors = [transform(image) for image in images]
        return tensors


class KITTIDataset_sequences(Dataset):
    def __init__(self, root_dir, sequences, phase, dataset_item_range="0.0-4.0", pickle_names=["pickle.txt"], IS_MONO=True, NO_DEPTH=True):
        
        all_sequences = []
        for sequence, pickle_name in zip(sequences, pickle_names):
            ds = KITTIDataset(root_dir, sequence, phase, dataset_item_range=dataset_item_range, pickle_name=pickle_name, IS_MONO=IS_MONO, NO_DEPTH=NO_DEPTH)
            all_sequences.append(ds)
            print(f'!!!!!!more_sequences: {len(ds)}!!!!!!')

        self.dataset = ConcatDataset(all_sequences)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]