import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

from datasets.common import image_to_tensor, load_pair_index, sample_training_pair
from utils.common import xyyaw_to_matrix


class NCLTSequence(Dataset):
    """One NCLT sequence (forward camera Cam5, undistorted and resized).

    train: random frame pairs drawn from a pre-computed pair index.
    test:  consecutive pairs of frames sub-sampled with a fixed stride.
    """

    def __init__(self, root_dir, csv_path, split, pair_index_path=None, test_stride=5, rot_pair_ratio=0.7):
        assert split in ['train', 'test']
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_path, header=None)
        self.split = split
        self.rot_threshold = int(round(rot_pair_ratio * 10))
        self.image_names = sorted(os.listdir(os.path.join(root_dir, "Cam1")))

        if self.split == "train":
            self.nearby_points, self.nearby_points_rot = load_pair_index(pair_index_path)
            self.image_names_part = self.image_names[:-1]
        else:
            self.image_names_part = self.image_names[::test_stride]

    def __len__(self):
        if self.split == "train":
            return len(self.image_names_part)
        return len(self.image_names_part) - 1

    def __getitem__(self, idx):
        if self.split == "train":
            idx1, idx2 = sample_training_pair(idx, self.nearby_points, self.nearby_points_rot,
                                              len(self.image_names_part), self.rot_threshold)
            names = self.image_names
        else:
            idx1, idx2 = idx, idx + 1
            names = self.image_names_part

        images = []
        poses = np.zeros((2, 4, 4), dtype=np.float64)
        timestamp = None
        for k, idx_temp in enumerate([idx1, idx2]):
            image_name = names[idx_temp].split('.')[0]
            row = self.find_nearest_value(float(image_name)).iloc[:].tolist()
            if timestamp is None:
                timestamp = float(row[0])
            poses[k] = xyyaw_to_matrix(float(row[1]), float(row[2]), float(row[6]))
            images.append(torch.stack([image_to_tensor(self.load_image(image_name))]))

        return torch.stack(images), poses, timestamp

    def load_image(self, image_name):
        image_path = os.path.join(self.root_dir, "Cam5", f"{image_name}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        return cv2.rotate(cv2.imread(image_path), cv2.ROTATE_90_CLOCKWISE)

    def find_nearest_value(self, target_value):
        return self.df.iloc[(self.df[0] - target_value).abs().idxmin()]


class NCLTSequences(Dataset):

    def __init__(self, base_dir, dates, split, pair_index_paths=None, test_stride=5, rot_pair_ratio=0.7):
        pair_index_paths = pair_index_paths or [None] * len(dates)
        sequences = []
        for date, pair_index_path in zip(dates, pair_index_paths):
            sequences.append(NCLTSequence(
                root_dir=f"{base_dir}/{date}/lb3_u_s_384",
                csv_path=f"{base_dir}/{date}/ground_truth/groundtruth_{date}.csv",
                split=split,
                pair_index_path=pair_index_path,
                test_stride=test_stride,
                rot_pair_ratio=rot_pair_ratio,
            ))
        self.dataset = ConcatDataset(sequences)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]
