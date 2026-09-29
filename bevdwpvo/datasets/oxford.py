import os
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from datasets.common import image_to_tensor, load_pair_index, sample_training_pair
from utils.common import find_nearest_ndx, read_ts_file, xyyaw_to_matrix

# Maximum time offset (ns) between a frame and its INS pose.
POSE_TIME_TOLERANCE_NS = 10_000_000


def read_frame_poses(ins_file: str, frame_dir: str):
    """Reads the INS poses and associates them with the frame timestamps.

    Frames are indexed by the timestamps of the ``velodyne_left`` scans, the rear camera
    image closest in time is used for each frame. Frames without an INS pose within
    POSE_TIME_TOLERANCE_NS are discarded.

    Returns:
        frame_timestamps (np.ndarray): (N,) int64 timestamps in ns.
        frame_poses (np.ndarray): (N, 4, 4) 3-DoF (northing, easting, yaw) poses.
    """
    with open(ins_file, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    ins_timestamps = np.zeros((n,), dtype=np.int64)
    ins_poses = np.zeros((n, 4, 4), dtype=np.float64)
    for ndx, line in enumerate(txt_poses):
        if ndx == 0:  # header
            continue
        temp = [e.strip() for e in line.split(',')]
        assert len(temp) == 15, f'Invalid line in INS file: {temp}'
        ins_timestamps[ndx - 1] = int(temp[0])
        ins_poses[ndx - 1] = xyyaw_to_matrix(float(temp[5]), float(temp[6]), float(temp[14]))

    sorted_ndx = np.argsort(ins_timestamps, axis=0)
    ins_timestamps = ins_timestamps[sorted_ndx]
    ins_poses = ins_poses[sorted_ndx]

    frame_ts_all = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(frame_dir)
                          if os.path.splitext(f)[1] == '.bin')

    frame_timestamps = []
    frame_poses = []
    for ts in frame_ts_all:
        closest = find_nearest_ndx(ts, ins_timestamps)
        if abs(ins_timestamps[closest] - ts) > POSE_TIME_TOLERANCE_NS:
            continue
        frame_timestamps.append(ts)
        frame_poses.append(ins_poses[closest])

    return np.array(frame_timestamps, dtype=np.int64), np.array(frame_poses, dtype=np.float64)


class OxfordSequence(Dataset):
    """One Oxford Radar RobotCar sequence (rectified rear monocular camera).

    train: random frame pairs drawn from a pre-computed pair index.
    test:  consecutive pairs of frames sub-sampled with a fixed stride.
    """

    def __init__(self, dataset_root: str, sequence_name: str, split: str, pair_index_path=None,
                 test_stride=8, rot_pair_ratio=0.7):
        assert os.path.exists(dataset_root), f'Cannot access dataset root: {dataset_root}'
        assert split in ['train', 'test']
        self.dataset_root = dataset_root
        self.split = split
        self.rot_threshold = int(round(rot_pair_ratio * 10))

        sequence_path = os.path.join(dataset_root, sequence_name)
        assert os.path.exists(sequence_path), f'Cannot access sequence: {sequence_path}'
        self.mono_rear_ts = read_ts_file(os.path.join(sequence_path, 'mono_rear.timestamps'))

        ins_file = os.path.join(sequence_path, 'gps/ins.csv')
        assert os.path.exists(ins_file), f'Cannot access ground truth file: {ins_file}'
        frame_rel_dir = os.path.join(sequence_name, 'velodyne_left')
        frame_dir = os.path.join(dataset_root, frame_rel_dir)
        assert os.path.exists(frame_dir), f'Cannot access frame index directory: {frame_dir}'

        self.timestamps, self.poses = read_frame_poses(ins_file, frame_dir)
        self.rel_frame_paths = [os.path.join(frame_rel_dir, str(e) + '.bin') for e in self.timestamps]

        if self.split == "train":
            self.nearby_points, self.nearby_points_rot = load_pair_index(pair_index_path)
        else:
            self.rel_frame_paths = self.rel_frame_paths[::test_stride]
            self.timestamps = self.timestamps[::test_stride]
            self.poses = self.poses[::test_stride]

        assert len(self.timestamps) == len(self.poses) == len(self.rel_frame_paths)
        print(f'{len(self.timestamps)} frames in {sequence_name}-{split}')

    def __len__(self):
        if self.split == "train":
            return len(self.rel_frame_paths)
        return len(self.rel_frame_paths) - 1

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
            image = cv2.cvtColor(cv2.imread(self.get_image_path(idx_temp)), cv2.COLOR_BGR2RGB)
            images.append(torch.stack([image_to_tensor(image)]))
            if timestamp is None:
                timestamp = float(self.timestamps[idx_temp])
            poses[k] = self.poses[idx_temp]

        return torch.stack(images), poses, timestamp

    def get_image_path(self, ndx):
        frame_path = os.path.join(self.dataset_root, self.rel_frame_paths[ndx])
        ts = self.timestamps[ndx]
        rear_ts = self.mono_rear_ts[find_nearest_ndx(ts, self.mono_rear_ts)]
        image_path = frame_path.replace('velodyne_left', 'mono_rear_rect').replace('bin', 'png')
        image_path = image_path.replace(str(ts), str(rear_ts))
        assert os.path.exists(image_path), f'Cannot access image file: {image_path}'
        return image_path


class OxfordSequences(Dataset):

    def __init__(self, dataset_root: str, sequence_names: List[str], split: str, pair_index_paths=None,
                 test_stride=8, rot_pair_ratio=0.7):
        pair_index_paths = pair_index_paths or [None] * len(sequence_names)
        sequences = [OxfordSequence(dataset_root, name, split, path, test_stride, rot_pair_ratio)
                     for name, path in zip(sequence_names, pair_index_paths)]
        self.dataset = ConcatDataset(sequences)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ndx):
        return self.dataset[ndx]
