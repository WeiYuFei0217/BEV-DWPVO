"""Builds models, camera matrices and datasets from a BEV-DWPVO config file."""
import copy
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import (KITTI_K, KITTI_T_CAM_BODY, KITTISequences, NCLTSequences, OxfordSequences)
from models import BEVDWPVO
from utils.camera import build_mats_dict

REPO_DIR = Path(__file__).resolve().parent.parent

# BEV grids used in the paper: (grid size, resolution) -> LSS bounds.
GRID_PRESETS = {
    (256, 0.4): dict(x_bound=[-51.2, 51.2, 0.4], y_bound=[-51.2, 51.2, 0.4], z_bound=[-5, 3, 8], d_bound=[2.0, 58, 0.5]),
    (256, 0.2): dict(x_bound=[-25.6, 25.6, 0.2], y_bound=[-25.6, 25.6, 0.2], z_bound=[-5, 3, 8], d_bound=[2.0, 30, 0.25]),
    (128, 0.8): dict(x_bound=[-51.2, 51.2, 0.8], y_bound=[-51.2, 51.2, 0.8], z_bound=[-5, 3, 8], d_bound=[2.0, 58, 0.5]),
}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def repo_path(path):
    """Resolves a path given relative to the repository root."""
    return path if os.path.isabs(path) else str(REPO_DIR / path)


def encoder_conf(cfg):
    model_cfg = cfg['model']
    key = (model_cfg['bev_size'], model_cfg['bev_resolution'])
    if key not in GRID_PRESETS:
        raise ValueError(f'Unsupported BEV grid {key}, choose from {list(GRID_PRESETS)}')
    conf = copy.deepcopy(model_cfg['encoder'])
    conf['final_dim'] = tuple(conf['final_dim'])
    conf.update(copy.deepcopy(GRID_PRESETS[key]))
    return conf


def solver_conf(cfg):
    model_cfg = cfg['model']
    conf = copy.deepcopy(model_cfg['solver'])
    bev_width = model_cfg['bev_size']
    if model_cfg.get('bev_half') is not None:
        bev_width = bev_width // 2
    conf['bev_width'] = bev_width
    conf['bev_resolution'] = model_cfg['bev_resolution']
    conf['in_channels'] = model_cfg['encoder']['output_channels']
    return conf


def build_model(cfg):
    return BEVDWPVO(encoder_conf(cfg), solver_conf(cfg),
                    bev_half=cfg['model'].get('bev_half'),
                    disable_validity_weights=cfg['train'].get('disable_validity_weights', False))


def load_weights(model, path):
    """Loads either a released weight file (plain state_dict) or a training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def camera_params(cfg):
    """Returns (K, T_cam_body) of the monocular camera used for the dataset."""
    name, root = cfg['dataset']['name'], cfg['dataset']['data_root']
    if name == 'kitti':
        return KITTI_K, KITTI_T_CAM_BODY
    if name == 'NCLT':
        meta_path, cam = os.path.join(root, 'format_data', 'image_meta.pkl'), -1   # forward camera (Cam5)
    elif name == 'oxford':
        meta_path, cam = os.path.join(root, 'image_meta.pkl'), 2                   # rear camera
    else:
        raise ValueError(f'Unknown dataset: {name}')
    with open(meta_path, 'rb') as f:
        image_meta = pickle.load(f)
    return np.array(image_meta['K'][cam]), np.array(image_meta['T'][cam])


def build_mats(cfg, batch_size):
    K, T_cam_body = camera_params(cfg)
    return build_mats_dict(K, T_cam_body, batch_size)


def _pair_index_paths(cfg):
    ds = cfg['dataset']
    return [os.path.join(repo_path(ds['pair_index_dir']), seq, ds['pair_index_file']) for seq in ds['train_sequences']]


def build_train_set(cfg):
    ds = cfg['dataset']
    name, root = ds['name'], ds['data_root']
    paths = _pair_index_paths(cfg)
    if name == 'NCLT':
        return NCLTSequences(os.path.join(root, 'format_data'), ds['train_sequences'], 'train', paths,
                             rot_pair_ratio=ds['rot_pair_ratio'])
    if name == 'oxford':
        return OxfordSequences(root, ds['train_sequences'], 'train', paths, rot_pair_ratio=ds['rot_pair_ratio'])
    if name == 'kitti':
        return KITTISequences(root, ds['train_sequences'], 'train', paths, rot_pair_ratio=ds['rot_pair_ratio'])
    raise ValueError(f'Unknown dataset: {name}')


def build_test_sets(cfg):
    """Returns a list of (sequence name, dataset) for evaluation."""
    ds = cfg['dataset']
    name, root, stride = ds['name'], ds['data_root'], ds['test_stride']
    test_sets = []
    for seq in ds['test_sequences']:
        if name == 'NCLT':
            test_sets.append((seq, NCLTSequences(os.path.join(root, 'format_data'), [seq], 'test', test_stride=stride)))
        elif name == 'oxford':
            test_sets.append((seq, OxfordSequences(root, [seq], 'test', test_stride=stride)))
        elif name == 'kitti':
            test_sets.append((seq, KITTISequences(root, [seq], 'test', test_strides=stride)))
        else:
            raise ValueError(f'Unknown dataset: {name}')
    return test_sets
