import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------------
# BEV coordinates
# ----------------------------------------------------------------------------
def normalize_coords(coords_2D, width, height):
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)


def get_indices(batch_size, window_size):
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        for j in range(window_size - 1):
            idx = i * window_size + j
            src_ids.append(idx)
            tgt_ids.append(idx + 1)
    return src_ids, tgt_ids


def pixel_to_metric(pixel_coords, config, bev_half=None):
    """Converts BEV pixel coordinates (u, v) to metric vehicle coordinates (x, y).

    If the BEV map was cropped to its front / rear half (see models.crop_bev_half),
    the pixel coordinates are first shifted back to the full BEV grid (in place).
    """
    bev_width = config['bev_width']
    bev_resolution = config['bev_resolution']
    device = config['device']

    if bev_half is not None:
        bev_width = bev_width * 2
        pixel_coords[:, :, 0] += bev_width // 4
        if bev_half == 'rear':
            pixel_coords[:, :, 1] += bev_width // 2

    if (bev_width % 2) == 0:
        min_range = (bev_width / 2 - 0.5) * bev_resolution
    else:
        min_range = bev_width // 2 * bev_resolution

    B, N, _ = pixel_coords.size()
    Rot = torch.tensor([[0, -bev_resolution], [bev_resolution, 0]]).expand(B, 2, 2).to(device)
    t = torch.tensor([[min_range], [-min_range]]).expand(B, 2, N).to(device)
    return (torch.bmm(Rot, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)


# ----------------------------------------------------------------------------
# Poses and evaluation
# ----------------------------------------------------------------------------
def xyyaw_to_matrix(x, y, yaw):
    """3-DoF (x, y, yaw) pose as a 4x4 matrix."""
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    T[0, 3] = x
    T[1, 3] = y
    return T


def get_inverse_tf(T):
    T2 = np.identity(4, dtype=np.float32)
    Rot = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = Rot.transpose()
    T2[0:3, 3:] = np.matmul(-1 * Rot.transpose(), t)
    return T2


def rotation_error(T):
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(T, dim=2):
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)


def compute_relative_pose_error(T_gt, T_pred, delta):
    rpe_t_error = []
    rpe_r_error = []
    for i in range(0, len(T_gt) - delta, delta):
        T_gt_rel = np.matmul(get_inverse_tf(T_gt[i]), T_gt[i + delta])
        T_pred_rel = np.matmul(get_inverse_tf(T_pred[i]), T_pred[i + delta])
        T_error_rel = np.matmul(get_inverse_tf(T_gt_rel), T_pred_rel)
        rpe_t_error.append(translation_error(T_error_rel))
        rpe_r_error.append(180 * rotation_error(T_error_rel) / np.pi)
    return rpe_t_error, rpe_r_error


def compute_median_error(T_gt, T_pred, delta=1):
    """Per-frame relative pose errors. Returns
    [median_t, std_t, median_r, std_r, mean_t, mean_r, mean_rpe_t, mean_rpe_r]."""
    t_error = []
    r_error = []
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translation_error(T_error))
        r_error.append(180 * rotation_error(T_error) / np.pi)

    t_error = np.array(t_error)
    r_error = np.array(r_error)

    rpe_t_error, rpe_r_error = compute_relative_pose_error(T_gt, T_pred, delta)
    rpe_t_error = np.array(rpe_t_error)
    rpe_r_error = np.array(rpe_r_error)

    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error),
            np.mean(t_error), np.mean(r_error), np.mean(rpe_t_error), np.mean(rpe_r_error)]


def save_tum_trajectory(file_path, timestamps, poses):
    """Accumulates relative poses and writes them in TUM format."""
    with open(file_path, 'w') as f:
        T_cumulative = np.identity(4)
        for i in range(len(timestamps)):
            timestamp = timestamps[i]
            T_cumulative = np.dot(T_cumulative, poses[i])

            translation = T_cumulative[:3, 3]
            rotation = R.from_matrix(T_cumulative[:3, :3]).as_quat()

            pose_str = f"{translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}"
            f.write(f"{timestamp.item()} {pose_str}\n")


# ----------------------------------------------------------------------------
# Timestamps
# ----------------------------------------------------------------------------
def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx - 1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx - 1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def read_ts_file(ts_filepath: str):
    with open(ts_filepath, "r") as h:
        txt_ts = h.readlines()

    ts = np.zeros((len(txt_ts),), dtype=np.int64)
    for ndx, timestamp in enumerate(txt_ts):
        temp = [e.strip() for e in timestamp.split(' ')]
        assert len(temp) == 2, f'Invalid line in timestamp file: {temp}'
        ts[ndx] = int(temp[0])
    return ts
