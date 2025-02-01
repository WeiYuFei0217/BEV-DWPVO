import pickle
import numpy as np
import torch
import torch.nn.functional as F
import csv
import os
from scipy.spatial import distance_matrix
import sys

from scipy.spatial.transform import Rotation as R

from utils import transform


# Faulty point clouds (with 0 points)
FAULTY_POINTCLOUDS = []

# Coordinates of test region centres (in Oxford sequences)
TEST_REGION_CENTRES = np.array([[5735400, 620000]])

# Radius of the test region
TEST_REGION_RADIUS = 220

# Boundary between training and test region - to ensure there's no overlap between training and test clouds
TEST_TRAIN_BOUNDARY = 50

def get_inverse_tf(T):
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2

def get_transform(x, y, theta):
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    T[0, 3] = x
    T[1, 3] = y
    return T

def get_transform2(R, t):
    T = np.identity(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.squeeze()
    return T

def enforce_orthog(T, dim=3):
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        s = np.sqrt(a**2 + b**2)
        a /= s
        b /= s
        R[0, 0] = a
        R[0, 1] = b
        R[1, 0] = -b
        R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T

def carrot(xbar):
    x = xbar.squeeze()
    if x.shape[0] == 3:
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
    elif x.shape[0] == 6:
        return np.array([[0, -x[5], x[4], x[0]],
                         [x[5], 0, -x[3], x[1]],
                         [-x[4], x[3], 0, x[2]],
                         [0, 0, 0, 1]])
    print('WARNING: attempted carrot operator on invalid vector shape')
    return xbar

def se3ToSE3(xi):
    T = np.identity(4, dtype=np.float32)
    rho = xi[0:3].reshape(3, 1)
    phibar = xi[3:6].reshape(3, 1)
    phi = np.linalg.norm(phibar)
    R = np.identity(3)
    if phi != 0:
        phibar /= phi  # normalize
        I = np.identity(3)
        R = np.cos(phi) * I + (1 - np.cos(phi)) * phibar @ phibar.T + np.sin(phi) * carrot(phibar)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * phibar @ phibar.T + \
            carrot(phibar) * (1 - np.cos(phi)) / phi
        rho = J @ rho
    T[0:3, 0:3] = R
    T[0:3, 3:] = rho
    return T

def SE3tose3(T):
    R = T[0:3, 0:3]
    evals, evecs = np.linalg.eig(R)
    idx = -1
    for i in range(3):
        if evals[i].real != 0 and evals[i].imag == 0:
            idx = i
            break
    assert(idx != -1)
    abar = evecs[idx].real.reshape(3, 1)
    phi = np.arccos((np.trace(R) - 1) / 2)
    rho = T[0:3, 3:]
    if phi != 0:
        I = np.identity(3)
        J = I * np.sin(phi) / phi + (1 - np.sin(phi) / phi) * abar @ abar.T + \
            carrot(abar) * (1 - np.cos(phi)) / phi
        rho = np.linalg.inv(J) @ rho
    xi = np.zeros((6, 1))
    xi[0:3, 0:] = rho
    xi[3:, 0:] = phi * abar
    return xi

def rotationError(T):
    d = 0.5 * (np.trace(T[0:3, 0:3]) - 1)
    return np.arccos(max(min(d, 1.0), -1.0))

def translationError(T, dim=2):
    if dim == 2:
        return np.sqrt(T[0, 3]**2 + T[1, 3]**2)
    return np.sqrt(T[0, 3]**2 + T[1, 3]**2 + T[2, 3]**2)

def computeRelativePoseError(T_gt, T_pred, delta):
    rpe_t_error = []
    rpe_r_error = []
    
    for i in range(0, len(T_gt) - delta, delta):
        T_gt_rel = np.matmul(get_inverse_tf(T_gt[i]), T_gt[i + delta])
        T_pred_rel = np.matmul(get_inverse_tf(T_pred[i]), T_pred[i + delta])
        
        T_error_rel = np.matmul(get_inverse_tf(T_gt_rel), T_pred_rel)
        
        rpe_t_error.append(translationError(T_error_rel))
        rpe_r_error.append(180 * rotationError(T_error_rel) / np.pi)
    
    return rpe_t_error, rpe_r_error

def computeMedianError(T_gt, T_pred, delta=1):
    t_error = []
    r_error = []
    
    for i, T in enumerate(T_gt):
        T_error = np.matmul(T, get_inverse_tf(T_pred[i]))
        t_error.append(translationError(T_error))
        r_error.append(180 * rotationError(T_error) / np.pi)
    
    t_error = np.array(t_error)
    r_error = np.array(r_error)
    
    rpe_t_error, rpe_r_error = computeRelativePoseError(T_gt, T_pred, delta)
    rpe_t_error = np.array(rpe_t_error)
    rpe_r_error = np.array(rpe_r_error)
    
    return [np.median(t_error), np.std(t_error), np.median(r_error), np.std(r_error),
            np.mean(t_error), np.mean(r_error), np.mean(rpe_t_error), np.mean(rpe_r_error)]

def trajectoryDistances(poses):
    dist = [0]
    for i in range(1, len(poses)):
        P1 = get_inverse_tf(poses[i - 1])
        P2 = get_inverse_tf(poses[i])
        dx = P1[0, 3] - P2[0, 3]
        dy = P1[1, 3] - P2[1, 3]
        dist.append(dist[i-1] + np.sqrt(dx**2 + dy**2))
    return dist

def lastFrameFromSegmentLength(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1

def calcSequenceErrors(poses_gt, poses_pred):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    err = []
    step_size = 4  # Every second
    # Pre-compute distances from ground truth as reference
    dist = trajectoryDistances(poses_gt)

    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, length)
            if last_frame == -1:
                continue
            # Compute rotational and translation errors
            pose_delta_gt = np.matmul(poses_gt[last_frame], get_inverse_tf(poses_gt[first_frame]))
            pose_delta_res = np.matmul(poses_pred[last_frame], get_inverse_tf(poses_pred[first_frame]))
            pose_error = np.matmul(pose_delta_gt, get_inverse_tf(pose_delta_res))
            r_err = rotationError(pose_error)
            t_err = translationError(pose_error)
            # Approx speed
            num_frames = float(last_frame - first_frame + 1)
            speed = float(length) / (0.25 * num_frames)
            err.append([first_frame, r_err/float(length), t_err/float(length), length, speed])
    return err

def getStats(err):
    t_err = 0
    r_err = 0
    for e in err:
        t_err += e[2]
        r_err += e[1]
    t_err /= float(len(err))
    r_err /= float(len(err))
    return t_err, r_err

def computeKittiMetrics(T_gt, T_pred, seq_lens):
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)
    err_list = []
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        poses_gt = []
        poses_pred = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            poses_gt.append(T_gt_)
            poses_pred.append(T_pred_)
        err = calcSequenceErrors(poses_gt, poses_pred)
        t_err, r_err = getStats(err)
        err_list.append([t_err, r_err])
    err_list = np.asarray(err_list)
    avg = np.mean(err_list, axis=0)
    t_err = avg[0]
    r_err = avg[1]
    return t_err * 100, r_err * 180 / np.pi

def saveKittiErrors(err, fname):
    pickle.dump(err, open(fname, 'wb'))

def loadKittiErrors(fname):
    return pickle.load(open(fname, 'rb'))

def save_in_yeti_format(T_gt, T_pred, timestamps, seq_lens, seq_names, root='./'):
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    for s, indices in enumerate(seq_indices):
        fname = root + 'accuracy' + seq_names[s] + '.csv'
        with open(fname, 'w') as f:
            f.write('x,y,yaw,gtx,gty,gtyaw,time1,time2\n')
            for i in indices:
                R_pred = T_pred[i][:3, :3]
                t_pred = T_pred[i][:3, 3:]
                yaw = -1 * np.arcsin(R_pred[0, 1])
                gtyaw = -1 * np.arcsin(T_gt[i][0, 1])
                t = np.matmul(-1 * R_pred.transpose(), np.reshape(t_pred, (3, 1)))
                T = get_inverse_tf(T_gt[i])
                f.write('{},{},{},{},{},{},{},{}\n'.format(t[0, 0], t[1, 0], yaw, T[0, 3], T[1, 3], gtyaw,
                                                           timestamps[i][0], timestamps[i][1]))

def load_icra21_results(results_loc, seq_names, seq_lens):
    T_icra = []
    for i, seq_name in enumerate(seq_names):
        fname = results_loc + 'accuracy' + seq_name + '.csv'
        with open(fname, 'r') as f:
            f.readline()  # Clear out the header
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(',')
                # Retrieve the transform estimated by MC-RANSAC + DOPPLER compensation
                T_icra.append(get_inverse_tf(get_transform(float(line[11]), float(line[12]), float(line[13]))))
                count += 1
            # Append identity transforms at the end in case the ICRA results ended early by a couple frames
            if count < seq_lens[i]:
                print('WARNING: ICRA results shorter than seq_len by {}. Append last TF.'.format((seq_lens[i] - count)))
            while count < seq_lens[i]:
                T_icra.append(T_icra[-1])
                count += 1
    return T_icra

def normalize_coords(coords_2D, width, height):
    batch_size = coords_2D.size(0)
    u_norm = (2 * coords_2D[:, :, 0].reshape(batch_size, -1) / (width - 1)) - 1
    v_norm = (2 * coords_2D[:, :, 1].reshape(batch_size, -1) / (height - 1)) - 1
    return torch.stack([u_norm, v_norm], dim=2)  # BW x num_patches x 2

def convert_to_radar_frame(pixel_coords, config):
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    # print("B,N,R.shape,t.shape:")
    # print(B,N,R.shape,t.shape)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

def convert_to_radar_frame_mono_cut(pixel_coords, config, dataset_type="NCLT"):
    cart_pixel_width = config['cart_pixel_width'] * 2
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']

    width = cart_pixel_width
    height = cart_pixel_width

    if dataset_type == 'NCLT' or dataset_type == 'kitti':
        pixel_coords[:, :, 0] += width // 4
    elif dataset_type == "oxford":
        pixel_coords[:, :, 0] += width // 4
        pixel_coords[:, :, 1] += height // 2

    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)

    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

def get_indices(batch_size, window_size):
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        for j in range(window_size - 1):
            idx = i * window_size + j
            src_ids.append(idx)
            tgt_ids.append(idx + 1)
    return src_ids, tgt_ids

def get_indices2(batch_size, window_size, asTensor=False):
    src_ids = []
    tgt_ids = []
    for i in range(batch_size):
        idx = i * window_size
        for j in range(idx + 1, idx + window_size):
            tgt_ids.append(j)
            src_ids.append(idx)
    if asTensor:
        src_ids = np.asarray(src_ids, dtype=np.int64)
        tgt_ids = np.asarray(tgt_ids, dtype=np.int64)
        return torch.from_numpy(src_ids), torch.from_numpy(tgt_ids)
    return src_ids, tgt_ids

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_T_ba(out, a, b):
    T_b0 = np.eye(4)
    T_b0[:3, :3] = out['R'][0, b].detach().cpu().numpy()
    T_b0[:3, 3:4] = out['t'][0, b].detach().cpu().numpy()
    T_a0 = np.eye(4)
    T_a0[:3, :3] = out['R'][0, a].detach().cpu().numpy()
    T_a0[:3, 3:4] = out['t'][0, a].detach().cpu().numpy()
    return np.matmul(T_b0, get_inverse_tf(T_a0))

def convert_to_weight_matrix(w, window_id, T_aug=[]):
    z_weight = 9.2103  # 9.2103 = log(1e4), 1e4 is inverse variance of 1cm std dev
    if w.size(1) == 1:
        # scalar weight
        A = torch.zeros(w.size(0), 9, device=w.device)
        A[:, (0, 4)] = torch.exp(w)
        A[:, 8] = torch.exp(torch.tensor(z_weight))
        A = A.reshape((-1, 3, 3))
        d = torch.zeros(w.size(0), 3, device=w.device)
        d[:, 0:2] += w
        d[:, 2] += z_weight
    elif w.size(1) == 3:
        # 2x2 matrix
        L = torch.zeros(w.size(0), 4, device=w.device)
        L[:, (0, 3)] = 1
        L[:, 2] = w[:, 0]
        L = L.reshape((-1, 2, 2))
        D = torch.zeros(w.size(0), 4, device=w.device)
        D[:, (0, 3)] = torch.exp(w[:, 1:])
        D = D.reshape((-1, 2, 2))
        A2x2 = L @ D @ L.transpose(1, 2)

        if T_aug:  # if list is not empty
            Rot = T_aug[window_id].to(w.device)[:2, :2].unsqueeze(0)
            A2x2 = Rot.transpose(1, 2) @ A2x2 @ Rot

        A = torch.zeros(w.size(0), 3, 3, device=w.device)
        A[:, 0:2, 0:2] = A2x2
        A[:, 2, 2] = torch.exp(torch.tensor(z_weight))
        d = torch.ones(w.size(0), 3, device=w.device)*z_weight
        d[:, 0:2] = w[:, 1:]
    else:
        assert False, "Weight scores should be dim 1 or 3"

    return A, d

def mask_intensity_filter(data, patch_size, patch_mean_thres=0.05):
    int_patches = F.unfold(data, kernel_size=patch_size, stride=patch_size)
    keypoint_int = torch.mean(int_patches, dim=1, keepdim=True)  # BW x 1 x num_patches
    return keypoint_int >= patch_mean_thres

def wrapto2pi(phi):
    if phi < 0:
        return phi + 2 * np.pi * np.ceil(phi / (-2 * np.pi))
    elif phi >= 2 * np.pi:
        return (phi / (2 * np.pi) % 1) * 2 * np.pi
    return phi

def getApproxTimeStamps(points, times, flip_y=False):
    azimuth_step = (2 * np.pi) / 400
    timestamps = []
    for i, p in enumerate(points):
        p = points[i]
        ptimes = times[i]
        delta_t = ptimes[-1] - ptimes[-2]
        ptimes = np.append(ptimes, int(ptimes[-1] + delta_t))
        point_times = []
        for k in range(p.shape[0]):
            x = p[k, 0]
            y = p[k, 1]
            if flip_y:
                y *= -1
            phi = np.arctan2(y, x)
            phi = wrapto2pi(phi)
            time_idx = phi / azimuth_step
            t1 = ptimes[int(np.floor(time_idx))]
            t2 = ptimes[int(np.ceil(time_idx))]
            # interpolate to get slightly more precise timestamp
            ratio = time_idx % 1
            t = int(t1 + ratio * (t2 - t1))
            point_times.append(t)
        timestamps.append(np.array(point_times))
    return timestamps

def undistort_pointcloud(points, point_times, t_refs, solver):
    for i, p in enumerate(points):
        p = points[i]
        ptimes = point_times[i]
        t_ref = t_refs[i]
        for j, ptime in enumerate(ptimes):
            T_0a = np.identity(4, dtype=np.float32)
            solver.getPoseBetweenTimes(T_0a, ptime, t_ref)
            pbar = T_0a @ p[j].reshape(4, 1)
            p[j, :] = pbar[:]
        points[i] = p
    return points

def save_tum_trajectory(file_path, timestamps, poses):
    with open(file_path, 'w') as f:
        T_cumulative = np.identity(4)

        for i in range(len(timestamps)):
            timestamp = timestamps[i]
            T = poses[i]

            T_cumulative = np.dot(T_cumulative, T)

            translation = T_cumulative[:3, 3]
            rotation_matrix = T_cumulative[:3, :3]
            rotation = R.from_matrix(rotation_matrix).as_quat()

            pose_str = f"{translation[0]} {translation[1]} {translation[2]} {rotation[0]} {rotation[1]} {rotation[2]} {rotation[3]}"

            f.write(f"{timestamp.item()} {pose_str}\n")



# utils_oxford

def in_train_split(pos):
    # returns true if pos is in train split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist > TEST_REGION_RADIUS + TEST_TRAIN_BOUNDARY).all(axis=1)
    return mask


def in_test_split(pos):
    # returns true if position is in evaluation split
    assert pos.ndim == 2
    assert pos.shape[1] == 2
    dist = distance_matrix(pos, TEST_REGION_CENTRES)
    mask = (dist < TEST_REGION_RADIUS).any(axis=1)
    return mask


def find_nearest_ndx(ts, timestamps):
    ndx = np.searchsorted(timestamps, ts)
    if ndx == 0:
        return ndx
    elif ndx == len(timestamps):
        return ndx - 1
    else:
        assert timestamps[ndx-1] <= ts <= timestamps[ndx]
        if ts - timestamps[ndx-1] < timestamps[ndx] - ts:
            return ndx - 1
        else:
            return ndx


def read_ts_file(ts_filepath: str):
    with open(ts_filepath, "r") as h:
        txt_ts = h.readlines()

    n = len(txt_ts)
    ts = np.zeros((n,), dtype=np.int64)

    for ndx, timestamp in enumerate(txt_ts):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in timestamp.split(' ')]
        assert len(temp) == 2, f'Invalid line in timestamp file: {temp}'

        ts[ndx] = int(temp[0])

    return ts


def read_lidar_poses(poses_filepath: str, left_lidar_filepath: str, pose_time_tolerance: float = 1.):
    # Read global poses from .csv file and link each lidar_scan with the nearest pose
    # threshold: threshold in seconds
    # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)    

    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    print("pose num:", n)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)       # 4x4 pose matrix

    for ndx, pose in enumerate(txt_poses):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in pose.split(',')]
        #print(temp)
        if ndx == 0:
            continue
        ndx -= 1
        #print("?:", ndx)
        assert len(temp) == 15, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = RPY2Rot(float(temp[5]), float(temp[6]), float(temp[7]), float(temp[12]), float(temp[13]), float(temp[14]))

    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]

    # List LiDAR scan timestamps
    left_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(left_lidar_filepath) if
                            os.path.splitext(f)[1] == '.bin']
    left_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(left_lidar_timestamps):
        # Skip faulty point clouds
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        # Find index of the closest timestamp
        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta > pose_time_tolerance * 10000000:  # 之前是1s容差，改成0.01s
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])
        #print(poses[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)     # (northing, easting) position

    print(f'{len(lidar_timestamps)} scans with valid pose, {count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses


def read_lidar_poses_RPY(poses_filepath: str, left_lidar_filepath: str, pose_time_tolerance: float = 1.):
    # Read global poses from .csv file and link each lidar_scan with the nearest pose
    # threshold: threshold in seconds
    # Returns a dictionary with (4, 4) pose matrix indexed by a timestamp (as integer)    

    with open(poses_filepath, "r") as h:
        txt_poses = h.readlines()

    n = len(txt_poses)
    print("pose num:", n)
    system_timestamps = np.zeros((n,), dtype=np.int64)
    poses = np.zeros((n, 4, 4), dtype=np.float64)       # 4x4 pose matrix
    poses_RPY = np.zeros((n, 6), dtype=np.float64)       # 6 pose matrix

    for ndx, pose in enumerate(txt_poses):
        # Split by comma and remove whitespaces
        temp = [e.strip() for e in pose.split(',')]
        #print(temp)
        if ndx == 0:
            continue
        ndx -= 1
        #print("?:", ndx)
        assert len(temp) == 15, f'Invalid line in global poses file: {temp}'
        system_timestamps[ndx] = int(temp[0])
        poses[ndx] = RPY2Rot(float(temp[5]), float(temp[6]), float(temp[7]), float(temp[12]), float(temp[13]), float(temp[14]))
        poses_RPY[ndx] = [float(temp[5]), float(temp[6]), float(temp[7]), float(temp[12]), float(temp[13]), float(temp[14])]

    # Ensure timestamps and poses are sorted in ascending order
    sorted_ndx = np.argsort(system_timestamps, axis=0)
    system_timestamps = system_timestamps[sorted_ndx]
    poses = poses[sorted_ndx]
    poses_RPY = poses_RPY[sorted_ndx]

    # List LiDAR scan timestamps
    left_lidar_timestamps = [int(os.path.splitext(f)[0]) for f in os.listdir(left_lidar_filepath) if
                            os.path.splitext(f)[1] == '.bin']
    left_lidar_timestamps.sort()

    lidar_timestamps = []
    lidar_poses = []
    lidar_poses_RPY = []
    count_rejected = 0

    for ndx, lidar_ts in enumerate(left_lidar_timestamps):
        # Skip faulty point clouds
        if lidar_ts in FAULTY_POINTCLOUDS:
            continue

        # Find index of the closest timestamp
        closest_ts_ndx = find_nearest_ndx(lidar_ts, system_timestamps)
        delta = abs(system_timestamps[closest_ts_ndx] - lidar_ts)
        # Timestamp is in nanoseconds = 1e-9 second
        if delta > pose_time_tolerance * 10000000:  # 之前是1s容差，改成0.01s
            # Reject point cloud without corresponding pose
            count_rejected += 1
            continue

        lidar_timestamps.append(lidar_ts)
        lidar_poses.append(poses[closest_ts_ndx])
        lidar_poses_RPY.append(poses_RPY[closest_ts_ndx])
        #print(poses[closest_ts_ndx])

    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)
    lidar_poses = np.array(lidar_poses, dtype=np.float64)     # (northing, easting) position
    lidar_poses_RPY = np.array(lidar_poses_RPY, dtype=np.float64)     # (northing, easting) position

    print(f'{len(lidar_timestamps)} scans with valid pose, {count_rejected} rejected due to unknown pose')
    return lidar_timestamps, lidar_poses, lidar_poses_RPY


def relative_pose(m1, m2):
    m = np.linalg.inv(m2) @ m1
    return m


def RPY2Rot(x, y, z, roll, pitch, yaw):
    T = np.identity(4, dtype=np.float32)
    T[0:2, 0:2] = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    T[0, 3] = x
    T[1, 3] = y
    return T


# apply random rotation along z axis
def random_rotation(xyz, angle_range=(-np.pi, np.pi)):
    angle = np.random.uniform(*angle_range)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]]).transpose()
    return np.dot(xyz, rotation_matrix)


# get the 4x4 SE(3) transformation matrix from euler angles
def euler2se3(x, y, z, roll, pitch, yaw):
    se3 = np.eye(4, dtype=np.float64)   
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    se3[:3, :3] = R
    se3[:3, 3] = np.array([x, y, z])

    return se3


# apply 4x4 SE(3) transformation matrix on (N, 3) point cloud or 3x3 transformation on (N, 2) point cloud
def apply_transform(pc: torch.Tensor, m: torch.Tensor):
    assert pc.ndim == 2
    n_dim = pc.shape[1]
    assert n_dim == 2 or n_dim == 3
    assert m.shape == (n_dim + 1, n_dim + 1)
    # (m @ pc.t).t = pc @ m.t
    pc = pc @ m[:n_dim, :n_dim].transpose(1, 0) + m[:n_dim, -1]

    return pc