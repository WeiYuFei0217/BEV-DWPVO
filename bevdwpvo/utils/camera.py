import numpy as np
import torch


def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert B == B_
    otherdims = shapelist[2:]
    return torch.reshape(tensor, [B * S] + otherdims)


def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert BS % B == 0
    otherdims = shapelist[1:]
    S = int(BS / B)
    return torch.reshape(tensor, [B, S] + otherdims)


def safe_inverse(a):
    """Inverse of a batch of rigid transforms (B x 4 x 4)."""
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1, 2)
    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])
    return inv


def split_intrinsics(K):
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    x0 = K[:, 0, 2]
    y0 = K[:, 1, 2]
    return fx, fy, x0, y0


def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = x0
    K[:, 1, 2] = y0
    K[:, 2, 2] = 1.0
    K[:, 3, 3] = 1.0
    return K


def build_mats_dict(K, T_cam_body, batch_size):
    """Builds the camera matrices consumed by the PV-BEV encoder for a monocular camera.

    Args:
        K (np.ndarray): (3, 3) camera intrinsics.
        T_cam_body (np.ndarray): (4, 4) transform from the vehicle (body) frame to the camera frame.
        batch_size (int): number of images in the batch.
    """
    B, S = batch_size, 1
    intrins = torch.from_numpy(np.array([K])).float()
    pix_T_cams = merge_intrinsics(*split_intrinsics(intrins)).unsqueeze(0)
    cams_T_body = torch.from_numpy(np.array([T_cam_body])).unsqueeze(0).float()

    pix_T_cams = pix_T_cams.repeat(B, 1, 1, 1).cuda()
    cams_T_body = cams_T_body.repeat(B, 1, 1, 1).cuda()
    body_T_cams = unpack_seqdim(safe_inverse(pack_seqdim(cams_T_body, B)), B)
    pix_T_cams = pix_T_cams.view(B, 1, S, 4, 4)
    body_T_cams = body_T_cams.view(B, 1, S, 4, 4)
    ida_mats = torch.from_numpy(np.eye(4)).repeat(B * S, 1, 1).cuda().view(B, 1, S, 4, 4)
    bda_mat = torch.from_numpy(np.eye(4)).repeat(B, 1, 1).cuda()

    return {
        'sensor2ego_mats': body_T_cams.float(),
        'intrin_mats': pix_T_cams.float(),
        'ida_mats': ida_mats.float(),
        'bda_mat': bda_mat.float(),
    }
