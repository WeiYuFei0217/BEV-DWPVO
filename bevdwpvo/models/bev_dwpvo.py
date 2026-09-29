import torch
from torch import nn

from .pv_bev_encoder import PVBEVEncoder
from .dwp_solver import DWPSolver

__all__ = ['BEVDWPVO', 'crop_bev_half']


def crop_bev_half(x, bev_half):
    """Keeps the half of the BEV map observed by the monocular camera.

    Args:
        x (Tensor): (B, C, H, W) BEV feature map (after rotation).
        bev_half (str or None): 'front', 'rear' or None (no cropping).
    """
    if bev_half is None:
        return x
    x = x[:, :, :, x.shape[3] // 4:x.shape[3] * 3 // 4]
    if bev_half == 'front':
        return x[:, :, :x.shape[2] // 2, :]
    if bev_half == 'rear':
        return x[:, :, x.shape[2] // 2:x.shape[2], :]
    raise ValueError(f'Unknown bev_half: {bev_half}')


class BEVDWPVO(nn.Module):
    """BEV-DWPVO: PV-BEV encoder + BEV keypoint matching + differentiable weighted Procrustes solver."""

    def __init__(self, encoder_conf, solver_conf, bev_half=None, disable_validity_weights=False):
        super(BEVDWPVO, self).__init__()
        self.pv_bev_encoder = PVBEVEncoder(**encoder_conf)
        self.dwp_solver = DWPSolver(solver_conf, bev_half, disable_validity_weights)
        self.bev_half = bev_half

    def forward(self, x, mats_dict, gt_Rt=None, use_match_range=True, match_range=3.0, gkp=False):
        x = self.pv_bev_encoder(x, mats_dict)
        x = torch.rot90(x, k=1, dims=[2, 3])
        x = crop_bev_half(x, self.bev_half)
        return self.dwp_solver(x, gt_Rt=gt_Rt, use_match_range=use_match_range, match_range=match_range, gkp=gkp)
