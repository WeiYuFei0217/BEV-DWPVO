import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import get_indices, normalize_coords


class SoftmaxMatcher(nn.Module):
    """Matches keypoints of the first BEV map against the dense descriptors of the second one."""

    def __init__(self, config):
        super().__init__()
        self.softmax_temp = config['softmax_temp']
        self.window_size = config['window_size']
        self.device = config['device']
        self.score_comp = config['score_comp']
        self.bev_resolution = config['bev_resolution']

    def forward(self, keypoint_validity, keypoint_desc, validity_dense, desc_dense, keypoint_coords,
                use_match_range=True, match_range=3.0):
        BW, encoder_dim, n_points = keypoint_desc.size()
        batch_size = int(BW / self.window_size)
        _, _, height, width = desc_dense.size()
        kp_inds, dense_inds = get_indices(batch_size, self.window_size)
        src_coords = keypoint_coords[kp_inds]

        src_desc = keypoint_desc[kp_inds]
        src_desc = F.normalize(src_desc, dim=1)
        B = src_desc.size(0)

        tgt_desc_dense = desc_dense[dense_inds]
        tgt_desc_unrolled = F.normalize(tgt_desc_dense.view(B, encoder_dim, -1), dim=1)

        match_vals = torch.matmul(src_desc.transpose(2, 1), tgt_desc_unrolled)

        if use_match_range:
            keep_range = int(match_range // self.bev_resolution + 1)
            match_vals_bnhw = match_vals.view(B, n_points, height, width)
            mask = torch.zeros_like(match_vals_bnhw)
            for i in range(B):
                for j in range(n_points):
                    u, v = src_coords[i, j].int()
                    mask[i, j, max(0, v - keep_range):min(height, v + keep_range + 1),
                         max(0, u - keep_range):min(width, u + keep_range + 1)] = 1
            match_vals = match_vals * mask.view(B, n_points, height * width)

        soft_match_vals = F.softmax(match_vals / self.softmax_temp, dim=2)

        v_coord, u_coord = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        v_coord = v_coord.reshape(height * width).float()
        u_coord = u_coord.reshape(height * width).float()
        coords = torch.stack((u_coord, v_coord), dim=1)
        tgt_coords_dense = coords.unsqueeze(0).expand(B, height * width, 2).to(self.device)

        pseudo_coords = torch.matmul(tgt_coords_dense.transpose(2, 1),
                                     soft_match_vals.transpose(2, 1)).transpose(2, 1)

        pseudo_norm = normalize_coords(pseudo_coords, height, width).unsqueeze(1)
        tgt_validity_dense = validity_dense[dense_inds]

        pseudo_validity = F.grid_sample(tgt_validity_dense, pseudo_norm, mode='bilinear')
        pseudo_validity = pseudo_validity.reshape(B, 1, n_points)

        pseudo_desc = F.grid_sample(tgt_desc_dense, pseudo_norm, mode='bilinear')
        pseudo_desc = pseudo_desc.reshape(B, encoder_dim, n_points)

        desc_match_score = torch.sum(src_desc * pseudo_desc, dim=1, keepdim=True) / float(encoder_dim)

        src_validity = keypoint_validity[kp_inds]

        if self.score_comp:
            match_weights = 0.5 * (desc_match_score + 1) * src_validity * pseudo_validity
        else:
            match_weights = src_validity

        return pseudo_coords, match_weights, kp_inds
