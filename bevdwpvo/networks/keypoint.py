import torch
import torch.nn.functional as F
from utils.common import normalize_coords


class Keypoint(torch.nn.Module):
    """Extracts one candidate keypoint per block by a softmax-weighted average of pixel coordinates."""

    def __init__(self, config):
        super().__init__()
        self.patch_size = config['keypoint_patch_size']
        self.device = config['device']
        self.width = config['bev_width']
        v_coords, u_coords = torch.meshgrid([torch.arange(0, self.width), torch.arange(0, self.width)])
        self.v_coords = v_coords.unsqueeze(0).float()
        self.u_coords = u_coords.unsqueeze(0).float()

    def forward(self, position_scores, validity, descriptors):
        BW, encoder_dim, _, _ = descriptors.size()
        v_patches = F.unfold(self.v_coords.expand(BW, 1, self.width, self.width), kernel_size=self.patch_size,
                             stride=self.patch_size).to(self.device)
        u_patches = F.unfold(self.u_coords.expand(BW, 1, self.width, self.width), kernel_size=self.patch_size,
                             stride=self.patch_size).to(self.device)
        score_dim = validity.size(1)
        position_patches = F.unfold(position_scores, kernel_size=self.patch_size, stride=self.patch_size)
        softmax_attention = F.softmax(position_patches, dim=1)

        expected_v = torch.sum(v_patches * softmax_attention, dim=1)
        expected_u = torch.sum(u_patches * softmax_attention, dim=1)
        keypoint_coords = torch.stack([expected_u, expected_v], dim=2)
        num_patches = keypoint_coords.size(1)

        norm_keypoints2D = normalize_coords(keypoint_coords, self.width, self.width).unsqueeze(1)

        keypoint_desc = F.grid_sample(descriptors, norm_keypoints2D, mode='bilinear', align_corners=True)
        keypoint_desc = keypoint_desc.view(BW, encoder_dim, num_patches)

        keypoint_validity = F.grid_sample(validity, norm_keypoints2D, mode='bilinear', align_corners=True)
        keypoint_validity = keypoint_validity.view(BW, score_dim, num_patches)

        return keypoint_coords, keypoint_validity, keypoint_desc
