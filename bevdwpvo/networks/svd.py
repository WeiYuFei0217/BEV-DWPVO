import torch
import torch.nn.functional as F
from utils.common import pixel_to_metric


class SVD(torch.nn.Module):
    """Differentiable weighted Procrustes solver (weighted SVD)."""

    def __init__(self, config, bev_half=None):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.bev_half = bev_half

    def compute_weighted_svd(self, src_coords, tgt_coords, weights, B, gt_Rt):
        w = torch.sum(weights, dim=2, keepdim=True) + 1e-4
        src_centroid = torch.sum(src_coords * weights, dim=2, keepdim=True) / w
        tgt_centroid = torch.sum(tgt_coords * weights, dim=2, keepdim=True) / w

        src_centered = src_coords - src_centroid
        tgt_centered = tgt_coords - tgt_centroid

        W = torch.bmm(tgt_centered * weights, src_centered.transpose(2, 1)) / w

        try:
            U, _, V = torch.svd(W)
        except RuntimeError:
            U, _, V = torch.svd(W + 1e-4 * W.mean() * torch.rand(1, 3).to(self.device))

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(B, 2).type_as(V)
        S = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))

        R_tgt_src = torch.bmm(U, torch.bmm(S, V.transpose(2, 1)))

        # Guided Convergence for Translation (GCT): use the ground-truth rotation to solve
        # the translation during the first training steps.
        R_for_t = gt_Rt[:, :3, :3].float() if gt_Rt is not None else R_tgt_src
        t_tgt_src_insrc = src_centroid - torch.bmm(R_for_t.transpose(2, 1), tgt_centroid)
        t_src_tgt_intgt = -R_for_t.bmm(t_tgt_src_insrc)

        return R_tgt_src, t_src_tgt_intgt

    def forward(self, src_coords, tgt_coords, weights, gt_Rt=None):
        assert src_coords.size() == tgt_coords.size()
        B = src_coords.size(0)

        src_coords = pixel_to_metric(src_coords, self.config, self.bev_half)
        tgt_coords = pixel_to_metric(tgt_coords, self.config, self.bev_half)

        if src_coords.size(2) < 3:
            pad = 3 - src_coords.size(2)
            src_coords = F.pad(src_coords, [0, pad, 0, 0])
        if tgt_coords.size(2) < 3:
            pad = 3 - tgt_coords.size(2)
            tgt_coords = F.pad(tgt_coords, [0, pad, 0, 0])

        src_coords = src_coords.transpose(2, 1)
        tgt_coords = tgt_coords.transpose(2, 1)
        return self.compute_weighted_svd(src_coords, tgt_coords, weights, B, gt_Rt)
