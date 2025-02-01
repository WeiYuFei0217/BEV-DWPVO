import torch
import torch.nn.functional as F
from utils.utils import convert_to_radar_frame, get_indices, convert_to_radar_frame_mono_cut

class SVD(torch.nn.Module):
    def __init__(self, config, IS_MONO_CUT=False, dataset_type="NCLT"):
        super().__init__()
        self.config = config
        self.window_size = config['window_size']
        self.gpuid = config['gpuid']
        self.IS_MONO_CUT = IS_MONO_CUT
        self.dataset_type = dataset_type

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
            U, _, V = torch.svd(W + 1e-4 * W.mean() * torch.rand(1, 3).to(self.gpuid))

        det_UV = torch.det(U) * torch.det(V)
        ones = torch.ones(B, 2).type_as(V)
        S = torch.diag_embed(torch.cat((ones, det_UV.unsqueeze(1)), dim=1))

        R_tgt_src = torch.bmm(U, torch.bmm(S, V.transpose(2, 1)))

        if gt_Rt is not None:
            t_tgt_src_insrc = src_centroid - torch.bmm(gt_Rt[:,:3,:3].float().transpose(2, 1), tgt_centroid)
            t_src_tgt_intgt = -gt_Rt[:,:3,:3].float().bmm(t_tgt_src_insrc)
        else:
            t_tgt_src_insrc = src_centroid - torch.bmm(R_tgt_src.transpose(2, 1), tgt_centroid)
            t_src_tgt_intgt = -R_tgt_src.bmm(t_tgt_src_insrc)
        
        return R_tgt_src, t_src_tgt_intgt

    def count_inliers(self, src_coords, tgt_coords, R, t, threshold):

        transformed_src = torch.bmm(R, src_coords.transpose(2, 1)) + t

        distances = torch.norm(transformed_src - tgt_coords.transpose(2, 1), dim=1)
        
        inliers = torch.sum(distances < threshold)
        return inliers
        
    def forward(self, src_coords, tgt_coords, weights, convert_from_pixels=True, gt_Rt=None):
        
        if src_coords.size(0) > tgt_coords.size(0):
            BW = src_coords.size(0)
            B = int(BW / self.window_size)
            kp_inds, _ = get_indices(B, self.window_size)
            src_coords = src_coords[kp_inds]

        assert(src_coords.size() == tgt_coords.size())
        B = src_coords.size(0)

        if convert_from_pixels:
            if self.IS_MONO_CUT:
                src_coords = convert_to_radar_frame_mono_cut(src_coords, self.config, self.dataset_type)
                tgt_coords = convert_to_radar_frame_mono_cut(tgt_coords, self.config, self.dataset_type)
            else:
                src_coords = convert_to_radar_frame(src_coords, self.config)
                tgt_coords = convert_to_radar_frame(tgt_coords, self.config)
        
        if src_coords.size(2) < 3:
            pad = 3 - src_coords.size(2)
            src_coords = F.pad(src_coords, [0, pad, 0, 0])
        if tgt_coords.size(2) < 3:
            pad = 3 - tgt_coords.size(2)
            tgt_coords = F.pad(tgt_coords, [0, pad, 0, 0])

        src_coords = src_coords.transpose(2, 1)  # B x 3 x N
        tgt_coords = tgt_coords.transpose(2, 1)
        R_tgt_src, t_src_tgt_intgt = self.compute_weighted_svd(src_coords, tgt_coords, weights, B, gt_Rt)

        return R_tgt_src, t_src_tgt_intgt