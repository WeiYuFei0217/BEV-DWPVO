import torch

from networks import UNet, Keypoint, SoftmaxMatcher, SVD

__all__ = ['DWPSolver', 'pose_loss']


class DWPSolver(torch.nn.Module):
    """BEV keypoint extraction, matching and differentiable weighted Procrustes solver."""

    def __init__(self, config, bev_half=None, disable_validity_weights=False):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.svd = SVD(config, bev_half)
        self.disable_validity_weights = disable_validity_weights

    def forward(self, x, gt_Rt=None, use_match_range=True, match_range=3.0, gkp=False):
        """
        Args:
            x (Tensor): (2B, C, H, W) BEV feature maps, ordered as frame pairs.
            gt_Rt (Tensor, optional): (B, 4, 4) ground-truth relative poses, only used by
                Guided Convergence for Translation (GCT) during training.
            use_match_range (bool): restrict matching to a local window around each keypoint.
            match_range (float): half size of the matching window in meters.
            gkp (bool): Global Keypoint Pre-training (GKP), i.e. set all validity weights to 1.
        """
        x = x.to(self.device)

        position_scores, validity, descriptors = self.unet(x)
        # Without keypoint validity weights (UKVW ablation) or during GKP, all validity weights are set to 1.
        if self.disable_validity_weights or gkp:
            validity = torch.ones_like(validity)

        position_scores_all = position_scores.reshape(position_scores.shape[0] // 2, 2, *position_scores.shape[1:])
        validity_all = validity.reshape(validity.shape[0] // 2, 2, *validity.shape[1:])
        descriptors_all = descriptors.reshape(descriptors.shape[0] // 2, 2, *descriptors.shape[1:])
        R_tgt_src_pred_all = []
        t_tgt_src_pred_all = []
        outputs_all = []
        for i in range(position_scores.shape[0] // 2):
            position_scores = position_scores_all[i]
            validity = validity_all[i]
            descriptors = descriptors_all[i]
            keypoint_coords, keypoint_validity, keypoint_desc = self.keypoint(position_scores, validity, descriptors)
            pseudo_coords, match_weights, kp_inds = self.softmax_matcher(
                keypoint_validity, keypoint_desc, validity, descriptors, keypoint_coords,
                use_match_range=use_match_range, match_range=match_range)
            src_coords = keypoint_coords[kp_inds]
            pair_gt_Rt = gt_Rt[i:i + 1] if gt_Rt is not None else None
            R_tgt_src_pred, t_tgt_src_pred = self.svd(src_coords, pseudo_coords, match_weights, gt_Rt=pair_gt_Rt)

            R_tgt_src_pred_all.append(R_tgt_src_pred)
            t_tgt_src_pred_all.append(t_tgt_src_pred)
            outputs_all.append({'bev': x, 'validity': validity, 'src': src_coords, 'tgt': pseudo_coords,
                                'match_weights': match_weights})

        return R_tgt_src_pred_all, t_tgt_src_pred_all, outputs_all


def pose_loss(R_tgt_src_pred, t_tgt_src_pred, T_tgt_src, rot_loss_weight=10):
    """L1 pose loss on the 3-DoF (x, y, yaw) relative pose."""
    batch_size = R_tgt_src_pred.shape[0]

    R_tgt_src = T_tgt_src[:, :2, :2].float().cuda()
    R_tgt_src_pred = R_tgt_src_pred[:, :2, :2].float().cuda()
    t_tgt_src = T_tgt_src[:, :2, 3].unsqueeze(-1).float().cuda()
    t_tgt_src_pred = t_tgt_src_pred[:, :2, :]

    identity = torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    loss_fn = torch.nn.L1Loss()
    R_loss = loss_fn(torch.matmul(R_tgt_src_pred.transpose(2, 1), R_tgt_src), identity)
    t_loss = loss_fn(t_tgt_src_pred, t_tgt_src)
    loss = t_loss + rot_loss_weight * R_loss
    return loss, {'R_loss': R_loss, 't_loss': t_loss}
