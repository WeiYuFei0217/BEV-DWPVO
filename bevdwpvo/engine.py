"""Shared training / evaluation steps of BEV-DWPVO."""
import os
import random

import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def relative_poses(poses):
    """(B, 2, 4, 4) absolute poses -> (B, 4, 4) relative poses T_1^-1 @ T_0."""
    return torch.matmul(torch.from_numpy(np.linalg.inv(poses[:, 1])), poses[:, 0])


def evaluate(model, test_loaders, mats_dict, train_cfg, monitor, out_dir, tag, gkp=False):
    """Runs the model on the test sequences, accumulating relative poses without any post-processing.

    Returns:
        dict: sequence name -> (mean translation error, mean rotation error) per frame pair.
    """
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for seq_name, loader in test_loaders:
        T_gt_all, T_pred_all, timestamps = [], [], []
        outputs_draw = None
        draw_at = min(1000, len(loader) // 5)
        for images, poses, timestamp in tqdm(loader, desc=seq_name):
            images = images.cuda()
            timestamps.append(timestamp)
            T_gt = relative_poses(poses).cuda()
            for i in range(images.shape[0]):
                R_pred, t_pred, outputs = model(images[i:i + 1].transpose(0, 1), mats_dict,
                                                use_match_range=train_cfg['use_match_range'],
                                                match_range=train_cfg['match_range'], gkp=gkp)
                T_gt_all.append(T_gt[i:i + 1].cpu().numpy().reshape((4, 4)))
                T_pred = np.eye(4)
                T_pred[:3, :3] = R_pred[0].cpu().detach().numpy().reshape((3, 3))
                T_pred[:3, 3] = t_pred[0].cpu().detach().numpy().reshape((3,))
                T_pred_all.append(T_pred)
                if outputs_draw is None and len(T_pred_all) >= draw_at:
                    outputs_draw = outputs[0]

        t_err, R_err = monitor.step_val(
            T_gt_all, T_pred_all, outputs_draw, timestamps,
            file_path_gt=os.path.join(out_dir, f'{tag}_{seq_name}_gt.txt'),
            file_path_pred=os.path.join(out_dir, f'{tag}_{seq_name}_pred.txt'))
        results[seq_name] = (t_err, R_err)

    monitor.log_val_average(np.mean([v[0] for v in results.values()]), np.mean([v[1] for v in results.values()]))
    return results
