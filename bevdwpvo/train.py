"""Trains BEV-DWPVO with pose supervision only.

Usage (from the bevdwpvo/ directory):
    python train.py --config configs/nclt.yaml [--resume outputs/<run>/checkpoints/epoch_10.pth]

Logs, checkpoints and evaluation trajectories are written to outputs/<run>/.
"""
import argparse
import datetime
import os
import warnings

import torch
from torch.multiprocessing import set_start_method
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from builder import (build_mats, build_model, build_test_sets, build_train_set, load_config, load_weights,
                     repo_path, solver_conf)
from engine import evaluate, relative_poses, set_seed
from models import pose_loss
from utils import Monitor

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description='Train BEV-DWPVO')
    parser.add_argument('-c', '--config', type=str, required=True, help='yaml config file')
    parser.add_argument('--resume', type=str, default=None, help='training checkpoint to resume from')
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg, test_cfg = cfg['train'], cfg['test']
    set_seed(train_cfg['seed'])

    model_cfg = cfg['model']
    run_name = (datetime.datetime.now().strftime('%y%m%d_%H%M%S')
                + f"_{cfg['dataset']['name']}_{model_cfg['bev_size']}x{model_cfg['bev_resolution']}m")
    run_dir = repo_path(os.path.join('outputs', run_name))
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    model = build_model(cfg).cuda()
    monitor = Monitor(os.path.join(run_dir, 'logs'), solver_conf(cfg), model_cfg.get('bev_half'))
    optimizer = Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['lr_decay'])

    gkp, gct = train_cfg['gkp'], train_cfg['gct']
    start_epoch = 0
    if args.resume is not None:
        print('Resuming from checkpoint: ' + args.resume)
        checkpoint = load_weights(model, args.resume)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        monitor.counter = checkpoint['counter']
        gkp, gct = False, False

    batch_size = train_cfg['batch_size']
    mats_train = build_mats(cfg, batch_size=2 * batch_size)
    mats_test = build_mats(cfg, batch_size=2)
    train_loader = DataLoader(build_train_set(cfg), batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=train_cfg['num_workers'])
    test_loaders = [(name, DataLoader(ds, batch_size=test_cfg['batch_size'], shuffle=False, drop_last=True,
                                      pin_memory=True, num_workers=test_cfg['num_workers']))
                    for name, ds in build_test_sets(cfg)]

    total_steps = 0
    for epoch in range(start_epoch, train_cfg['epochs']):
        if gkp and total_steps > train_cfg['gkp_steps']:
            gkp = False
        if gct and total_steps > train_cfg['gct_steps']:
            gct = False
        print(f'{run_name}  epoch {epoch + 1}')

        model.train()
        for images, poses, _ in tqdm(train_loader):
            images = images.cuda()
            T_gt = relative_poses(poses).cuda()
            images = images.reshape(images.shape[0] * images.shape[1], *images.shape[2:]).unsqueeze(1)

            R_pred_all, t_pred_all, _ = model(images, mats_train, gt_Rt=T_gt if gct else None,
                                              use_match_range=train_cfg['use_match_range'],
                                              match_range=train_cfg['match_range'], gkp=gkp)

            loss, R_loss, t_loss = 0, 0, 0
            for i in range(batch_size):
                loss_i, loss_dict = pose_loss(R_pred_all[i], t_pred_all[i], T_gt[i:i + 1],
                                              rot_loss_weight=train_cfg['rot_loss_weight'])
                loss += loss_i
                R_loss += loss_dict['R_loss'].detach().cpu().item()
                t_loss += loss_dict['t_loss'].detach().cpu().item()
            loss = loss / batch_size

            optimizer.zero_grad()
            loss.backward()
            if train_cfg['grad_clip']:
                clip_grad_norm_(model.parameters(), train_cfg['max_norm'])
            optimizer.step()

            monitor.step(loss, R_loss / batch_size, t_loss / batch_size)
            total_steps += 1

        scheduler.step()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'counter': monitor.counter,
            'epoch': epoch + 1,
        }, os.path.join(ckpt_dir, f'epoch_{epoch + 1}.pth'))

        with torch.no_grad():
            results = evaluate(model, test_loaders, mats_test, train_cfg, monitor,
                               os.path.join(run_dir, 'trajectories'), tag=f'epoch_{epoch + 1}', gkp=gkp)
        for seq_name, (t_err, R_err) in results.items():
            print(f'epoch {epoch + 1}  {seq_name}: t_err_avg {t_err:.6f}  R_err_avg {R_err:.6f}')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
