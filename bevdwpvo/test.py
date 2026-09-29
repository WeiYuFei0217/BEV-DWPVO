"""Evaluates BEV-DWPVO on the test sequences of a dataset.

Usage (from the bevdwpvo/ directory):
    python test.py --config configs/nclt.yaml [--checkpoint weights/bevdwpvo_nclt.pth]

Trajectories (TUM format) are written to outputs/<run>/trajectories/.
"""
import argparse
import datetime
import os
import warnings

import torch
from torch.utils.data import DataLoader

from builder import build_mats, build_model, build_test_sets, load_config, load_weights, repo_path, solver_conf
from engine import evaluate, set_seed
from utils import Monitor

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description='Evaluate BEV-DWPVO')
    parser.add_argument('-c', '--config', type=str, required=True, help='yaml config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='weights (default: test.checkpoint in config)')
    args = parser.parse_args()
    cfg = load_config(args.config)
    test_cfg = cfg['test']
    set_seed(cfg['train']['seed'])

    run_name = datetime.datetime.now().strftime('%y%m%d_%H%M%S') + f"_{cfg['dataset']['name']}_test"
    run_dir = repo_path(os.path.join('outputs', run_name))

    model = build_model(cfg).cuda()
    checkpoint_path = repo_path(args.checkpoint or test_cfg['checkpoint'])
    print('Loading weights: ' + checkpoint_path)
    load_weights(model, checkpoint_path)

    monitor = Monitor(os.path.join(run_dir, 'logs'), solver_conf(cfg), cfg['model'].get('bev_half'))
    mats_dict = build_mats(cfg, batch_size=2)
    test_loaders = [(name, DataLoader(ds, batch_size=test_cfg['batch_size'], shuffle=False, drop_last=True,
                                      pin_memory=True, num_workers=test_cfg['num_workers']))
                    for name, ds in build_test_sets(cfg)]

    with torch.no_grad():
        results = evaluate(model, test_loaders, mats_dict, cfg['train'], monitor,
                           os.path.join(run_dir, 'trajectories'), tag='test')
    for seq_name, (t_err, R_err) in results.items():
        print(f'{seq_name}: t_err_avg {t_err:.6f}  R_err_avg {R_err:.6f}')


if __name__ == '__main__':
    main()
