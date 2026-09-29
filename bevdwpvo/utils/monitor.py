import os
from time import time

from torch.utils.tensorboard import SummaryWriter

from utils.common import compute_median_error, save_tum_trajectory
from utils.vis import draw_batch, plot_sequences


class Monitor(object):
    """TensorBoard logging of training losses and evaluation results."""

    def __init__(self, log_dir, solver_conf, bev_half=None):
        self.log_dir = log_dir
        self.solver_conf = solver_conf
        self.bev_half = bev_half
        self.counter = 0
        self.current_time = 0
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        print('monitor running and saving to {}'.format(self.log_dir))

    def step(self, loss, R_loss, t_loss):
        self.counter += 1
        self.current_time = time()
        self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
        self.writer.add_scalar('train/Rloss', R_loss, self.counter)
        self.writer.add_scalar('train/tloss', t_loss, self.counter)
        return self.counter

    def step_val(self, T_gt, T_pred, outputs_draw, timestamps, file_path_gt, file_path_pred):
        results = compute_median_error(T_gt, T_pred)
        self.writer.add_scalar('val/t_err_avg', results[4], self.counter)
        self.writer.add_scalar('val/R_err_avg', results[5], self.counter)
        print("t_err_avg:{}, R_err_avg:{}".format(results[4], results[5]))

        save_tum_trajectory(file_path_gt, timestamps, T_gt)
        save_tum_trajectory(file_path_pred, timestamps, T_pred)

        for img in plot_sequences(T_gt, T_pred, [len(T_pred)]):
            self.writer.add_image('val/trajectory', img, self.counter)
        if outputs_draw is not None:
            self.writer.add_image('val/batch_img', draw_batch(outputs_draw, self.solver_conf, self.bev_half), self.counter)
        self.current_time = time()
        return results[4], results[5]

    def log_val_average(self, t_err_avg, R_err_avg):
        self.writer.add_scalar('val/t_err_avg', t_err_avg, self.counter)
        self.writer.add_scalar('val/R_err_avg', R_err_avg, self.counter)
        self.current_time = time()
