import os
from time import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2

import sys
from utils.utils import computeMedianError, save_tum_trajectory
from utils.vis import draw_batch, plot_sequences, plot_2d_trajectory
from torchvision.transforms import ToTensor

def convert_to_radar_frame(pixel_coords, config):
    cart_pixel_width = config['cart_pixel_width']
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)
    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

def convert_to_radar_frame_mono_cut(pixel_coords, config, dataset_type="NCLT"):
    cart_pixel_width = config['cart_pixel_width'] * 2
    cart_resolution = config['cart_resolution']
    gpuid = config['gpuid']

    width = cart_pixel_width
    height = cart_pixel_width

    if dataset_type == 'NCLT' or dataset_type == 'kitti':
        pixel_coords[:, :, 0] += width // 4
    elif dataset_type == "oxford":
        pixel_coords[:, :, 0] += width // 4
        pixel_coords[:, :, 1] += height // 2

    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    B, N, _ = pixel_coords.size()
    R = torch.tensor([[0, -cart_resolution], [cart_resolution, 0]]).expand(B, 2, 2).to(gpuid)
    t = torch.tensor([[cart_min_range], [-cart_min_range]]).expand(B, 2, N).to(gpuid)

    return (torch.bmm(R, pixel_coords.transpose(2, 1)) + t).transpose(2, 1)

def project_to_image(coords, sensor2ego_mats, intrin_mats, match_weights, nms, max_w, bda_mats=None, image_size=(224, 384)):
    coords_images = []
    
    coords_with_height = torch.cat((coords, torch.zeros(coords.shape[0], coords.shape[1], 1), torch.ones(coords.shape[0], coords.shape[1], 1)), dim=2)
    coords_with_height = coords_with_height.squeeze(0)

    for i in range(len(sensor2ego_mats)):
        sensor2ego_mat = sensor2ego_mats[i]
        intrin_mat = intrin_mats[i]
        bda_mat = bda_mats[i] if bda_mats is not None else None
        
        ego_to_sensor_mat = torch.inverse(sensor2ego_mat)
        
        coords_cam = torch.matmul(ego_to_sensor_mat, coords_with_height.transpose(0, 1)).transpose(0, 1)
        
        valid_indices = (coords_cam[:, 2] > 0) & (match_weights > nms * max_w)
        coords_cam = coords_cam[valid_indices]

        coords_img_homogeneous = torch.matmul(intrin_mat[:3, :3], coords_cam[:, :3].transpose(0, 1)).transpose(0, 1)
        
        coords_img = coords_img_homogeneous[:, :2] / coords_img_homogeneous[:, 2].unsqueeze(1)
        
        coords_images.append(coords_img)

    return coords_images

def denormalize(img_np):
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    img_np = img_np * np.array(normalize_std) + np.array(normalize_mean)
    img_np = np.clip(img_np, 0, 1)
    return img_np


class MonitorBase(object):
    def __init__(self, model, config):
        self.model = model
        self.log_dir = config['log_dir']
        self.config = config
        self.gpuid = config['gpuid']
        self.counter = 0
        self.dt = 0
        self.current_time = 0
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        print('monitor running and saving to {}'.format(self.log_dir))

    def step(self, loss, R_loss, t_loss, depth_loss):
        self.counter += 1
        self.dt = time() - self.current_time
        self.current_time = time()

        self.writer.add_scalar('train/loss', loss.detach().cpu().item(), self.counter)
        self.writer.add_scalar('train/Rloss', R_loss, self.counter)
        self.writer.add_scalar('train/tloss', t_loss, self.counter)
        self.writer.add_scalar('train/depth_loss', depth_loss, self.counter)
        
        return self.counter

    def step_val(self, feature_size, loss, R_loss, t_loss, T_gt, T_pred, dict_DWP_Solver_draw, timestamps, mats_dict, file_path_gt='./evo/tum_trajectory_gt.txt', file_path_pred='./evo/tum_trajectory_pred.txt', IS_MONO_CUT=False, dataset_type="NCLT"):        

        results = computeMedianError(T_gt, T_pred)
        self.writer.add_scalar('val/t_err_avg', results[4], self.counter)
        self.writer.add_scalar('val/R_err_avg', results[5], self.counter)
        print("t_err_avg:{}, R_err_avg:{}".format(results[4], results[5]))

        save_tum_trajectory(file_path_gt, timestamps, T_gt)
        save_tum_trajectory(file_path_pred, timestamps, T_pred)

        imgs = plot_sequences(T_gt, T_pred, [len(T_pred)])
        for i, img in enumerate(imgs):
            self.writer.add_image('val/trajectory', img, self.counter)
        
        batch_img = draw_batch(feature_size, dict_DWP_Solver_draw, self.config, IS_MONO_CUT, dataset_type)
        self.writer.add_image('val/batch_img', batch_img, self.counter)
        self.current_time = time()

        return results[4], results[5]
    
    def step_val_ral_record(self, t_err_avg_record, R_err_avg_record):        

        self.writer.add_scalar('val/t_err_avg', t_err_avg_record, self.counter)
        self.writer.add_scalar('val/R_err_avg', R_err_avg_record, self.counter)

        self.current_time = time()