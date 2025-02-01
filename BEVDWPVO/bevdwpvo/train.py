import sys
from models import BaseBEVDWPVO, DepthLoss
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np
import os
from tqdm import tqdm
import random
import cv2
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from train_model import NCLTDataset, NCLTDataset_sequences, OxfordSequence, OxfordSequences, KITTIDataset, KITTIDataset_sequences
from torch.optim.lr_scheduler import ExponentialLR
import datetime
import yaml
import argparse
from torch.optim import Adam
from torch.multiprocessing import set_start_method
from torch.nn.utils import clip_grad_norm_
try:
    set_start_method('spawn')
except RuntimeError:
    pass 


from utils import geom
from utils import MonitorBase
from config_files import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import subprocess
import re

def get_target_ip_suffix(prefix="10.12.120."):
    try:
        result = subprocess.run(['ip', 'addr'], stdout=subprocess.PIPE, text=True, check=True)
        ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)/\d+', result.stdout)
        for ip in ips:
            if ip.startswith(prefix):
                return ip.split('.')[-1]
    except subprocess.CalledProcessError as e:
        print(f"Error executing ip command: {e}")
    return "unknown"

def list_and_sort_files(folder_path):
    file_names = os.listdir(folder_path)
    sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
    sorted_file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in sorted_file_names]
    return sorted_file_names_without_extension

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_configs(defaults, overrides):
    for key, value in overrides.items():
        if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
            merge_configs(defaults[key], value)
        else:
            defaults[key] = value
    return defaults

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="load yaml")
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='yaml path'
    )
    args = parser.parse_args()
    config = load_config(args.config)

    model_conf = config.get('model_conf', {'use_pretrained_model': False})
    backbone_conf = config.get('backbone_conf', {})
    config['backbone_conf']['final_dim'] = tuple(config['backbone_conf']['final_dim'])
    head_conf = config.get('head_conf', {})
    DWP_Solver_conf = config.get('DWP_Solver_conf', {})
    training_params = config.get('training_params', {})

    seed = training_params.get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ip_suffix = get_target_ip_suffix()
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%m%d_%H%M%S")

    NUM_WORKERS = training_params.get('NUM_WORKERS', 6)
    batch_size_train = training_params.get('batch_size_train', 4)
    batch_size_test = training_params.get('batch_size_test', 1)
    epoch_save = training_params.get('epoch_save', 1)

    NO_DEPTH = training_params.get('NO_DEPTH', True)

    DEBUG = training_params.get('DEBUG', False)
    JUST_TEST = training_params.get('JUST_TEST', False)
    TEST_model_dir = training_params.get('TEST_model_dir', 'model_save')

    feature_size = training_params.get('feature_size', 128)
    division = training_params.get('division', 0.8)
    dataset_type = training_params.get('dataset_type', 'NCLT')
    dataset_item_range = training_params.get('dataset_item_range', '0-1')
    pickle_name = training_params.get('pickle_name', 'pickle_d0_d2_r15_r45.txt')
    other_features = training_params.get('other_features', '')

    IS_MONO = training_params.get('IS_MONO', True)
    IS_MONO_CUT = training_params.get('IS_MONO_CUT', True)

    other_downsample_factor = training_params.get('other_downsample_factor', 16)

    use_keep_range = training_params.get('use_keep_range', False)
    range_keep = training_params.get('range_keep', 12.8)

    stop_weight_scores_firstNsteps = training_params.get('stop_weight_scores_firstNsteps', True)
    firstNsteps = training_params.get('firstNsteps', 5000)

    SPLIT_R_t = training_params.get('SPLIT_R_t', False)
    split_firstNsteps = training_params.get('split_firstNsteps', 5000)

    freeze_bev = training_params.get('freeze_bev', False)
    freeze_bev_firstNsteps = training_params.get('freeze_bev_firstNsteps', 5000)

    NO_WEIGHT_SCORES = training_params.get('NO_WEIGHT_SCORES', False)
    max_norm = training_params.get('max_norm', 1.0)
    USE_GRAD_CLIP = training_params.get('USE_GRAD_CLIP', False)
    depthloss_rate = training_params.get('depthloss_rate', 1)
    R_LOSS_enhenace = training_params.get('R_LOSS_enhenace', 10)
    use_global_mask = training_params.get('use_global_mask', False)
    patch_size = training_params.get('patch_size', "normal")  
    lr_start = training_params.get('lr_start', 0.0001)

    NAME = "25" + current_time_str + "_" + dataset_type + "_" + str(division) + "m-pix_" + str(feature_size) + "_" + dataset_item_range + other_features
    if NO_DEPTH:
        NAME = NAME + "_NO_DEPTH"
    if IS_MONO:
        if IS_MONO_CUT:
            NAME = NAME + "_MONO_CUT"
        else:
            NAME = NAME + "_MONO"
    if USE_GRAD_CLIP:
        NAME = NAME + "_GRAD_CLIP"
    if SPLIT_R_t:
        NAME = NAME + "_SPLIT_Rt"
    if use_keep_range:
        NAME = NAME + "_keep_range_" + str(range_keep)
    if stop_weight_scores_firstNsteps:
        NAME = NAME + "_NoWS_Nsteps"
    if other_downsample_factor != 16:
        NAME = NAME + "_DS_" + str(other_downsample_factor)

    NAME += f"_IP_{ip_suffix}"

    ckpt_path = None
    START = 0

    if 'NAME' in training_params and 'START' in training_params:
        NAME = training_params['NAME']
        START = training_params['START']
        ckpt_path = str(Root_dir) + "/model_save/model_"+str(NAME)+"/model_"+str(START)+".pth"
        stop_weight_scores_firstNsteps = False
        SPLIT_R_t = False
        freeze_bev = False
    if JUST_TEST:
        ckpt_path = TEST_model_dir
        stop_weight_scores_firstNsteps = False
        SPLIT_R_t = False
        freeze_bev = False

    DWP_Solver_conf["log_dir"] = f"./train_model/log_{NAME}/"
    log_file_path = f"./train_model/log_{NAME}/loss_record.txt"

    if feature_size == 256 and division == 0.1:  
        DWP_Solver_conf["cart_resolution"] = 0.1
        DWP_Solver_conf["cart_pixel_width"] = 256
        backbone_conf['x_bound'] = [-12.8, 12.8, 0.1]
        backbone_conf['y_bound'] = [-12.8, 12.8, 0.1]
        backbone_conf['z_bound'] = [-5, 3, 8]
        backbone_conf['d_bound'] = [2.0, 16, 0.125]
    if feature_size == 256 and division == 0.2:  
        DWP_Solver_conf["cart_resolution"] = 0.2
        DWP_Solver_conf["cart_pixel_width"] = 256
        backbone_conf['x_bound'] = [-25.6, 25.6, 0.2]
        backbone_conf['y_bound'] = [-25.6, 25.6, 0.2]
        backbone_conf['z_bound'] = [-5, 3, 8]
        backbone_conf['d_bound'] = [2.0, 30, 0.25]
    if feature_size == 256 and division == 0.4:  
        DWP_Solver_conf["cart_resolution"] = 0.4
        DWP_Solver_conf["cart_pixel_width"] = 256
        backbone_conf['x_bound'] = [-51.2, 51.2, 0.4]
        backbone_conf['y_bound'] = [-51.2, 51.2, 0.4]
        backbone_conf['z_bound'] = [-5, 3, 8]
        backbone_conf['d_bound'] = [2.0, 58, 0.5]
    if feature_size == 128 and division == 0.4: 
        DWP_Solver_conf["cart_resolution"] = 0.4
        DWP_Solver_conf["cart_pixel_width"] = 128
        backbone_conf['x_bound'] = [-25.6, 25.6, 0.4]
        backbone_conf['y_bound'] = [-25.6, 25.6, 0.4]
        backbone_conf['z_bound'] = [-5, 3, 8]
        backbone_conf['d_bound'] = [2.0, 30, 0.25]
    if feature_size == 128 and division == 0.8:  
        DWP_Solver_conf["cart_resolution"] = 0.8
        DWP_Solver_conf["cart_pixel_width"] = 128
        backbone_conf['x_bound'] = [-51.2, 51.2, 0.8]
        backbone_conf['y_bound'] = [-51.2, 51.2, 0.8]
        backbone_conf['z_bound'] = [-5, 3, 8]
        backbone_conf['d_bound'] = [2.0, 58, 0.5]
        model_conf['use_pretrained_model'] = True

    if dataset_type == 'oxford':
        backbone_conf['final_dim'] = (320, 640)
    if dataset_type == 'kitti':
        model_conf['use_pretrained_model'] = False
        backbone_conf['final_dim'] = (384, 1216)

    if IS_MONO:
        if IS_MONO_CUT:
            DWP_Solver_conf["cart_pixel_width"] = int(DWP_Solver_conf["cart_pixel_width"] / 2)
    else:
        IS_MONO_CUT = False

    if other_downsample_factor != 16:
        backbone_conf['downsample_factor'] = other_downsample_factor
        if other_downsample_factor == 8:
            backbone_conf['img_neck_conf'] = dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[0.5, 1, 2, 4],
                out_channels=[128, 128, 128, 128]
            )
            model_conf['use_pretrained_model'] = False
        if other_downsample_factor == 4:
            backbone_conf['img_neck_conf'] = dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[128, 128, 128, 128]
            )
            model_conf['use_pretrained_model'] = False
    
    if patch_size == "small":
        if feature_size == 128:
            DWP_Solver_conf["networks"]["keypoint_block"]["patch_size"] = 4
        if feature_size == 256:
            DWP_Solver_conf["networks"]["keypoint_block"]["patch_size"] = 6

    print(DWP_Solver_conf, backbone_conf, model_conf)
    print("batch_size_train, NUM_WORKERS:", batch_size_train, NUM_WORKERS)

    test_net = BaseBEVDWPVO(backbone_conf=backbone_conf,
                head_conf=head_conf,
                DWP_Solver_conf=DWP_Solver_conf,
                model_conf = model_conf,
                is_train_depth=True,
                IS_MONO_CUT=IS_MONO_CUT,
                NO_WEIGHT_SCORES=NO_WEIGHT_SCORES,
                dataset_type=dataset_type,
                feature_size=feature_size,
                use_global_mask=use_global_mask,
                freeze_bev=freeze_bev).cuda()

    monitor = MonitorBase(test_net, DWP_Solver_conf)

    optimizer = Adam(test_net.parameters(), lr=lr_start, weight_decay=1e-4)


    scheduler = ExponentialLR(optimizer, gamma=0.95)
    start_epoch = None

    if True:
        if ckpt_path is not None:
            try:
                print('Loading from checkpoint: ' + ckpt_path)
                checkpoint = torch.load(ckpt_path, map_location=torch.device(DWP_Solver_conf['gpuid']))
                test_net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] - 1
                if JUST_TEST:
                    monitor.counter = checkpoint['counter'] + 1
                else:
                    monitor.counter = checkpoint['counter']
                print('success')
            except Exception as e:
                print(e)
                print('WRONG IN LOADING CHECKPOINT!!!')
                test_net.load_state_dict(checkpoint, strict=False)
                print('success')

        if dataset_type == 'NCLT':
            NCLT_path = os.path.join(data_NCLT_dir, "format_data", "image_meta.pkl")
            with open(NCLT_path, 'rb') as handle:
                image_meta = pickle.load(handle)
                S = 5  
        if dataset_type == 'oxford':
            oxford_path = os.path.join(data_oxford_dir, "image_meta.pkl")
            with open(oxford_path, 'rb') as handle:
                image_meta = pickle.load(handle)
                S = 3  
        if dataset_type == 'kitti':
            K = np.array([
                [7.070912000000e+02, 0.000000000000e+00, 5.068873000000e+02],
                [0.000000000000e+00, 7.070912000000e+02, 1.901104000000e+02],
                [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]
            ])
            T = np.array([ 
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]
            ])
            image_meta = {
                'K': [K],
                'T': [T]
            }
            S = 1
        if IS_MONO:
            S = 1

        mats_dict = {}
        B = 2 * batch_size_train 
        __p = lambda x: geom.pack_seqdim(x, B)
        __u = lambda x: geom.unpack_seqdim(x, B)

        if IS_MONO and dataset_type == 'NCLT':
            intrins = torch.from_numpy(np.array(image_meta['K'])).float()[-1:, ...]
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1, -1:, ...]  
        elif IS_MONO and dataset_type == 'oxford':
            intrins = torch.from_numpy(np.array(image_meta['K'][:3])).float()[-1:, ...]
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'][:3])).unsqueeze(0).float()[:1, -1:, ...]
        elif IS_MONO and dataset_type == 'kitti':
            intrins = torch.from_numpy(np.array(image_meta['K'])).float()
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1,]
        else:
            intrins = torch.from_numpy(np.array(image_meta['K'][:S])).float()
            pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
            cams_T_body = torch.from_numpy(np.array(image_meta['T'][:S])).unsqueeze(0).float()

        pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
        cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
        body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
        pix_T_cams = pix_T_cams.view(B,1,S,4,4)
        cams_T_body = cams_T_body.view(B,1,S,4,4)
        body_T_cams = body_T_cams.view(B,1,S,4,4)
        ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
        bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

        mats_dict['sensor2ego_mats'] = body_T_cams.float()
        mats_dict['intrin_mats'] = pix_T_cams.float()
        mats_dict['ida_mats'] = ida_mats.float()
        mats_dict['bda_mat'] = bda_mat.float()

        if dataset_type == 'NCLT':
            base_dir = data_NCLT_dir + "/format_data"
            
            train_dates = ['2013-04-05', '2012-01-08', '2012-02-04']
            
            root_dirs = [f"{base_dir}/{date}/lb3_u_s_384" for date in train_dates]
            csv_paths = [f"{base_dir}/{date}/ground_truth/groundtruth_{date}.csv" for date in train_dates]
            pickle_names = [f"{Root_dir}/NCLT_pickle/{date}/{pickle_name}" for date in train_dates]

            nclt_dataset = NCLTDataset_sequences(
                root_dirs=root_dirs, 
                csv_paths=csv_paths, 
                phase="train",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH,
            )

            dataloader_params = {
                'batch_size': batch_size_train,
                'shuffle': True,
                'drop_last': True,
                'pin_memory': True
            }
            
            if (not JUST_TEST) and (not DEBUG):
                dataloader_params['num_workers'] = NUM_WORKERS
            
            dataloader = DataLoader(nclt_dataset, **dataloader_params)

            test_date = '2012-03-17'
            test_root = [f"{base_dir}/{test_date}/lb3_u_s_384"]
            test_csv = [f"{base_dir}/{test_date}/ground_truth/groundtruth_{test_date}.csv"]
            pickle_names = [f"{Root_dir}/NCLT_pickle/{test_date}/{pickle_name}"]

            test_date_2 = '2012-02-02'
            test_data_3 = '2012-02-19'
            test_data_4 = '2012-08-20'
            test_root_2 = [f"{base_dir}/{test_date_2}/lb3_u_s_384"]
            test_root_3 = [f"{base_dir}/{test_data_3}/lb3_u_s_384"]
            test_root_4 = [f"{base_dir}/{test_data_4}/lb3_u_s_384"]
            test_csv_2 = [f"{base_dir}/{test_date_2}/ground_truth/groundtruth_{test_date_2}.csv"]
            test_csv_3 = [f"{base_dir}/{test_data_3}/ground_truth/groundtruth_{test_data_3}.csv"]
            test_csv_4 = [f"{base_dir}/{test_data_4}/ground_truth/groundtruth_{test_data_4}.csv"]
            pickle_names_2 = [f"{Root_dir}/NCLT_pickle/{test_date_2}/{pickle_name}"]
            pickle_names_3 = [f"{Root_dir}/NCLT_pickle/{test_data_3}/{pickle_name}"]
            pickle_names_4 = [f"{Root_dir}/NCLT_pickle/{test_data_4}/{pickle_name}"]

            nclt_dataset_test = NCLTDataset_sequences(
                root_dirs=test_root,
                csv_paths=test_csv,
                phase="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH,
            )
            nclt_dataset_test_2 = NCLTDataset_sequences(
                root_dirs=test_root_2,
                csv_paths=test_csv_2,
                phase="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names_2,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH,
            )
            nclt_dataset_test_3 = NCLTDataset_sequences(
                root_dirs=test_root_3,
                csv_paths=test_csv_3,
                phase="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names_3,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH,
            )
            nclt_dataset_test_4 = NCLTDataset_sequences(
                root_dirs=test_root_4,
                csv_paths=test_csv_4,
                phase="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names_4,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH,
            )
            
            test_loader_params = {
                'batch_size': batch_size_test,
                'shuffle': False,
                'drop_last': True,
                'pin_memory': True
            }
            
            test_loader_params['num_workers'] = NUM_WORKERS
                
            dataloader_test = DataLoader(nclt_dataset_test, **test_loader_params)
            dataloader_test_2 = DataLoader(nclt_dataset_test_2, **test_loader_params)
            dataloader_test_3 = DataLoader(nclt_dataset_test_3, **test_loader_params)
            dataloader_test_4 = DataLoader(nclt_dataset_test_4, **test_loader_params)

        if dataset_type == 'oxford':
            root_dir = data_oxford_dir + "/"
            sequences_name = ["2019-01-11-13-24-51", "2019-01-14-14-15-12", "2019-01-15-14-24-38"]
            pickle_names = [f"{Root_dir}/oxford_pickle/{date}/{pickle_name}" for date in sequences_name]

            oxford_dataset = OxfordSequences(
                dataset_root=root_dir, 
                sequence_names=sequences_name, 
                split="train", 
                dataset_item_range=dataset_item_range, 
                pickle_names=pickle_names, 
                IS_MONO=IS_MONO, 
                NO_DEPTH=NO_DEPTH)
            
            dataloader_params = {
                'batch_size': batch_size_train,
                'shuffle': True,
                'drop_last': True,
                'pin_memory': True
            }

            if (not JUST_TEST) and (not DEBUG):
                dataloader_params['num_workers'] = NUM_WORKERS
            
            dataloader = DataLoader(oxford_dataset, **dataloader_params)

            sequence_name_test = ["2019-01-15-13-06-37"]
            pickle_names_test = [f"{Root_dir}/oxford_pickle/{date}/{pickle_name}" for date in sequence_name_test]
            sequence_name_test_2 = ["2019-01-11-12-26-55"]
            sequence_name_test_3 = ["2019-01-16-14-15-33"]
            sequence_name_test_4 = ["2019-01-17-12-48-25"]
            pickle_names_test_2 = [f"{Root_dir}/oxford_pickle/{date}/{pickle_name}" for date in sequence_name_test_2]
            pickle_names_test_3 = [f"{Root_dir}/oxford_pickle/{date}/{pickle_name}" for date in sequence_name_test_3]
            pickle_names_test_4 = [f"{Root_dir}/oxford_pickle/{date}/{pickle_name}" for date in sequence_name_test_4]

            oxford_dataset_test = OxfordSequences(
                dataset_root=root_dir, 
                sequence_names=sequence_name_test, 
                split="test", 
                dataset_item_range=dataset_item_range, 
                pickle_names=pickle_names_test, 
                IS_MONO=IS_MONO, 
                NO_DEPTH=NO_DEPTH)
            oxford_dataset_test_2 = OxfordSequences(
                dataset_root=root_dir,
                sequence_names=sequence_name_test_2,
                split="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names_test_2,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH)
            oxford_dataset_test_3 = OxfordSequences(
                dataset_root=root_dir,
                sequence_names=sequence_name_test_3,
                split="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names_test_3,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH)
            oxford_dataset_test_4 = OxfordSequences(
                dataset_root=root_dir,
                sequence_names=sequence_name_test_4,
                split="test",
                dataset_item_range=dataset_item_range,
                pickle_names=pickle_names_test_4,
                IS_MONO=IS_MONO,
                NO_DEPTH=NO_DEPTH)
            
            test_loader_params = {
                'batch_size': batch_size_test,
                'shuffle': False,
                'drop_last': True,
                'pin_memory': True
            }

            test_loader_params['num_workers'] = NUM_WORKERS

            dataloader_test = DataLoader(oxford_dataset_test, **test_loader_params)
            dataloader_test_2 = DataLoader(oxford_dataset_test_2, **test_loader_params)
            dataloader_test_3 = DataLoader(oxford_dataset_test_3, **test_loader_params)
            dataloader_test_4 = DataLoader(oxford_dataset_test_4, **test_loader_params)
        
        if dataset_type == 'kitti':
            sequences_train = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
            sequences_test = ['09']
            sequences_test_2 = ['10']
            root_dir_kitti = data_kitti_dir + "/"

            pickle_names = [f"{Root_dir}/kitti_pickle/{date}/{pickle_name}" for date in sequences_train]
            pickle_names_test = [f"{Root_dir}/kitti_pickle/{date}/{pickle_name}" for date in sequences_test]
            pickle_names_test_2 = [f"{Root_dir}/kitti_pickle/{date}/{pickle_name}" for date in sequences_test_2]

            kitti_dataset = KITTIDataset_sequences(
                root_dir=root_dir_kitti, 
                sequences=sequences_train, 
                phase="train", 
                dataset_item_range=dataset_item_range, 
                pickle_names=pickle_names, 
                IS_MONO=IS_MONO, 
                NO_DEPTH=NO_DEPTH)

            dataloader_params = {
                'batch_size': batch_size_train,
                'shuffle': True,
                'drop_last': True,
                'pin_memory': True,
            }

            if (not JUST_TEST) and (not DEBUG):
                dataloader_params['num_workers'] = NUM_WORKERS

            dataloader = DataLoader(kitti_dataset, **dataloader_params)

            kitti_dataset_test = KITTIDataset_sequences(
                root_dir=root_dir_kitti, 
                sequences=sequences_test, 
                phase="test", 
                dataset_item_range=dataset_item_range, 
                pickle_names=pickle_names, 
                IS_MONO=IS_MONO, 
                NO_DEPTH=NO_DEPTH)
            kitti_dataset_test_2 = KITTIDataset_sequences(
                root_dir=root_dir_kitti, 
                sequences=sequences_test_2, 
                phase="test", 
                dataset_item_range=dataset_item_range, 
                pickle_names=pickle_names, 
                IS_MONO=IS_MONO, 
                NO_DEPTH=NO_DEPTH)

            test_loader_params = {
                'batch_size': batch_size_test,
                'shuffle': False,
                'drop_last': True,
                'pin_memory': True,
            }

            test_loader_params['num_workers'] = NUM_WORKERS
                
            dataloader_test = DataLoader(kitti_dataset_test, **test_loader_params)
            dataloader_test_2 = DataLoader(kitti_dataset_test_2, **test_loader_params)

        folder_path = str(Root_dir) + "/model_save/model_" + str(NAME)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        depth_loss_fn = DepthLoss(backbone_conf)

        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        with open(log_file_path, "a") as f:
            f.write("Current Time: {}\n".format(current_time_str))

        num_epoch = 0
        total_steps = 0
        while num_epoch < 1000:  
            print(NAME)
            if stop_weight_scores_firstNsteps and total_steps > firstNsteps:
                stop_weight_scores_firstNsteps = False
            if SPLIT_R_t and total_steps > split_firstNsteps:
                SPLIT_R_t = False
            if freeze_bev and total_steps > freeze_bev_firstNsteps:
                freeze_bev = False
                test_net.freeze_bev = False
                for param in test_net.backbone.parameters():
                    param.requires_grad = True

            mats_dict = {}
            B = 2 * batch_size_train 
            __p = lambda x: geom.pack_seqdim(x, B)
            __u = lambda x: geom.unpack_seqdim(x, B)
            
            if IS_MONO and dataset_type == 'NCLT':
                intrins = torch.from_numpy(np.array(image_meta['K'])).float()[-1:, ...]
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1, -1:, ...]
            elif IS_MONO and dataset_type == 'oxford':
                intrins = torch.from_numpy(np.array(image_meta['K'][:3])).float()[-1:, ...]
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'][:3])).unsqueeze(0).float()[:1, -1:, ...]
            else:
                intrins = torch.from_numpy(np.array(image_meta['K'][:S])).float()
                pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                cams_T_body = torch.from_numpy(np.array(image_meta['T'][:S])).unsqueeze(0).float()

            pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
            cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
            body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
            pix_T_cams = pix_T_cams.view(B,1,S,4,4)
            cams_T_body = cams_T_body.view(B,1,S,4,4)
            body_T_cams = body_T_cams.view(B,1,S,4,4)
            ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
            bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

            mats_dict['sensor2ego_mats'] = body_T_cams.float()
            mats_dict['intrin_mats'] = pix_T_cams.float()
            mats_dict['ida_mats'] = ida_mats.float()
            mats_dict['bda_mat'] = bda_mat.float()


            if start_epoch is not None:
                print(f"training from epoch {start_epoch + 2}")
                num_epoch = start_epoch + 1  
                start_epoch = None
            print("num_epoch:", num_epoch + 1)
            total_svd_loss = 0
            total_R_loss = 0
            total_t_loss = 0
            total_depth_loss = 0
            total_num = 0

            temp_svd_loss = 0
            temp_R_loss = 0
            temp_t_loss = 0
            temp_depth_loss = 0
            temp_num = 0

            model_save_path = str(Root_dir) + "/model_save/model_"+str(NAME)+f"/model_{num_epoch + 1}.pth"
            test_net.train()
            if (num_epoch + 1) % epoch_save == 0:
                print("----------------------------------------")
                time_last = time.time()
            for batch_all_2_5_images, poses, batch_all_2_5_depth, timestamp_train_all, idx1_all, idx2_all in tqdm(dataloader):
                batch_all_2_5_images = batch_all_2_5_images.cuda()
                batch_all_2_5_depth = batch_all_2_5_depth.cuda()
                if total_num >= 100 and DEBUG:
                    break
                if JUST_TEST:
                    break

                pose0 = poses[:, 0]  
                pose1 = poses[:, 1]  
                pose1_inv = torch.from_numpy(np.linalg.inv(pose1))  
                t1_T_t = torch.matmul(pose1_inv, pose0).cuda()


                batch_all_2_5_images_new = batch_all_2_5_images.reshape(batch_all_2_5_images.shape[0] * batch_all_2_5_images.shape[1], batch_all_2_5_images.shape[2], batch_all_2_5_images.shape[3], batch_all_2_5_images.shape[4], batch_all_2_5_images.shape[5])
                batch_all_2_5_images_new = batch_all_2_5_images_new.unsqueeze(1)

                if SPLIT_R_t:
                    R_tgt_src_pred_all, t_tgt_src_pred_all, dict_DWP_Solver_all, depth_pred_all = test_net(batch_all_2_5_images_new, mats_dict, gt_Rt=t1_T_t, use_keep_range=use_keep_range, range_keep=range_keep, stop_weight_scores_firstNsteps=stop_weight_scores_firstNsteps)
                else:
                    R_tgt_src_pred_all, t_tgt_src_pred_all, dict_DWP_Solver_all, depth_pred_all = test_net(batch_all_2_5_images_new, mats_dict, use_keep_range=use_keep_range, range_keep=range_keep, stop_weight_scores_firstNsteps=stop_weight_scores_firstNsteps)
                depth_pred_all = depth_pred_all.reshape(batch_size_train, depth_pred_all.shape[0] // batch_size_train, depth_pred_all.shape[1], depth_pred_all.shape[2], depth_pred_all.shape[3])

                for iii in range(batch_size_train):
                    R_tgt_src_pred, t_tgt_src_pred, dict_DWP_Solver, depth_pred, timestamp_train = R_tgt_src_pred_all[iii], t_tgt_src_pred_all[iii], dict_DWP_Solver_all[iii], depth_pred_all[iii], timestamp_train_all[iii]
                    svd_loss, dict_loss = BaseBEVDWPVO.RTloss(R_tgt_src_pred, t_tgt_src_pred, t1_T_t[iii:iii+1], R_LOSS_enhenace=R_LOSS_enhenace)

                    if NO_DEPTH == False:
                        depth_loss, _ = depth_loss_fn(batch_all_2_5_depth[iii], depth_pred)
                        total_depth_loss += depth_loss.detach().cpu().item()*depthloss_rate
                        temp_depth_loss += depth_loss.detach().cpu().item()*depthloss_rate
                        temp_svd_loss += (svd_loss + depth_loss*depthloss_rate)  
                        total_svd_loss += (svd_loss.detach().cpu().item() + depth_loss.detach().cpu().item()*depthloss_rate)  
                    else:
                        temp_svd_loss += svd_loss  
                        total_svd_loss += svd_loss.detach().cpu().item()

                    total_R_loss += dict_loss["R_loss"].detach().cpu().item()
                    total_t_loss += dict_loss["t_loss"].detach().cpu().item()
                    total_num += 1
                    temp_R_loss += dict_loss["R_loss"].detach().cpu().item()
                    temp_t_loss += dict_loss["t_loss"].detach().cpu().item()
                    temp_num += 1

                if temp_num == 0:
                    temp_svd_loss = 0
                    temp_R_loss = 0
                    temp_t_loss = 0
                    temp_depth_loss = 0
                    time_last = time.time()
                    continue
                
                back_loss_record = temp_svd_loss / temp_num
                R_loss_record = temp_R_loss / temp_num
                t_loss_record = temp_t_loss / temp_num
                depth_loss_record = temp_depth_loss / temp_num

                optimizer.zero_grad()  
                back_loss_record.backward()  

                if USE_GRAD_CLIP:
                    clip_grad_norm_(test_net.parameters(), max_norm)

                optimizer.step() 

                monitor_count = monitor.step(back_loss_record, R_loss_record, t_loss_record, depth_loss_record)
                total_steps += 1

                temp_svd_loss = 0
                temp_R_loss = 0
                temp_t_loss = 0
                temp_depth_loss = 0
                temp_num = 0

                time_last = time.time()

            if not JUST_TEST:   
                scheduler.step()  
                
            if (num_epoch + 1) % epoch_save == 0:

                mats_dict = {}
                B = 2   
                __p = lambda x: geom.pack_seqdim(x, B)
                __u = lambda x: geom.unpack_seqdim(x, B)
                
                if IS_MONO and dataset_type == 'NCLT':
                    intrins = torch.from_numpy(np.array(image_meta['K'])).float()[-1:, ...]
                    pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                    cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1, -1:, ...]
                elif IS_MONO and dataset_type == 'oxford':
                    intrins = torch.from_numpy(np.array(image_meta['K'][:3])).float()[-1:, ...]
                    pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                    cams_T_body = torch.from_numpy(np.array(image_meta['T'][:3])).unsqueeze(0).float()[:1, -1:, ...]
                elif IS_MONO and dataset_type == 'kitti':
                    intrins = torch.from_numpy(np.array(image_meta['K'])).float()
                    pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                    cams_T_body = torch.from_numpy(np.array(image_meta['T'])).unsqueeze(0).float()[:1,]
                else:
                    intrins = torch.from_numpy(np.array(image_meta['K'][:S])).float()
                    pix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(intrins)).unsqueeze(0)
                    cams_T_body = torch.from_numpy(np.array(image_meta['T'][:S])).unsqueeze(0).float()

                pix_T_cams = pix_T_cams.repeat(B,1,1,1).cuda()
                cams_T_body = cams_T_body.repeat(B,1,1,1).cuda()
                body_T_cams = __u(geom.safe_inverse(__p(cams_T_body)))
                pix_T_cams = pix_T_cams.view(B,1,S,4,4)
                cams_T_body = cams_T_body.view(B,1,S,4,4)
                body_T_cams = body_T_cams.view(B,1,S,4,4)
                ida_mats = torch.from_numpy(np.eye(4)).repeat(B*S,1,1).cuda().view(B,1,S,4,4)
                bda_mat = torch.from_numpy(np.eye(4)).repeat(B,1,1).cuda()

                mats_dict['sensor2ego_mats'] = body_T_cams.float()
                mats_dict['intrin_mats'] = pix_T_cams.float()
                mats_dict['ida_mats'] = ida_mats.float()
                mats_dict['bda_mat'] = bda_mat.float()


                if not JUST_TEST:
                    print(f"Model saved at epoch {num_epoch + 1}")
                    torch.save({
                        'model_state_dict': test_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'counter': monitor.counter,
                        'epoch': num_epoch + 1,
                        }, model_save_path)

                test_net.eval()
                
                if dataset_type in ['NCLT', 'oxford']:
                    all_test = [dataloader_test, dataloader_test_2, dataloader_test_3, dataloader_test_4]
                elif dataset_type == 'kitti':
                    all_test = [dataloader_test, dataloader_test_2]

                test_num_seq = 0
                t_err_avg_all = []
                R_err_avg_all = []

                for dataloader_test_now in all_test:
                    test_num_seq += 1
                    
                    all_t1_T_t = []
                    all_homogeneous_transform = []
                    timestamps_val = []
                    dict_DWP_Solver_draw = None
                    test_num = 0

                    total_svd_loss_val = 0
                    total_R_loss_val = 0
                    total_t_loss_val = 0
                    total_num_val = 0

                    for batch_all_2_5_images, poses, batch_all_2_5_depth, timestamp_val in tqdm(dataloader_test_now):
                        batch_all_2_5_images = batch_all_2_5_images.cuda()
                        timestamps_val.append(timestamp_val)
                        pose0 = poses[:, 0]  
                        pose1 = poses[:, 1]  
                        pose1_inv = torch.from_numpy(np.linalg.inv(pose1)) 
                        t1_T_t = torch.matmul(pose1_inv, pose0).cuda()

                        for iii in range(batch_size_test):
                            R_tgt_src_pred, t_tgt_src_pred, dict_DWP_Solver, _ = test_net(batch_all_2_5_images[iii:iii+1].transpose(0,1), mats_dict, use_keep_range=use_keep_range, range_keep=range_keep, stop_weight_scores_firstNsteps=stop_weight_scores_firstNsteps)
                            test_num += 1
                            R_tgt_src_pred, t_tgt_src_pred, dict_DWP_Solver = R_tgt_src_pred[0], t_tgt_src_pred[0], dict_DWP_Solver[0]
                            svd_loss_val, dict_loss_val = BaseBEVDWPVO.RTloss(R_tgt_src_pred, t_tgt_src_pred, t1_T_t[iii:iii+1], R_LOSS_enhenace=R_LOSS_enhenace)
                            all_t1_T_t.append(t1_T_t[iii:iii+1].cpu().numpy().reshape((4, 4)))
                            homogeneous_transform = np.eye(4)
                            homogeneous_transform[:3, :3] = R_tgt_src_pred.cpu().detach().numpy()
                            homogeneous_transform[:3, 3] = t_tgt_src_pred.cpu().detach().numpy().reshape((3,))
                            all_homogeneous_transform.append(homogeneous_transform)

                            total_svd_loss_val += svd_loss_val.detach().cpu().item()
                            total_R_loss_val += dict_loss_val["R_loss"].detach().cpu().item()
                            total_t_loss_val += dict_loss_val["t_loss"].detach().cpu().item()
                            total_num_val += 1
                            
                            if not DEBUG and (total_num_val >= 1000 or total_num_val >= len(dataloader_test) // 5) and dict_DWP_Solver_draw is None:  
                                dict_DWP_Solver_draw = dict_DWP_Solver
                                dict_DWP_Solver_draw["batch_all_2_5_images"] = batch_all_2_5_images[iii]
                                dict_DWP_Solver_draw["batch_all_2_5_depth"] = batch_all_2_5_depth[iii]
                            
                            if DEBUG and total_num_val >= len(dataloader_test) // 200 and dict_DWP_Solver_draw is None: 
                                dict_DWP_Solver_draw = dict_DWP_Solver
                                dict_DWP_Solver_draw["batch_all_2_5_images"] = batch_all_2_5_images[iii]
                                dict_DWP_Solver_draw["batch_all_2_5_depth"] = batch_all_2_5_depth[iii]

                        if (total_num_val >= (len(dataloader_test) // 200 + 10)) and DEBUG:
                            break

                    with open(log_file_path, "a") as f:
                        original_stdout = sys.stdout
                        sys.stdout = f

                        if not JUST_TEST and total_num != 0 and test_num_seq == 1:
                            avg_svd_loss = total_svd_loss / total_num  
                            avg_R_loss = total_R_loss / total_num 
                            avg_t_loss = total_t_loss / total_num 
                            print("num_epoch: {:d}\tavg_svd_loss: {:.6f}\tavg_R_loss: {:.6f}\tavg_t_loss: {:.6f}".format((num_epoch + 1), avg_svd_loss, avg_R_loss, avg_t_loss))

                        avg_svd_loss_val = total_svd_loss_val / total_num_val 
                        avg_R_loss_val = total_R_loss_val / total_num_val 
                        avg_t_loss_val = total_t_loss_val / total_num_val  

                        t_err_avg, R_err_avg = monitor.step_val(feature_size, avg_svd_loss_val, avg_R_loss_val, avg_t_loss_val, all_t1_T_t, all_homogeneous_transform, dict_DWP_Solver_draw, \
                            timestamps_val, mats_dict, file_path_gt='./train_model/evo/tum_trajectory_gt_'+str(NAME)+'_'+str(num_epoch)+'.txt', \
                            file_path_pred='./train_model/evo/tum_trajectory_pred_'+str(NAME)+'_'+str(num_epoch)+'.txt', IS_MONO_CUT=IS_MONO_CUT, dataset_type=dataset_type)
                        
                        t_err_avg_all.append(t_err_avg)
                        R_err_avg_all.append(R_err_avg)

                        if test_num_seq == len(all_test):
                            t_err_avg_record = np.mean(t_err_avg_all)
                            R_err_avg_record = np.mean(R_err_avg_all)
                            monitor.step_val_ral_record(t_err_avg_record, R_err_avg_record)

                        print("num_epoch: {:d}\tavg_svd_loss_val: {:.6f}\tavg_R_loss_val: {:.6f}\tavg_t_loss_val: {:.6f}\n----------------------------------------" \
                        .format((num_epoch + 1), avg_svd_loss_val, avg_R_loss_val, avg_t_loss_val))
                        print("t_err_avg: {:.6f}\tR_err_avg: {:.6f}".format(t_err_avg, R_err_avg))

                        sys.stdout = original_stdout

                torch.cuda.empty_cache()
                if JUST_TEST:
                    print("finish test")
                    break

            else:
                avg_svd_loss = total_svd_loss / total_num  
                avg_R_loss = total_R_loss / total_num  
                avg_t_loss = total_t_loss / total_num 
                print("num_epoch: {:d}\tavg_svd_loss: {:.6f}\tavg_R_loss: {:.6f}\tavg_t_loss: {:.6f}".format((num_epoch + 1), avg_svd_loss, avg_R_loss, avg_t_loss))
            num_epoch += 1