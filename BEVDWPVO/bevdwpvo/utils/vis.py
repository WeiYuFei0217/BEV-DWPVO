import io
import PIL.Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch
import torchvision.utils as vutils
from torchvision.transforms import ToTensor

from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import os
from matplotlib import cm
from tqdm import tqdm
import cv2

def enforce_orthog(T, dim=3):
    if dim == 2:
        if abs(np.linalg.det(T[0:2, 0:2]) - 1) < 1e-10:
            return T
        R = T[0:2, 0:2]
        epsilon = 0.001
        if abs(R[0, 0] - R[1, 1]) > epsilon or abs(R[1, 0] + R[0, 1]) > epsilon:
            print("WARNING: this is not a proper rigid transformation:", R)
            return T
        a = (R[0, 0] + R[1, 1]) / 2
        b = (-R[1, 0] + R[0, 1]) / 2
        s = np.sqrt(a**2 + b**2)
        a /= s
        b /= s
        R[0, 0] = a
        R[0, 1] = b
        R[1, 0] = -b
        R[1, 1] = a
        T[0:2, 0:2] = R
    if dim == 3:
        if abs(np.linalg.det(T[0:3, 0:3]) - 1) < 1e-10:
            return T
        c1 = T[0:3, 1]
        c2 = T[0:3, 2]
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        newcol0 = np.cross(c1, c2)
        newcol1 = np.cross(c2, newcol0)
        T[0:3, 0] = newcol0
        T[0:3, 1] = newcol1
        T[0:3, 2] = c2
    return T

def get_inverse_tf(T):
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2

def convert_plt_to_img():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return PIL.Image.open(buf)

def convert_plt_to_tensor():
    return ToTensor()(convert_plt_to_img())

def draw_batch(feature_size, dict_DWP_Solver_draw, config, IS_MONO_CUT=False, dataset_type='NCLT'):
    """Creates an image of the radar scan, scores, and keypoint matches for a single batch."""
    radar_img = []
    BEV_feature = dict_DWP_Solver_draw['x']
    x_save = torch.mean(BEV_feature, dim=1)
    x_save = x_save.reshape(BEV_feature.shape[0], BEV_feature.shape[2], BEV_feature.shape[3])
    for shape_0 in range(x_save.shape[0]):
        img = x_save[shape_0].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        
        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img)

        img_rgb = np.stack((img, img, img), axis=0)
        radar_img.append(torch.from_numpy(img_rgb.astype('uint8')))
    src = dict_DWP_Solver_draw['src'][0].squeeze().detach().cpu().numpy()
    tgt = dict_DWP_Solver_draw['tgt'][0].squeeze().detach().cpu().numpy()
    match_weights = dict_DWP_Solver_draw['match_weights'][0].squeeze().detach().cpu().numpy()
    nms = config['vis_keypoint_nms']
    max_w = np.max(match_weights)
    
    if IS_MONO_CUT:
        match_img = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
        radar_np = radar_img[0].numpy()
        radar_np = np.transpose(radar_np, (1, 2, 0))
        radar_img_part = Image.fromarray(radar_np.astype('uint8'))
        if dataset_type == 'NCLT' or dataset_type == 'kitti':
            match_img.paste(radar_img_part, (config['cart_pixel_width'] // 2, 0))
        elif dataset_type == 'oxford':
            match_img.paste(radar_img_part, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))
    else:
        match_img = radar_img[0].numpy()
        match_img = np.transpose(match_img, (1, 2, 0))
        match_img = Image.fromarray(match_img.astype('uint8'))

    draw = ImageDraw.Draw(match_img)
    for i in range(src.shape[0]):
        if match_weights[i] < nms * max_w:
            continue
        blue_color = (0, 0, 255)
        draw.line([(int(src[i, 0]), int(src[i, 1])), (int(tgt[i, 0]), int(tgt[i, 1]))], fill=blue_color, width=1)
        green_color = (0, 255, 0)
        draw.point((int(src[i, 0]), int(src[i, 1])), fill=green_color)
        red_color = (255, 0, 0)
        draw.point((int(tgt[i, 0]), int(tgt[i, 1])), fill=red_color)
    match_img = transforms.ToTensor()(match_img)

    scores = dict_DWP_Solver_draw['scores'][0].squeeze().detach().cpu().numpy()
    print("scores.min(), scores.max(), scores.mean():", scores.min(), scores.max(), scores.mean())
    scaled_scores = ((scores - scores.min()) / (scores.max() - scores.min()) * 255).astype('uint8')
    print("scaled_scores.min(), scaled_scores.max(), scaled_scores.mean():", scaled_scores.min(), scaled_scores.max(), scaled_scores.mean())
    score_img = Image.fromarray(scaled_scores)
    score_img = score_img.convert('RGB')
    score_img = transforms.ToTensor()(score_img)

    if IS_MONO_CUT:
        radar_img_0 = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
        radar_np = radar_img[0].numpy()
        radar_np = np.transpose(radar_np, (1, 2, 0))
        radar_img_part = Image.fromarray(radar_np.astype('uint8'))
        if dataset_type == 'NCLT' or dataset_type == 'kitti':
            radar_img_0.paste(radar_img_part, (config['cart_pixel_width'] // 2, 0))
        elif dataset_type == 'oxford':
            radar_img_0.paste(radar_img_part, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))
        radar_img_1 = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
        radar_np = radar_img[1].numpy()
        radar_np = np.transpose(radar_np, (1, 2, 0))
        radar_img_part = Image.fromarray(radar_np.astype('uint8'))
        if dataset_type == 'NCLT' or dataset_type == 'kitti':
            radar_img_1.paste(radar_img_part, (config['cart_pixel_width'] // 2, 0))
        elif dataset_type == 'oxford':
            radar_img_1.paste(radar_img_part, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))
        score_img_new = Image.new('RGB', (config['cart_pixel_width'] * 2, config['cart_pixel_width'] * 2), color='black')
        score_img *= 255
        score_img = score_img.numpy()
        score_img = np.transpose(score_img, (1, 2, 0))
        score_img = Image.fromarray(score_img.astype('uint8'))
        if dataset_type == 'NCLT' or dataset_type == 'kitti':
            score_img_new.paste(score_img, (config['cart_pixel_width'] // 2, 0))
        elif dataset_type == 'oxford':
            score_img_new.paste(score_img, (config['cart_pixel_width'] // 2, config['cart_pixel_width']))
        radar_img_0 = transforms.ToTensor()(radar_img_0)
        radar_img_1 = transforms.ToTensor()(radar_img_1)
        score_img_new = transforms.ToTensor()(score_img_new)
        return vutils.make_grid([radar_img_0, radar_img_1, score_img_new, match_img])

    return vutils.make_grid([radar_img[0], radar_img[1], score_img, match_img])

def histogram_equalization(img):
    flat_img = img.flatten()
    
    histogram, bin_edges = np.histogram(flat_img, bins=256, range=(flat_img.min(), flat_img.max()))
    cdf = histogram.cumsum()
    cdf_normalized = cdf / cdf.max()

    equalized_img = np.interp(flat_img, bin_edges[:-1], cdf_normalized)
    equalized_img = equalized_img.reshape(img.shape)
    
    return equalized_img

def min_max_scaling(img):
    img_min = img.min()
    img_max = img.max()
    scaled_img = (img - img_min) / (img_max - img_min)
    
    return scaled_img

def plot_sequences(T_gt, T_pred, seq_lens, returnTensor=True, T_icra=None, savePDF=False, fnames=None, flip=True):
    """Creates a top-down plot of the predicted odometry results vs. ground truth."""
    seq_indices = []
    idx = 0
    for s in seq_lens:
        seq_indices.append(list(range(idx, idx + s - 1)))
        idx += (s - 1)

    T_flip = np.identity(4)
    T_flip[1, 1] = -1
    T_flip[2, 2] = -1
    imgs = []
    for seq_i, indices in enumerate(seq_indices):
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        if flip:
            T_gt_ = np.matmul(T_flip, T_gt_)
            T_pred_ = np.matmul(T_flip, T_pred_)
        x_gt = []
        y_gt = []
        x_pred = []
        y_pred = []
        x_icra = []
        y_icra = []
        for i in indices:
            T_gt_ = np.matmul(T_gt[i], T_gt_)
            T_pred_ = np.matmul(T_pred[i], T_pred_)
            enforce_orthog(T_gt_)
            enforce_orthog(T_pred_)
            T_gt_temp = get_inverse_tf(T_gt_)
            T_pred_temp = get_inverse_tf(T_pred_)
            x_gt.append(T_gt_temp[0, 3])
            y_gt.append(T_gt_temp[1, 3])
            x_pred.append(T_pred_temp[0, 3])
            y_pred.append(T_pred_temp[1, 3])

        img = draw_plot(x_gt, y_gt, x_pred, y_pred)
        if returnTensor:
            imgs.append(transforms.ToTensor()(img))
        else:
            imgs.append(img)

    return imgs

def plot_2d_trajectory(T_gt, T_pred, plane='xy'):
    fig, ax = plt.subplots()

    x_gt, y_gt = [], []
    x_pred, y_pred = [], []

    T_gt_accum = np.identity(4)
    T_pred_accum = np.identity(4)

    for T_g, T_p in zip(T_gt, T_pred):
        T_gt_accum = np.matmul(T_gt_accum, T_g)
        T_pred_accum = np.matmul(T_pred_accum, T_p)
        
        if plane == 'xy':
            x_gt.append(T_gt_accum[0, 3])
            y_gt.append(T_gt_accum[1, 3])
            x_pred.append(T_pred_accum[0, 3])
            y_pred.append(T_pred_accum[1, 3])
        elif plane == 'yz':
            x_gt.append(T_gt_accum[1, 3])
            y_gt.append(T_gt_accum[2, 3])
            x_pred.append(T_pred_accum[1, 3])
            y_pred.append(T_pred_accum[2, 3])
        elif plane == 'xz':
            x_gt.append(T_gt_accum[0, 3])
            y_gt.append(T_gt_accum[2, 3])
            x_pred.append(T_pred_accum[0, 3])
            y_pred.append(T_pred_accum[2, 3])

    ax.plot(x_gt, y_gt, label='Ground Truth')
    ax.plot(x_pred, y_pred, label='Prediction')

    ax.set_xlabel(plane[0].upper())
    ax.set_ylabel(plane[1].upper())
    ax.legend()

    return fig

def draw_plot(x_gt, y_gt, x_pred, y_pred):
    """Draws a top-down plot of the predicted odometry results vs. ground truth using PIL."""

    if max(x_gt + x_pred) == float('inf') or min(x_gt + x_pred) == float('-inf') or max(y_gt + y_pred) == float('inf') or min(y_gt + y_pred) == float('-inf'):
        return Image.new('RGB', (1000, 1000), color='white')

    center_x = (max(x_gt + x_pred) + min(x_gt + x_pred)) / 2
    center_y = (max(y_gt + y_pred) + min(y_gt + y_pred)) / 2

    img_width = 1000
    img_height = 1000

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Calculate the scaling factor
    scale_factor = min((img_width-100) / (max(x_gt + x_pred) - min(x_gt + x_pred)),
                      (img_height-100) / (max(y_gt + y_pred) - min(y_gt + y_pred)))

    # Draw X and Y axis
    draw.line([(0, 950), (img_width, 950)], fill='black', width=2)
    draw.line([(50, 0), (50, img_height)], fill='black', width=2)

    axis_bounds_width = img_width / scale_factor
    axis_bounds_height = img_height / scale_factor

    if int(axis_bounds_width/22) != 0 and int(axis_bounds_width/22) != 0:
        # Draw tick marks and labels on X axis
        for i in range(-int(axis_bounds_width/2), int(axis_bounds_width/2), int(axis_bounds_width/22)):
            tick_x = int((i * scale_factor) + img_width / 2)
            tick_y = int(950)
            draw.line([(tick_x, tick_y - 5), (tick_x, tick_y + 5)], fill='black', width=2)
            draw.text((tick_x - 5, tick_y + 10), str(i), fill='black', font=None)

        # Draw tick marks and labels on Y axis
        for i in range(-int(axis_bounds_height/2), int(axis_bounds_height/2), int(axis_bounds_width/22)):
            tick_x = int(50)
            tick_y = int((-i * scale_factor) + img_height / 2)
            draw.line([(tick_x - 5, tick_y), (tick_x + 5, tick_y)], fill='black', width=2)
            draw.text((tick_x - 20, tick_y - 5), str(i), fill='black', font=None)

        # Draw ground truth
        for i in range(len(x_gt) - 1):
            draw.line([(x_gt[i] - center_x) * scale_factor + img_width / 2,
                    (y_gt[i] - center_y) * scale_factor + img_height / 2,
                    (x_gt[i + 1] - center_x) * scale_factor + img_width / 2,
                    (y_gt[i + 1] - center_y) * scale_factor + img_height / 2],
                    fill='black', width=3)
        draw.text((100, 50), 'black-gt', fill='black', font=None)

        # Draw predicted
        for i in range(len(x_pred) - 1):
            draw.line([(x_pred[i] - center_x) * scale_factor + img_width / 2,
                    (y_pred[i] - center_y) * scale_factor + img_height / 2,
                    (x_pred[i + 1] - center_x) * scale_factor + img_width / 2,
                    (y_pred[i + 1] - center_y) * scale_factor + img_height / 2],
                    fill='blue', width=3)
        draw.text((100, 65), 'blue-pred', fill='blue', font=None)

    return img