import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageDraw

from utils.common import get_inverse_tf

plt.switch_backend('agg')


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


def _to_full_canvas(img, width, bev_half):
    """Pastes a half-cropped BEV image back onto a full-size canvas."""
    canvas = Image.new('RGB', (width * 2, width * 2), color='black')
    offset = (width // 2, 0) if bev_half == 'front' else (width // 2, width)
    canvas.paste(img, offset)
    return canvas


def draw_batch(outputs, config, bev_half=None):
    """Creates an image of the two BEV feature maps, validity weights and keypoint matches of one frame pair."""
    bev_imgs = []
    bev = outputs['bev']
    bev_mean = torch.mean(bev, dim=1).reshape(bev.shape[0], bev.shape[2], bev.shape[3])
    for k in range(bev_mean.shape[0]):
        img = bev_mean[k].detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = cv2.equalizeHist(img.astype(np.uint8))
        bev_imgs.append(Image.fromarray(np.stack((img, img, img), axis=-1)))

    src = outputs['src'][0].squeeze().detach().cpu().numpy()
    tgt = outputs['tgt'][0].squeeze().detach().cpu().numpy()
    match_weights = outputs['match_weights'][0].squeeze().detach().cpu().numpy()
    nms = config['vis_keypoint_nms']
    max_w = np.max(match_weights)
    width = config['bev_width']

    validity = outputs['validity'][0].squeeze().detach().cpu().numpy()
    validity = ((validity - validity.min()) / (validity.max() - validity.min()) * 255).astype('uint8')
    validity_img = Image.fromarray(validity).convert('RGB')

    if bev_half is not None:
        bev_imgs = [_to_full_canvas(img, width, bev_half) for img in bev_imgs]
        validity_img = _to_full_canvas(validity_img, width, bev_half)

    match_img = bev_imgs[0].copy()
    draw = ImageDraw.Draw(match_img)
    for i in range(src.shape[0]):
        if match_weights[i] < nms * max_w:
            continue
        draw.line([(int(src[i, 0]), int(src[i, 1])), (int(tgt[i, 0]), int(tgt[i, 1]))], fill=(0, 0, 255), width=1)
        draw.point((int(src[i, 0]), int(src[i, 1])), fill=(0, 255, 0))
        draw.point((int(tgt[i, 0]), int(tgt[i, 1])), fill=(255, 0, 0))

    to_tensor = transforms.ToTensor()
    return vutils.make_grid([to_tensor(bev_imgs[0]), to_tensor(bev_imgs[1]), to_tensor(validity_img), to_tensor(match_img)])


def plot_sequences(T_gt, T_pred, seq_lens, returnTensor=True, flip=True):
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
    for indices in seq_indices:
        T_gt_ = np.identity(4)
        T_pred_ = np.identity(4)
        if flip:
            T_gt_ = np.matmul(T_flip, T_gt_)
            T_pred_ = np.matmul(T_flip, T_pred_)
        x_gt, y_gt, x_pred, y_pred = [], [], [], []
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
        imgs.append(transforms.ToTensor()(img) if returnTensor else img)
    return imgs


def draw_plot(x_gt, y_gt, x_pred, y_pred):
    """Draws a top-down plot of the predicted odometry results vs. ground truth using PIL."""
    xs, ys = x_gt + x_pred, y_gt + y_pred
    if max(xs) == float('inf') or min(xs) == float('-inf') or max(ys) == float('inf') or min(ys) == float('-inf'):
        return Image.new('RGB', (1000, 1000), color='white')

    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    img_width = 1000
    img_height = 1000

    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    scale_factor = min((img_width - 100) / (max(xs) - min(xs)), (img_height - 100) / (max(ys) - min(ys)))

    draw.line([(0, 950), (img_width, 950)], fill='black', width=2)
    draw.line([(50, 0), (50, img_height)], fill='black', width=2)

    axis_bounds_width = img_width / scale_factor
    axis_bounds_height = img_height / scale_factor
    tick = int(axis_bounds_width / 22)
    if tick != 0:
        for i in range(-int(axis_bounds_width / 2), int(axis_bounds_width / 2), tick):
            tick_x = int((i * scale_factor) + img_width / 2)
            draw.line([(tick_x, 945), (tick_x, 955)], fill='black', width=2)
            draw.text((tick_x - 5, 960), str(i), fill='black')
        for i in range(-int(axis_bounds_height / 2), int(axis_bounds_height / 2), tick):
            tick_y = int((-i * scale_factor) + img_height / 2)
            draw.line([(45, tick_y), (55, tick_y)], fill='black', width=2)
            draw.text((30, tick_y - 5), str(i), fill='black')

        for i in range(len(x_gt) - 1):
            draw.line([(x_gt[i] - center_x) * scale_factor + img_width / 2,
                       (y_gt[i] - center_y) * scale_factor + img_height / 2,
                       (x_gt[i + 1] - center_x) * scale_factor + img_width / 2,
                       (y_gt[i + 1] - center_y) * scale_factor + img_height / 2],
                      fill='black', width=3)
        draw.text((100, 50), 'black-gt', fill='black')

        for i in range(len(x_pred) - 1):
            draw.line([(x_pred[i] - center_x) * scale_factor + img_width / 2,
                       (y_pred[i] - center_y) * scale_factor + img_height / 2,
                       (x_pred[i + 1] - center_x) * scale_factor + img_width / 2,
                       (y_pred[i + 1] - center_y) * scale_factor + img_height / 2],
                      fill='blue', width=3)
        draw.text((100, 65), 'blue-pred', fill='blue')

    return img
