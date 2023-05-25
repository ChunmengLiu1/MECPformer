import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import embed
import scipy.misc
import cmapy

def save_att(label, norm_cam, save_path, img_name):
    for i in range(20):
        if label[i] == 1:
            # embed(header='-----------infer_utils.py.py:251------------')
            att = norm_cam[i]
            # att[att < 0] = 0    # exp2 attention
            # att[att < 0.31] = 0    # exp1 attention
            # embed(header='-----------infer_utils.py.py:254------------')
            att = att/(np.max(att) + 1e-8)
            att = np.array(att * 255, dtype=np.uint8)
            # att = [281, 500], minmax = [0, 255]
            # embed(header='-----------infer_utils.py.py:257------------')
            # save_path = './save/final_result'
            out_name = save_path + '/' + img_name + '_{}.png'.format(i)
            # out_name = './save/attention/train_aug/2007_000032_0.png'
            # embed(header='-----------infer_utils.py.py:260------------')
            cv2.imwrite(out_name, att)
    return label


def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam,  cmapy.cmap('seismic'))
    return cam

def draw_heatmap_array(img, hm):     # hm = norm_cam[gt]
    # img = [1, 3, 281, 500]

    # 想使用一下th, 先把th去掉
    # hm[hm <= th] = 0
    # embed(header='-----------infer_utils.py:106------------')

    img = img.squeeze(0)
    img = img.permute(1, 2, 0)  # [224, 224, 3], minmax = [0, 255]
    hm = plt.cm.hot(hm)[:, :, :3]  # hm = [224, 224, 3], minmax = [0, 1]

    # embed(header='-----------infer_utils.py:112------------')
    hm = np.array(
        Image.fromarray((hm * 255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(
        np.float) * 2

    # embed(header='-----------infer_utils.py:117------------')
    if hm.shape == np.array(img).astype(np.float).shape:
        # hm.shape = [224, 224, 3], np.array(img).astype(np.float).shape = [224, 224, 3]
        # embed(header='-----------infer_utils.py:120------------')
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(
            "hm.shape not equal np.array(img).astype(np.float).shape")  # 这里刚开始报错：out未定义就使用，原因是传起来的shape顺序不对，需要squeeze,permute
    # embed(header='-----------infer_utils.py:126------------')
    # out.shape = [224, 224, 3], out_minmax = [0, 255], hm.shape = [224, 224, 3], hm_minmax = [0, 510], very strange value

    return hm, out  # 保存为了out0428.png


def draw_single_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label == 1)[0]
    orig_img = orig_img.squeeze(0)
    orig_img = orig_img.permute(1, 2, 0)
    orig_img = orig_img.numpy()     #  [281, 500, 3], minmax = [0, 255], numpy
    for i, gt in enumerate(gt_cat):
        # embed(header='-----------infer_utils.py:274------------')
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))  # './save/heatmap/train_aug/2007_000032_0.png'
        # embed(header='-----------infer_utils.py:79------------')
        show_cam_on_image(orig_img, norm_cam[gt], cam_viz_path)
        # orig_img = [1, 3, 366, 500], minmax = [0, 255], numpy
        # norm_cam[gt] = [366, 500], minmax = [0, 1], numpy



def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)