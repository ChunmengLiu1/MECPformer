import cv2
from PIL import Image
import numpy as np
import os
from os.path import exists
import torch

'''
This is Adaptation Conflict Module
'''

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_lst_path = './voc12/train_aug_cls.txt'
sal_path = '../VOCdevkit/VOC2012/saliency_map_DRS/'
att_path = './save/attention/train_aug/'
save_path = './save/0525_ACM_03/'
gt_folder_path = '../VOCdevkit/VOC2012/SegmentationClassAug/'



if not exists(save_path):
    os.makedirs(save_path)

with open(train_lst_path) as f:
    lines = f.readlines()


for i, line in enumerate(lines):
    print(i)
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    bg_name = sal_path + name + '.png'
    if not os.path.exists(bg_name):
        print('=====do not exist bg_name')
        continue

    sal = cv2.imread(bg_name, 0)
    height, width = sal.shape
    gt = np.zeros((21, height, width), dtype=np.float32)
    sal = np.array(sal, dtype=np.float32)

    # some thresholds
    conflict_rate = 0.9
    fg_thr = 0.45
    bg_thr = 32
    att_thr = 0.8

    # use saliency map to provide background cues
    gt[0] = (1 - (sal / 255))
    sal_att = sal.copy()

    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])

        att_name = att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            print('======do not exist att_name======')
            continue

        # normalize attention to [0, 1]
        att = cv2.imread(att_name, 0)

        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        gt[cls + 1] = att.copy()
        sal_att = np.maximum(sal_att, (att > att_thr) * 255)

    # throw low confidence values for all classes
    gt[gt < fg_thr] = 0

    # adaptation conflict
    max_gt = np.max(gt, axis=0, keepdims=True)  # max_gt = [1, 281, 500]
    gt = torch.from_numpy(gt)
    max_gt = torch.from_numpy(max_gt)
    max_gt = max_gt.expand_as(gt)
    conflict = max_gt * conflict_rate
    gt = gt.numpy()
    conflict = conflict.numpy()

    # conflict pixels with multiple confidence values
    bg = np.array(gt > conflict, dtype=np.uint8)
    bg = np.sum(bg, axis=0)

    gt = gt.argmax(0).astype(np.uint8)
    gt[bg > 1] = 255

    # pixels regarded as background but confidence saliency values
    bg = np.array(sal_att >= bg_thr, dtype=np.uint8) * np.array(gt == 0, dtype=np.uint8)
    gt[bg > 0] = 255

    out = gt
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        # ratio = 0,
        out[...] = 255


    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)

print('over')
