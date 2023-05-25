
import numpy as np
import torch
import os
import voc12.data_copy
import voc12.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, infer_utils
import argparse
from PIL import Image
import torch.nn.functional as F


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default='network.conformer_CAM', type=str)
    parser.add_argument("--infer_list", default='voc12/train_aug.txt', type=str)    # or 'voc12/val.txt', 'voc12/train_aug.txt'
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", default='../VOCdevkit/VOC2012', type=str)
    parser.add_argument("--save", default='./save', type=str)
    parser.add_argument("--out_cam", default='save/out_cam', type=str)
    parser.add_argument("--arch", default='sm21', type=str)
    parser.add_argument("--method", default='transcam', type=str)

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if not os.path.exists(args.out_cam):
        os.makedirs(args.out_cam)


    attention_folder = args.save + '/attention' + args.infer_list[5:-4]
    if not os.path.exists(attention_folder):
        os.makedirs(attention_folder)

    heatmap_folder = args.save + '/heatmap' + args.infer_list[5:-4]
    if not os.path.exists(heatmap_folder):
        os.makedirs(heatmap_folder)

    pmod_folder = args.save + '/pmod' + args.infer_list[5:-4]
    if not os.path.exists(pmod_folder):
        os.makedirs(pmod_folder)

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()

    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    infer_dataset = voc12.data_copy.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  inter_transform=torchvision.transforms.Compose(
                                                      [
                                                       np.asarray,
                                                       imutils.Normalize(),
                                                       imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print('infer beginning...')
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]; label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        cam_list = []
        sal_att_list = []

        with torch.no_grad():
            for i, img in enumerate(img_list):
                logits_conv, logits_trans, trans_patch_logits, cam = model(args.method, img.cuda())
                cam = F.interpolate(cam[:, 1:, :, :], orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()

                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)

                cam_list.append(cam)


        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0


        cam_max = np.max(sum_cam, (1,2), keepdims=True)
        cam_min = np.min(sum_cam, (1,2), keepdims=True)
        sum_cam[sum_cam < cam_min+1e-5] = 0
        norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

        ZERO = infer_utils.save_att(label, norm_cam, attention_folder, img_name)

        orig_img_ht = torch.from_numpy(orig_img)
        orig_img_ht = orig_img_ht.permute(2, 0, 1)
        orig_img_ht = orig_img_ht.unsqueeze(0)
        infer_utils.draw_single_heatmap(norm_cam, label, orig_img_ht, heatmap_folder, img_name)

        # generate pmod initial seed
        bg = [np.ones((orig_img.shape[0], orig_img.shape[1])) * 0.40]   # 0.4 is the simple threshold to delete noise information
        cam_21 = np.concatenate((bg, norm_cam), axis=0)    #
        seg_map = np.asarray(np.argmax(cam_21, axis=0), dtype=int)
        out = Image.fromarray(seg_map.astype(np.uint8), mode='P')
        out.putpalette(palette)
        out_name = pmod_folder + '/' + img_name + '.png'
        out.save(out_name)

        cam_dict = {}
        for i in range(20):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

        h, w = list(cam_dict.values())[0].shape
        tensor = np.zeros((21, h, w), np.float32)
        for key in cam_dict.keys():
            tensor[key+1] = cam_dict[key]

        if iter % 500 == 0:
            print('over iter:', iter)
