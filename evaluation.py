import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
from IPython import embed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']

pmod_folder = './save/230304_pmod/'
if not os.path.exists(pmod_folder):
    os.makedirs(pmod_folder)


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, input_type, threshold):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder, '%s.npy' % name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((21, h, w), np.float32)
                for key in predict_dict.keys():
                    tensor[key + 1] = predict_dict[key]
                tensor[0, :, :] = threshold
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

                # save pmod pseudo labels
                out = Image.fromarray(predict.astype(np.uint8), mode='P')
                out.putpalette(palette)
                out_name = pmod_folder + '/' + name + '.png'
                out.save(out_name)

            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='voc12/train_id.txt', type=str)   # or 'voc12/val_id.txt', 'voc12/trainaug_id.txt'
    parser.add_argument("--predict_dir", default='save/out_cam', type=str)
    parser.add_argument("--gt_dir", default='../VOCdevkit/VOC2012/SegmentationClassAug', type=str)
    parser.add_argument('--logfile', default='./evallog_MECPformer.txt', type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    # parser.add_argument('--t', default=0.45, type=float)
    # parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=True, type=bool)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        print('=====begining...=====')
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, args.t, printlog=True)
        writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        for i in range(30, 60):
            t = i / 100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, t)
            l.append(loglist['mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%%' % (i, t, loglist['mIoU']))
        writelog(args.logfile, {'mIoU': l}, args.comment)
