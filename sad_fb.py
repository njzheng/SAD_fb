import argparse
import time
import os
import h5py
from tqdm import tqdm
import glob
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tdnn_model import TDNN_LSTM
from common import AverageMeter, slide_window, unslide_window

from dataset_sad import SADDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio


def flabel2seg(flabel, wlen, hop=0.01):
    # hop = 10ms
    seg = []
    start_t = 0.0
    for i in range(len(flabel)-1):
        if flabel[i] == 0 and flabel[i+1] == 1:
            start_t = (i+1)*hop
        elif flabel[i] == 1 and flabel[i+1] == 0:
            end_t = min((i+1)*hop, wlen)
            seg.append([start_t, end_t])

    if flabel[-1] == 1:
        if start_t < wlen:
            seg.append([start_t, wlen])

    return seg

# first gap and then fill
def smooth_seg_gf(seg, gap=0.05):
    seg_new = []
    # first delete the short active regions
    for s in seg:
        if s[-1] - s[0] >= gap:
            seg_new.append(s)

    seg_new2 = []
    starti = seg_new[0][0]
    endi = seg_new[0][1]
    for i in range(len(seg_new)-1):
        if endi + gap > seg_new[i+1][0]:
            endi = seg_new[i+1][1]
        else:
            # write the seg
            seg_new2.append([starti, endi])
            # update
            starti = seg_new[i+1][0]
            endi = seg_new[i+1][1]

    seg_new2.append([starti, endi])
    return seg_new2

# first fill and then gap
def smooth_seg_fg(seg, gap=0.05):
    seg_new = []
    starti = seg[0][0]
    endi = seg[0][1]
    for i in range(len(seg)-1):
        if endi + gap > seg[i+1][0]:
            endi = seg[i+1][1]
        else:
            # write the seg
            seg_new.append([starti, endi])
            # update
            starti = seg[i+1][0]
            endi = seg[i+1][1]

    seg_new.append([starti, endi])

    seg_new2 = []
    # first delete the short active regions
    for s in seg_new:
        if s[-1] - s[0] >= gap:
            seg_new2.append(s)

    return seg_new2

def write_vad_fb(model, fb_file, out_file, threshold=0.5):
    # import ipdb; ipdb.set_trace()
    frame_shift = 0.01
    slide_len = int(500)
    slide_hop = int(0.5*slide_len)

    model.eval()
    act_dur = 0.0
    tot_dur = 0.0

    # wav_batch = []
    fb = torch.tensor(np.load(fb_file))
    assert fb.dim()==2

    # wav = wav[:fs*100]
    wlen = float(len(fb)*frame_shift)
    fb_seg = slide_window(fb, slide_len, slide_hop)
    fb_seg = fb_seg.cuda()
    with torch.no_grad():
        _, act_seg = model.infer(fb_seg)

    # convert from flabel to seg
    act_seg = act_seg.squeeze(-1).cpu()
    # concate act_seg, average the overlap regions
    act_temp = unslide_window(act_seg, hop=0.5)
    act = (act_temp > threshold).int()
    act = act[:len(fb)]

    act_dur += act.sum().float()
    tot_dur += len(act)
    print(f"Prob of act is {act.sum().float()/len(act)*100}% in {wlen} s")

    seg = flabel2seg(act, wlen, frame_shift)
    # delete the short active and fill the short silence < 0.05s
    seg = smooth_seg_fg(seg, gap=0.05)

    with open(out_file, "w") as lab_file:
        lab_file.write("file: %s\n" % (fb_file))
        for i in range(len(seg)):
            lab_file.write("%.3f %.3f speech\n" % (seg[i][0], seg[i][1]))


def main():
    parser = argparse.ArgumentParser(description="SAD Net")
    # Data loader
    parser.add_argument('--frame_range', type=int, default=[500, 500],  help='Input length to the network')
    # Training details
    # parser.add_argument('--load_model_path', type=str, default="", help='Load model weights')
    parser.add_argument('--load_model_path', type=str, default="./example/tdnn_lstm_4_sad_aspire_5s_fb256.tar", help='Load model weights')
    # Model definition
    parser.add_argument('--fb_file', type=str, default="./example/feat_mat.npy", help='np mat file')
    parser.add_argument('--out_file', type=str, default="./example/lab.txt", help='seg lab txt written')
    parser.add_argument('--model', type=str, default="tdnn_lstm", help='Name of model definition')
    args = parser.parse_args()

    if args.model == 'tdnn':
        model = TDNN()
    elif args.model == 'tdnn_lstm':
        model = TDNN_LSTM(hidden_size=256)

    model = model.cuda()
    torch.backends.cudnn.benchmark = True
    threshold = 0.5
    # reload model
    if args.load_model_path != '':
        if os.path.isfile(args.load_model_path):
            print("=> loading checkpoint '{}'".format(args.load_model_path))
            checkpoint = torch.load(args.load_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            threshold = checkpoint['threshold']
            print(f"with miss_fa_error={checkpoint['miss_fa_error']*100}% with threshold={checkpoint['threshold']}")
        else:
            print("=> no checkpoint found at '{}'".format(args.load_model_path))
            exit()

    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # import ipdb; ipdb.set_trace()
    write_vad_fb(model, fb_file=args.fb_file, out_file=args.out_file, threshold=threshold)
    # exit()


if __name__ == '__main__':
    main()
