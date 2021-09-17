#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Authors: Lukas Burget, Federico Landini
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz

import numpy as np
import features
import struct
import kaldi_io
import os
import sys
from tqdm import tqdm
from tdnn_model import TDNN_LSTM
import torch
import torchaudio
import soundfile as sf
import features
from common import AverageMeter, slide_window, unslide_window

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

def write_vad_fb(model, fb, out_file, threshold=0.5, slide_len=500):
    frame_shift = 0.01
    slide_len = int(500)
    slide_hop = int(0.5*slide_len)
    model.eval()
    act_dur = 0.0
    tot_dur = 0.0

    # wav_batch = []
    assert fb.dim()==2
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
    print(f"Act prob of {os.path.basename(out_file)} is {act.sum().float()/len(act)*100}% in {wlen} s")

    seg = flabel2seg(act, wlen, frame_shift)
    # delete the short active and fill the short silence < 0.05s
    if len(seg) > 0:
        # seg = smooth_seg_fg(seg, gap=0.05)
        seg = smooth_seg_gf(seg, gap=0.05)  # gap + fill

    with open(out_file, "w") as lab_file:
        if len(seg) == 0:
            lab_file.write("0.00 0.001 speech\n")
            return None
        for i in range(len(seg)):
            lab_file.write("%.3f %.3f speech\n" % (seg[i][0], seg[i][1]))

#####################################################
def main():
    parser = argparse.ArgumentParser(description="SAD Net")
    # Data loader
    parser.add_argument('--frame_range', type=int, default=[300, 300],  help='Input length to the network')
    # Training details
    # parser.add_argument('--load_model_path', type=str, default="", help='Load model weights')
    parser.add_argument('--load_model_path', type=str, default="/apdcephfs/private_naijunzheng/DATA/SAD/model/tdnn_lstm_4_sad_aspire_5s_fb256.tar", help='Load model weights')
    # Model definition
    parser.add_argument('--out_file_dir', type=str, default="./example", help='seg lab txt written')
    # parser.add_argument('--model', type=str, default="tdnn_lstm", help='Name of model definition')
    parser.add_argument('--threshold', type=float, default=-1, help='thr for VAD')
    args = parser.parse_args()

    in_file_list = args.file_list
    in_flac_dir = args.file_dir  # directory with flac files
    # find flac or wav
    if 'wav' in in_flac_dir.split('/')[-2:]:
        suffix = 'wav'
    elif 'flac' in in_flac_dir.split('/')[-2:]:
        suffix = 'flac'

    out_lab_dir = args.out_file_dir  # directory with files with VAD information

    file_names = np.loadtxt(in_file_list, dtype=object, ndmin=1)
    assert file_names.ndim >= 1

    # import ipdb; ipdb.set_trace()
    # load sad net
    model = TDNN_LSTM(hidden_size=256)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True
    load_model_path = args.load_model_path  # sad_model
    # reload model
    if os.path.isfile(load_model_path):
        print("=> loading SAD model checkpoint '{}'".format(load_model_path))
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        if args.threshold <= 0.0:
            threshold = checkpoint['threshold']
        else:
            threshold = args.threshold
        print(f"with miss_fa_error={checkpoint['miss_fa_error']*100}% with threshold={threshold}")
    else:
        assert False

    if not os.path.isdir(out_lab_dir):
        os.makedirs(out_lab_dir)

    noverlap = 240
    winlen = 400
    fs = 16000
    window = features.povey_window(winlen)
    fbank_mx = features.mel_fbank_mx(winlen, fs, NUMCHANS=40, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
    LC = 150
    RC = 149
    # import ipdb; ipdb.set_trace()
    for fn in tqdm(file_names):
        inpwave_name = in_flac_dir+"/"+fn+"."+suffix
        # extract fbank
        signal, samplerate = sf.read(in_flac_dir+"/"+fn+"."+suffix)
        # signal = features.add_dither((signal_o*2**(samplerate/1000 - 1)).astype(int))
        signal = signal*2**(samplerate/1000 - 1)
        fb = features.fbank_htk(signal, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        fb = features.cmvn_floating_kaldi(fb, LC, RC, norm_vars=False)
        fb = torch.tensor(fb).float()
        out_file = out_lab_dir+'/'+fn+'.lab'
        write_vad_fb(model, fb, out_file, threshold=threshold, slide_len=args.frame_range[0])


    print("Finish VAD!")

if __name__ == '__main__':
    main()
