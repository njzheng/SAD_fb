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
import torchaudio

def get_lab(lab_file):
    seg = []
    with open(lab_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            data = lines[i].split()
            sub_seg = [float(data[0]), float(data[1])]
            seg.append(sub_seg)

    return seg

def get_stm(stm_file):
    # read and get the seg based on stm file
    seg_dict = {}
    with open(stm_file, 'r') as f:
        lines = f.readlines()
        print(f" list {stm_file} has {len(lines)} lines")

        for i in range(len(lines)):
            data = lines[i].split()
            file_name = data[0]
            # start and end time in second
            sub_seg = [float(data[3]), float(data[4])]
            if file_name not in seg_dict:
                seg_dict[file_name] = []

            seg_dict[file_name].append(sub_seg)

    wav_list = list(seg_dict.keys())
    print(f"totally {len(wav_list)} files")

    total_duration = 0
    for wav in seg_dict:
        total_duration += seg_dict[wav][-1][-1]

    print(f"totally {total_duration/60} mins")
    return seg_dict


def get_rttm(rttm_file):
    # read and get the seg based on rttm file
    seg_dict = {}
    with open(rttm_file, 'r') as f:
        lines = f.readlines()
        print(f" list {rttm_file} has {len(lines)} lines")

        for i in range(len(lines)):
            data = lines[i].split()
            file_name = data[1]
            # start and duration time in second
            sub_seg = [float(data[3]), float(data[4])]
            sub_seg = [sub_seg[0], sub_seg[0]+sub_seg[1]]
            if file_name not in seg_dict:
                seg_dict[file_name] = []

            seg_dict[file_name].append(sub_seg)

    wav_list = list(seg_dict.keys())
    print(f"totally {len(wav_list)} files")

    total_duration = 0
    for wav in seg_dict:
        total_duration += seg_dict[wav][-1][-1]

    print(f"totally {total_duration/60} mins")
    return seg_dict


def get_ovd_rttm(rttm_file):
    # read and get the ovd seg based on rttm file
    seg_dict = {}
    with open(rttm_file, 'r') as f:
        lines = f.readlines()
        print(f" list {rttm_file} has {len(lines)} lines")

        for i in range(len(lines)):
            data = lines[i].split()
            file_name = data[1]
            # start and duration time in second
            sub_seg = [float(data[3]), float(data[4])]
            sub_seg = [sub_seg[0], sub_seg[0]+sub_seg[1]]
            if file_name not in seg_dict:
                seg_dict[file_name] = []

            seg_dict[file_name].append(sub_seg)

    wav_list = list(seg_dict.keys())
    print(f"totally {len(wav_list)} files")

    total_duration = 0
    for wav in seg_dict:
        total_duration += seg_dict[wav][-1][-1]
    print(f"totally {total_duration/60} mins")

    ovd_dict = {}
    # import ipdb; ipdb.set_trace()
    for fn in seg_dict.keys():
        segs = seg_dict[fn].copy()
        segs.sort()
        ovd_seg = []
        s_f, e_f = -0.1, -0.1  # to record former one
        s_ovd, e_ovd = -0.1, -0.1  # to record the overlap
        has_ovd = False  # symbol index
        for seg in segs:
            if seg[0] >= e_f:
                s_f, e_f = seg
                if has_ovd:
                    ovd_seg.append([s_ovd, e_ovd])
                    has_ovd = False
            elif seg[0] <= e_ovd:  # extend ovd end
                e_ovd = max(min(e_f, seg[1]), e_ovd)
                s_f, e_f = seg[0], max(e_f, seg[1])
                has_ovd = True
            elif seg[0] > e_ovd:
                if has_ovd:
                    ovd_seg.append([s_ovd, e_ovd])
                    has_ovd = False
                # new ovd
                s_ovd = seg[0]
                e_ovd = min(e_f, seg[1])
                s_f, e_f = seg[0], max(e_f, seg[1])
                has_ovd = True
            else:
                assert False

        if has_ovd:
            ovd_seg.append([s_ovd, e_ovd])

        # save the seq
        ovd_dict[fn] = ovd_seg
    return ovd_dict


def seg2tlabel(seg, fs, tlen):
    tlabel = torch.zeros(max(tlen, 0), dtype=torch.int)
    for i in range(len(seg)):
        tseg = (torch.tensor(seg[i])*fs).int()
        tlabel[tseg[0]:tseg[1]] = 1
    return tlabel


def get_accurate(act, flabel, threshold=0.5):
    # collect the sample number
    # import ipdb; ipdb.set_trace()
    act = (act.float() >= threshold).float()
    flabel = flabel.float()
    miss = ((1-act)*flabel).sum(-1)
    fa = (act*(1-flabel)).sum(-1)
    return miss, fa


def Deterr(lab_dir, ref_file, mode='stm'):
    # reference one
    # import ipdb; ipdb.set_trace()
    if mode == 'stm':
        seg_dict = get_stm(ref_file)
    elif mode == 'rttm':
        seg_dict = get_rttm(ref_file)
    file_list = []
    for file in sorted(glob.glob(lab_dir+'/**/*.lab', recursive=True), key=os.path.getmtime):
        file_list.append(file)
    file_list.sort()
    print(f"The number of files is {len(file_list)}")

    # import ipdb; ipdb.set_trace()
    miss_t, fa_t = 0.0, 0.0
    len_t = 0.0
    fs = 16000
    for file in file_list:
        out_seg = get_lab(file)
        file_basename = os.path.basename(file)[:-4]
        ref_seg = seg_dict[file_basename]
        tlen = max([seg[-1] for seg in ref_seg])*fs
        tlen = int(max(tlen, ref_seg[-1][-1])*fs)  # roughly short

        out_label = seg2tlabel(out_seg, fs, tlen)
        ref_label = seg2tlabel(ref_seg, fs, tlen)

        miss, fa = get_accurate(out_label, ref_label)
        slen = out_label.shape[0]
        missr = miss/slen
        far = fa/slen
        if missr > 0.1 or far > 0.1:
            print(f"{file_basename}: Miss={missr*100:.5f}%, FA={far*100:.5f}% with len {tlen/fs:.3f} s")

        miss_t += miss
        fa_t += fa
        len_t += slen

    miss_tr = miss_t/len_t
    fa_tr = fa_t/len_t
    print(f"Total: Miss={miss_tr*100:.5f}%, FA={fa_tr*100:.5f}% ")

    return None


def Deterr_ovd(lab_dir, ref_file, mode='rttm'):
    # reference one
    seg_dict = get_rttm(ref_file)
    seg_ovd_dict = get_ovd_rttm(ref_file)

    file_list = []
    for file in sorted(glob.glob(lab_dir+'/**/*.lab', recursive=True), key=os.path.getmtime):
        file_list.append(file)
    file_list.sort()
    print(f"The number of files is {len(file_list)}")

    # import ipdb; ipdb.set_trace()
    miss_t, fa_t = 0.0, 0.0
    len_t = 0.0
    fs = 16000
    for file in file_list:
        # import ipdb; ipdb.set_trace()
        out_seg = get_lab(file)
        file_basename = os.path.basename(file)[:-4]
        ref_ovd_seg = seg_ovd_dict[file_basename]
        ref_seg = seg_dict[file_basename]
        tlen = int(max([seg[-1] for seg in ref_seg])*fs)
        out_label = seg2tlabel(out_seg, fs, tlen)
        ref_ovd_label = seg2tlabel(ref_ovd_seg, fs, tlen)

        miss, fa = get_accurate(out_label, ref_ovd_label)
        slen = out_label.shape[0]
        missr = miss/slen
        far = fa/slen
        if missr > 0.1 or far > 0.1:
            print(f"{file_basename}: Miss={missr*100:.5f}%, FA={far*100:.5f}% with len {tlen/fs:.3f} s")

        miss_t += miss
        fa_t += fa
        len_t += out_label.shape[0]

    miss_tr = miss_t/len_t
    fa_tr = fa_t/len_t
    print(f"Total: Miss={miss_tr*100:.5f}%, FA={fa_tr*100:.5f}% ")

    return miss_tr, fa_tr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab_dir', type=str, default="./EXAMPLE/data/pynnest_sad")
    parser.add_argument('--ref_file', type=str, default="./EXAMPLE/data/test_all.rttm")
    parser.add_argument('--mode', type=str, default='rttm')
    parser.add_argument('--isovd', action='store_true')
    args = parser.parse_args()

    #Deterr_stm(lab_dir="/apdcephfs/share_1316500/naijunzheng/corpus/AMI/data/nnest_sad",
    #       ref_file="/apdcephfs/share_1316500/naijunzheng/corpus/AMI/data/segments_test/test_all.stm",
    #       mode = 'stm')

    # Deterr(lab_dir="./EXAMPLE/data/pynnest_sad",
    if not args.isovd:
        Deterr(lab_dir=args.lab_dir, ref_file=args.ref_file, mode = args.mode)
    else:
        Deterr_ovd(lab_dir=args.lab_dir, ref_file=args.ref_file, mode = args.mode)


    """
    Deterr_ovd(lab_dir="/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse/data/oracle_ovd",
            ref_file="/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse/data/test_all.rttm",
            mode = 'rttm')
    """


if __name__ == '__main__':
    main()
