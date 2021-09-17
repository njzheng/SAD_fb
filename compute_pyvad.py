#!/usr/bin/env python
# use pyannone to do vad

import argparse
import glob
import numpy as np
import struct
import os
import sys
from tqdm import tqdm
import torch
import torchaudio
import soundfile as sf
from pyannote.audio.utils.signal import Binarize

#####################################################

def main():
    parser = argparse.ArgumentParser(description="SAD PYNet")
    # Data loader
    # Training details
    parser.add_argument('--file_list', type=str, default=None)
    parser.add_argument('--in_file_dir', type=str, default="/apdcephfs/share_1316500/naijunzheng/corpus/King-216-sub/concate5/data/wav")
    parser.add_argument('--out_file_dir', type=str, default='/apdcephfs/share_1316500/naijunzheng/corpus/King-216-sub/concate5/data/nnest_vad', help='seg lab txt written')
    parser.add_argument('--threshold', type=float, default=0.52, help='thr for VAD')
    args = parser.parse_args()

    in_file_list = args.file_list
    in_flac_dir = args.in_file_dir  # directory with flac files
    # find flac or wav
    if 'wav' in in_flac_dir.split('/')[-2:]:
        suffix = 'wav'
    elif 'flac' in in_flac_dir.split('/')[-2:]:
        suffix = 'flac'

    out_lab_dir = args.out_file_dir  # directory with files with VAD information

    if in_file_list is not None:
        file_names = np.loadtxt(in_file_list, dtype=object, ndmin=1)
    else:
        file_names = sorted(glob.glob(in_flac_dir+'/*.wav', recursive=True))
        file_names = [os.path.splitext(os.path.basename(fn))[0] for fn in file_names]
        file_names = np.array(file_names)
    assert file_names.ndim >= 1
    print(f"file number is {len(file_names)}")

    # overlapped speech detection model trained on AMI training set
    # obtain raw SAD scores (as `pyannote.core.SlidingWindowFeature` instance)
    sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
    threshold = args.threshold
    binarize = Binarize(offset=threshold, onset=threshold, log_scale=True,
                                min_duration_off=0.1, min_duration_on=0.1)

    if not os.path.isdir(out_lab_dir):
        os.makedirs(out_lab_dir)

    import ipdb; ipdb.set_trace()
    for fn in tqdm(file_names):
        inpwave_name = in_flac_dir+"/"+fn+"."+suffix
        out_file = out_lab_dir+'/'+fn+'.lab'
        test_file = {'uri': 'wav', 'audio': inpwave_name}
        # obtain raw SAD scores
        sad_scores = sad(test_file)
        speech = binarize.apply(sad_scores, dimension=1)
        with open(out_file, 'w') as lab_file:
            for speech_region in speech:
                lab_file.write("%.3f %.3f speech\n" % (speech_region.start, speech_region.end))


    print("Finish VAD!")

if __name__ == '__main__':
    main()
