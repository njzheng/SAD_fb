import torch
import torchaudio
import numpy as np
import os
from tqdm import tqdm
import argparse
import librosa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAD Net")

    # Data loader
    parser.add_argument('--time_range', type=int, default=[30, 150], help='Input length to split')
    parser.add_argument('--nutts', type=int, default=5, help='Minimum number of utterances per file')
    parser.add_argument('--input_path', type=str,
            default="/apdcephfs/share_1316500/naijunzheng/corpus/adv_org/ft_local",
            help='input path for split noise')
    parser.add_argument('--out_path', type=str, default="/apdcephfs/share_1316500/naijunzheng/corpus/noise/adv", help='Path for split noise')
    parser.add_argument('--prefix', type=str,
            default="adv",
            help='prefix to for naming')
    args = parser.parse_args()

    wav_list = []
    if os.path.isdir(args.input_path):
        wav_list = os.listdir(args.input_path)
        # only for .wav files
        wav_list = [os.path.join(args.input_path, wav) for wav in wav_list if wav[-4:]=='.wav']
    elif os.path.isfile(args.input_path):
        wav_list = [args.input_path]

    os.makedirs(args.out_path, exist_ok=True)

    print(f"There are totally {len(wav_list)} wav files")
    fs = 16000

    # import ipdb; ipdb.set_trace()
    wav_id = 0
    for wav in tqdm(wav_list):
        seg_id = 0
        basename = os.path.splitext(os.path.basename(wav))[0]
        output_basename = os.path.join(args.out_path, args.prefix)
        # directly resampled to fs 
        X, _ = librosa.load(wav, fs, False, dtype=np.float32, res_type='kaiser_fast')
        X = torch.from_numpy(X)
        if X.ndim > 1:
            X = X[0]
        if X.abs().max() > 1:
            X = X/(X.abs().max()+1e-8)
        tlen = X.shape[0]/fs  # seconds
        if tlen < args.time_range[0]:
            torchaudio.save(f"{output_basename}-{wav_id:04d}-{seg_id:04d}.wav",
                    X, fs)
        else:
            nseg = int(tlen//args.time_range[0])
            nseg = min(nseg, args.nutts)
            nseg_start = np.linspace(0, tlen-args.time_range[0], nseg)
            for si in range(nseg):
                torchaudio.save(f"{output_basename}-{wav_id:04d}-{si:04d}.wav",
                        X[int(nseg_start[si]*fs):int((nseg_start[si]+args.time_range[0])*fs)], fs)

        wav_id += 1

