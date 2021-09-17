import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchaudio
import scipy.signal as signal
import kaldi_io
import features
from common import getFileList, getRirFileName, getNoiseFileName, butter_bandpass_filter

class SAD_fb_Dataset(Dataset):
    def __init__(
            self, mode, wav_scp_file, stm_file=None, rttm_file=None,
            smp_flen_range=[300, 500], num_seg=1):
        super(SAD_fb_Dataset).__init__()

        # import ipdb; ipdb.set_trace()
        self.mode = mode  # train
        self.stm_file = stm_file
        self.rttm_file = rttm_file
        if stm_file is None and rttm_file is not None:
            print(f"use rttm_file {rttm_file}")
        elif stm_file is not None:
            print(f"use stm file {stm_file}")
        else:
            print("no stm or rttm file here, exit")
            exit()

        self.wav_list, self.seg_dict = self.get_wav_list()
        self.wav_scp = self.read_scp(wav_scp_file)
        self.num_wav = len(self.wav_scp)
        print(f" have {self.num_wav} scp files")

        self.num_seg = num_seg  # split wav into short dur
        self.smp_flen_range = smp_flen_range
        self.smp_flen = smp_flen_range[0]
        # self.smp_tlen = int(self.smp_flen*10/1000*16000)

    def read_scp(self, scp_file):
        with open(scp_file, 'r') as sf:
            lines = sf.readlines()
        return lines

    def change_flength(self):
        self.smp_flen = random.randrange(self.smp_flen_range[0], self.smp_flen_range[1]+1, 1)
        # self.smp_tlen = int(self.smp_flen*10/1000*16000)
        print(f"change smp_frame_length to {self.smp_flen}")

    def get_wav_list(self):
        # read and get the seg based on stm file
        seg_dict = {}
        if self.stm_file is not None:
            with open(self.stm_file, 'r') as f:
                lines = f.readlines()
                print(f" list {self.stm_file} has {len(lines)} lines")

                for i in range(len(lines)):
                    data = lines[i].split()
                    file_name = data[0]
                    # start and end time in second
                    sub_seg = [float(data[3]), float(data[4])]
                    if file_name not in seg_dict:
                        seg_dict[file_name] = []
                    seg_dict[file_name].append(sub_seg)

        elif self.rttm_file is not None:
            with open(self.rttm_file, 'r') as f:
                lines = f.readlines()
                print(f" list {self.rttm_file} has {len(lines)} lines")
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
        return wav_list, seg_dict

    def get_wav(self, utt_file):
        # load wav for vox; m4a for vox2
        if utt_file[-4:] == '.wav' or utt_file[-5] == '.flac':
            wav, fs = torchaudio.load(utt_file)
            wav = wav[0]
        else:
            raise NotImplementedError
        return wav, fs

    def seg2flabel(self, seg, hop, flen):
        tlabel = torch.zeros(max(flen, 0), dtype=torch.int)
        for i in range(len(seg)):
            tseg = (torch.tensor(seg[i])/hop).int()
            tlabel[tseg[0]:min(tseg[1],flen)] = 1
        return tlabel

    def load_fb_seg(self, wav_id, ark_id, seg):
        fb = torch.tensor(kaldi_io.read_mat(ark_id))
        flabel = self.seg2flabel(seg, hop=0.01, flen=len(fb))
        assert len(flabel) == len(fb)
        return fb, flabel

    def getitem_whole(self, i):
        assert i < self.num_wav
        scp_line = self.wav_scp[i % self.num_wav]
        wav_id, ark_id = scp_line.strip().split()
        # assume the test utt has no augmentation 
        # wav_id = '-'.join(wav_id.split('-')[:-1])
        seg = self.seg_dict[wav_id]

        fb, flabel = self.load_fb_seg(wav_id, ark_id, seg)
        return fb, flabel

    def __getitem__(self, i):
        # import ipdb; ipdb.set_trace()
        # select based on scp due to the data_aug
        scp_line = self.wav_scp[i % self.num_wav]
        wav_id, ark_id = scp_line.strip().split()
        wav_id = '-'.join(wav_id.split('-')[:-1])  # discard the suffix noise

        seg = self.seg_dict[wav_id]
        fb, flabel = self.load_fb_seg(wav_id, ark_id, seg)

        if self.smp_flen > len(fb):
            repeat_time = smp_flen//len(fb) + 1
            fb = fb.repeat(repeat_time, 1)
            flabel = flabel.repeat(repeat_time)

        # randomly select num_seg*seg
        # start_t = random.randint(0, len(wav)-tot_tlen - 1)
        start_t = np.random.randint(0, len(fb)-self.smp_flen - 1, self.num_seg)
        end_t = start_t + self.smp_flen
        fb_seg, flabel_seg = [], []
        for i in range(self.num_seg):
            fb_mod = fb[start_t[i]:end_t[i]].float()
            fb_seg.append(fb_mod)
            flabel_seg.append(flabel[start_t[i]:end_t[i]])

        # wav_seg = wav_seg.reshape(self.num_seg, -1)
        fb_seg = torch.stack(fb_seg, 0)
        flabel_seg = torch.stack(flabel_seg, 0)

        return fb_seg, flabel_seg

    def __len__(self):
        # spk number * times : for one epoch, it can have more utts
        return self.num_wav * 30



############################################################################
def extract_fb(signal, fs=16000, window=None, fbank_mx=None):
    noverlap = 240
    winlen = 400
    if window is None:
        window = features.povey_window(winlen)
    if fbank_mx is None:
        fbank_mx = features.mel_fbank_mx(winlen, fs, NUMCHANS=40, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
    LC = 150
    RC = 149
    # avoid overflow
    # signal = signal/(signal.abs().max() + 0.1)
    signal = np.array(signal*2**(fs/1000 - 1)).astype(np.float32)
    signal = np.r_[signal[noverlap//2-1::-1], signal, signal[-1:-winlen//2-1:-1]]
    fb = features.fbank_htk(np.array(signal).astype(np.float32), window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
    fb = features.cmvn_floating_kaldi(fb, LC, RC, norm_vars=False)
    fb = torch.tensor(fb).float()
    return fb

def extract_spec(signal, fs=16000):
    win_tlen=0.025
    hop_tlen=0.010
    LC = 150
    RC = 149
    # pitch = torchaudio.functional.detect_pitch_frequency(signal, fs, frame_time=hop_tlen)
    # spec = torchaudio.compliance.kaldi.spectrogram(signal)
    Spectrogram = torchaudio.transforms.Spectrogram(n_fft = int(fs*win_tlen))
    Pow2dB = torchaudio.transforms.AmplitudeToDB()
    # CMVN = torchaudio.transforms.SlidingWindowCmn(300)
    spec = Spectrogram(signal)
    spec = Pow2dB(spec).transpose(0, 1)
    spec = features.cmvn_floating_kaldi(spec, LC, RC, norm_vars=False)
    #spec = CMVN(spec)
    return spec.float()
#################################################################################


# class to get vad train dataset from wavs and convert to fbanks
class SADDataset(Dataset):
    def __init__(
            self, mode, wav_dir, stm_file=None, rttm_file=None, noise_list=None, reverb_list=None,
            smp_flen_range=[300, 500], num_seg=1, wav_range=None):
        super(SADDataset).__init__()

        self.fs = 16000
        self.noverlap = 240
        self.winlen = 400
        self.window = features.povey_window(self.winlen)
        self.fbank_mx = features.mel_fbank_mx(self.winlen, self.fs, NUMCHANS=40, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)

        # import ipdb; ipdb.set_trace()
        self.mode = mode  # train
        self.wav_dir = wav_dir
        self.stm_file = stm_file
        self.rttm_file = rttm_file
        if stm_file is None and rttm_file is not None:
            print(f"use rttm_file {rttm_file}")
        elif stm_file is not None:
            print(f"use stm file {stm_file}")
        else:
            print("no stm or rttm file here, exit")
            exit()

        self.wav_list, self.seg_dict = self.get_wav_list()
        self.num_wav = len(self.wav_list)


        self.noise_list = noise_list
        self.reverb_list = reverb_list
        # self.SNR = [-6, 0, 6, 12, 18, 24]  # dB

        self.num_seg = num_seg  # split wav into short dur
        self.smp_flen_range = smp_flen_range
        self.smp_flen = smp_flen_range[0]
        self.smp_tlen = int(self.smp_flen*10/1000*16000)

        self.noiseFiles = self.rirFiles = []
        if self.reverb_list is not None:
            self.rirFiles = getFileList(self.reverb_list, '\.wav', [])
            self.rirFiles.sort()
            print(f"rir files num is {len(self.rirFiles)}")
            # rirNames, rirTypeIdx, rirTypeName = getRirFileName(self.rirFiles)
        if self.noise_list is not None:
            self.noiseFiles = getFileList(self.noise_list, '\.wav', [])
            self.noiseFiles.sort()
            print(f"noise files num is {len(self.noiseFiles)}")
            # classify
            # import ipdb; ipdb.set_trace()
            self.noiseFiles_dic = {}
            for noise in self.noiseFiles:
                noise_type = os.path.basename(noise).split('-')[0]
                if noise_type not in ['song', 'speech', 'clap']:  # remove some kind of noise
                    if noise_type not in self.noiseFiles_dic:
                        self.noiseFiles_dic[noise_type] = []
                    self.noiseFiles_dic[noise_type].append(noise)

            print(f"There are total {len(self.noiseFiles_dic)} types noise")
            print([(key, len(self.noiseFiles_dic[key])) for key in self.noiseFiles_dic])


        # indicate the valid range of the wavs
        self.wav_range = None
        if isinstance(wav_range, list):
            print(f"valid range of the wavs are inside of {wav_range} seconds")
            self.wav_range = wav_range

    def change_flength(self):
        self.smp_flen = random.randrange(self.smp_flen_range[0], self.smp_flen_range[1]+1, 1)
        self.smp_tlen = int(self.smp_flen*10/1000*16000)
        print(f"change smp_frame_length to {self.smp_flen}")

    def get_wav_list(self):
        # read and get the seg based on stm file
        seg_dict = {}
        if self.stm_file is not None:
            with open(self.stm_file, 'r') as f:
                lines = f.readlines()
                print(f" list {self.stm_file} has {len(lines)} lines")

                for i in range(len(lines)):
                    data = lines[i].split()
                    file_name = data[0]
                    # start and end time in second
                    sub_seg = [float(data[3]), float(data[4])]
                    if file_name not in seg_dict:
                        seg_dict[file_name] = []
                    seg_dict[file_name].append(sub_seg)

        elif self.rttm_file is not None:
            with open(self.rttm_file, 'r') as f:
                lines = f.readlines()
                print(f" list {self.rttm_file} has {len(lines)} lines")
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
        return wav_list, seg_dict

    def get_wav(self, utt_file):
        # load wav for vox; m4a for vox2
        if utt_file[-4:] == '.wav' or utt_file[-5] == '.flac':
            wav, fs = torchaudio.load(utt_file)
            wav = wav[0]
        else:
            raise NotImplementedError
        return wav, fs

    def addNoise(self, wav, noi, SNR_range, spEn=None):
        if len(noi) < len(wav):
            noi = noi.repeat(len(wav)//len(noi)+1)
        noi_len = len(wav)
        start_n = random.randint(0, len(noi)-noi_len)
        noi = noi[start_n:start_n+noi_len]
        start = random.randint(0, len(wav)-noi_len)
        if spEn is None:
            spEn = (wav**2).mean()
        noiEn = (noi**2).mean()
        SNR = random.choice(SNR_range)

        r = (pow(10, (SNR/10))*noiEn/spEn).sqrt()
        if r > 1e-8:
            wav = wav + 1/r * noi

        if wav.abs().max() > 1:
            wav = wav/(wav.abs().max()+0.01)
        return wav

    def applyRIR(self, speech, rir, energyNorm=0, delayNorm=1):
        sp_max_ori = speech.abs().max()
        sp_len_ori = len(speech)
        speech_ori = speech
        speech = torch.tensor(signal.fftconvolve(speech_ori, rir, 'full'))

        # normalized energy:
        if energyNorm == 1:
            speech = speech / (speech.abs().max() + 1e-12) * sp_max_ori * 0.9

        # import ipdb; ipdb.set_trace()
        # remove delay introduce by rir:
        if delayNorm == 1:
            rir_sid = rir.abs().argmax()
            speech = speech[rir_sid:]
            if len(speech) > sp_len_ori:
                speech = speech[:sp_len_ori]
            else:
                speech = torch.cat((speech, torch.zeros((sp_len_ori - len(speech)))), 0)
        return speech

    def applyRIR2(self, speech, rir):
        rir         = np.expand_dims(rir.astype(np.float32),0)
        rir         = rir / np.sqrt(np.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

    def seg2tlabel(self, seg, fs, tlen):
        tlabel = torch.zeros(max(tlen, 0), dtype=torch.int)
        for i in range(len(seg)):
            tseg = (torch.tensor(seg[i])*fs).int()
            tlabel[tseg[0]:tseg[1]] = 1
        return tlabel

    def load_wav_seg(self, wav_id, seg):
        wav_path = os.path.join(self.wav_dir, wav_id+'.wav')
        wav, fs = self.get_wav(wav_path)
        assert fs == 16000

        # get seg labe
        # import ipdb; ipdb.set_trace()
        # if len(wav)/fs < seg[-1][-1]:      
            # print(f"{wav_id}: {len(wav/fs)}  {seg[-1]}")
        tlabel = self.seg2tlabel(seg, fs, len(wav))
        assert len(tlabel) == len(wav)

        if self.wav_range is not None:
            wrange = [int(max(t*fs, -1)) for t in self.wav_range]
            assert wrange[0] < len(wav)
            wav = wav[min(wrange[0], len(wav)-1):min(wrange[1], len(wav))]
            tlabel = tlabel[min(wrange[0], len(tlabel)-1):min(wrange[1], len(tlabel))]

        return wav, tlabel

    def getitem_whole(self, i):
        assert i < self.num_wav
        wav_id = self.wav_list[i % self.num_wav]
        seg = self.seg_dict[wav_id]
        # import ipdb; ipdb.set_trace()
        wav_path = os.path.join(self.wav_dir, wav_id+'.wav')
        wav, fs = self.get_wav(wav_path)
        assert fs == 16000
        # get seg labe
        tlabel = self.seg2tlabel(seg, fs, len(wav))
        # return wav, tlabel

        if self.wav_range is not None:
            wrange = [int(max(t*fs, -1)) for t in self.wav_range]
            assert wrange[0] < len(wav)
            wav = wav[min(wrange[0], len(wav)-1):min(wrange[1], len(wav))]
            tlabel = tlabel[min(wrange[0], len(tlabel)-1):min(wrange[1], len(tlabel))]

        # convert to fbank
        fb = extract_fb(wav, self.fs, self.window, self.fbank_mx)
        # fb = extract_spec(wav, self.fs)
        # fb0 = torchaudio.compliance.kaldi.fbank(wav.unsqueeze(0), dither=0.0, energy_floor=1.0, subtract_mean=True, remove_dc_offset=True, high_freq=7600, low_freq=20, num_mel_bins=40)
        fb = fb.squeeze(0).float()
        flabel_sad = tlabel[80::160]
        assert flabel_sad.shape[-1] >= fb.shape[0]
        flabel_sad = flabel_sad[:fb.shape[0]]
        return fb, flabel_sad

    def __getitem__(self, i):
        # import ipdb; ipdb.set_trace()
        wav_id = self.wav_list[i % self.num_wav]
        seg = self.seg_dict[wav_id]

        wav, tlabel = self.load_wav_seg(wav_id, seg)
        spEn = (wav**2).mean()
        tot_tlen = self.smp_tlen * self.num_seg
        if self.smp_tlen > len(wav):
            print(len(wav), self.smp_tlen)
            wav = wav.repeat(self.smp_tlen//len(wav) + 1)
            tlabel = tlabel.repeat(self.smp_tlen//len(wav) + 1)

        # randomly select num_seg*seg
        # start_t = random.randint(0, len(wav)-tot_tlen - 1)
        start_t = np.random.randint(0, len(wav)-self.smp_tlen - 1, self.num_seg)
        end_t = start_t + self.smp_tlen
        wav_seg, tlabel_seg = [], []
        for i in range(self.num_seg):
            wav_mod = wav[start_t[i]:end_t[i]].float()
            tlabel_seg.append(tlabel[start_t[i]:end_t[i]])
            # telephone channnel simulation
            if random.random() < 0.05:
                wav_mod = butter_bandpass_filter(wav_mod, low_freq=300, high_freq=3400, fs=16000, order=6)
                wav_mod = torch.tensor(wav_mod).float()
            wav_seg.append(wav_mod)

        # import ipdb; ipdb.set_trace()
        if random.random() > 0.5 and len(self.rirFiles) > 0:
            rir_fn = random.choice(self.rirFiles)
            rir, fs = self.get_wav(rir_fn)
            wav_seg = [self.applyRIR(seg, rir) for seg in wav_seg]

        # add noise
        temp = random.random()
        if temp > 0.05 and len(self.noiseFiles) > 0:
        # if len(self.noiseFiles) > 0:
            # remove the babble noise
            type_list = list(self.noiseFiles_dic.keys())
            # can delete some kind of noise
            noise_type = random.choice(type_list)
            if noise_type == 'speech':
                SNR_range = list(range(20, 30, 5))
                # do not add babble speech here
                # return wav_seg, tlabel_seg
            elif noise_type == 'music':
                SNR_range = list(range(-3, 10, 3))
            elif noise_type == 'noise':
                SNR_range = list(range(5, 30, 5))
            elif noise_type == "clap" :
                SNR_range = list(range(0, 20, 5))
            elif noise_type == "bgm" :
                SNR_range = list(range(-3, 20, 3))
            elif noise_type == "song" :
                SNR_range = list(range(15, 30, 5))
            elif noise_type == "adv" :
                SNR_range = list(range(-3, 15, 3))
            else:
                SNR_range = list(range(5, 20, 5))
                raise NotImplementedError
            noi_i = random.randint(0, len(self.noiseFiles_dic[noise_type])-1)
            noi, fs = self.get_wav(self.noiseFiles_dic[noise_type][noi_i])
            wav_seg_noi = [self.addNoise(seg, noi, SNR_range, spEn) for seg in wav_seg]
            wav_seg = wav_seg_noi

        # wav_seg = wav_seg.reshape(self.num_seg, -1)
        # tlabel_seg = tlabel_seg.reshape(self.num_seg, -1)
        wav_seg = torch.stack(wav_seg, 0)
        tlabel_seg = torch.stack(tlabel_seg, 0)
        # return wav_seg, tlabel_seg

        if torch.isnan(wav_seg).any():
            print("wav_seg has nan")
            assert False
        # convert to fbank
        fb_seg = [extract_fb(wav_seg[i], self.fs, self.window, self.fbank_mx) for i in range(len(wav_seg))]
        # fb_seg = [extract_spec(wav_seg[i], self.fs) for i in range(len(wav_seg))]
        fb_seg = torch.stack(fb_seg, 0).float()
        flabel_seg_sad = tlabel_seg[:, 80::160]
        assert flabel_seg_sad.shape[-1] >= fb_seg.shape[1]
        flabel_seg_sad = flabel_seg_sad[:, :fb_seg.shape[1]]
        if torch.isnan(fb_seg).any():
            print("fb_seg has nan")
            assert False


        return fb_seg, flabel_seg_sad

    def __len__(self):
        # spk number * times : for one epoch, it can have more utts
        return self.num_wav*20


if __name__ == '__main__':
    print("none")
