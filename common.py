import os, glob,re
import torch
import numpy as np
from scipy.signal import lfilter, butter
from scipy.io.wavfile import read,write
from numpy import array, int16
import sys
'''
list dir
'''


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def getFileList(strDir, flag, fileList):
    #for file in glob.glob(strDir+'/*'):
    for file in sorted(glob.glob(strDir+'/*'), key=os.path.getmtime):
        if os.path.isdir(file): # the 'file' is a dir
            getFileList(file, flag, fileList)
        elif (re.search(flag, file, re.IGNORECASE)): 
            fileList.append(file)
    return fileList


def getNoiseFileName(noiseFiles):
    noiseNames      = []
    noiseTypeIdx    = []
    noiseTypeName   = []
    noiseTypes      = {}
    for i in range(len(noiseFiles)):
        vec         = noiseFiles[i].split('/')
        noiseType   = vec[-2]
        if noiseType not in noiseTypes.keys():
            noiseTypes[noiseType] = 1
            noiseTypeName.append(noiseType)
            noiseTypeIdx.append(i)
        else:
            noiseTypes[noiseType] = noiseTypes[noiseType] + 1
        noiseNames.append('NOI_'+noiseType[0:3]+num2str(noiseTypes[noiseType],4))
    return noiseNames, noiseTypeIdx, noiseTypeName

def num2str(num, tarlen):
    outstr = str(num)
    while len(outstr) < tarlen:
        outstr = '0'+outstr
    return outstr 

def getAzmFromName(rirName):
    azimuth = -1000 #default illegal value
    if not detectRirAngleInfo(rirName):
        return azimuth

    vec = rirName.split('_')
    temp = vec[-2]
    azmStr = temp[3:]
    azimuth = float(azmStr)
    return azimuth

def detectRirAngleInfo(inputStr):
    detectedFlag = 0
    m   = re.search(r'[_-]ang[0-9]*\.[0-9]+[_-]', inputStr) or re.search(r'[_-]ang[0-9]+[_-]', inputStr)
    if m:
        detectedFlag = 1
    #else:
    #    print 'warning: no angle info detected in rir file name'

    return detectedFlag


def getRirFileName(rirFiles):
    rirNames    = []
    rirTypeIdx  = []   
    rirTypeName = []   
    rirTypes    = {}
    for i in range(len(rirFiles)):
        vec     = rirFiles[i].split("/")
        rirType = vec[-2]
        if rirType not in rirTypes.keys():
            rirTypes[rirType] = 1
            rirTypeName.append(rirType)   
            rirTypeIdx.append(i)
        else:
            rirTypes[rirType] = rirTypes[rirType] + 1
        
        vec1    = vec[-1].split("-")
        match   = re.search('\D+(\d+)', vec1[0])
        if match:
            temp    = match.group(1)
            #import pdb;pdb.set_trace()
            if detectRirAngleInfo(rirFiles[i]):
                rirNames.append('RIR_'+rirType[0:4]+num2str(int(temp),6)+'_'+vec1[-3]+'_'+vec1[-1][0:-4])
            else:
                rirNames.append('RIR_'+rirType[0:4]+num2str(int(temp),6)+'_'+vec1[-1][0:-4])
        else:
            rirNames.append('RIR_'+rirType[0:4]+'_'+vec1[-1][0:-4])

    return rirNames, rirTypeIdx, rirTypeName


def slide_window(data, slide_len, slide_hop, padzero=True):
    # convert inputs into slides
    assert data.dim() <= 2
    # assume the first dim is the seq
    l = data.shape[0]
    if padzero and (l % slide_hop > 0):
        num_pad = slide_hop - (l%slide_hop)
        if data.dim() == 1:
            data = torch.cat((data, torch.zeros(num_pad, dtype=data.dtype)), 0)
        else:
            data = torch.cat((data, torch.zeros((num_pad, data.shape[1]), dtype=data.dtype)), 0)
    num_slides = data.shape[0] // slide_hop
    assert data.shape[0] % slide_hop == 0
    data_seg = data.unfold(0, slide_len, slide_hop)
    return data_seg


def unslide_window(data, hop=0.5):
    # average the slices
    slide_len =  data.shape[1]
    slide_hop = int(slide_len*hop)
    num_seg = data.shape[0]
    sum_temp = torch.zeros((num_seg-1)*slide_hop+slide_len)
    times_temp = torch.zeros((num_seg-1)*slide_hop+slide_len)
    for i in range(num_seg):
        start_i = i*slide_hop
        sum_temp[start_i:start_i+slide_len] += data[i]
        times_temp[start_i:start_i+slide_len] += 1
    return sum_temp/times_temp


# telephone channel
def butter_params(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_params(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    file_name = "hqyok.wav"
    fs,audio = read(file_name)
    low_freq = 300.0
    high_freq = 3400.0
    filtered_signal = butter_bandpass_filter(audio, low_freq=300, high_freq=3400, fs=16000, order=6)
    fname = file_name.split('.wav')[0] + '_moded.wav'
    # write(fname,fs,array(filtered_signal,dtype=int16))
