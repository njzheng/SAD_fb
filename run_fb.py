import argparse
import time
import os
import h5py
import glob
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tdnn_model import TDNN_LSTM, LSTM_FD
from common import AverageMeter, slide_window, unslide_window
from utils import report_progress

from dataset_sad import SADDataset, SAD_fb_Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import torchaudio


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    fs = 16000
    winlen = int(0.025*fs)
    hop = int(0.01*fs)
    model.train()
    end = time.time()
    # import ipdb; ipdb.set_trace()
    for batch_idx, (inputs, flabel) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs = inputs.transpose(-1, -2)
        # if inputs.dim() == flabel.dim()+1 == 4:
            # convert to 2D
        # [B, 40, 500]
        inputs = inputs.flatten(0, 1)
        flabel = flabel.flatten(0, 1)
        inputs = Variable(inputs.cuda())
        flabel = Variable(flabel.float().cuda())

        optimizer.zero_grad()
        out, act = model(inputs)

        nf = min(flabel.shape[1], act.shape[1])
        out = out[:, :nf].squeeze(-1)
        flabel = flabel[:, :nf]
        loss = criterion(out, flabel)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        losses.update(loss.data, act.shape[0])
        if batch_idx % 10 == 0 or (batch_idx == len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg


def get_accurate2(act_set, flabel_set):
    # import ipdb; ipdb.set_trace()
    num = len(act_set)
    miss_avg_best, fa_avg_best = 10.0, 10.0
    thr = 0.0
    for t in torch.arange(0.0, 1, 0.02):
        miss_total, fa_total, sample_total = 0.0, 0.0, 0.0
        for i in range(num):
            miss, fa = get_accurate(act_set[i], flabel_set[i], t)
            miss_total += miss
            fa_total += fa
            sample_total += act_set[i].shape[0]

        miss_avg = miss_total/sample_total
        fa_avg = fa_total/sample_total
        if miss_avg + fa_avg < miss_avg_best + fa_avg_best:
            miss_avg_best = miss_avg
            fa_avg_best = fa_avg
            thr = t

    return miss_avg_best, fa_avg_best, thr


def get_accurate(act, flabel, threshold=0.5):
    # collect the sample number
    # import ipdb; ipdb.set_trace()
    act = (act >= threshold).float()
    flabel = flabel.float()
    miss = ((1-act)*flabel).sum(-1)
    fa = (act*(1-flabel)).sum(-1)
    return miss, fa


def validate(vset, model, criterion, slide_length, tmp_path):
    fs = 16000
    winlen = int(0.025*fs)
    hop = int(0.01*fs)
    losses = AverageMeter()
    model.eval()
    # num_trials = min(vset.num_wav, 20)
    num_trials = vset.num_wav
    print(f"number of trials is {num_trials}")
    slide_len = int(slide_length)
    slide_hop = int(0.5*slide_len)
    act_set = []
    flabel_set = []
    # import ipdb; ipdb.set_trace()
    for i in tqdm(range(num_trials)):
        inputs, flabel = vset.getitem_whole(i)
        flabel = flabel.bool()  # seemed to reduce memory
        flabel_len = flabel.shape[0]
        # quick test:
        # inputs = inputs[:200]
        # flabel = flabel[:200]
        inputs_seg = slide_window(inputs, slide_len, slide_hop) # [B, 40, 500]
        flabel_seg = slide_window(flabel, slide_len, slide_hop)
        flabel_seg = flabel_seg.float().cuda()
        del inputs

        with torch.no_grad():
            out, act_seg = model(inputs_seg.cuda())

            nf = min(flabel_seg.shape[1], act_seg.shape[1])
            act_seg = act_seg[:, :nf].squeeze(-1).cpu()
            out = out[:, :nf].squeeze(-1)
            flabel_seg = flabel_seg[:, :nf]
            loss = criterion(out, flabel_seg)
            losses.update(loss.data, act_seg.shape[0])

        # concate act_seg
        act_temp = unslide_window(act_seg, hop=0.5)
        assert act_temp.shape[0] >= flabel_len

        act_set.append(act_temp[:flabel_len])
        flabel_set.append(flabel)

    miss, fa, t = get_accurate2(act_set, flabel_set)
    error = losses.avg

    print(f"loss error={error:.5f}, Miss={miss*100:.5f}%, FA={fa*100:.5f}% with threshold = {t}")
    return miss+fa, t


def main():
    parser = argparse.ArgumentParser(description="SAD Net")

    # Data loader
    parser.add_argument('--frame_range', type=int, default=[200, 200],  help='Input length to the network')
    parser.add_argument('--batch_size', type=int, default=16,  help='Batch size')
    parser.add_argument('--min_utts_per_spk', type=int, default=8, help='Minimum number of utterances per speaker')
    # Training details
    parser.add_argument('--test_interval', type=int, default=1, help='Test and save every [test_interval] epochs')
    parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam')
    # Learning rates
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    # Load and save

    parser.add_argument('--load_model_path', type=str, default="", help='Load model weights')
    # parser.add_argument('--load_model_path', type=str, default="/apdcephfs/private_naijunzheng/DATA/SAD/model/tdnn_lstm_21_sad_vox_vox_200_noi.tar", help='Load model weights')
    # parser.add_argument('--load_model_path', type=str, default="/apdcephfs/private_naijunzheng/DATA/SAD/model/tdnn_lstm_134_sad_ami_vox_200.tar", help='Load model weights')
    parser.add_argument('--save_model_path', type=str, default="/apdcephfs/private_naijunzheng/DATA/SAD/model/", help='Path for model')
    # Training and test data
    parser.add_argument('--tmp_path',  type=str, default="/apdcephfs/private_naijunzheng/DATA/SAD/model/",  help='tmp path to store log')
    # Model definition
    parser.add_argument('--model', type=str, default="tdnn_lstm", help='Name of model definition')
    parser.add_argument('--wav_range', type=int, default=None, help='indicate the range of the wavs')
    # parser.add_argument('--model', type=str, default="lstm_fd", help='Name of model definition')
    args = parser.parse_args()

    # rir and noise
    # args.reverb_list = "/apdcephfs/share_1316500/naijunzheng/corpus/RIRS_NOISES/simulated_rirs/"
    args.noise_list = "/apdcephfs/share_1316500/naijunzheng/corpus/noise/noise"
    args.reverb_list = None
    # args.noise_list = None
    # process the dataset

    train_dataset = "vox20"
    test_dataset = "vox20"
    print(f"Train dataset is {train_dataset}, and test_dataset is {test_dataset}")

    if train_dataset == "ami":
        # ami dataset
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/AMI/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/AMI/data/rttm/MixHeadset.train.rttm"
        args.batch_size = 2
        args.num_seg = 64
    elif train_dataset == "king":
        # king chinese dataset
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/King-216-sub/concate20/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/King-216-sub/concate20/data/rttm/train.rttm"
        args.batch_size = 16
        args.num_seg = 4
    elif train_dataset == "moca2":
        # guangdong hua elder 
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/Moca/iPhone_MoCA_0031/data/wav"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/Moca/iPhone_MoCA_0031/data/train_sad.rttm"
        args.batch_size = 4
        args.num_seg = 64
    elif train_dataset == "chime6":
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/chime/train/data/wav"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/chime/train/data/train_sad.rttm"
        args.batch_size = 2
        args.num_seg = 64
        args.wav_range = [80.0, -1]  # remain this part and discard the rest
    elif train_dataset == "vox":  # 216 files
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse/data/test_all.rttm"
        args.batch_size = 16
        args.num_seg = 4
    elif train_dataset == "vox20":  # 404 files
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse20/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse20/data/train_all.rttm"
        args.batch_size = 16
        args.num_seg = 4
    else:
        raise NotImplementedError
    # import ipdb; ipdb.set_trace()
    dset = SADDataset(
            mode='train', wav_dir=args.wav_dir,
            rttm_file=args.rttm_file,
            noise_list=args.noise_list, reverb_list=args.reverb_list,
            smp_flen_range=args.frame_range, num_seg=args.num_seg,
            wav_range=args.wav_range)
    dset.__getitem__(np.random.randint(0, dset.__len__()-1))

    args.wav_range = None
    if test_dataset == "vox": # 232 files
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse_eval/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse_eval/data/test_all.rttm"
    elif test_dataset == "vox20":  # 44 files
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse20/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/voxconverse20/data/test_all.rttm"
    elif test_dataset == "biaozhu":
        # biao zhu chinese test dataset
        args.wav_dir =  "/apdcephfs/share_1316500/naijunzheng/corpus/biaozhu/data/wav/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/biaozhu/data/test_all.rttm"
    elif test_dataset == "king":
        # king chinese test dataset
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/King-216-sub/concate20/data/wav_test/"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/King-216-sub/concate20/data/rttm/test.rttm"
    elif test_dataset == "moca2":
        # guangdong hua elder
        args.wav_dir =  "/apdcephfs/share_1316500/naijunzheng/corpus/Moca/iPhone_MoCA_0032/data/wav"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/Moca/iPhone_MoCA_0032/data/test_sad.rttm"
    elif test_dataset == "chime6":
        args.wav_dir = "/apdcephfs/share_1316500/naijunzheng/corpus/chime/dev/data/wav"
        args.rttm_file = "/apdcephfs/share_1316500/naijunzheng/corpus/chime/dev/data/test_sad.rttm"
        args.wav_range = [80.0, 900.0]  # remain this part and discard the rest
        # args.wav_range = None
    else:
        raise NotImplementedError

    vset = SADDataset(
            mode='test', wav_dir=args.wav_dir,
            rttm_file=args.rttm_file,
            smp_flen_range=args.frame_range,
            wav_range=args.wav_range)
    vset.getitem_whole(0)


    # print data stats
    train_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True,
            num_workers=8)

    # define loss function and optimizer
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.MSELoss()
    criterion = criterion.cuda()

    if args.model == 'tdnn':
        model = TDNN()
    elif args.model == 'tdnn_lstm':
        model = TDNN_LSTM(feat_dim=40, hidden_size=256)
    elif args.model == 'lstm_fd':
        model = LSTM_FD(dropout=0)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr,  betas=(0.9, 0.99),
    #                         eps=1e-09, weight_decay=1e-4, amsgrad=False)
    model = model.cuda()

    """
    def lambda1(epoch): return 0.5 ** (epoch // 20)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    """
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr*2, step_size_up=20)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
            patience=3, verbose=True, cooldown=0, min_lr=1e-7, eps=1e-08)
    # """
    torch.backends.cudnn.benchmark = True
    # reload model
    start_epoch = 0
    best_error = 1000   # miss + FA
    if args.load_model_path != '':
        if os.path.isfile(args.load_model_path):
            print("=> loading checkpoint '{}'".format(args.load_model_path))
            checkpoint = torch.load(args.load_model_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if 'scheduler_state_dict' in checkpoint:
            #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))
            print(f"with loss error={checkpoint['miss_fa_error']}")
            # best_error = checkpoint['miss_fa_error']
        else:
            print("=> no checkpoint found at '{}'".format(args.load_model_path))
            exit()

    slide_length = args.frame_range[0]  # length of slide window, frames  
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # test_error, threshold = validate(vset, model, criterion, slide_length, args.tmp_path)

    # import ipdb; ipdb.set_trace()
    print("Start training---")
    for epoch in range(start_epoch, args.max_epoch):
        train_loader.dataset.change_flength()
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
        progress = {"step": epoch, "type": "train", "loss": float(train_loss)}
        ret, msg = report_progress(progress)

        print("learning rate: {:.6f}".format(optimizer.param_groups[0]['lr']))
        # evaluate on validation set
        if epoch % args.test_interval == 0:
            test_error, threshold = validate(vset, model, criterion, slide_length, args.tmp_path)
            with open(f"{args.tmp_path}/error.log", "a") as f:
                f.write(f"epoch {epoch}: loss miss_fa={test_error}, with threshold = {threshold}\n")
            progress = {"step": epoch, "type": "test", "loss": float(test_error)}
            ret, msg = report_progress(progress)
            scheduler.step(test_error)
            # scheduler.step()

            if test_error >= best_error:
                print(f"The best miss+fa error is still {best_error*100}%")
            else:
                best_error = test_error

                # save the model
                save_name = os.path.join(args.save_model_path, \
                        args.model+'_'+str(epoch)+'_sad_'+train_dataset+'_'+test_dataset+'_'+str(slide_length)+'_noi.tar')
                # save_checkpoint
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'miss_fa_error': test_error,
                    'threshold': threshold
                }, save_name)
                print(f"save model to {save_name} with miss+fa:{test_error*100}%")


if __name__ == '__main__':
    main()
