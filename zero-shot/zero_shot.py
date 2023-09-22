from __future__ import print_function
import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
import torch.optim as optim 
from torch.cuda.amp import autocast
import random
import pandas as pd
from ipdb import set_trace
from PIL import Image
from einops import rearrange
from torch.cuda import amp
import certifi
import numpy as np
import argparse
import sys

import argparse
from base_options import BaseOptions
import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from utils.utils import *
from utils.Recorder import Recorder
import torch.nn.functional as F


args = BaseOptions()

args.parser.add_argument('--zero_audio_folder', type=str, default="", help="raw audio path for zero-shot")
args.parser.add_argument('--zero_video_folder', type=str, default="", help="video frame path for zero-shot")
args.parser.add_argument('--backbone', type=str, default='', help="name of the pretrained model")
args.parser.add_argument('--is_event_score', type=int, default=1, help="compute is_event_score")
args.parser.add_argument('--test_dataset_name', type=str, default="", help="the downstream dataset")

args = args.parse()
is_event_score = args.is_event_score
dataset_name = args.test_dataset_name
if is_event_score:
    args.weak = True
else:
    args.weak = True
    
if dataset_name == "AVE":
    from zero_shot_AVE_dataset import AVE_dataset as dataset
    args.zero_audio_folder = "/root/autodl-tmp/duanhaoyi/data/AVE/wave"
    args.zero_video_folder = "/root/autodl-tmp/duanhaoyi/data/AVE/frames"
    args.CTX_INIT = "a photo of a"
    args.test_batch_size = 16
elif dataset_name == "LLP":
    from zero_shot_AVE_dataset import AVE_dataset as dataset
    args.zero_audio_folder = "/root/autodl-tmp/duanhaoyi/data/AVVP/LLP_dataset/wave"
    args.zero_video_folder = "/root/autodl-tmp/duanhaoyi/data/AVVP/LLP_dataset/frame"
    args.CTX_INIT = "a photo of a"
    args.test_batch_size = 32
else:
    raise NotimplementedError

from zero_shot_model import main_model

from nets.net_trans import MMIL_Net

from utils.utils import *
from utils.Recorder import Recorder
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pretrain_model = MMIL_Net(args).to('cuda')
# 加载vggsound预训练的模型
checkpoint = torch.load("../pretrain/models/{}".format(args.backbone))
pretrain_model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint.items() if 'prompt_learner.token_prefix' not in k and 'prompt_learner.token_suffix' not in k}, strict=False)

# =============================================================================

def main():
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Dataset'''
    
    test_dataloader = DataLoader(
        dataset(args, mode='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''model setting'''
    mainModel = main_model(args, pretrain_model)
    mainModel = nn.DataParallel(mainModel).cuda()
    
    SEED = 43
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False 
    '''Only Evaluate'''
    print(f"\nStart Evaluation..")
    acc = validate_epoch(args, mainModel, test_dataloader, epoch=0, eval_only=True)
    
    return
    

@torch.no_grad()
def validate_epoch(args, model, test_dataloader, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end_time = time.time()

    model.eval()
    # model.double()
    
    total_acc = 0
    total_num = 0
    
    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        wave, gt = batch_data['wave'].to('cuda'), batch_data['GT'].to('cuda') 
        image = batch_data['image'].to('cuda')
        
        bs = gt.size(0)
        event_scores, _, _ = model(image, wave)
        
        if dataset_name == "AVE":
            if is_event_score:
                acc = (event_scores.argmax(dim=-1) == rearrange(gt, 'b t class -> (b t) class').argmax(dim=-1)).sum()
            else:
                gt = gt[:, :, :-1]
                gt_list = []
                for gt_i in gt:
                    ready = False
                    index, _ = gt_i.max(dim=-1)
                    for i, index_i in enumerate(index):
                        if index_i.item() != 0:
                            gt_list.append(gt_i[i])
                            ready = True
                            break
                    if not ready:
                        gt_list.append(torch.zeros_like(gt_i[0]))                        
                gt = torch.stack(gt_list)
    
                event_scores = event_scores.view(bs, 10, -1)
                event_scores = torch.mean(event_scores, dim=1)
                acc = (event_scores.argmax(dim=-1) == gt.argmax(dim=-1)).sum()
        elif dataset_name == "LLP":
            event_scores = event_scores.view(bs, 10, -1)
            event_scores = torch.mean(event_scores, dim=1)
            acc = (event_scores.argmax(dim=-1) == gt.argmax(dim=-1)).sum()
        num = event_scores.size(0)
        total_acc += acc
        total_num += num

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Print logs in Terminal'''
        if n_iter % 1 == 0:
            print(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                # f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {acc:.3f} ({total_acc/total_num * 100:.3f})'
            )

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        # writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', total_acc/total_num, epoch)

    print(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {total_acc/total_num * 100:.4f}%."
    )
    return total_acc/total_num * 100


if __name__ == '__main__':
    main()

