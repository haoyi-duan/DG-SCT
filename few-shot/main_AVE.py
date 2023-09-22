from __future__ import print_function
import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo 
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

args = BaseOptions().parse()
if args.dataset_name == "AVE":
    args.audio_folder = 'data/AVE/audio'
    args.video_folder = 'data/AVE/frames'
elif args.dataset_name == "vggsound":
    args.audio_folder = 'data/vggsound/audio'
    args.video_folder = 'data/vggsound/frames'
    
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.cuda.amp import autocast
import random
from dataloader import *
from nets.net_trans import MMIL_Net
# from nets.net_trans_805 import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
from ipdb import set_trace
import wandb
from PIL import Image
from criterion import YBLoss,YBLoss2, InfoNCELoss, MaskInfoNCELoss

from torch.optim.lr_scheduler import StepLR
from einops import rearrange

import certifi
import sys
from torch.cuda import amp

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

SEED = 43
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def train(args, model, train_loader, optimizer, criterion, criterion_event, epoch):
    
	model.train()
 
	optimizer.zero_grad()
	mean_loss = torch.zeros(1).to(device)
	mean_loss_audio_image = torch.zeros(1).to(device)
	mean_loss_image_audio = torch.zeros(1).to(device)
	mean_acc = torch.zeros(1).to(device)
        
	for batch_idx, sample in enumerate(train_loader):
		gt = sample['GT']
		# print("gt", gt.shape)
		image = sample['image']
		wave = sample['wave']

		bs = image.size(0)

		if args.dataset_name in ['AVE', 'vggsound']:
			event_scores, logits_audio_image, logits_image_audio = model(wave.to(device), image.to(device))
			if args.weak:
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
		else:
			raise NotImplementedError
                        
		labels = gt.to(device)
		if args.dataset_name in ['AVE', 'vggsound']:
			if args.weak:
				event_scores = event_scores.view(bs, 10, -1)
				event_scores = torch.mean(event_scores, dim=1)
				loss = criterion_event(event_scores, labels)
			else:
				loss = criterion_event(event_scores, rearrange(labels, 'b t class -> (b t) class'))
		else:
			raise NotImplementedError
                
		audio_visual_labels = torch.eye(bs).to(device)
		loss_audio_image = criterion(logits_audio_image, audio_visual_labels)       
		loss_image_audio = criterion(logits_image_audio, audio_visual_labels) 
		w1 = loss / (epoch + loss_audio_image + loss_image_audio)
		w2 = loss_audio_image / (epoch + loss_audio_image + loss_image_audio)
		w3 = loss_image_audio / (epoch + loss_audio_image + loss_image_audio)
		if epoch <= 10:
			#loss = w1 * loss + w2 * loss_audio_image + w3 * loss_image_audio
			loss = 500 * loss + 2 * loss_audio_image + 1.5 * loss_image_audio
			#loss = loss +  2 * loss_audio_image + 1.5 * loss_image_audio
		else:
			loss = 5 * loss + 2 * loss_audio_image + 1.5 * loss_image_audio
			#loss = loss +  loss_audio_image + loss_image_audio
		loss.backward(retain_graph=True)
		loss_audio_image = reduce_value(loss_audio_image, average=True)
		loss_image_audio = reduce_value(loss_image_audio, average=True)
		mean_loss = (mean_loss * batch_idx + loss.detach()) / (batch_idx + 1)
		mean_loss_image_audio = (mean_loss_image_audio * batch_idx + loss_image_audio.detach()) / (batch_idx + 1)
		mean_loss_audio_image = (mean_loss_audio_image * batch_idx + loss_audio_image.detach()) / (batch_idx + 1)
		'''Compute Accuracy'''
		if not args.weak:        
			val = (event_scores.argmax(dim=-1) == rearrange(labels, 'b t class -> (b t) class').argmax(dim=-1)).sum()
		else:
			val = (event_scores.argmax(dim=-1) == labels.argmax(dim=-1)).sum()
		num = event_scores.size(0)
		acc = val / num * 100
		mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
		
            
		'''Clip Gradient'''
		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        
		optimizer.step()
		optimizer.zero_grad()
        
		# # weights update
		# if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
		# 	optimizer.step()

		if batch_idx % 50 == 0:
				# print(model.fusion_weight['blocks-0-attn-qkv-weight'])

			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_total: {:.6f}'.format(
				epoch, batch_idx * bs, len(train_loader.dataset),
						100. * batch_idx / len(train_loader), mean_loss.item()))
			print('loss_is_event: {:.6f} loss_audio_image: {:.6f} loss_image_audio: {:.6f}'.format((mean_loss-mean_loss_image_audio-mean_loss_audio_image).item(), mean_loss_audio_image.item(), mean_loss_image_audio.item()))
			print('train acc: %.3f'%(mean_acc.item()))

@torch.no_grad()
def eval(model, val_loader, args):

	model.eval()
	mean_acc = torch.zeros(1).to(device)
    
	for batch_idx, sample in enumerate(val_loader):
		
		gt= sample['GT']
		image = sample['image']
		wave = sample['wave']
		bs = image.size(0)
		labels = gt.to(device)
		if args.dataset_name in ['AVE', 'vggsound']:
			event_scores, _, _ = model(wave.to(device), image.to(device))
			val = (event_scores.argmax(dim=-1) == rearrange(labels, 'b t class -> (b t) class').argmax(dim=-1)).sum()
		else:
			raise NotImplementedError

		num = event_scores.size(0)
		acc = val / num * 100
		mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)

	print('val acc: %.3f'%(mean_acc.item()))
        
	return mean_acc

def main():
	
	# if args.wandb:
	# 	wandb.init(config=args, project="ada_av",name=args.model_name)

	model = MMIL_Net(args)
	if args.resume != "":
		checkpoint_path = os.path.join("./models", args.resume)
		tmp = torch.load(checkpoint_path)
		tmp = {k[7:]:v for k, v in tmp.items() if k[7:] not in ['prompt_learner.token_prefix', 'prompt_learner.token_suffix']}
		model.load_state_dict(tmp, strict=False)
    
	"""Freeze clip model"""
	for name, param in model.named_parameters():
		if "text_encoder" in name:
			param.requires_grad_(False)
	
	# convert to DDP model
	#model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
	model.to(device)
	## -------> condition for wandb tune
	if args.start_tune_layers > args.start_fusion_layers: 
		exit()
	#### <------
	if args.mode == 'train':

		train_dataset = AVE_dataset(opt = args)
		val_dataset = AVE_dataset(opt = args, mode='test')
            
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
		val_loader = DataLoader(val_dataset, batch_size=2, num_workers=args.num_workers, shuffle=False, pin_memory=True)

		param_group = []
		train_params = 0
		total_params = 0
		additional_params = 0
        
        # print the names of parameters
		print("-- #### The names of parameters #### --")
		if os.path.exists("module.txt"):
			os.remove("module.txt")
		f = open("module.txt", "a")
		for name, _ in model.named_parameters():
			print(name)
			f.write(name)
			f.write("\n")
		f.close()
        
		for name, param in model.named_parameters():
			
			param.requires_grad = False
			### ---> compute params
			tmp = 1
			for num in param.shape:
				tmp *= num

			if 'ViT' in name:
				param.requires_grad = False
				total_params += tmp
			elif 'htsat' in name:
				param.requires_grad = False
				total_params += tmp
			elif 'text_encoder' in name:
				param.requires_grad = False
				total_params += tmp
			elif 'token_embedding' in name:
				param.requires_grad = False
				total_params += tmp
			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name, param.shape, tmp)
			elif 'clip_adapter' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
			elif 'audio_adapter' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
			elif 'prompt_learner' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
			elif 'CMBS' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
			elif 'audio_visual_contrastive_learner' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
			
			if 'mlp_class' in name:
				param_group.append({"params": param, "lr":args.lr_mlp})
			else:
				param_group.append({"params": param, "lr":args.lr})
		print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
		print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
		print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))


		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
        
		scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

		criterion = nn.CrossEntropyLoss()
		criterion_event = nn.CrossEntropyLoss()

		best_F = 0 if args.resume == "" else float(args.resume.split("_")[-1][:-3])
		count = 0
		checkpoint_name = args.resume
		for epoch in range(1, args.epochs + 1):
            
			train(args, model, train_loader, optimizer, criterion, criterion_event, epoch=epoch)

			F_event  = eval(model, val_loader, args)

			count +=1
			scheduler.step()
            
			if  F_event >= best_F:
				count = 0
				best_F = F_event
				print('#################### save model #####################')
				print('[epoch {}] accuracy: {}'.format(epoch, best_F))
				if checkpoint_name != "":
					os.system("rm {}".format(args.model_save_dir + checkpoint_name))
				torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + "_%0.2f.pt"%(best_F))
				checkpoint_name = args.checkpoint + "_%0.2f.pt"%(best_F)                    
				# if args.wandb:
				# 	wandb.log({"val-best": F_event})
			if count == args.early_stop:
				exit()
        
	elif args.mode == 'val':
		test_dataset = AVE_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
									st_dir=args.st_dir, transform=transforms.Compose([
				ToTensor()]))
            
		test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
		# model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		eval(model, test_loader, args.label_val, args)
	else:
		test_dataset = AVE_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
											   ToTensor()]))
		            
		test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
		# model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
		eval(model, test_loader, args.label_test, args)

def compute_accuracy_supervised(is_event_scores, event_scores, labels):
	# labels = labels[:, :, :-1]  # 28 denote background
	_, targets = labels.max(-1)
	# pos pred
	is_event_scores = is_event_scores.sigmoid()
	scores_pos_ind = is_event_scores > 0.5
	scores_mask = scores_pos_ind == 0
	_, event_class = event_scores.max(-1)  # foreground classification
	pred = scores_pos_ind.long()
	pred *= event_class[:, None]
	# add mask
	pred[scores_mask] = 141  # 141 denotes bg
	correct = pred.eq(targets)
	correct_num = correct.sum().double()
	acc = correct_num * (100. / correct.numel())

	return acc

if __name__ == '__main__':
	main()

