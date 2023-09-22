import argparse
from base_options import BaseOptions
from gpuinfo import GPUInfo 
from nets.utils import do_mixup, get_mix_lambda, do_mixup_label
import os
args = BaseOptions().parse()

current_dir = os.getcwd()
print("current_dir", current_dir)

parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
grandparent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
print("root_path", grandparent_dir)

args.root_path = grandparent_dir

mygpu = GPUInfo.get_info()[0]
gpu_source = {}

if 'N/A' in mygpu.keys():
	for info in mygpu['N/A']:
		if info in gpu_source.keys():
			gpu_source[info] +=1
		else:
			gpu_source[info] =1

for gpu_id in args.gpu:
	gpu_id = str(gpu_id)

	if gpu_id not in gpu_source.keys():
		print('go gpu:', gpu_id)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
		break
	elif gpu_source[gpu_id] < 1:
		print('go gpu:', gpu_id)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
		break

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
from einops import rearrange, repeat

import certifi
import sys
from torch.cuda import amp

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler

os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(os.path.dirname(sys.argv[0]), certifi.where())

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)




def train(args, model, train_loader, optimizer, criterion, criterion_event, epoch):
	model.train()
	nceloss = InfoNCELoss(τ=args.margin1)
	mseloss = torch.nn.MSELoss()
	ybloss = YBLoss()
	ybloss2 = YBLoss2()
	


	scaler = amp.GradScaler()
	rand_train_idx = 11

	for batch_idx, sample in enumerate(train_loader):
		

		gt = sample['GT'].to('cuda')
		wave = sample['wave'].to('cuda')

		mean_acc = torch.zeros(1).cuda()
		image = sample['image'].to('cuda')
		bs = image.size(0)
		if args.backbone_type == "audioset":
			mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*10)).to('cuda')
		else:
			mixup_lambda = None
            
	
		optimizer.zero_grad()
        
		is_event_scores, event_scores, audio_visual_gate, av_score = model([wave], image, rand_train_idx=rand_train_idx, stage='train')   
		is_event_scores = is_event_scores.transpose(1, 0).squeeze(-1).contiguous()
		labels = gt.cuda()
		acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
		mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
		audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze(-1).contiguous()

		labels_foreground = labels[:, :, :-1]  # [32, 10, n_cls]
		labels_BCE, labels_evn = labels_foreground.max(-1)
		# _, labels_CAS = labels.max(-1)

		labels_event, _ = labels_evn.max(-1)
		loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
		label_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())
		loss_cas = criterion_event(av_score, labels_event.cuda())
		loss_event_class = criterion_event(event_scores, labels_event.cuda())
		# loss_cas_event = criterion_event(cas_out, labels_CAS.cuda())

		loss = loss_is_event + label_is_gate + loss_event_class + loss_cas

		loss.backward()

		# weights update
		if ((batch_idx + 1) % args.accum_itr == 0) or (batch_idx + 1 == len(train_loader)):
			optimizer.step()


		if batch_idx % 50 == 0:
			# print(model.fusion_weight['blocks-0-attn-qkv-weight'])

			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_total: {:.6f}'.format(
				epoch, batch_idx * bs, len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item())) #

def eval(model, val_loader, args):

	model.eval()
	mean_acc = torch.zeros(1).cuda()

	total_acc = 0
	total_num = 0

	for batch_idx, sample in enumerate(val_loader):
		
		gt= sample['GT'].to('cuda')
		wave = sample['wave'].to('cuda')
		image = sample['image'].to('cuda')
		bs = image.size(0)
        
		if args.backbone_type == "audioset":
			mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*10)).to('cuda')
		else:
			mixup_lambda = None

		with torch.no_grad():
			labels = gt.cuda()
			is_event_scores, event_scores, audio_visual_gate, av_score = model([wave], image)
			is_event_scores = is_event_scores.transpose(1, 0).squeeze(-1).contiguous()
			acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
			mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)

	print('val acc: %.2f'%(mean_acc.item()))
	return mean_acc.item()



def main():


	if args.wandb:
		wandb.init(config=args, project="ada_av",name=args.model_name)

	if args.model == 'MMIL_Net':
		model = MMIL_Net(args).to('cuda')
		# model.double()
	else:
		raise ('not recognized')

	## -------> condition for wandb tune
	if args.start_tune_layers > args.start_fusion_layers: 
		exit()
	#### <------
	if args.mode == 'train':
		########## note for fast training #########
		# print('loading pre-training!!!!!!!!!!!!!!!!!')
		# model.load_state_dict(torch.load(args.model_save_dir + "MMIL_Net_52.35.pt"))
		########## note for fast training #########
		train_dataset = AVE_dataset(opt = args)
		val_dataset = AVE_dataset(opt = args, mode='test')
		# val_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
		# 									   ToTensor()]))
		
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory = True)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory = True)

		param_group = []
		train_params = 0
		total_params = 0
		additional_params = 0
		for name, param in model.named_parameters():
			
			param.requires_grad = False
			### ---> compute params
			tmp = 1
			for num in param.shape:
				tmp *= num

			if 'ViT' in name or 'swin' in name:
				if 'norm' in name and args.is_vit_ln:
					param.requires_grad = bool(args.is_vit_ln)
					total_params += tmp
					train_params += tmp
				else:
					param.requires_grad = False
					total_params += tmp
			elif 'htsat' in name:
				param.requires_grad = False
				total_params += tmp
                    
			# ### <----



			# if  'audio_adapter_blocks' in name :  #'my_blocks' in name or 'my_mlp_forward' in name or 'adapter' in name or 'my_mlp_forward' in name 
			# 	print(name)
			# 	param.requires_grad = False
			# 	train_params += tmp
			# 	additional_params += tmp
			# 	total_params += tmp
			
			elif 'adapter_blocks' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
				print('########### train layer:', name, param.shape , tmp)
			elif 'CMBS' in name:
				param.requires_grad = True
				train_params += tmp
				additional_params += tmp
				total_params += tmp
			# elif 'norm' in name:
			# 	param.requires_grad = True
			# 	train_params += tmp
				# print('########### train layer:', name)
			elif 'mlp_class' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
			elif 'temporal_attn' in name:
				param.requires_grad = True
				train_params += tmp
				total_params += tmp
				additional_params += tmp
			if 'mlp_class' in name:
				param_group.append({"params": param, "lr":args.lr_mlp})
			else:
				param_group.append({"params": param, "lr":args.lr})
		print('####### Trainable params: %0.4f  #######'%(train_params*100/total_params))
		print('####### Additional params: %0.4f  ######'%(additional_params*100/(total_params-additional_params)))
		print('####### Total params in M: %0.1f M  #######'%(total_params/1000000))


		optimizer = optim.Adam(param_group)
		# optimizer = optim.Adam(model.parameters(), lr=args.lr)

		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay)
		# scheduler = StepLR(optimizer, step_size=15000, gamma=0.1)
		criterion = nn.BCEWithLogitsLoss()
		# criterion = nn.CrossEntropyLoss()
		criterion_event = nn.CrossEntropyLoss()
		best_F = 0
		count = 0
		for epoch in range(1, args.epochs + 1):
			train(args, model, train_loader, optimizer, criterion, criterion_event, epoch=epoch)
			scheduler.step()
			F_event  = eval(model, val_loader, args)
			

			count +=1

			if  F_event >= best_F:
				count = 0
				best_F = F_event
				print('#################### save model #####################')
				torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + "_%0.2f.pt"%(best_F))
				if args.wandb:
					wandb.log({"val-best": F_event})
			if count == args.early_stop:
				exit()
	else:
		test_dataset = AVE_dataset(opt=args, mode='test')
		test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
		model.load_state_dict(torch.load(args.model_save_dir + "best_82.18" + ".pt"), strict=False)
		eval(model, test_loader, args)
        
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
	pred[scores_mask] = 28  # 141 denotes bg
	correct = pred.eq(targets)
	correct_num = correct.sum().double()
	acc = correct_num * (100. / correct.numel())

	return acc

if __name__ == '__main__':
	main()

