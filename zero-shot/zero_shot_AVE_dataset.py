import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from ipdb import set_trace
import pickle as pkl
import h5py

import torchaudio
import torchvision
import glob

import ast
from PIL import Image
from munch import munchify
import torch.nn as nn
import time
import random

from tqdm import tqdm
import pickle
import zipfile
from io import BytesIO
import pdb
import json

from einops import rearrange, repeat

### VGGSound
from scipy import signal
###

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from nets.clip import clip
from nets.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


import warnings
warnings.filterwarnings('ignore')


def ids_to_multinomial(ids):
	""" label encoding

	Returns:
		1d array, multimonial representation, e.g. [1,0,1,0,0,...]
	"""
	categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
					'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
					'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
					'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
					'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
					'Clapping']
	id_to_idx = {id: index for index, id in enumerate(categories)}

	y = np.zeros(len(categories))
	for id in ids:
		index = id_to_idx[id]
		y[index] = 1
	return y


class AVE_dataset(Dataset):

	def __init__(self, opt, mode='train'):

		self.opt = opt
		self.mode = mode
		self.root_path = opt.root_path
		self.dataset_name = self.opt.test_dataset_name
		if self.dataset_name == "AVE":
			with h5py.File(os.path.join(self.root_path, 'data/AVE/labels.h5'), 'r') as hf:
				self.labels = hf['avadataset'][:]

			with h5py.File(os.path.join(self.root_path, 'data/AVE/mil_labels.h5'), 'r') as hf:
				self.mil_labels = hf['avadataset'][:]
            
			if mode == 'train':
				with h5py.File(os.path.join(self.root_path, 'data/AVE/train_order.h5'), 'r') as hf:
					order = hf['order'][:]
			elif mode == 'test':
				with h5py.File('data/AVE/test_order.h5', 'r') as hf:
					order = hf['order'][:]

			self.lis = order
			self.len = len(self.lis)
            
			self.raw_gt = pd.read_csv("data/AVE/Annotations.txt", sep="&")
		elif self.dataset_name == "LLP":
			self.audio_folder = 'data/AVVP/LLP_dataset/audio'
			self.video_folder = 'data/AVVP/LLP_dataset/frame'
			self.df = pd.read_csv('data/AVVP/AVVP_test_pd.csv', header=0, sep='\t')
			self.filenames = self.df["filename"]
			self.len = len(self.filenames)

		self.norm_mean = -4.1426
		self.norm_std = 3.2001
		### <----
		self.my_normalize = Compose([
			# Resize([384,384], interpolation=Image.BICUBIC),
			Resize([224,224], interpolation=Image.BICUBIC),
			# Resize([192,192], interpolation=Image.BICUBIC),
			# CenterCrop(224),
			Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
		])

	def getVggoud_proc(self, filename, idx=None):

		audio_length = 1
		samples, samplerate = torchaudio.load(filename)
		samples = samples.transpose(1, 0)
		if samples.shape[0] > 16000*(audio_length+0.1):
			sample_indx = np.linspace(0, samples.shape[0] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
			samples = samples[sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]

		else:
			# repeat in case audio is too short
			samples = np.tile(samples,int(self.opt.audio_length))[:int(16000*self.opt.audio_length)]

		samples[samples > 1.] = 1.
		samples[samples < -1.] = -1.

		frequencies, times, spectrogram = signal.spectrogram(samples, samplerate, nperseg=512,noverlap=353)
		spectrogram = np.log(spectrogram+ 1e-7)

		mean = np.mean(spectrogram)
		std = np.std(spectrogram)
		spectrogram = np.divide(spectrogram-mean,std+1e-9)

		
		return torch.tensor(spectrogram).unsqueeze(0).float()

	def _wav2fbank(self, filename, filename2=None, idx=None):
		# mixup
		if filename2 == None:
			waveform, sr = torchaudio.load(filename)
			waveform = waveform - waveform.mean()
		# mixup
		else:
			waveform1, sr = torchaudio.load(filename)
			waveform2, _ = torchaudio.load(filename2)

			waveform1 = waveform1 - waveform1.mean()
			waveform2 = waveform2 - waveform2.mean()

			if waveform1.shape[1] != waveform2.shape[1]:
				if waveform1.shape[1] > waveform2.shape[1]:
					# padding
					temp_wav = torch.zeros(1, waveform1.shape[1])
					temp_wav[0, 0:waveform2.shape[1]] = waveform2
					waveform2 = temp_wav
				else:
					# cutting
					waveform2 = waveform2[0, 0:waveform1.shape[1]]

			# sample lambda from uniform distribution
			#mix_lambda = random.random()
			# sample lambda from beta distribtion
			mix_lambda = np.random.beta(10, 10)

			mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
			waveform = mix_waveform - mix_waveform.mean()
		

		## yb: align ##
		if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
			sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
			waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)] # [2, 16000]
		## align end ##


		# if self.opt.vis_encoder_type == 'vit':
		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=224, dither=0.0, frame_shift=10) ## original
			# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
		# elif self.opt.vis_encoder_type == 'swin':
		# 	fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------
		# if self.opt.vis_encoder_type == 'vit':
		target_length = 224 ## for audioset: 10s
		# elif self.opt.vis_encoder_type == 'swin':
			# target_length = 192 ## yb: overwrite for swin

		# target_length = 512 ## 5s
		# target_length = 256 ## 2.5s
		n_frames = fbank.shape[0]

		p = target_length - n_frames

		# cut and pad
		if p > 0:
			m = torch.nn.ZeroPad2d((0, 0, 0, p))
			fbank = m(fbank)
		elif p < 0:
			fbank = fbank[0:target_length, :]

		if filename2 == None:
			return fbank, 0
		else:
			return fbank, mix_lambda
        
	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		if self.dataset_name == "AVE":
			
			real_idx = self.lis[idx]		
			file_name = self.raw_gt.iloc[real_idx][1]
            
			filepath = '/root/autodl-tmp/duanhaoyi/data/AVE/wave'
			filepath = os.path.join(filepath, file_name + '.npy')
			wave = np.load(filepath, allow_pickle=True)
			#wave = wave[file_name + '.wav']
			wave = torch.from_numpy(wave)
			wave = wave.view(10, 32000)
			while wave.size(-1) < 32000 * 10:
				wave = torch.cat((wave, wave), dim=-1)
			wave = wave[:, :32000 * 10]
        
		else:
			row = self.df.loc[idx, :]  
			file_name = row[0][:11]
			ids = row[-1].split(',')                
			label = ids_to_multinomial(ids)
			wave = np.load('/root/autodl-tmp/duanhaoyi/data/AVVP/LLP_dataset/wave/{}.npy'.format(file_name), allow_pickle=True)
			wave = torch.from_numpy(wave)
			wave = wave.view(10, 32000)
			while wave.size(-1) < 32000 * 10:
				wave = torch.cat((wave, wave), dim=-1)     
			wave = wave[:, :32000*10]
            
            
		### ---> video frame process 
		total_num_frames = len(glob.glob(self.opt.zero_video_folder+'/'+file_name+'/*.jpg'))
		sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		total_img = []
		for vis_idx in range(10):
			tmp_idx = sample_indx[vis_idx]
			tmp_img = torchvision.io.read_image(self.opt.zero_video_folder+'/'+file_name+'/'+ str("{:08d}".format(tmp_idx))+ '.jpg')/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)
		### <---



		if self.dataset_name == "AVE":
			return {
				'GT':self.labels[real_idx],
				'mil_GT':self.mil_labels[real_idx], 
				'wave': wave, 
				'image':total_img
		}
		else:
			return {
				'GT':label, 
				'wave': wave, 
				'image':total_img
		}           
