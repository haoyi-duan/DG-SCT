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

def generate_category_list(root_path, dataset_name):
    if dataset_name == "vggsound":
        file_path = os.path.join(root_path, 'data/vggsound/VggsoundAVEL40kCategories.txt')
    elif dataset_name == "AVE":
        file_path = os.path.join(root_path, 'data/AVE/categories.txt')
    else:
        raise NotImplementedError
        
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    id_to_idx = {id: index for index, id in enumerate(category_list)}
    return category_list, id_to_idx

class AVE_dataset(Dataset):

	def __init__(self, opt, mode='train'):

		self.opt = opt
		self.mode = mode
		self.root_path = opt.root_path
		self.dataset_name = opt.dataset_name
		if self.dataset_name == "vggsound":
			all_df = pd.read_csv(os.path.join(opt.root_path, "data/vggsound/vggsound-avel40k_labels.csv"))
			self.split_df = all_df[all_df['split'] == mode]

			# 输出train、test、valid的占比
			print(f'{len(self.split_df)}/{len(all_df)} videos are used for {mode}')
			self.all_categories, self.id_to_idx = generate_category_list(opt.root_path, self.dataset_name)
			print(f'total {len(self.all_categories)} classes in VggsoundAVEL40k')
			self.len = len(self.split_df)
			if mode == "train" and self.opt.shot > 0:
				category_dict = dict()
				for category in self.all_categories:
					category_dict[category] = []
				all_lst = []
				for i in range(len(self.split_df)):
					one_video_df = self.split_df.iloc[i]
					cat = one_video_df['category']
					category_dict[cat].append(i)
				for category, lst in category_dict.items():
					lst = lst[:self.opt.shot]
					all_lst += lst
				self.order = all_lst
				self.len = len(all_lst)
		elif self.dataset_name == "AVE":
			self.all_categories, self.id_to_idx = generate_category_list(self.root_path, self.dataset_name)
			with h5py.File(os.path.join(opt.root_path, 'data/AVE/labels.h5'), 'r') as hf:
				self.labels = hf['avadataset'][:]

			if mode == 'train':
				with h5py.File(os.path.join(opt.root_path, 'data/AVE/train_order.h5'), 'r') as hf:
					order = hf['order'][:]
			elif mode == 'test':
				with h5py.File(os.path.join(opt.root_path, 'data/AVE/test_order.h5'), 'r') as hf:
					order = hf['order'][:]

			self.raw_gt = pd.read_csv(os.path.join(opt.root_path, "data/AVE/Annotations.txt"), sep="&")
			category_dict = dict()
			for category in self.all_categories:
				category_dict[category] = []
                
			if mode == "train":
				if self.opt.shot > 0:
					all_lst = []
					for idx in order:
						cat = self.raw_gt.iloc[idx][0]
						category_dict[cat].append(idx)      
					for category, lst in category_dict.items():
						lst = lst[:self.opt.shot]
						all_lst += lst
					order = all_lst
			self.lis = order
			self.len = len(self.lis)
		elif self.dataset_name == "LLP":
			if mode == "train":
				self.audio_folder = os.path.join(self.root_path, 'data/AVVP/LLP_dataset/audio')
				self.video_folder = os.path.join(self.root_path, 'data/AVVP/LLP_dataset/frame')
				self.df = pd.read_csv(os.path.join(self.root_path, 'data/AVVP/AVVP_train.csv'), header=0, sep='\t')
				if opt.shot > 0:
					categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
						'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
						'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
						'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
						'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
						'Clapping']
					cat_list = dict()
					for cat in categories:
						cat_list[cat] = []
					for idx in range(len(self.df)):
						row = self.df.loc[idx, :]
						file_name = row[0][:11]
						ids = row[-1].split(',')
						if len(ids) != 1:
							continue
						cat_list[ids[0]].append(idx)
					self.all_lst = []
					for category, lst in cat_list.items():
						lst = lst[:self.opt.shot]
						self.all_lst += lst
				self.filenames = self.df["filename"]
				self.len = len(self.filenames)
				if opt.shot > 0:
					self.len = len(self.all_lst)
			
			elif mode == "test":
				self.audio_folder = os.path.join(self.root_path, 'data/AVVP/LLP_dataset/audio')
				self.video_folder = os.path.join(self.root_path, 'data/AVVP/LLP_dataset/frame')
				self.df = pd.read_csv(os.path.join(self.root_path, 'data/AVVP/AVVP_test_pd.csv'), header=0, sep='\t')
				self.filenames = self.df["filename"]
				self.len = len(self.filenames)


			self.norm_mean = -4.984795570373535
			self.norm_std = 3.7079780101776123

			self.my_normalize = Compose([
				Resize([192, 192], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])

		else:
			raise NotImplementedError

        
		### ---> yb calculate: vggsound dataset
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


		fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10) ## original
		# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)

		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------
		target_length = int(1024 * (1/10)) ## for audioset: 10s

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
		if self.dataset_name == "vggsound":
			if self.mode == "train" and self.opt.shot > 0:
				video_id = self.order[idx]
			else:
				video_id = idx
			one_video_df = self.split_df.iloc[video_id]
			category, video_name, labels = one_video_df['category'], one_video_df['video_id'], one_video_df['label']
			video_name = str(int(video_name)).zfill(6)
			labels = ast.literal_eval(labels)

			category_id = self.id_to_idx[category]
			label = torch.zeros(10, 142)
  
			labels = torch.tensor(labels)==1
			for id, is_event in enumerate(labels):
				if is_event == True:
					label[id][category_id] = 1
				else:
					label[id][-1] = 1
		
			file_name = video_name
            
		elif self.dataset_name == "AVE":
			real_idx = self.lis[idx]

			file_name = self.raw_gt.iloc[real_idx][1]

			### ---> video frame process
			total_num_frames = len(glob.glob(os.path.join(self.root_path, self.opt.video_folder) + '/' + file_name + '/*.jpg'))
			sample_indx = np.linspace(1, total_num_frames, num=10, dtype=int)
			total_img = []
			for vis_idx in range(10):
				tmp_idx = sample_indx[vis_idx]
				tmp_img = torchvision.io.read_image(
					os.path.join(self.root_path, self.opt.video_folder) + '/' + file_name + '/' + str("{:08d}".format(tmp_idx)) + '.jpg') / 255
				tmp_img = self.my_normalize(tmp_img)
				total_img.append(tmp_img)
			total_img = torch.stack(total_img)
			### <---

			filepath = os.path.join(self.root_path, 'data/AVE/wave')
			file_name += '.npy'
			filepath = os.path.join(filepath, file_name)
			wave = np.load(filepath, allow_pickle=True)
			wave = torch.from_numpy(wave)
			wave = wave.view(10, 32000)
			while wave.size(-1) < 32000 * 10:
				wave = torch.cat((wave, wave), dim=-1)
			wave = wave[:, :32000 * 10]
			#需要查看一下wave的shape
			return {
					'GT': self.labels[real_idx],
					'image': total_img,
					'wave': wave
					}

		elif self.dataset_name == "LLP":
			if self.mode == 'train' and self.opt.shot > 0:
				real_idx = self.all_lst[idx]
				row = self.df.loc[real_idx, :]
			else:
				row = self.df.loc[idx, :]  
			file_name = row[0][:11]
			ids = row[-1].split(',')                
			label = ids_to_multinomial(ids)

			### ---> video frame process 
			total_num_frames = len(glob.glob(os.path.join(self.root_path, self.video_folder)+'/'+file_name+'/*.jpg'))
			total_img = []
			for vis_idx in range(10):
				tmp_img = torchvision.io.read_image(self.video_folder+'/'+file_name+'/'+ str("{:08d}".format(vis_idx+1))+ '.jpg')/255
				tmp_img = self.my_normalize(tmp_img)
				total_img.append(tmp_img)
			total_img = torch.stack(total_img)
			### <---
   
			wave = np.load(os.path.join(self.root_path, 'data/AVVP/LLP_dataset/wave/{}.npy'.format(file_name)), allow_pickle=True)
			wave = torch.from_numpy(wave)
			wave = wave.view(10, 32000)
			while wave.size(-1) < 32000 * 10:
				wave = torch.cat((wave, wave), dim=-1)     
			wave = wave[:, :32000*10]
            
			return {
					'GT': label, 
					'wave': wave,
					'image':total_img
			}
		
		else:
			raise NotImplementedError

		### ---> video frame process 
		total_num_frames = len(glob.glob(os.path.join(self.root_path, self.opt.video_folder)+'/'+file_name+'/*.jpg'))
		sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		total_img = []
		for vis_idx in range(10):
			tmp_idx = sample_indx[vis_idx]
			tmp_img = torchvision.io.read_image(os.path.join(self.root_path, self.opt.video_folder)+'/'+file_name+'/'+ str("{:08d}".format(tmp_idx))+ '.jpg')/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)
		### <---
   

		return {
					'GT': label, 
					'image':total_img
			}
	
	def _load_fea(self, fea_base_path, video_id):
		fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
		with zipfile.ZipFile(fea_path, mode='r') as zfile:
			for name in zfile.namelist():
				if '.pkl' not in name:
					continue
				with zfile.open(name, mode='r') as fea_file:#fea_file是.pkl文件
					content = BytesIO(fea_file.read())
					fea = pkl.load(content)
		return fea

