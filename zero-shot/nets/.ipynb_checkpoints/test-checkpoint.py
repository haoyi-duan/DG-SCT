import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import pandas as pd
import ast
import json
from PIL import Image
from munch import munchify

import time
import random

import h5py
from PIL import Image
from tqdm import tqdm
import pickle
import zipfile
from io import BytesIO
import pdb
import time
import torchaudio

import h5py
from PIL import Image
from tqdm import tqdm
import pickle
import zipfile
from io import BytesIO
import pdb
import time
import torchaudio
import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from collections import OrderedDict
import math

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def load_clip_to_cpu():
    backbone_name = "ViT-B/32"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    
    return model

if __name__ == "__main__":
    clip_model = load_clip_to_cpu()
    clip_model.float()
    path = '/root/autodl-tmp/data/AVE/caption/'
    with open('/root/autodl-tmp/data/AVE/AVE_caption.pkl', 'rb') as tf:
        dic = pickle.load(tf)
    i = 0
    for k, v in dic.items():
        i += 1
        text = []
        for p in v:
            text_i = clip.tokenize(p['sentence'])
            text.append(text_i.squeeze())
        text = torch.stack(text)
        with torch.no_grad():
            result = clip_model.encode_text(text)
        result = result.numpy()
        outfile = os.path.join(path, k+'.npy')
        np.save(outfile, result)
        print('-- {} / {} finish!'.format(i, len(dic)))
            
# def operate(files):
#     with tqdm(total=len(files)) as pbar:
#         for file in files:
#             if '.tar' in file:
#                 os.system('tar -xf {}'.format(file))
#                 os.system('rm {}'.format(file))
#             pbar.update(1)
            
# if __name__ == "__main__":
#     files = os.listdir('./')
#     operate(files)
    
# files = os.listdir('/root/autodl-tmp/data/vggsound/audio_features/vggish')
# print(files)

# cat_list = []
# categories = open('./data/AVE/categories.txt', 'a')
# with open('./data/AVE/Annotations.txt', 'r') as f:
#     for line in f.readlines():
#         cat = line.split('&')[0]
#         if cat not in cat_list:
#             cat_list.append(cat)
#             categories.write(cat)
#             categories.write('\n')
        
# with h5py.File('./data/AVE/vgg/audio_feature_noisy.h5', 'r') as hf:
#     order = hf['avadataset'][:]
    # order = hf['order'][:]
# print(order)
# print('=============================ls')
# os.system('rm a.py')

# files = os.listdir('./data/AVE/video')
# files = os.listdir('./data/vggsound/audio_features/vggish')
# files.sort()

# print(files)
# all_df = pd.read_csv('./data/vggsound/new.csv')
# train_df = all_df[all_df['split'] == 'train']
# valid_df = all_df[all_df['split'] == 'val']
# test_df = all_df[all_df['split'] == 'test']

# print("train: {} / {}".format(len(train_df), len(all_df)))
# print("valid: {} / {}".format(len(valid_df), len(all_df)))
# print("test: {} / {}".format(len(test_df), len(all_df)))
# print(len(files))
# print(len(all_df))

# video_id = [(str(i)).zfill(6) for i in range(len(all_df))]
# all_df['video_id'] = video_id
# all_df.to_csv('./data/vggsound/vggsound-avel40k_labels.csv', index=False)

# n = 0
# with tqdm(total=len(files)) as pbar:
#     for y in files:
#         for x in all_df.values:
#             if x[1] in y:
#                 # os.system('mv ./data/vggsound/video_features/vgg19/{} ./data/vggsound/video_features/vgg19/{}'.format(y, str(int(x[4])).zfill(6)+'.zip'))
#                 os.system('mv ./data/vggsound/audio_features/vggish/{} ./data/vggsound/audio_features/vggish/{}'.format(y, str(int(x[4])).zfill(6)+'.zip'))
#                 break
#         if n % 1000 == 0:
#             # print("{}: rename file {}".format(n, str(int(x[4])).zfill(6)+'.zip'))
#             print("{}: rename file {}".format(n, str(int(x[4])).zfill(6)+'.zip'))
#         n += 1
#         pbar.update(1)

# n = 0
# with tqdm(total=len(files)) as pbar:
#     for y in files:
#         flag = False
#         for x in all_df.values:
#             if str(int(x[4])).zfill(6) in y:
#                 flag = True
#                 break
#         if not flag:
#             n += 1
#             os.system('rm ./data/vggsound/video_features/vgg19/{}'.format(y))
#             # os.system('rm ./data/vggsound/audio_features/vggish/{}'.format(y))
#             print("{}: delete file {}".format(n, y))
#         pbar.update(1)
        
# print("{} / {} files not needed!".format(n, len(files)))

# n = 0
# with tqdm(total=len(all_df)) as pbar:
#     for x in all_df.values:
#         flag = False
#         for y in files:
#             if x[1] in y:
#                 flag = True
#                 break
#         if not flag:
#             n += 1
#             # print("file {} not exists!".format(x[1]))
#         pbar.update(1)
        
# print("{} / {} files not exists!".format(n, len(all_df)))


