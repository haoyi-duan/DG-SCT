import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import torchaudio
import torchvision
import glob
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from scipy import signal
import soundfile as sf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from einops import rearrange, repeat

import warnings
warnings.filterwarnings('ignore')


categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
              'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
              'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
              'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
              'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
              'Clapping']

def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y



class LLP_dataset(Dataset):

    def __init__(self, args, label, audio_dir, video_dir, st_dir, transform=None):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.transform = transform
        self.opt = args
        self.norm_mean =  -4.984795570373535
        self.norm_std =  3.7079780101776123
        self.my_normalize = Compose([
            Resize([192,192], interpolation=Image.BICUBIC),
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


        if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
            sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
            waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)] # [2, 16000]
        ## align end ##

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

        ########### ------> very important: audio normalized
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ### <--------
        target_length = 192 

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
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        name = row[0][:11]
        
        total_num_frames = len(glob.glob(self.opt.video_folder+'/'+name+'/*.jpg'))
        # print("total_num_frames", total_num_frames)
        # sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
        total_img = []
        for vis_idx in range(10):
            # tmp_idx = sample_indx[vis_idx]
            # print("debug here: ", tmp_idx)
            # print(os.path.join(self.opt.root_path, self.opt.video_folder)+'/'+name+'/'+ str("{:08d}".format(tmp_idx))+ '.jpg')
            tmp_img = torchvision.io.read_image(os.path.join(self.opt.root_path, self.opt.video_folder)+'/'+name+'/'+ str("{:08d}".format(vis_idx+1))+ '.jpg')/255
            tmp_img = self.my_normalize(tmp_img)
            total_img.append(tmp_img)
        total_img = torch.stack(total_img)
        ### <---
        
        wave = np.load(os.path.join(self.opt.root_path, 'data/AVVP/LLP_dataset/wave/{}.npy'.format(name)), allow_pickle=True)
        wave = torch.from_numpy(wave)
        wave = wave.view(10, 32000)
        while wave.size(-1) < 32000 * 10:
            wave = torch.cat((wave, wave), dim=-1)     
        wave = wave[:, :32000*10]
        
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)
        sample = {'audio': wave, 'video_s': total_img, 'video_st': video_st, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):
        if len(sample) == 2:
            audio = sample['audio']
            label = sample['label']
            return {'audio': torch.from_numpy(audio), 'label': torch.from_numpy(label)}
        else:
            audio = sample['audio']
            video_s = sample['video_s']
            video_st = sample['video_st']
            label = sample['label']
            return {'audio': audio, 'video_s': video_s,
                    'video_st': torch.from_numpy(video_st),
                    'label': torch.from_numpy(label)}