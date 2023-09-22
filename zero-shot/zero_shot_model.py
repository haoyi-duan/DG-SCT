import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import copy
import copy
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from collections import OrderedDict
import math
from einops import rearrange, repeat
import os
import copy
from nets.prompt_learner import *
from nets.clip import clip
from nets.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import esc_fig

class main_model(nn.Module):
    def __init__(self, config, pretrain_model):
        super(main_model, self).__init__()
        self.config = config
        self.lavish_forward = pretrain_model.lavish_forward
        self.htsat = pretrain_model.htsat
        self.audio_projection = pretrain_model.audio_projection
        
        checkpoint_path = os.path.join(esc_fig.checkpoint_path, esc_fig.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        text_branch_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_branch' in k}
        text_transform_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_transform' in k}
        text_projection_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_projection' in k}
        
        
        self.logit_scale_a = pretrain_model.logit_scale_a
        self.logit_scale_t = pretrain_model.logit_scale_a
        self.audio_visual_contrastive_learner = pretrain_model.audio_visual_contrastive_learner
        
        self.clip_adapter = pretrain_model.clip_adapter
        self.clip_adapter_text = pretrain_model.clip_adapter_text
        
        self.classnames, _ = generate_category_list(self.config)
        classnames = copy.deepcopy(self.classnames)
        self.clap_text_encoder = CLAPTextEncoder(self.config, self.classnames, [text_branch_list, text_transform_list, text_projection_list])
        print(f"Loading CLIP (backbone: {self.config.ViT})")
        clip_model = load_clip_to_cpu(self.config)
        
        
        self.prompt_learner = PromptLearner(self.config, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = pretrain_model.text_encoder
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        
        
    def clip_matching(self, visual_grd):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        x = self.clip_adapter(visual_grd)
        ratio = 0.2
        visual_grd = ratio * x + (1 - ratio) * visual_grd
        visual_grd = visual_grd / visual_grd.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(visual_grd)

        text_features = self.text_encoder(prompts, tokenized_prompts)  # [n_cls, 512]
        x = self.clip_adapter_text(text_features)
        ratio = 0.2
        text_features = ratio * x + (1 - ratio) * text_features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * visual_grd @ text_features.t()
        return logits
    
    def clap_matching(self, audio_features):
        text_features = self.clap_text_encoder()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale_a * audio_features @ text_features.t()
        
        return logits
    
    def forward(self, video, audio):
        b, t, c, w, h = video.shape
        output_dict=self.lavish_forward(rearrange(video, 'b t c w h -> (b t) c w h'), audio)  # [B*10, 512]
        v_cls=output_dict['x']
        a_cls=output_dict['embedding']
        loss_audio_image=output_dict['logits_audio_image']
        loss_image_audio=output_dict['logits_image_audio']
        
        logits_v = self.clip_matching(v_cls)
        logits_a = self.clap_matching(a_cls)
        
        # print('logits_v', logits_v.shape)
        # print('logits_a', logits_a.shape)
        
        # 视频 和 音频 的 匹配分数
        w1 = logits_v / (logits_v + logits_a)
        w2 = logits_a / (logits_v + logits_a)
        event_scores = w1 * logits_v + w2 * logits_a
        
        return event_scores, loss_audio_image, loss_image_audio  # [B*10, 142]


def load_clip_to_cpu(cfg):
    backbone_name = cfg.ViT
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


def generate_category_list(cfg):
    
    if cfg.test_dataset_name == "AVE":
        file_path = '/root/autodl-tmp/duanhaoyi/data/AVE/categories.txt'
    elif cfg.test_dataset_name == "LLP":
        categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                    'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                    'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                    'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                    'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                    'Clapping']
        id_to_idx = {id: index for index, id in enumerate(categories)}
        return categories, id_to_idx
    else:
        raise NotImplementedError
        
    category_list = []
    
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    id_to_idx = {id: index for index, id in enumerate(category_list)}
    
    return category_list, id_to_idx