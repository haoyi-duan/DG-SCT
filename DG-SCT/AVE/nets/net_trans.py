import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace
import os

from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat

from timm.models.vision_transformer import Attention
import timm
import loralib as lora

from transformers.activations import get_activation
from .models import *
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint

### VGGSound
from nets import Resnet_VGGSound
from .htsat import HTSAT_Swin_Transformer
import nets.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label
    
# torch.manual_seed(0)
# np.random.seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)


from nets.my_vit import my_vit

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output



class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class+1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, content):

        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model=d_model, num_class=28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores

class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)
    
    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class TemporalAttention(nn.Module):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.beta = 0.4
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.video_fc_dim = 512
        self.audio_fc_dim = 128
        self.d_model = 256

        self.v_fc = nn.Linear(1536, self.video_fc_dim)
        self.a_fc = nn.Linear(768, self.audio_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )
        self.video_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )

        self.alpha = 0.1
        self.gamma = 0.1

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 1536]
        # [batch, 10, 768]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_feature = self.a_fc(audio_feature)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]


        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)


        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.gamma
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.gamma

        return video_query_output, audio_query_output, audio_visual_gate
    
    
class CMBS(nn.Module):
    def __init__(self, config):
        super(CMBS, self).__init__()
        self.config = config
        self.beta = 0.4
        self.d_model = 256

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model)
        self.video_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.alpha = 0.1
        self.gamma = 0.3

    def forward(self, visual_feature, audio_feature):

        video_cas = self.video_cas(visual_feature)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_feature)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(visual_feature, audio_feature)
        audio_query_output = self.VAInter(audio_feature, visual_feature)

        is_event_scores, event_scores = self.localize_module((visual_feature + audio_feature)/2)
        event_scores = event_scores + self.gamma*av_score

        return is_event_scores, event_scores, av_score
    

class AudioVisualContrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.len = 10
        self.fc_v1 = nn.Linear(1536, 512)
        self.fc_a1 = nn.Linear(768, 512)
        self.tanh = nn.Tanh()
        self.fc_gl=nn.Linear(1024,512)
        
        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 1)
        self.relu4 = nn.ReLU()
        
    def forward(self, video, audio, f_v_spatial_att_maps):
        bs = video.size(0) // self.len
        audio = audio.squeeze(dim=1)
        video = self.fc_v1(video)
        audio_feat = self.fc_a1(audio)
        temp_visual = video.permute(0, 2, 1)
        temp_visual = temp_visual.view(temp_visual.size(0), temp_visual.size(1), 6, 6)
        v_feat = torch.bmm(f_v_spatial_att_maps, video)
        visual_feat_before_grounding_posi = v_feat.squeeze(1)    # [B*T, C]
        
        B, C, H, W = temp_visual.size()
        v_feat = temp_visual.view(B, C, H*W)
        v_feat = v_feat.permute(0, 2, 1)                            # [B*T, HxW, C]
        visual_feat_posi = nn.functional.normalize(v_feat, dim=2)   # [B*T, HxW, C]
        
        ## audio-visual grounding posi
        audio_feat_aa = audio_feat.unsqueeze(-1)                        # [B*T, C, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B*T, C, 1]
        
        _, C = visual_feat_before_grounding_posi.size()
        visual_feat_before_grounding_posi = visual_feat_before_grounding_posi.view(bs, 10, C) # [B, T, C]
        _, hw, C = visual_feat_posi.size()
        visual_feat_posi = visual_feat_posi.view(bs, 10, hw, C) # [B, T, HxW, C]
        _, C, D = audio_feat_aa.size()
        audio_feat_aa = audio_feat_aa.view(bs, 10, C, D) # [B, T, C, 1]
        
        visual_gl = []
        for i, audio_feat_aa_i in enumerate(audio_feat_aa): 
            for j, (visual_feat_before_grounding_posi_j, visual_feat_posi_j) in enumerate(zip(visual_feat_before_grounding_posi, visual_feat_posi)):
            
                x2_va = torch.matmul(visual_feat_posi_j, audio_feat_aa_i).squeeze() # [T, HxW]

                x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [T, 1, HxW]
                visual_feat_grd = torch.matmul(x2_p, visual_feat_posi_j)
                visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [T, C]   

                visual_gl_ij = torch.cat((visual_feat_before_grounding_posi_j, visual_feat_grd_after_grounding_posi), dim=-1)
                visual_gl.append(visual_gl_ij)
        
        visual_gl = torch.stack(visual_gl) # [B*B, T, C*2]
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*B, T, C]
        
        _, C = audio_feat.size()
        audio_feat = audio_feat.view(bs, 10, C)
        audio_feat_list = []
        for i, audio_feat_i in enumerate(audio_feat):
            for j in range(bs):
                 audio_feat_list.append(audio_feat_i)
        
        audio_feat = torch.stack(audio_feat_list)
                    
        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*B, T, C*2], [B*B, T, 1024]

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_posi = self.fc4(feat)     # (128, 1)
        
        
        return out_match_posi
    

class AudioVisualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.6
        self.d_model = 256
        self.audio_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )
        self.video_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )
        
        self.video_encoder = InternalTemporalRelationModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=512, d_model=self.d_model, num_layers=1)
        self.fc_a = nn.Linear(768, 128)
        self.fc_a = nn.Linear(1536, 512)
        
    def forward(self, x, audio):
        x = x.squeeze(1)
        audio = audio.squeeze(1)
        
        bs = x.size(0) // 10
        x = x.view(bs, 10, -1) # [B, 10, 1536]
        x = x.permute(1, 0, 2) # [10, B, 1536]
        audio = audio.view(bs, 10, -1) # [B, 10, 768]
        f_v = self.fc_v(x)
        f_a = self.fc_a(audio)
        # audio query
        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio, x)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, B, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, B, 512]
        
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        
        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)
        
        x = x + audio_gate * x * self.alpha
        
        x = x.permute(1, 0, 2) # [B, 10, 1536]
        x = x.view(bs*10, -1)
        
        audio = audio + video_gate * audio * self.alpha
        x = x.permute(1, 0, 2) # [B, 10, 768]
        x = x.view(bs*10, -1)
        
        audio_key_value_feature = audio_key_value_feature.permute(1, 0, 2).contiguous() # [B, 10, 256]
        audio_key_value_feature = audio_key_value_feature.view(bs*10, -1)
        
        return x, audio
    
    
class VisualAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True, num_tk=87, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt
		self.num_tk = num_tk
		self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)  
		self.fc = nn.Linear(linear_in, linear_out)
		self.conv_dim_out = conv_dim_out
        
		d_model = linear_out // 2
		self.fc_affine_audio_1 = nn.Linear(linear_out, linear_out)
		self.fc_affine_video_1 = nn.Linear(linear_out, linear_out)
		self.fc_affine_bottleneck = nn.Linear(linear_out, d_model)
		self.fc_affine_video_2 = nn.Linear(linear_out, d_model)
		self.fc_affine_audio_2 = nn.Linear(linear_out, d_model)     
		self.fc_affine_v_s_att = nn.Linear(d_model, 1) 
		self.fc_tanh = nn.Tanh()
		self.fc_softmax = nn.Softmax(dim=-1)
		self.fc_affine_v_c_att = nn.Linear(d_model, linear_out)

        
		if use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // reduction_factor
			### -----> attetnion 
			# self.cm1_att = nn.MultiheadAttention(embed_dim=self.down_sample_size, num_heads=1)

			# self.cm2_att = nn.MultiheadAttention(embed_dim=self.down_sample_size, num_heads=1)




			# self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, self.down_sample_size)))
			# self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))


			self.my_tokens = nn.Parameter(torch.rand((num_tk, input_dim)))

			# self.ln_z = nn.LayerNorm(self.down_sample_size)
			# self.ln_tk = nn.LayerNorm(self.down_sample_size)

			# self.mapping = nn.Conv2d(input_dim, input_dim, 1, groups=self.opt.num_conv_group, bias=False)
			

			self.gate_tk = nn.Parameter(torch.ones(1))


			self.gate_av = nn.Parameter(torch.zeros(1))
	

			
			

			### <------

			self.activation = nn.ReLU(inplace=True)
			# self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			
			# self.down_sampler_vis = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

			# self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)
			
			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // reduction_factor
			self.activation = nn.ReLU(inplace=True)
			
			# self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			nn.init.zeros_(self.down_sampler) # yb:for lora

			# self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "basic":
			self.activation = nn.ReLU(inplace=True)
			# self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
			self.conv = nn.Linear(input_dim, output_dim, bias=False)

			if use_bn:
				# self.bn = nn.BatchNorm2d(output_dim)
				self.bn = nn.BatchNorm1d(output_dim)

		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		vis_token = self.conv_adapter(vis_token.transpose(2, 1))
		vis_token = self.fc(vis_token.squeeze(-1))
		vis_token = vis_token.permute(0, 2, 1).unsqueeze(-1)

		# vis_token = vis_token.squeeze(-1)
		# vis_token = vis_token.transpose(2, 1)
		# vis_token = self.fc(vis_token)
		# hw = int(math.sqrt(vis_token.size(1)))
		# vis_token = vis_token.view(vis_token.size(0), hw, hw, -1)
		# vis_token = F.interpolate(rearrange(vis_token, 'BF w h c -> BF c w h'), mode='bicubic',size=[int(math.sqrt(self.conv_dim_out)), int(math.sqrt(self.conv_dim_out))])
		# BF, C, _, _ = vis_token.size()
		# vis_token = vis_token.view(BF, C, -1).unsqueeze(-1)
        
		spatial_att_maps = None
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			
			

			### -------> high dim att
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))              

			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

			att_v2tk = F.softmax(att_v2tk, dim=-1)
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))


			rep_token = rep_token + rep_token_res
			

			att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))

			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)


			x = x + self.gate_av*x_res.contiguous()
            
			# ============================== Channel Attention ====================================    
			audio = vis_token.mean(dim=2).squeeze(-1) # [B*10, dim]
			audio_query_1 = F.relu(self.fc_affine_audio_1(audio)).unsqueeze(-2)  
			video_query_1 = F.relu(self.fc_affine_video_1(x.squeeze(-1).permute(0, 2, 1))) # [*, grid ** 2, width]       
			audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2) #  [*, width] 
			audio_video_query = F.relu(self.fc_affine_bottleneck(audio_video_query_raw))
			channel_att_maps = self.fc_affine_v_c_att(audio_video_query).sigmoid().reshape(x.size(0), 1, -1)      
			c_att_visual_feat = (x.squeeze(-1).permute(0, 2, 1) * (channel_att_maps + 1)) # [B*10, 36, 768]  

			# ============================== Spatial Attention =====================================
			# channel attended visual feature: [batch * 10, 36, v_dim]
			c_att_visual_query = F.relu(self.fc_affine_video_2(c_att_visual_feat))
			audio_query_2 = F.relu(self.fc_affine_audio_2(audio)).unsqueeze(-2)
			audio_video_query_2 = c_att_visual_query * audio_query_2
			spatial_att_maps_tmp = self.fc_affine_v_s_att(audio_video_query_2) 
			spatial_att_maps_sigmoid = spatial_att_maps_tmp.transpose(2, 1).sigmoid()
			spatial_att_maps_sigmoid = spatial_att_maps_sigmoid.transpose(2, 1)
			spatial_att_maps = self.fc_softmax(self.fc_tanh(spatial_att_maps_tmp).transpose(2, 1))
			c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat)
            
			alpha, beta = 0.3, 0.05            
			x = x.squeeze(-1).permute(0, 2, 1) * (alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
			x = x.permute(0, 2, 1).unsqueeze(-1)
            
			# spatial_att_maps = spatial_att_maps.squeeze(1) * rearrange(temporal_att_maps, 'b t c d-> (b t) (c d)')
			# spatial_att_maps = spatial_att_maps.unsqueeze(-2)
            
			# 	gamma = 0.3
			# 	x = x.squeeze(-1).permute(0, 2, 1)
			# 	x = x.view(x.size(0) // 10, 10, x.size(-2), x.size(-1)) # [B, 10, HxW, C]
			# 	x = x + x * temporal_att_maps * gamma
			# 	x = rearrange(x, 'b t h c -> (b t) h c')
			# 	x = x.permute(0, 2, 1).unsqueeze(-1)
			### <----------
            
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)
			

			if self.use_bn:
				# z = self.bn1(rearrange(z, 'N C L -> N L C') )
				# z = rearrange(z, 'N L C -> N C L')

				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				# output = self.bn2(rearrange(output, 'N C L -> N L C') ) 
				# output = rearrange(output, 'N L C -> N C L')
				output = self.bn2(output)
	
		elif self.adapter_kind == "bottleneck":

			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)
			# z = self.activation(z)
			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)
			

		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')


		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

		if self.gate is not None:
			output = self.gate * output
	

		return output, spatial_att_maps


class MMIL_Net(nn.Module):

	def __init__(self, opt):
		super(MMIL_Net, self).__init__()

		# self.AST = ASTModel(label_dim=512, fstride=10, tstride=10, input_fdim=128,
		# 						  input_tdim=102, imagenet_pretrain=True,
		# 						  audioset_pretrain=True, model_size='base384') # input_tdim=1024 for 10s 



		self.opt = opt
		self.CMBS = CMBS(self.opt)
		self.temporal_attn = TemporalAttention()


		self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
		# self.swin = timm.create_model('swinv2_base_window12_192_22k', pretrained=True)
        
		if opt.backbone_type == "esc-50":
			esc_config.dataset_path = "your processed ESC-50 folder"
			esc_config.dataset_type = "esc-50"
			esc_config.loss_type = "clip_ce"
			esc_config.sample_rate = 32000
			esc_config.hop_size = 320 
			esc_config.classes_num = 50
			esc_config.checkpoint_path = "../checkpoints/ESC-50/"
			esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
		elif opt.backbone_type == "audioset":
			esc_config.dataset_path = "your processed audioset folder"
			esc_config.dataset_type = "audioset"
			esc_config.balanced_data = True
			esc_config.loss_type = "clip_bce"
			esc_config.sample_rate = 32000
			esc_config.hop_size = 320 
			esc_config.classes_num = 527
			esc_config.checkpoint_path = "../checkpoints/AudioSet/"
			esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
		elif opt.backbone_type == "scv2":
			esc_config.dataset_path = "your processed SCV2 folder"
			esc_config.dataset_type = "scv2"
			esc_config.loss_type = "clip_bce"
			esc_config.sample_rate = 16000
			esc_config.hop_size = 160
			esc_config.classes_num = 35
			esc_config.checkpoint_path = "../checkpoints/SCV2/"
			esc_config.checkpoint = "HTSAT_SCV2_Saved_3.ckpt"
		else:
			raise NotImplementedError
    
		self.htsat = HTSAT_Swin_Transformer(
			spec_size=esc_config.htsat_spec_size,
			patch_size=esc_config.htsat_patch_size,
			in_chans=1,
			num_classes=esc_config.classes_num,
			window_size=esc_config.htsat_window_size,
			config = esc_config,
			depths = esc_config.htsat_depth,
			embed_dim = esc_config.htsat_dim,
			patch_stride=esc_config.htsat_stride,
			num_heads=esc_config.htsat_num_head
		)
        
		checkpoint_path = os.path.join(esc_config.checkpoint_path, esc_config.checkpoint)
		tmp = torch.load(checkpoint_path, map_location='cpu')
		tmp = {k[10:]:v for k, v in tmp['state_dict'].items()}
		self.htsat.load_state_dict(tmp, strict=True)



		
		# self.mlp_class = nn.Linear(hidden_d_size*2, 512)
		# self.mlp_class_2 = nn.Linear(512, 29)
        
		# self.mlp_class_audio_visual_contrastive_learner = AudioVisualContrastive()
		# self.mlp_class_fc = nn.Linear(768, 1536)
		# self.mlp_class = nn.Linear(1536+768, 512) # swinv2-Large
		# self.mlp_class_2 = nn.Linear(512, 29)
		# self.audio_visual_adapter = AudioVisualAdapter()
		# self.mlp_map_vggish = nn.Linear(128, 1536) 

		# self.mlp_class = nn.Linear(1024*2, 512) # swinv2-Base and vit-large
		# self.mlp_class_2 = nn.Linear(512, 29)


		# self.mlp_class = nn.Linear(768*2, 512) # ViT-Base
		# self.mlp_class_2 = nn.Linear(512, 29)

		# self.mlp_class = nn.Linear(384*2, 512) # ViT-small
		# self.mlp_class_2 = nn.Linear(512, 29)
		# self.mlp_class = nn.Linear(192*2, 64) # ViT-tiny
		# self.mlp_class_2 = nn.Linear(64, 29)
		

		# self.mlp_class_audio_map = nn.Linear(128, 768)


		
		hidden_list, hidden_list_a = [], []
		down_in_dim, down_in_dim_a = [], []
		down_out_dim, down_out_dim_a = [], []
		conv_dim, conv_dim_a = [], []
        
		## ------------> for swin and htsat 
		for idx_layer, (my_blk, my_blk_a) in enumerate(zip(self.swin.layers, self.htsat.layers)):
			conv_dim_tmp = (my_blk.input_resolution[0]*my_blk.input_resolution[1])
			conv_dim_tmp_a = (my_blk_a.input_resolution[0]*my_blk_a.input_resolution[1])
			if not isinstance(my_blk.downsample, nn.Identity):
				down_in_dim.append(my_blk.downsample.reduction.in_features)
				down_out_dim.append(my_blk.downsample.reduction.out_features)
			if my_blk_a.downsample is not None:
				down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
				down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)
            
			for blk, blk_a in zip(my_blk.blocks, my_blk_a.blocks):
				hidden_d_size = blk.norm1.normalized_shape[0]
				hidden_list.append(hidden_d_size)
				conv_dim.append(conv_dim_tmp)
				hidden_d_size_a = blk_a.norm1.normalized_shape[0]
				hidden_list_a.append(hidden_d_size_a)
				conv_dim_a.append(conv_dim_tmp_a)

                
		self.adapter_token_downsampler = nn.ModuleList([
				nn.Linear(down_out_dim[i]//(self.opt.Adapter_downsample*2), down_out_dim[i]//self.opt.Adapter_downsample, bias=False)
				for i in range(len(down_in_dim))])
		self.adapter_token_downsampler.append(nn.Identity())
		## <--------------


		if self.opt.is_audio_adapter_p1:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], 
				adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i,
				reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate,
				num_tk=opt.num_tokens, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
				linear_in=hidden_list[i], linear_out=hidden_list_a[i]       
				)
				for i in range(len(hidden_list_a))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], 
				output_dim=hidden_list[i], adapter_kind="bottleneck", 
				dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=True,
				num_tk=opt.num_tokens, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
				linear_in=hidden_list_a[i], linear_out=hidden_list[i]   
				)
				for i in range(len(hidden_list))])

		if self.opt.is_audio_adapter_p2:
			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck", 
				dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate,
				num_tk=opt.num_tokens, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i],
				linear_in=hidden_list[i], linear_out=hidden_list_a[i]
				)
				for i in range(len(hidden_list_a))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", 
				dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, 
				opt=opt, use_bn=self.opt.is_bn, use_gate=True,
				num_tk=opt.num_tokens, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i],
				linear_in=hidden_list_a[i], linear_out=hidden_list[i]   
				)
				for i in range(len(hidden_list))])


	def forward_swin(self, audio, vis, mixup_lambda, rand_train_idx=12, stage='eval'):
            
		audio = audio[0]
		audio = audio.view(audio.size(0)*audio.size(1), -1)
		waveform = audio
		bs = vis.size(0)
		vis = rearrange(vis, 'b t c w h -> (b t) c w h')
		f_v = self.swin.patch_embed(vis)
            
		audio = self.htsat.spectrogram_extractor(audio)
		audio = self.htsat.logmel_extractor(audio)        
		audio = audio.transpose(1, 3)
		audio = self.htsat.bn0(audio)
		audio = audio.transpose(1, 3)
		if self.htsat.training:
			audio = self.htsat.spec_augmenter(audio)
		if self.htsat.training and mixup_lambda is not None:
			audio = do_mixup(audio, mixup_lambda)

		if audio.shape[2] > self.htsat.freq_ratio * self.htsat.spec_size:
			audio = self.htsat.crop_wav(audio, crop_size=self.htsat.freq_ratio * self.htsat.spec_size)
			audio = self.htsat.reshape_wav2img(audio)
		else: # this part is typically used, and most easy one
			audio = self.htsat.reshape_wav2img(audio)
		frames_num = audio.shape[2]
		f_a = self.htsat.patch_embed(audio)
		if self.htsat.ape:
			f_a = f_a + self.htsat.absolute_pos_embed
		f_a = self.htsat.pos_drop(f_a)   
        
		idx_layer = 0
		out_idx_layer = 0
		for _, (my_blk, htsat_blk) in enumerate(zip(self.swin.layers, self.htsat.layers)) :

			if len(my_blk.blocks) == len(htsat_blk.blocks):
				aud_blocks = htsat_blk.blocks
			else:
				aud_blocks = [None, None, htsat_blk.blocks[0], None, None, htsat_blk.blocks[1], None, None, htsat_blk.blocks[2], None, None, htsat_blk.blocks[3], None, None, htsat_blk.blocks[4], None, None, htsat_blk.blocks[5]]
				assert len(aud_blocks) == len(my_blk.blocks)
                
			for (blk, blk_a) in zip(my_blk.blocks, aud_blocks):
				if blk_a is not None:
                        
					f_a_res, f_a_spatial_att_maps = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
					f_v_res, f_v_spatial_att_maps = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

					f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
					f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

					f_a, _ = blk_a(f_a)
					f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
				
					f_a_res, f_a_spatial_att_maps = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
					f_v_res, f_v_spatial_att_maps = self.vis_adapter_blocks_p2[idx_layer]( f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

					f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))
					f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

					f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)
                    
					idx_layer = idx_layer +1
                    
				else:
					f_v = f_v + blk.drop_path1(blk.norm1(blk._attn(f_v)))
					f_v = f_v + blk.drop_path2(blk.norm2(blk.mlp(f_v)))

			f_v = my_blk.downsample(f_v)
			if htsat_blk.downsample is not None:
				f_a = htsat_blk.downsample(f_a)


		f_v = self.swin.norm(f_v)

		# f_v = f_v.mean(dim=1, keepdim=True)
		f_v = torch.bmm(f_v_spatial_att_maps, f_v)
		# f_a = f_a.mean(dim=1, keepdim=True)
		f_a = torch.bmm(f_a_spatial_att_maps, f_a)
		# f_v, f_a = self.audio_visual_adapter(f_v, f_a)    
        
		########## Temporal Attention ##########        
		f_v = f_v.view(bs, 10, -1)
		f_a = f_a.view(bs, 10, -1)
		video_feature, audio_feature, audio_visual_gate = self.temporal_attn(f_v, f_a)
        
		is_event_scores, event_scores, av_score = self.CMBS(video_feature, audio_feature)
        
		return is_event_scores, event_scores, audio_visual_gate, av_score

	def forward(self, audio, vis, mixup_lambda=None, rand_train_idx=12, stage='eval'):
		return self.forward_swin(audio, vis, mixup_lambda, rand_train_idx=12, stage='eval')
