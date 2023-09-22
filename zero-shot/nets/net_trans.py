import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace

from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat

from timm.models.vision_transformer import Attention
import timm
import loralib as lora
from .my_layers import PHMLinear
from transformers.activations import get_activation
from .models import *
from .prompt_learner import *
import copy
from torch.nn import MultiheadAttention
import random
from nets.HTSAT import VideoAudioAttentionAdapter
from nets.HTSAT import HTSAT_Swin_Transformer
import esc_fig
### VGGSound
from nets import Resnet_VGGSound

from .helper import do_mixup, interpolate

from nets.ast_models import ASTModel
from nets.my_vit import VisionTransformer


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
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class + 1, kernel_size=1, stride=1, padding=0,
                      bias=False)
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


class CMBS(nn.Module):
    def __init__(self, config):
        super(CMBS, self).__init__()
        self.config = config
        self.beta = 0.4
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta)
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.video_fc_dim = 512
        self.d_model = 256

        self.v_fc = nn.Linear(1536, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim,
                                                 d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

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
        # [batch, 10, 512]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
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

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha

        video_cas = self.video_cas(video_query_output)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        # event_visual_gate = score_video.sigmoid()
        # event_audio_gate = score_audio.sigmoid()

        av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        is_event_scores, event_scores = self.localize_module((video_query_output + audio_query_output) / 2)
        event_scores = event_scores + self.gamma * av_score
        # event_scores = event_scores + self.gamma * (event_visual_gate * event_audio_gate) * event_scores

        return is_event_scores, event_scores, audio_visual_gate, av_score


class AudioVisualContrastive(nn.Module):
    def __init__(self, logit_scale):
        super().__init__()
        self.fc_a1 = nn.Linear(512, 512)
        self.logit_scale = logit_scale.exp()

    def forward(self, video, audio):
        bs = audio.size(0) // 10
        audio = self.fc_a1(audio)
        video = video.view(bs, 10, -1)
        audio = audio.view(bs, 10, -1)
        video, audio = video.mean(dim=1), audio.mean(dim=1)
        video = video / video.norm(dim=-1, keepdim=True)
        audio = audio / audio.norm(dim=-1, keepdim=True)
        logits_audio_image = self.logit_scale * audio @ video.t()  # [B, B]
        logits_image_audio = self.logit_scale * video @ audio.t()  # [B, B]

        return logits_audio_image, logits_image_audio


class AudioAdapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = 0.6
        self.d_model = 256
        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=768, d_model=256, num_layers=1)

        #self.audio_encoder = VideoAudioAttentionAdapter()

    def forward(self, x, audio):
        bs = x.size(0) // 10
        x = x.view(bs, 10, -1)  # [B, 10, 768]
        x = x.permute(1, 0, 2)  # [10, B, 768]
        audio = audio.view(bs, 10, -1)  # [B, 10, 128]
        # audio query
        audio_rnn_output1 = self.audio_visual_rnn_layer.audio_rnn(audio)[0]
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, B, 256]
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)

        audio_gate = self.audio_gated(audio_key_value_feature)

        x = x + audio_gate * x * self.alpha

        x = x.permute(1, 0, 2)  # [B, 10, 768]
        x = x.view(bs * 10, -1)

        audio_key_value_feature = audio_key_value_feature.permute(1, 0, 2).contiguous()  # [B, 10, 256]
        audio_key_value_feature = audio_key_value_feature.view(bs * 10, -1)

        return x, audio_key_value_feature


class VisualAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True, num_tk=87, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt
		self.fc_caption = nn.Linear(512, 192)
		self.num_tk = num_tk
		self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)  
		self.fc = nn.Linear(linear_in, linear_out)
        
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

		self.temporal_gated = nn.Sequential(
                        nn.Linear(linear_out, 1),
                        nn.Sigmoid()
                    )
        
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

	def forward(self, x, vis_token=None, caption=None, is_temporal=False):
		#import pdb
		#pdb.set_trace()
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
		temporal_att_maps = None
		if self.adapter_kind == "bottleneck" and self.is_multimodal:
			
			

			### -------> high dim att
			if caption == None:
				rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
			else:
				caption = self.fc_caption(caption)
				rep_token = rearrange(caption, 'b l d -> (b l) d')
				rep_token = repeat(caption, 'b d -> b t d', t = self.num_tk)               

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

			# ============================== Temporal Attention =====================================
			audio = audio.view(audio.size(0) // 10, 10, -1)
			temporal_att_maps = self.temporal_gated(audio).unsqueeze(-1)
            
			alpha, beta = 0.3, 0.01
			gamma = 0.05
			# x = x.squeeze(-1).permute(0, 2, 1) * (alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
			# x = x.permute(0, 2, 1).unsqueeze(-1)

			x = x.squeeze(-1).permute(0, 2, 1)
			bs = x.size(0) // 10
			x = x.view(bs, 10, x.size(-2), x.size(-1)) # [B, 10, HxW, C]
			channel_att_maps_tmp = channel_att_maps.view(bs, 10, channel_att_maps.size(-2), channel_att_maps.size(-1))
			spatial_att_maps_sigmoid_tmp = spatial_att_maps_sigmoid.view(bs, 10, spatial_att_maps_sigmoid.size(-2), spatial_att_maps_sigmoid.size(-1))                
			x = x * (alpha * channel_att_maps_tmp + beta * spatial_att_maps_sigmoid_tmp + gamma * temporal_att_maps + 1 - alpha)
			x = rearrange(x, 'b t h c -> (b t) h c')
			x = x.permute(0, 2, 1).unsqueeze(-1)
			# <----------
            
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
	

		return output, spatial_att_maps, temporal_att_maps


class MMIL_Net(nn.Module):

    def __init__(self, opt):
        super(MMIL_Net, self).__init__()

        self.opt = opt
        self.classnames, _ = generate_category_list(self.opt.test_dataset_name)
        print(f"Loading CLIP (backbone: {self.opt.ViT})")
        clip_model = load_clip_to_cpu(self.opt)
        clip_model.float()
        print("Building custom CLIP")
        self.prompt_learner = PromptLearner(self.opt, self.classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_adapter = ClipAdapter(512, 4)
        self.clip_adapter_text = ClipAdapter(512, 4)
        self.CMBS = CMBS(self.opt)
        self.audio_adapter = AudioAdapter()
        # 这里换成HTSAT
        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_fig.htsat_spec_size,
            patch_size=esc_fig.htsat_patch_size,
            in_chans=1,
            num_classes=esc_fig.classes_num,
            window_size=esc_fig.htsat_window_size,
            config=esc_fig,
            depths=esc_fig.htsat_depth,
            embed_dim=esc_fig.htsat_dim,
            patch_stride=esc_fig.htsat_stride,
            num_heads=esc_fig.htsat_num_head
        )
        
        self.audio_projection = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )
        
        checkpoint_path = os.path.join(esc_fig.checkpoint_path, esc_fig.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        audio_projection_list = {k[24:]:v for k, v in tmp['state_dict'].items() if 'audio_projection' in k}
        self.audio_projection.load_state_dict(audio_projection_list)
        
        text_branch_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_branch' in k}
        text_transform_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_transform' in k}
        text_projection_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_projection' in k}
        self.clap_text_encoder = CLAPTextEncoder(self.opt, self.classnames, [text_branch_list, text_transform_list, text_projection_list])
        self.logit_scale_a = tmp['state_dict']['module.logit_scale_a']
        self.logit_scale_t = tmp['state_dict']['module.logit_scale_t']
        
        print("loading HTSAT")
        temp = dict()
        useless = ["logit_scale_a", "logit_scale_t", "patch_embed.mel_conv2d.weight", "patch_embed.mel_conv2d.bias",
                   "patch_embed.fusion_model.local_att.0.weight", "patch_embed.fusion_model.local_att.0.bias",
                   "patch_embed.fusion_model.local_att.1.weight", "patch_embed.fusion_model.local_att.1.bias",
                   "patch_embed.fusion_model.local_att.1.running_mean",
                   "patch_embed.fusion_model.local_att.1.running_var",
                   "patch_embed.fusion_model.local_att.1.num_batches_tracked",
                   "patch_embed.fusion_model.local_att.3.weight", "patch_embed.fusion_model.local_att.3.bias",
                   "patch_embed.fusion_model.local_att.4.weight", "patch_embed.fusion_model.local_att.4.bias",
                   "patch_embed.fusion_model.local_att.4.running_mean",
                   "patch_embed.fusion_model.local_att.4.running_var",
                   "patch_embed.fusion_model.local_att.4.num_batches_tracked",
                   "patch_embed.fusion_model.global_att.1.weight", "patch_embed.fusion_model.global_att.1.bias",
                   "patch_embed.fusion_model.global_att.2.weight", "patch_embed.fusion_model.global_att.2.bias",
                   "patch_embed.fusion_model.global_att.2.running_mean",
                   "patch_embed.fusion_model.global_att.2.running_var",
                   "patch_embed.fusion_model.global_att.2.num_batches_tracked",
                   "patch_embed.fusion_model.global_att.4.weight", "patch_embed.fusion_model.global_att.4.bias",
                   "patch_embed.fusion_model.global_att.5.weight", "patch_embed.fusion_model.global_att.5.bias",
                   "patch_embed.fusion_model.global_att.5.running_mean",
                   "patch_embed.fusion_model.global_att.5.running_var",
                   "patch_embed.fusion_model.global_att.5.num_batches_tracked", "logit_scale_a", "logit_scale_t"]
        for k, v in tmp['state_dict'].items():
            p = k.find(".", 7)
            if p != -1:
                if k[p + 1:] in useless:
                    continue
                temp[k[p + 1:]] = v
                # print(k[p+1:])
                if k[p + 1:] == "head.bias":
                    break
            else:
                p = k.find(".")
                if k[p + 1:] in useless:
                    continue
                temp[k[p + 1:]] = v
                # print(k[p+1:])
        # tmp = {k[10:]: v for k, v in tmp['state_dict'].items()}
        self.htsat.load_state_dict(temp, strict=True)

        self.audio_visual_contrastive_learner = AudioVisualContrastive(self.logit_scale)

        self.ViT = VisionTransformer(clip_model)

        
        hidden_list, hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []

        ## ------------> for swin and htsat
        for idx_layer, my_blk_a in enumerate(self.htsat.layers):
            conv_dim_tmp_a = (my_blk_a.input_resolution[0] * my_blk_a.input_resolution[1])
            if my_blk_a.downsample is not None:
                down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
                down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)

            for idx_layer, blk_a in enumerate(my_blk_a.blocks):
                hidden_d_size_a = blk_a.norm1.normalized_shape[0]
                hidden_list_a.append(hidden_d_size_a)
                conv_dim_a.append(conv_dim_tmp_a)
        

        ### ----------> for vit
        for idx_layer, my_blk in enumerate(self.ViT.transformer.resblocks) :
            hidden_d_size = my_blk.mlp.c_proj.in_features
            hidden_list.append(hidden_d_size)
            conv_dim_tmp = self.ViT.input_resolution
            conv_dim.append(conv_dim_tmp)
        # # ### <---------
        hidden_list = [768]*len(hidden_list)
        conv_dim = [50]*len(conv_dim)
        
        
        
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
        
    def forward_vit(self, wave, vis):
        b, t, c, w, h = vis.shape
        output_dict=self.lavish_forward(rearrange(vis, 'b t c w h -> (b t) c w h'), wave)  # [B*10, 512]
        v_cls=output_dict['x']
        a_cls=output_dict['embedding']
        loss_audio_image=output_dict['logits_audio_image']
        loss_image_audio=output_dict['logits_image_audio']
        
        logits_v = self.clip_matching(v_cls)
        logits_a = self.clap_matching(a_cls)
        
        # 视频 和 音频 的 匹配分数
        w1 = logits_v / (logits_v + logits_a)
        w2 = logits_a / (logits_v + logits_a)
        event_scores = w1 * logits_v + w2 * logits_a
        #########################
        
        return event_scores, loss_audio_image, loss_image_audio  # [B*10, 142]

    def forward(self, wave, vis, labels_evn=None):
        return self.forward_vit(wave, vis)

    def lavish_forward(self, vis, wave, mixup_lambda=None, longer_idx=None):
        # x is the vis
        x = self.ViT.conv1(vis)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2 ,width]
        x = torch.cat([self.ViT.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, witdh]
        x = x + self.ViT.positional_embedding.to(x.dtype)
        x = self.ViT.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        # y is the wave
        
        y = wave.to(device=wave.device, non_blocking=True)
        
        y = y.view(y.size(0) * y.size(1), -1)
        y = self.htsat.spectrogram_extractor(y)  # (batch_size, 1, time_steps, freq_bins)
        y = self.htsat.logmel_extractor(y)  # (batch_size, 1, time_steps, mel_bins)
        y = y.transpose(1, 3)
        y = self.htsat.bn0(y)
        y = y.transpose(1, 3)
        if self.htsat.training:
            y = self.htsat.spec_augmenter(y)

        if self.htsat.training and mixup_lambda is not None:
            y = do_mixup(y, mixup_lambda)

        y = self.htsat.reshape_wav2img(y)

        # handle x and y
        frames_num = y.shape[2]
        y = self.htsat.patch_embed(y, longer_idx=longer_idx)
        if self.htsat.ape:
            y = y + self.htsat.absolute_pos_embed(y)
        y = self.htsat.pos_drop(y)
        cnt = 0
        for idx_blk, blk in enumerate(self.htsat.layers):
            for idx_layer, layer in enumerate(blk.blocks):
                # compute audio
                attns = []
                if blk.use_checkpoint:
                    y = checkpoint.checkpoint(blk, y)
                else:
                    y, attn = layer(y)
                    if not layer.training:
                        attns.append(attn.unsqueeze(0))

                # compute vis
                x = x + self.ViT.transformer.resblocks[cnt].attention(self.ViT.transformer.resblocks[cnt].ln_1(x))
                
                # use LAVISH
                f_a = y
                f_v = x.permute(1, 0, 2)
                
                # self.audio_adapter_blocks_p1[idx_layer].is_multimodal = False
                
                f_a_res, spatial_att_maps, temporal_att_maps = self.audio_adapter_blocks_p1[cnt](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                  f_v.permute(0, 2, 1).unsqueeze(-1))
                f_v_res, spatial_att_maps, temporal_att_maps = self.vis_adapter_blocks_p1[cnt](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                f_a.permute(0, 2, 1).unsqueeze(-1))
                
                #because HTSAT and viT do not have the drop_path1 layers, so this part will not be used temporarily
                #f_v = f_v + self.swin.layers.drop_path1(self.swin.layers.ls1(self.swin.layers.attn(self.swin.layers.norm1(f_v))))
                f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                #f_a = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_a))))
                f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                # self.audio_adapter_blocks_p2[idx_layer].is_multimodal = False
                x = f_v.permute(1, 0, 2)
                x = x + self.ViT.transformer.resblocks[cnt].mlp(self.ViT.transformer.resblocks[cnt].ln_2(x))
                
                f_a_res, spatial_att_maps, temporal_att_maps = self.audio_adapter_blocks_p2[cnt](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                  f_v.permute(0, 2, 1).unsqueeze(-1))
                f_v_res, spatial_att_maps, temporal_att_maps = self.vis_adapter_blocks_p2[cnt](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                f_a.permute(0, 2, 1).unsqueeze(-1))
                
                f_v = x.permute(1, 0, 2)
                #f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
                f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                #f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))
                f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)
                cnt += 1
                
                #import pdb
                #pdb.set_trace()
                x = f_v.permute(1, 0, 2)
                y = f_a
            # compute audio
            if blk.downsample is not None:
                y = blk.downsample(y)
            if not blk.training:
                attn = torch.cat(attns, dim=0)
                attn = torch.mean(attn, dim=0)

        # handle audio
        y = self.htsat.norm(y)
        B, N, C = y.shape
        SF = frames_num // (2 ** (len(self.htsat.depths) - 1)) // self.htsat.patch_stride[0]
        ST = frames_num // (2 ** (len(self.htsat.depths) - 1)) // self.htsat.patch_stride[1]
        y = y.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, F, T = y.shape
        # group 2D CNN
        c_freq_bin = F // self.htsat.freq_ratio
        y = y.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        y = y.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        # get latent_output
        fine_grained_latent_output = torch.mean(y, dim=2)
        fine_grained_latent_output = interpolate(fine_grained_latent_output.permute(0, 2, 1).contiguous(),
                                                 8 * self.htsat.patch_stride[1])

        latent_output = self.htsat.avgpool(torch.flatten(y, 2))
        latent_output = torch.flatten(latent_output, 1)

        # display the attention map, if needed

        y = self.htsat.tscam_conv(y)
        y = torch.flatten(y, 2)  # B, C, T

        fpx = interpolate(torch.sigmoid(y).permute(0, 2, 1).contiguous(), 8 * self.htsat.patch_stride[1])

        y = self.htsat.avgpool(y)
        y = torch.flatten(y, 1)

        output_dict = {
            'framewise_output': fpx,  # already sigmoided
            'clipwise_output': torch.sigmoid(y),
            'fine_grained_embedding': fine_grained_latent_output,
            'embedding': latent_output
        }
        #

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ViT.ln_post(x[:, 0, :])
        
        if self.ViT.proj is not None:
            x = x @ self.ViT.proj
        latent_output=self.audio_projection(latent_output)
        
        logits_audio_image, logits_image_audio = self.audio_visual_contrastive_learner(x, latent_output)
        output_dict = {
            'framewise_output': fpx,  # already sigmoided
            'clipwise_output': torch.sigmoid(y),
            'fine_grained_embedding': fine_grained_latent_output,
            'embedding': latent_output,
            'x':x,
            'logits_audio_image':logits_audio_image,
            'logits_image_audio':logits_image_audio
        }
        return output_dict
