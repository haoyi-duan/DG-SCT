import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from nets.grouping import ModalityTrans
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
import copy
from torch.nn import MultiheadAttention
import random
import torch.utils.checkpoint as checkpoint
from .models import EncoderLayer, DecoderLayer, Decoder
from .models import Encoder as CMBS_Encoder

from .htsat import HTSAT_Swin_Transformer
import nets.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, d_model, num_layers=num_layers, batch_first=True,
                                bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output

    
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
    
    
class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = CMBS_Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature
    
class LabelSmoothingNCELoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingNCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -torch.mean(torch.log(torch.sum(true_dist * pred, dim=self.dim)))


class TemporalAttention(nn.Module):
    def __init__(self):
        super(TemporalAttention, self).__init__()
        self.beta = 0.4
        self.video_input_dim = 128
        self.audio_input_dim = 128

        self.video_fc_dim = 128
        self.audio_fc_dim = 128
        self.d_model = 64

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )
        self.video_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )

        self.alpha = 0.05
        self.gamma = 0.05

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 1536]
        # [batch, 10, 768]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 128]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 128]


        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)


        audio_gate = self.audio_gated(audio_key_value_feature).transpose(1, 0)
        video_gate = self.video_gated(video_key_value_feature).transpose(1, 0)

        video_query_output = visual_feature + audio_gate * visual_feature * self.gamma
        audio_query_output = audio_feature + video_gate * audio_feature * self.gamma

        return video_query_output, audio_query_output
    
    
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
			nn.init.zeros_(self.down_sampler) 

			# self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

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

	def forward(self, x, vis_token=None, caption=None):
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


class MGN_Net(nn.Module):

    def __init__(self, args):
        super(MGN_Net, self).__init__()

        opt = args
        self.opt = opt
        self.fc_a =  nn.Linear(768, args.dim)
        self.fc_v = nn.Linear(1536, args.dim)
        self.fc_st = nn.Linear(512, args.dim)
        self.fc_fusion = nn.Linear(args.dim * 2, args.dim)

        # hard or soft assignment
        self.unimodal_assgin = args.unimodal_assign
        self.crossmodal_assgin = args.crossmodal_assign

        unimodal_hard_assignment = True if args.unimodal_assign == 'hard' else False
        crossmodal_hard_assignment = True if args.crossmodal_assign == 'hard' else False

        # learnable tokens
        self.audio_token = nn.Parameter(torch.zeros(25, args.dim))
        self.visual_token = nn.Parameter(torch.zeros(25, args.dim))

        # class-aware uni-modal grouping
        self.audio_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_aud,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=True
                        )

        self.visual_cug = ModalityTrans(
                            args.dim,
                            depth=args.depth_vis,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=unimodal_hard_assignment,
                            use_han=False
                        )

        # modality cross-modal grouping
        self.av_mcg = ModalityTrans(
                            args.dim,
                            depth=args.depth_av,
                            num_heads=8,
                            mlp_ratio=4.,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.,
                            attn_drop=0.,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            out_dim_grouping=args.dim,
                            num_heads_grouping=8,
                            num_group_tokens=25,
                            num_output_groups=25,
                            hard_assignment=crossmodal_hard_assignment,
                            use_han=False                        
                        )

        # prediction
        self.fc_prob = nn.Linear(args.dim, 1)
        self.fc_prob_a = nn.Linear(args.dim, 1)
        self.fc_prob_v = nn.Linear(args.dim, 1)

        self.fc_cls = nn.Linear(args.dim, 25)

        self.apply(self._init_weights)

        self.temporal_attn = TemporalAttention()
        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
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
            esc_config.checkpoint = "HTSAT_SCV2_Saved_2.ckpt"
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
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, audio, visual, visual_st, mixup_lambda=None):
        b, t, d = visual_st.size()
        
        audio = audio.view(audio.size(0)*audio.size(1), -1)
        waveform = audio
        bs = visual.size(0)
        vis = rearrange(visual, 'b t c w h -> (b t) c w h')
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
        
        # f_v = f_v.mean(dim=1, keepdim=False) # [B*10, 1536]
        f_v = torch.bmm(f_v_spatial_att_maps, f_v).squeeze(1)
        # f_a = f_a.mean(dim=1, keepdim=False) # [B*10, 768]
        f_a = torch.bmm(f_a_spatial_att_maps, f_a).squeeze(1)
        f_v = f_v.view(f_v.size(0) // 10, 10, -1)
        f_a = f_a.view(f_a.size(0) // 10, 10, -1)
        
        x1_0 = self.fc_a(f_a)

        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(f_v)
        
        ################### Temporal Attention ######################
        vid_s, x1 = self.temporal_attn(vid_s, x1_0)
        
        vid_st = self.fc_st(visual_st)
        x2_0 = torch.cat((vid_s, vid_st), dim=-1)
        x2_0 = self.fc_fusion(x2_0)

        # visual uni-modal grouping
        x2, attn_visual_dict, _ = self.visual_cug(x2_0, self.visual_token, return_attn=True)

        # audio uni-modal grouping
        x1, attn_audio_dict, _ = self.audio_cug(x1_0, self.audio_token, x2_0, return_attn=True)

        # modality-aware cross-modal grouping
        x, _, _ = self.av_mcg(x1, x2, return_attn=True)

        # prediction
        av_prob = torch.sigmoid(self.fc_prob(x))                                # [B, 25, 1]
        global_prob = av_prob.sum(dim=-1)                                       # [B, 25]

        # cls token prediction
        aud_cls_prob = self.fc_cls(self.audio_token)                            # [25, 25]
        vis_cls_prob = self.fc_cls(self.visual_token)                           # [25, 25]

        # attentions
        attn_audio = attn_audio_dict[self.unimodal_assgin].squeeze(1)                    # [25, 10]
        attn_visual = attn_visual_dict[self.unimodal_assgin].squeeze(1)                  # [25, 10]

        # audio prediction
        a_prob = torch.sigmoid(self.fc_prob_a(x1))                                # [B, 25, 1]
        a_frame_prob = (a_prob * attn_audio).permute(0, 2, 1)                     # [B, 10, 25]
        a_prob = a_prob.sum(dim=-1)                                               # [B, 25]

        # visual prediction
        v_prob = torch.sigmoid(self.fc_prob_v(x2))                                # [B, 25, 1]
        v_frame_prob = (v_prob * attn_visual).permute(0, 2, 1)                    # [B, 10, 25]
        v_prob = v_prob.sum(dim=-1)                                               # [B, 25]

        return aud_cls_prob, vis_cls_prob, global_prob, a_prob, v_prob, a_frame_prob, v_frame_prob

