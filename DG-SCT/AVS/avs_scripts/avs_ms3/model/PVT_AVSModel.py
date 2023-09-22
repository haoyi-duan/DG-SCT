import torch
import torch.nn as nn
import torchvision.models as models
from model.pvt import pvt_v2_b5
from model.TPAVI import TPAVIModule
from ipdb import set_trace
import timm
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
import numpy as np
import copy
import math
from ipdb import set_trace
import os
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder

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

### VGGSound
from .htsat import HTSAT_Swin_Transformer
import model.esc_config as esc_config
from .utils import do_mixup, get_mix_lambda, do_mixup_label

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
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature
    
class VisualAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = opt.is_multimodal
        self.opt = opt
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

            self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))
            self.gate_av = nn.Parameter(torch.zeros(1))

            ### <------

            self.activation = nn.ReLU(inplace=True)

            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

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


            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)


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

            # ============================== Temporal Attention =====================================
            # audio = audio.view(audio.size(0) // 5, 5, -1)
            # temporal_att_maps = self.temporal_gated(audio).unsqueeze(-1)

#             alpha, beta = 0.3, 0.05            
#             x = x.squeeze(-1).permute(0, 2, 1) * (alpha * channel_att_maps + 1 - alpha)
#             x = x.permute(0, 2, 1).unsqueeze(-1)
            # adjust
            #best score = 52.5  alpha, beta = 0.4, 0.05
            #best score = 52.5  alpha, beta = 0.4, 0.1
            #best score = 52.6  alpha, beta = 0.2, 0.1
            alpha, beta = 0.2, 0.1            
            x = x.squeeze(-1).permute(0, 2, 1) * (alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
            x = x.permute(0, 2, 1).unsqueeze(-1)
            
            # spatial_att_maps = spatial_att_maps.squeeze(1) * rearrange(temporal_att_maps, 'b t c d-> (b t) (c d)')
            # spatial_att_maps = spatial_att_maps.unsqueeze(-2)
            
            # gamma = 0.3
            # x = x.squeeze(-1).permute(0, 2, 1)
            # x = x.view(x.size(0) // 5, 5, x.size(-2), x.size(-1)) # [B, 10, HxW, C]
            # x = x + x * temporal_att_maps * gamma
            # x = rearrange(x, 'b t h c -> (b t) h c')
            # x = x.permute(0, 2, 1).unsqueeze(-1)
            
            ### <----------

            z = self.down_sampler(x)

            ## <----

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)
                
        elif self.adapter_kind == "bottleneck":

            if self.opt.is_before_layernorm:
                x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)
            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)

        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C') )
                output = rearrange(output, 'N L C -> N C L')

        if self.gate is not None:
            output = self.gate * output


        if self.opt.is_post_layernorm:
            output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)


        return output, spatial_att_maps
    
    
class Classifier_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NoLabels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class TemporalAttention(nn.Module):
    def __init__(self, opt):
        super(TemporalAttention, self).__init__()
        #adjust
        #default alpha =0.05
        #best = 53.46 alpha =0.1
        self.gamma = opt.gamma
        self.video_input_dim = 256
        self.audio_input_dim = 128

        self.video_fc_dim = 256
        self.audio_fc_dim = 128
        self.d_model = 256

        self.v_fc = nn.ModuleList([nn.Linear(self.video_input_dim, self.video_fc_dim) for i in range(4)])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.video_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
        self.video_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=512, d_model=self.d_model, feedforward_dim=1024) for i in range(4)]) 
        self.audio_encoder = nn.ModuleList([InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
        self.audio_decoder = nn.ModuleList([CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024) for i in range(4)])
        self.audio_visual_rnn_layer = nn.ModuleList([RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1) for i in range(4)])

        self.audio_gated = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                        ) for i in range(4)])
        self.video_gated = nn.ModuleList([nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                        ) for i in range(4)])
        
    def forward(self, visual_feature_list, audio_feature):
        # shape for pvt-v2-b5
        # BF x 256 x 56 x 56
        # BF x 256 x 28 x 28
        # BF x 256 x 14 x 14
        # BF x 256 x  7 x  7

        bs = audio_feature.size(0)
        x1, x2, x3, x4 = visual_feature_list
        x1_ = self.avgpool(x1)
        x1_ = x1_.squeeze()
        x2_ = self.avgpool(x2)
        x2_ = x2_.squeeze()
        x3_ = self.avgpool(x3)
        x3_ = x3_.squeeze()
        x4_ = self.avgpool(x4)
        x4_ = x4_.squeeze()
        
        x1_ = x1_.view(bs, 5, -1)
        x2_ = x2_.view(bs, 5, -1)
        x3_ = x3_.view(bs, 5, -1)
        x4_ = x4_.view(bs, 5, -1)

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.view(-1, audio_feature.size(-1))
        x1_, x2_, x3_, x4_ = [self.v_fc[i](x) for i, x in enumerate([x1_, x2_, x3_, x4_])]
        x1_, x2_, x3_, x4_ = [self.dropout(self.relu(x)) for x in [x1_, x2_, x3_, x4_]]
        
        visual_rnn_input = [x1_, x2_, x3_, x4_]

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer[0](audio_rnn_input, visual_rnn_input[0])
        audio_rnn_output2, visual_rnn_output2 = self.audio_visual_rnn_layer[1](audio_rnn_input, visual_rnn_input[1])
        audio_rnn_output3, visual_rnn_output3 = self.audio_visual_rnn_layer[2](audio_rnn_input, visual_rnn_input[2])
        audio_rnn_output4, visual_rnn_output4 = self.audio_visual_rnn_layer[3](audio_rnn_input, visual_rnn_input[3])
        
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 256]
        audio_encoder_input2 = audio_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 256]
        audio_encoder_input3 = audio_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 256]
        audio_encoder_input4 = audio_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 256]
        
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [5, B, 512]
        visual_encoder_input2 = visual_rnn_output2.transpose(1, 0).contiguous()  # [5, B, 512]
        visual_encoder_input3 = visual_rnn_output3.transpose(1, 0).contiguous()  # [5, B, 512]
        visual_encoder_input4 = visual_rnn_output4.transpose(1, 0).contiguous()  # [5, B, 512]

        # audio query
        video_key_value_feature1 = self.video_encoder[0](visual_encoder_input1)
        video_key_value_feature2 = self.video_encoder[1](visual_encoder_input2)
        video_key_value_feature3 = self.video_encoder[2](visual_encoder_input3)
        video_key_value_feature4 = self.video_encoder[3](visual_encoder_input4)
        
        audio_query_output1 = self.audio_decoder[0](audio_encoder_input1, video_key_value_feature1)
        audio_query_output2 = self.audio_decoder[1](audio_encoder_input2, video_key_value_feature2)
        audio_query_output3 = self.audio_decoder[2](audio_encoder_input3, video_key_value_feature3)
        audio_query_output4 = self.audio_decoder[3](audio_encoder_input4, video_key_value_feature4)
        
        # video query
        audio_key_value_feature1 = self.audio_encoder[0](audio_encoder_input1)
        audio_key_value_feature2 = self.audio_encoder[1](audio_encoder_input2)
        audio_key_value_feature3 = self.audio_encoder[2](audio_encoder_input3)
        audio_key_value_feature4 = self.audio_encoder[3](audio_encoder_input4)
        
        video_query_output1 = self.video_decoder[0](visual_encoder_input1, audio_key_value_feature1)
        video_query_output2 = self.video_decoder[1](visual_encoder_input2, audio_key_value_feature2)
        video_query_output3 = self.video_decoder[2](visual_encoder_input3, audio_key_value_feature3)
        video_query_output4 = self.video_decoder[3](visual_encoder_input4, audio_key_value_feature4)

        audio_gate1 = self.audio_gated[0](audio_key_value_feature1) # [5, B, 1]
        audio_gate2 = self.audio_gated[1](audio_key_value_feature2)
        audio_gate3 = self.audio_gated[2](audio_key_value_feature3)
        audio_gate4 = self.audio_gated[3](audio_key_value_feature4)
        
        video_gate1 = self.video_gated[0](video_key_value_feature1) # [5, B, 1]
        video_gate2 = self.video_gated[1](video_key_value_feature2)
        video_gate3 = self.video_gated[2](video_key_value_feature3)
        video_gate4 = self.video_gated[3](video_key_value_feature4)

        audio_gate1 = audio_gate1.transpose(1, 0)
        audio_gate1 = audio_gate1.reshape(bs*5, 1, 1, 1)
        audio_gate2 = audio_gate2.transpose(1, 0)
        audio_gate2 = audio_gate2.reshape(bs*5, 1, 1, 1)
        audio_gate3 = audio_gate3.transpose(1, 0)
        audio_gate3 = audio_gate3.reshape(bs*5, 1, 1, 1)
        audio_gate4 = audio_gate4.transpose(1, 0)
        audio_gate4 = audio_gate4.reshape(bs*5, 1, 1, 1)

        video_gate1 = video_gate1.transpose(1, 0)
        video_gate1 = video_gate1.reshape(bs*5, 1)
        video_gate2 = video_gate2.transpose(1, 0)
        video_gate2 = video_gate2.reshape(bs*5, 1)
        video_gate3 = video_gate3.transpose(1, 0)
        video_gate3 = video_gate3.reshape(bs*5, 1)
        video_gate4 = video_gate4.transpose(1, 0)
        video_gate4 = video_gate4.reshape(bs*5, 1)
        
        x1 = x1 + audio_gate1 * x1 * self.gamma
        x2 = x2 + audio_gate2 * x2 * self.gamma
        x3 = x3 + audio_gate3 * x3 * self.gamma
        x4 = x4 + audio_gate4 * x4 * self.gamma
        
        video_gate = (video_gate1 + video_gate2 + video_gate3 + video_gate4) / 4
        audio_feature = audio_feature + video_gate * audio_feature * self.gamma
        
        return [x1, x2, x3, x4], audio_feature
    
class Pred_endecoder(nn.Module):
    # pvt-v2 based encoder decoder
    def __init__(self, channel=256, opt=None, config=None, vis_dim=[64, 128, 320, 512], tpavi_stages=[], tpavi_vv_flag=False, tpavi_va_flag=True):
        super(Pred_endecoder, self).__init__()
        self.cfg = config
        self.tpavi_stages = tpavi_stages
        self.tpavi_vv_flag = tpavi_vv_flag
        self.tpavi_va_flag = tpavi_va_flag
        self.vis_dim = vis_dim
        
        self.opt = opt
        
        self.encoder_backbone = pvt_v2_b5()
        self.relu = nn.ReLU(inplace=True)
        self.temporal_attn = TemporalAttention(opt)
        
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[3])
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[2])
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[1])
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, self.vis_dim[0])

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)
        
        self.x1_linear = nn.Linear(192,64)
        self.x2_linear = nn.Linear(384,128)
        self.x3_linear = nn.Linear(768,320)
        self.x4_linear = nn.Linear(1536,512)
        
        self.x1_linear_ = nn.Linear(192,256)
        self.x2_linear_ = nn.Linear(384,256)
        self.x3_linear_ = nn.Linear(768,256)
        self.x4_linear_ = nn.Linear(1536,256)
        
        self.audio_linear = nn.Linear(768,128)
        self.swin = timm.create_model('swinv2_large_window12_192_22k', pretrained=True)
        
        if opt.backbone_type == "esc-50":
            esc_config.dataset_path = "your processed ESC-50 folder"
            esc_config.dataset_type = "esc-50"
            esc_config.loss_type = "clip_ce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 50
            esc_config.checkpoint_path = os.path.join(opt.root_path, "DG-SCT/checkpoints/ESC-50/")
            esc_config.checkpoint = "HTSAT_ESC_exp=1_fold=1_acc=0.985.ckpt"
        elif opt.backbone_type == "audioset":
            esc_config.dataset_path = "your processed audioset folder"
            esc_config.dataset_type = "audioset"
            esc_config.balanced_data = True
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 32000
            esc_config.hop_size = 320 
            esc_config.classes_num = 527
            esc_config.checkpoint_path = os.path.join(opt.root_path, "DG-SCT/checkpoints/AudioSet/")
            esc_config.checkpoint = "HTSAT_AudioSet_Saved_1.ckpt"
        elif opt.backbone_type == "scv2":
            esc_config.dataset_path = "your processed SCV2 folder"
            esc_config.dataset_type = "scv2"
            esc_config.loss_type = "clip_bce"
            esc_config.sample_rate = 16000
            esc_config.hop_size = 160
            esc_config.classes_num = 35
            esc_config.checkpoint_path = os.path.join(opt.root_path, "DG-SCT/checkpoints/SCV2/")
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


        self.audio_adapter_blocks_p1 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck",dim_list=hidden_list_a, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i], linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list))])

        self.vis_adapter_blocks_p1 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i], linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list))])

        self.audio_adapter_blocks_p2 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list_a[i], output_dim=hidden_list_a[i], adapter_kind="bottleneck", dim_list=hidden_list_a, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate, conv_dim_in=conv_dim[i], conv_dim_out=conv_dim_a[i], linear_in=hidden_list[i], linear_out=hidden_list_a[i])
            for i in range(len(hidden_list))])

        self.vis_adapter_blocks_p2 = nn.ModuleList([
            VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True, conv_dim_in=conv_dim_a[i], conv_dim_out=conv_dim[i], linear_in=hidden_list_a[i], linear_out=hidden_list[i])
            for i in range(len(hidden_list))])

        
        for i in self.tpavi_stages:
            setattr(self, f"tpavi_b{i+1}", TPAVIModule(in_channels=channel, mode='dot'))
            print("==> Build TPAVI block...")

        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        if self.training:
            self.initialize_pvt_weights()


    def pre_reshape_for_tpavi(self, x):
        # x: [B*5, C, H, W]
        _, C, H, W = x.shape
        x = x.reshape(-1, 5, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous() # [B, C, T, H, W]
        return x

    def post_reshape_for_tpavi(self, x):
        # x: [B, C, T, H, W]
        # return: [B*T, C, H, W]
        _, C, _, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4) # [B, T, C, H, W]
        x = x.view(-1, C, H, W)
        return x

    def tpavi_vv(self, x, stage):
        # x: visual, [B*T, C=256, H, W]
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, _ = tpavi_b(x) # [B, C, T, H, W]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x

    def tpavi_va(self, x, audio, stage):
        # x: visual, [B*T, C=256, H, W]
        # audio: [B*T, 128]
        # ra_flag: return audio feature list or not
        tpavi_b = getattr(self, f'tpavi_b{stage+1}')
        audio = audio.view(-1, 5, audio.shape[-1]) # [B, T, 128]
        x = self.pre_reshape_for_tpavi(x) # [B, C, T, H, W]
        x, a = tpavi_b(x, audio) # [B, C, T, H, W], [B, T, C]
        x = self.post_reshape_for_tpavi(x) # [B*T, C, H, W]
        return x, a

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, audio_feature=None, mixup_lambda=None):
        B, frame, C, H, W = x.shape
        x = x.view(B*frame, C, H, W)
        
        audio = audio_feature
        audio = audio.view(audio.size(0)*audio.size(1), -1)
        waveform = audio
        
        x = F.interpolate(x, mode='bicubic',size=[192,192])
        f_v = self.swin.patch_embed(x)
        
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
        multi_scale = []
        out_idx_layer = 0
        idx_block = 0
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
            if idx_block != 3:
                multi_scale.append(f_v)
            else:
                multi_scale.append(self.swin.norm(f_v))
            idx_block += 1                
         
            f_v = my_blk.downsample(f_v)
            if htsat_blk.downsample is not None:
                f_a = htsat_blk.downsample(f_a)


        f_v = self.swin.norm(f_v)
        
        
        # audio_feature = rearrange(f_a.mean(dim=1), '(b t) d -> b t d', t=5)
        audio_feature = rearrange(torch.bmm(f_a_spatial_att_maps, f_a).squeeze(dim=1), '(b t) d -> b t d', t=5)
        audio_feature = self.audio_linear(audio_feature)
        
        x1 = multi_scale[0].view(multi_scale[0].size(0),48,48,-1)
        x2 = multi_scale[1].view(multi_scale[1].size(0),24,24,-1)
        x3 = multi_scale[2].view(multi_scale[2].size(0),12,12,-1)
        x4 = multi_scale[3].view(multi_scale[3].size(0),6,6,-1)
        
        x1 = self.x1_linear_(x1)
        x2 = self.x2_linear_(x2)
        x3 = self.x3_linear_(x3)
        x4 = self.x4_linear_(x4)

        x1 = F.interpolate(rearrange(x1, 'BF w h c -> BF c w h'), mode='bicubic',size=[56,56])
        x2 = F.interpolate(rearrange(x2, 'BF w h c -> BF c w h'), mode='bicubic',size=[28,28])
        x3 = F.interpolate(rearrange(x3, 'BF w h c -> BF c w h'), mode='bicubic',size=[14,14])
        x4 = F.interpolate(rearrange(x4, 'BF w h c -> BF c w h'), mode='bicubic',size=[7,7])
        
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        # shape for pvt-v2-b5
        # BF x  64 x 56 x 56
        # BF x 128 x 28 x 28
        # BF x 320 x 14 x 14
        # BF x 512 x  7 x  7

        # conv1_feat = self.conv1(x1)    # BF x 256 x 56 x 56
        # conv2_feat = self.conv2(x2)    # BF x 256 x 28 x 28
        # conv3_feat = self.conv3(x3)    # BF x 256 x 14 x 14
        # conv4_feat = self.conv4(x4)    # BF x 256 x  7 x  7
        
        conv1_feat = x1    # BF x 256 x 56 x 56
        conv2_feat = x2    # BF x 256 x 28 x 28
        conv3_feat = x3    # BF x 256 x 14 x 14
        conv4_feat = x4    # BF x 256 x  7 x  7
        
        # print(conv1_feat.shape, conv2_feat.shape, conv3_feat.shape, conv4_feat.shape)
        
        # feature_map_list = [conv1_feat, conv2_feat, conv3_feat, conv4_feat]
        feature_map_list, audio_feature = self.temporal_attn([conv1_feat, conv2_feat, conv3_feat, conv4_feat], audio_feature)
        a_fea_list = [None] * 4

        if len(self.tpavi_stages) > 0:
            if (not self.tpavi_vv_flag) and (not self.tpavi_va_flag):
                raise Exception('tpavi_vv_flag and tpavi_va_flag cannot be False at the same time if len(tpavi_stages)>0, \
                    tpavi_vv_flag is for video self-attention while tpavi_va_flag indicates the standard version (audio-visual attention)')
            for i in self.tpavi_stages:
                tpavi_count = 0
                conv_feat = torch.zeros_like(feature_map_list[i]).cuda()
                if self.tpavi_vv_flag:
                    conv_feat_vv = self.tpavi_vv(feature_map_list[i], stage=i)
                    conv_feat += conv_feat_vv
                    tpavi_count += 1
                if self.tpavi_va_flag:
                    conv_feat_va, a_fea = self.tpavi_va(feature_map_list[i], audio_feature, stage=i)
                    conv_feat += conv_feat_va
                    tpavi_count += 1
                    a_fea_list[i] = a_fea
                conv_feat /= tpavi_count
                feature_map_list[i] = conv_feat # update features of stage-i which conduct non-local

        conv4_feat = self.path4(feature_map_list[3])            # BF x 256 x 14 x 14
        conv43 = self.path3(conv4_feat, feature_map_list[2])    # BF x 256 x 28 x 28
        conv432 = self.path2(conv43, feature_map_list[1])       # BF x 256 x 56 x 56
        conv4321 = self.path1(conv432, feature_map_list[0])     # BF x 256 x 112 x 112

        pred = self.output_conv(conv4321)   # BF x 1 x 224 x 224
        # print(pred.shape)

        return pred, feature_map_list, a_fea_list


    def initialize_pvt_weights(self,):
        pvt_model_dict = self.encoder_backbone.state_dict()
        pretrained_state_dicts = torch.load(self.cfg.TRAIN.PRETRAINED_PVTV2_PATH)
        # for k, v in pretrained_state_dicts['model'].items():
        #     if k in pvt_model_dict.keys():
        #         print(k, v.requires_grad)
        state_dict = {k : v for k, v in pretrained_state_dicts.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(state_dict)
        self.encoder_backbone.load_state_dict(pvt_model_dict)
        print(f'==> Load pvt-v2-b5 parameters pretrained on ImageNet from {self.cfg.TRAIN.PRETRAINED_PVTV2_PATH}')
        # pdb.set_trace()


if __name__ == "__main__":
    imgs = torch.randn(10, 3, 224, 224)
    audio = torch.randn(2, 5, 128)
    # model = Pred_endecoder(channel=256)
    model = Pred_endecoder(channel=256, tpavi_stages=[0,1,2,3], tpavi_va_flag=True,)
    # output = model(imgs)
    output = model(imgs, audio)
    pdb.set_trace()