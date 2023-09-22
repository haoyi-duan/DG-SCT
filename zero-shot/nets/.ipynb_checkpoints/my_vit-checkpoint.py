import timm
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from ipdb import set_trace
import torch.nn.functional as F

from timm.models.layers import to_2tuple,trunc_normal_
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, clip_model, patch_size = 32, width = 768, output_dim = 512):
        super().__init__()
        
        scale = width ** -0.5
        layers = clip_model.visual.transformer.layers
        heads = width // 64
        
        self.input_resolution = clip_model.visual.input_resolution
        self.output_dim = clip_model.visual.output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.conv1.load_state_dict(clip_model.visual.conv1.state_dict(), strict=True)
        
        # self.class_embedding = nn.Parameter(clip_model.visual.class_embedding)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # self.positional_embedding = nn.Parameter(clip_model.visual.positional_embedding)
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # self.ln_pre.load_state_dict(clip_model.visual.ln_pre.state_dict(), strict=True)

        self.transformer = Transformer(width, layers, heads)
        # self.transformer.load_state_dict(clip_model.visual.transformer.state_dict(), strict=True)

        self.ln_post = LayerNorm(width)
        # self.ln_post.load_state_dict(clip_model.visual.ln_post.state_dict(), strict=True)
        # self.proj = nn.Parameter(clip_model.visual.proj)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    

class my_PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=1024, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x,is_shape_info=False):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        patch_info_4d = x.shape
        x = x.flatten(2).transpose(1, 2) # 768x12x101
        # x = self.norm(x)
        if is_shape_info:
            return x, patch_info_4d
        else:
            return x


class my_vit(nn.Module):
    """
    """
    def __init__(self, name=''):
        
        super(my_vit, self).__init__()

        # override timm input shape restriction (v0.4.5)
        # timm.models.vision_transformer.PatchEmbed = PatchEmbed #(img_size=224, patch_size=32, in_chans=3, embed_dim=1024)
        # timm.models.layers.patch_embed.PatchEmbed = PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=1024)
        # timm.models.layers.patch_embed.PatchEmbed = my_PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=1024)

        # timm.models.vision_transformer.patch_embed
        # timm.models.vision_transformer.VisionTransformer.patch_embed = my_PatchEmbed

        

        self.v = timm.create_model(name, pretrained=True)

        # self.ViT.v.patch_embed.proj.weight



        ### -------> yb: custom forward (v0.6.7)
        my_conv = my_PatchEmbed(img_size=self.v.patch_embed.img_size[0], patch_size=self.v.patch_embed.patch_size[0], in_chans=3, embed_dim=self.v.embed_dim)
        my_conv.proj.load_state_dict(self.v.patch_embed.proj.state_dict(), strict=True)
        self.v.patch_embed = my_conv
        ###### <-------


        # #### --------------> yb: adapt MAE weights. should only skip "head.weight", "head.bias"
        # self.v.load_state_dict(torch.load('./ckpt/mae_pretrain_vit_large.pth')['model'], strict=False)
        # ### <------




        

    def forward_patch(self,x, is_shape_info=False):
        x, patch_info_4d = self.v.patch_embed(x, is_shape_info=is_shape_info)

        ### deit ------>
        # x = torch.cat((
        #     self.v.cls_token.expand(x.shape[0], -1, -1),
        #     self.v.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        ### <-------

        ### standard vit -------->
        x = torch.cat((
            self.v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        ### <---------


        # adapt visual to audio spec
        if self.v.pos_embed.size(1) != x.size(1):
            x = self.v.pos_drop(x + F.interpolate(self.v.pos_embed.permute(0,2,1), x.size(1), mode='linear').permute(0,2,1))
        else:
            x = self.v.pos_drop(x + self.v.pos_embed)

        if is_shape_info:
            return x, patch_info_4d
        else:
            return x

    @autocast()
    def forward_features(self, x, additional_patch=None) -> torch.Tensor:
        x = self.v.patch_embed(x)


        # ### deit ------>
        # x = torch.cat((
        #     self.v.cls_token.expand(x.shape[0], -1, -1),
        #     self.v.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # ### <-------

        
        
        
        ## standard vit -------->
        x = torch.cat((
            self.v.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        ## <---------

        # -----------> adapt visual to audio spec
        if self.v.pos_embed.size(1) != x.size(1):
            x = self.v.pos_drop(x + F.interpolate(self.v.pos_embed.permute(0,2,1), x.size(1), mode='linear').permute(0,2,1))
        else:
            x = self.v.pos_drop(x + self.v.pos_embed)
        # ######### <---------
        


        if additional_patch is not None:
            x = torch.cat((x,additional_patch), dim=1) 


        # vis tokens: 578 ; audio tokens: 110 
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.v.head(x[:, 0]), self.v.head_dist(x[:, 1])
        if self.v.distilled_training and self.v.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2
    @autocast()
    def forward(self, x):
        x = self.v.forward_features(x)
        x = self.v.forward_head(x)
        return x