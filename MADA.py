# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import bias
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
# from pyrsistent import v
import torch
import torch.nn as nn
from .DCNv2.dcn import DeformableConv2d
import cv2
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


BN_MOMENTUM = 0.1


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class transposedconvolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride = 1, with_bn=True):
        super(transposedconvolution, self).__init__()
        self.with_bn = with_bn
        self.tconv = nn.ConvTranspose2d(inp_dim, out_dim, k, stride=stride, bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.tconv(x)
        x   = self.bn(x)
        relu = self.relu(x)
        return relu

class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, feature_maps=None):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return mySequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

class MergeUp(nn.Module):
    def forward(self, up1, up2, feature_maps=None):
        if feature_maps:
            tmp = up1 + up2
            for i in feature_maps:
                if up1.shape[-1] == i.shape[-1]:
                    tmp += i
            return tmp
        return up1 + up2

class MergeUp2(nn.Module):
    def forward(self, up1, up2):
        if up1.size() != up2.size():
            # Example: Resize up2 to match the size of up1, up1占썩뫅�� up2 占쎈툖�놅옙�뉎럦�좑옙 �좎룞�� �좎떬�댿뵛 �브퀗�ｏ옙占�
            up2 = F.interpolate(up2, size=up1.size()[2:], mode='nearest')
        return up1 + up2

# class MergeUp3(nn.Module):
#     def __init__(self, inp_dim):
#         super(MergeUp3, self).__init__()
#         self.conv = convolution(3, inp_dim*2, inp_dim, stride=1)

#     def forward(self, up1, up2, feature_maps=None):
#         tmp = up1 + up2
#         if feature_maps:
#             for i in feature_maps:
#                 if up1.shape[-1] == i.shape[-1]:
#                     concat = torch.cat([tmp, i], dim=1)
#                     concat = self.conv(concat)
#         return concat


def make_merge_layer(dim):
    return MergeUp()

def make_merge2_layer(dim):
    return MergeUp2()

# def make_merge3_layer(inp_dim):
#     return MergeUp3(inp_dim)

# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.UpsamplingBilinear2d(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

class DCN(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.Dconv_layers = DeformableConv2d(ic, oc, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.Dconv_layers(x)



######################
# decoder
######################



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        #self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        #x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2)
        #self.k = nn.Linear(dim, dim, bias=qkv_bias)
        #self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.q_proxy = nn.Linear(dim, dim)
        self.kv_proxy = nn.Linear(dim, dim * 2)

        self.q_proxy_ln = nn.LayerNorm(dim)

        self.p_ln = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path*1.0) if drop_path > 0. else nn.Identity()
        #self.mlp = PVT2FFN(dim, int(dim * mlp_ratio))
        self.ffn = PVT2FFN(dim, int(dim * 4))

        self.mlp_proxy = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim),
        )
        self.proxy_ln = nn.LayerNorm(dim)

        self.qkv_proxy = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*3)
        )

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def selfatt(self, featuremap, H=None, W=None):
        B, C, W, H = featuremap.shape
        featuremap = featuremap.flatten(2).transpose(1, 2)
        B, N, C = featuremap.shape

        qkv = self.qkv_proxy(featuremap).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        featuremap = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return featuremap
    

    def forward(self, x, featuremap, H=None, W=None):

        x1 = self.norm(x.flatten(2).transpose(1, 2))

        x1 = x1.transpose(1, 2) 
        x1 = x1.view(x1.shape[0], x1.shape[1], int(np.sqrt(x1.shape[2])), int(np.sqrt(x1.shape[2]))) 
        #[3, 1024, 384] -> [3, 384, 32, 32]

        x1 = self.drop_path(self.gamma1 * self.selfatt(x1))
        #x1 = self.drop_path(self.gamma1 * self.selfatt(x))
    
        x = x.flatten(2).transpose(1, 2)

        #featuremap = featuremap.mean(dim=1) #[3, 32, 32]
        #print(featuremap.size())
        #print(f1.size()) #[3, 1024, 384]

        x = x + x1

 
        featuremap = featuremap.expand(featuremap.shape[0], x1.shape[2], featuremap.shape[2], featuremap.shape[3])

        B, C, W, H = featuremap.shape
        featuremap = featuremap.flatten(2).transpose(1, 2)

        B, N, C = featuremap.shape
        
        B_p, N_p, C_p = x.shape

        featuremap1 = featuremap #�좎럥�놅옙醫묒삕占쏙옙�좑옙 濾곌쑨�ｏ옙洹⑥��좑옙 �좎럥遊울옙占�

        featuremap = self.norm(featuremap)

        q = self.q(featuremap).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q_HR = self.q_proxy(self.q_proxy_ln(x)).reshape(B_p, N_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k_HR = self.kv_proxy(featuremap).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #kp, vp = kv_LR[0], kv_LR[1]
        kp = k_HR[0]

        v_LR = self.kv_proxy(featuremap).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        vp = v_LR[1]

        attn = (q_HR @ kp.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _x = (attn @ vp).transpose(1, 2).reshape(B, N_p, C) * self.gamma2
        x = x + self.drop_path(_x)
        x = self.ffn(x + self.drop_path(self.gamma3 * self.mlp_proxy(self.p_ln(x))))

        #------------------------------------------------


        kv = self.kv(self.proxy_ln(x)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        #k = self.k(x)
        #v = self.v(x)
        #q = q.mean(dim=1)
        #n =  q.size()[-1]
        #k = k[:, :, :n]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn) 

        featuremap = (attn @ v).transpose(1, 2).reshape(B, N, C)
        featuremap = self.proj(featuremap)

        featuremap = featuremap + featuremap1
        featuremap1 = featuremap
        
        featuremap = self.ffn(self.norm(featuremap))
        featuremap = featuremap + featuremap1
       
        featuremap = featuremap.reshape(B, W, H, -1).permute(0, 3, 1, 2).contiguous()
        x = x.reshape(B, W, H, -1).permute(0, 3, 1, 2).contiguous()
        
        result = torch.cat([x, featuremap], dim=1)
        n =  result.size()[1]
        result = result[:, :n//2, :, :]
     
        return result
    


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, featuremap, H=None, W=None):
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        featuremap = featuremap.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        
        q = self.q(x)
        k = self.k(featuremap)
        # q = self.q(featuremap)
        # k = self.k(x)
        v = self.v(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, W, H, -1).permute(0, 3, 1, 2).contiguous()
        
        return x    
    


class PALayer(nn.Module):
    def __init__(self, dim):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
            self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)            
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, map):
        result = self.conv1(map)
        result = self.relu(result)
        result = self.conv2(result)
        result = self.sigmoid(result)

        result = result*map

        return result
    

class CALayer(nn.Module):
    def __init__(self, dim):
            super(CALayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, map):
        result = self.avg_pool(map)
        result = self.conv1(result)
        result = self.relu(result)
        result = self.conv2(result)
        result = self.sigmoid(result)

        result = result*map

        return result
    
    
class FAMLayer(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(FAMLayer, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1, 1, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, map):

        result = self.act1(self.conv1(map))

        result = F.interpolate(result, size=map.size()[2:], mode='nearest')

        result = result + map
        result = self.conv2(result)
        result = self.calayer(result)
        result = self.palayer(result)

        result = F.interpolate(result, size=map.size()[2:], mode='nearest')
        result += map
        
        return result

    
class Separate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.obj_attention = DualAttention(dim)
        self.mask_layer1 = FAMLayer(dim)
        self.attn = Attention(dim)
   
    def forward(self, x, featuremap, hm):
        
        '''
        obj_mask = self.mask_layer1(hm) 
        obj_x = x * obj_mask 
        obj_feature = featuremap * obj_mask 
        
        obj_attention = self.obj_attention(obj_x, obj_feature)
        attention_map = obj_attention + x   #skip
        attention_map = self.conv1(attention_map)
        '''
        obj_attention = self.obj_attention(x, hm)
        attention_map = self.attn(obj_attention, featuremap)
        #attention_map = obj_attention + featuremap   #skip
        attention_map = self.conv1(attention_map)   

        return attention_map

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual, stack=None,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_merge2_layer=make_merge2_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, stack=stack, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)
        self.merge2 = make_merge2_layer(curr_dim) # 筌�꼻�숃짆�노룾�좎럥�뽩뜝占� 筌�쥜�숋옙�뚯삕

        if self.n > 1:
            self.fam1 = Separate(curr_dim)
            self.fam2 = Separate(curr_dim)
        else:
            self.fam1 = Separate(curr_dim)
            self.fam2 = Separate(curr_dim)
            self.fam3 = Separate(next_dim)
            self.fam4 = Separate(next_dim)
              

    def forward(self, x, feature_maps = None, dwn_hm = None):
        if feature_maps is None:
            up1  = self.up1(x)
            max1 = self.max1(x)
            low1 = self.low1(max1)  # 1 : 64x64x256, 2 : 32x32x384, 3 : 16x16x384, 4 : 8x8x384, 5 : 4x4x512
            low2 = self.low2(low1)

            if len(low2) == 2: 
                low3 = self.low3(low2[0])
            else:
                low3 = self.low3(low2)          
            up2  = self.up2(low3)

            if len(low2) == 2:
                low2[1].append(up1) # �낅슣�섋땻占� �좎럥�놅옙占�
                low2[1].append(self.merge2(up1, up2))
                return self.merge2(up1, up2), low2[1]
            
            return self.merge2(up1, up2),  [up1, self.merge2(up1, up2)] # [up1, up2]
        else:
            for num, i in enumerate(feature_maps):  
                if x.shape[-1] == i.shape[-1] and num % 2 == 0 and x.shape[-1] <= 32:
                    x = self.fam1(x, i, dwn_hm[num])
                        
            up1  = self.up1(x)
            max1 = self.max1(x)
            low1 = self.low1(max1)  # 1 : 64x64x256, 2 : 32x32x384, 3 : 16x16x384, 4 : 8x8x384, 5 : 4x4x512
            if low1.shape[-1] == 8:
                low1 = self.fam3(low1, feature_maps[0], dwn_hm[0])
                low2 = self.low2(low1, feature_maps)
                low2 = self.fam4(low2, feature_maps[1], dwn_hm[1])
  
                low3 = self.low3(low2)
            else:
                low2 = self.low2(low1, feature_maps, dwn_hm)
                low3 = self.low3(low2)
            up2  = self.up2(low3)
            for num, i in enumerate(feature_maps):  
                if x.shape[-1] == i.shape[-1] and num % 2 == 1 and x.shape[-1] <= 32:
                    up2 = self.fam2(up2, i, dwn_hm[num])

                    
            return self.merge2(up1, up2)

class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre
        self.pre_1 = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=1)
        )
        # self.pre_2 = nn.Sequential(
        #     residual(3, 128, 256, stride=1)
        # )
        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        
        self.upfeature1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=256, out_dim=256, stride=1)
        )
        
        self.uphm = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=1, out_dim=1, stride=1, with_bn=False),
            nn.Conv2d(1, 1, (1, 1)),
        )
        self.upwh = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=2, out_dim=2, stride=1, with_bn=False),
            nn.Conv2d(2, 2, (1, 1))
        )
        self.upoff = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            convolution(k=3, inp_dim=2, out_dim=2, stride=1, with_bn=False),
            nn.Conv2d(2, 2, (1, 1))
        )
    
        self.dcn_layers   = nn.ModuleList([
            nn.Sequential(
                DeformableConv2d(ic, oc, kernel_size=(3, 3), stride=1,
                    padding=1),
                nn.BatchNorm2d(oc),
                nn.ReLU()
            ) for ic, oc in zip([384, 384, 384, 384, 384, 384, 256, 256], [512, 512, 384, 384, 384, 384, 384, 384])
        ])

        self.conv = convolution(k=3, inp_dim=517, out_dim=256, stride=1) ## multi-scale �좎럥援앾옙占� �브퀗�ｏ옙占�
        
        ## keypoint heatmaps
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        inter_256x256 = self.pre_1(image)
        outs  = []
        dwn_hms = []
    
        for ind in range(self.nstack):
            if ind == 0:
                kp_, cnv_  = self.kps[ind], self.cnvs[ind]
                kp, feature_maps = kp_(inter) # hourglass 嶺뚮ㅄ維�獄�옙
                cnv = cnv_(kp) # hourglass 嶺뚮ㅄ維�獄�옙
                
            elif ind == 1:
                kp_, cnv_  = self.kps[ind], self.cnvs[ind]
                kp = kp_(inter_1, feature_maps, dwn_hms) # hourglass 嶺뚮ㅄ維�獄�옙
                cnv = cnv_(kp) # hourglass 嶺뚮ㅄ維�獄�옙

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                # print(layer)
                y = layer(cnv) # �좎럥�뉛옙猿볦춹�좑옙, wh, offset�좎룞�� �좎룞�쇿뜝�숈삕 Loss占쎌뮋�� 占쎈땶�ル┃�ル�먯삕 �좎럡�들뜮占� 占쎈슓維�占쎈뱺�댐옙貫占쏙옙 �좎럡�ｆ쾮占� �좎럩瑗뱄쭛占�
                out[head] = y 

            outs.append(out)

            if ind < self.nstack - 1:

                cnv_1 = self.cnvs_[ind](cnv) # 繞벿살탪�뚳옙 �좎룞��
                cnv_1 = self.upfeature1(cnv_1) # �좎럥援욄틦占쎌삕占쏙옙異�

                hm = outs[0]['hm']
    
                ## multi scale input ##
                uphm = self.uphm(outs[0]['hm'])
                upwh = self.upwh(outs[0]['wh'])
                upoff = self.upoff(outs[0]['reg'])
                
                depthconcat = torch.cat((uphm, upwh, upoff, cnv_1, inter_256x256), dim=1)
                inter = self.conv(depthconcat)
                inter_1 = self.inters[ind](inter) # �좎럥�꾢첎癒щ눀�븐뼚夷� �좎떬�녿쐳�좎룞��
                
                ## deformable convolution ##
                for i in range(8):
                    dcn = self.dcn_layers[i]
                    feature_maps[i] = dcn(feature_maps[i])
                
                ## heatmap �좎떬�녿턄嶺뚯빢�� �브퀗�ｏ옙占� ##
                for i in feature_maps:
                    f_shape = i.shape[-1]
                    dwn_hm_map = F.interpolate(hm, size=(f_shape, f_shape), mode='bilinear')
                    dwn_hms.append(dwn_hm_map)                              

        return outs

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=2):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

def get_large_hourglass_net(num_layers, heads, head_conv, num_stack = 2):
    if num_stack:
        pass
    else:
        num_stack = 2
        
    model = HourglassNet(heads, num_stacks = num_stack)
    return model
