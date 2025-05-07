#!/usr/bin/env python
# coding=utf-8
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from torch import nn, einsum
import torchvision.models as models
from einops import rearrange
from CLIP import clip
from transformers import MambaConfig
import torch.nn.functional as F
from Tiny_model_4_CD.models.layers import MixingMaskAttentionBlock
from RSCaMa.model.mamba_block import CaMambaModel
from DTCDSCN import SEBasicBlock, Dblock, CDNet_model, Dblock_more_dilate
# from transformers.models.mamba.modeling_mamba import MambaRMSNorm
from model_SEIFNet import CoDEM2, SupervisedAttentionModule
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]

        elif 'CLIP' in self.network:
            clip_model_type = self.network.replace('CLIP-', '')
            self.clip_model, preprocess = clip.load(clip_model_type, jit=False)  #
            self.clip_model = self.clip_model.to(dtype=torch.float32)

        # self.cnn_list = nn.ModuleList(modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        if "CLIP" in self.network:
            img_A = imageA.to(dtype=torch.float32)
            img_B = imageB.to(dtype=torch.float32)
            clip_emb_A, img_feat_A = self.clip_model.encode_image(img_A)
            clip_emb_B, img_feat_B = self.clip_model.encode_image(img_B)

            # import numpy as np
            # token_all = np.zeros((token.shape[0], 77), dtype=int)
            # token_all = torch.from_numpy(token_all).cuda()
            # token_all[:, :token.shape[1]] = token
            # _, token = self.clip_model.encode_text(token_all)
            # token /= token.norm(dim=-1, keepdim=True)

        else:
            # feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
            # feat2 = self.cnn(imageB)
            feat1 = imageA
            feat2 = imageB
            feat1_list = []
            feat2_list = []
            cnn_list = list(self.cnn.children())
            for module in cnn_list:
                feat1 = module(feat1)
                feat2 = module(feat2)
                feat1_list.append(feat1)
                feat2_list.append(feat2)
            feat1_list = feat1_list[-4:]
            feat2_list = feat2_list[-4:]
        return img_feat_A, img_feat_B


    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 3 through 4
        if 'CLIP' in self.network and fine_tune:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            # If fine-tuning, only fine-tune last 2 trans and ln_post
            children_list = list(self.clip_model.visual.transformer.resblocks.children())[-6:]
            children_list.append(self.clip_model.visual.ln_post)
            for c in children_list:
                for p in c.parameters():
                    p.requires_grad = True
        elif 'CLIP' not in self.network and fine_tune:
            for c in list(self.cnn.children())[:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class resblock(nn.Module):
    '''
    module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, int(outchannel / 1), kernel_size=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), int(outchannel / 1), kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm(int(outchannel/2),dim=1),
            nn.BatchNorm2d(int(outchannel / 1)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 1), outchannel, kernel_size=1),
            # nn.LayerNorm(int(outchannel / 1),dim=1)
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return self.act(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads = 8, dropout = 0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias = False)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2, x3):
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_k(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)#(b,n,dim)

class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout = 0., norm_first = False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads = heads, dropout = dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout = dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3)[0] + x1)
            x = self.norm2(self.feedforward(x) + x)

        return x

class CrossTransformer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.soft = nn.Softmax()

    def forward(self, input1, input2):
        attn_output, attn_weight = self.attention(input1, input2, input2)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 定义隐藏层，包含两个隐藏层，使用ReLU激活函数
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入到第一个隐藏层
        x = torch.relu(self.fc1(x))
        # 输入到第二个隐藏层
        x = torch.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim=512):
        super(SpatialAttention, self).__init__()
        # 1x1 卷积层，用于生成空间注意力
        self.conv = nn.Conv1d(embed_dim, 1, kernel_size=1)  # 输出大小为1，表示每个token的权重
        self.sigmoid = nn.Sigmoid()  # 激活函数将权重压缩到0和1之间

    def forward(self, x):
        # 转换为 (B, C, N) 作为 Conv1d 的输入
        x = x.transpose(1, 2)  # (B, C, N)
        # 通过卷积生成空间注意力分数
        attn_map = self.conv(x)  # (B, 1, N)
        # 将空间注意力分数进行sigmoid激活，得到权重
        attn_map = self.sigmoid(attn_map)  # (B, 1, N)
        # 将注意力权重应用到特征上
        attn_map = attn_map.squeeze(1)  # (B, N)
        # 对每个token的特征加权
        x = x.transpose(1, 2)  # 转回原来的 (B, N, C)
        x = x * attn_map.unsqueeze(2)  # 广播注意力权重到特征维度
        return x

class Dynamic_conv(nn.Module):
    def __init__(self, dim):
        super(Dynamic_conv, self).__init__()
        self.d_conv_3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim
        )
        self.d_conv_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim)
        self.d_conv_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(3*dim)
        self.conv_1 = nn.Conv2d(3*dim, dim, 1)
    def forward(self, x):
        x1 = self.d_conv_3x3(x)
        x2 = self.d_conv_1x5(x)
        x3 = self.d_conv_5x1(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.BN(x)
        x = self.activation(x)
        x = self.conv_1(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
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

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, dropout=0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.h_feat = h_feat
        self.w_feat = w_feat
        self.n_layers = n_layers
        self.channels = channels
        # position embedding
        self.LN_norm = nn.LayerNorm(channels)
        attention_dim = 768
        hidden_dim = 768
        heads = 16

        self.t1 = Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False)
        # self.t2 = Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False)
        # self.ds = nn.Conv1d(512, 256, kernel_size=1)

        self.crstransformer = CrossTransformer(768, 8)

        self.relu = nn.ReLU()

        # self.mlp = MLP(768, 768, 768)

        self.mha = nn.MultiheadAttention(768, 8, dropout)
        # self.mha_dif = nn.MultiheadAttention(768, 8, dropout=0.)
        self.ln = nn.LayerNorm(768)
        self.ln1 = nn.LayerNorm(768)
        self.fforward = FeedForward(768, 768, dropout=dropout)
        self.ln2 = nn.LayerNorm(768)
        # Feed-forward network (2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.SiLU(),
            nn.Linear(768*2, 768),
            nn.Dropout(dropout)
        )
        # self.ffn2 = nn.Sequential(
        #     nn.Linear(768, 768*2),
        #     nn.SiLU(),
        #     nn.Linear(768*2, 768)
        # )
        self.sa = SpatialAttention()

        self.drop = nn.Dropout(dropout)
        # self.pos_embedding = nn.Embedding(256, 768)
        self.sig = nn.Sigmoid()
        # self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dblock_master = Dblock(768)
        # self.dblock_master_3 = Dblock_more_dilate(768)
        self.dif_encode = SEBasicBlock(768)
        dim = 768
        self.dwconv_5x5 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(dim)
        )
        self.dwconv_7x7 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(dim)
        )
        self.dwconv_9x9 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(dim)
        )
        self.dwconv_1x1 = nn.Sequential(
            nn.BatchNorm1d(dim*3),
            nn.GELU(),
            nn.Conv1d(dim * 3, dim, 1))
        self.norm = nn.LayerNorm(768)
        self.h_embedding = nn.Embedding(16, int(768/2))
        self.w_embedding = nn.Embedding(16, int(768/2))

        self.patch_embed1 = OverlapPatchEmbed(patch_size=2, stride=2, embed_dim=768)
        # dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]
        # self.block1 = nn.ModuleList([Block(
        #     dim=768, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        #     drop=0., attn_drop=0., drop_path=dpr[0 + i], norm_layer=nn.LayerNorm,
        #     sr_ratio=4)
        #     for i in range(2)])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=4, stride=4, embed_dim=768)
        # self.block2 = nn.ModuleList([Block(
        #     dim=768, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        #     drop=0., attn_drop=0., drop_path=dpr[0 + i], norm_layer=nn.LayerNorm,
        #     sr_ratio=4)
        #     for i in range(2)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, img_A, img_B):
    #     img_A = self.sa(img_A) + img_A
    #     img_B = self.sa(img_B) + img_B
    #
    #     mid_A = img_A.transpose(0, 1)
    #     mid_B = img_B.transpose(0, 1)
    #     img_A = self.mha(mid_A, mid_A, mid_A)[0].transpose(0, 1) + img_A
    #     img_B = self.mha(mid_B, mid_B, mid_B)[0].transpose(0, 1) + img_B
    #     mid_A1 = self.ffn(self.ln(img_A))
    #     mid_B1 = self.ffn(self.ln(img_B))
    #
    #     mid_A1 = self.sa(mid_A1) + mid_A1
    #     mid_B1 = self.sa(mid_B1) + mid_B1
    #     img_A = self.t1(mid_A.transpose(0, 1), mid_A1, mid_A1) + img_A
    #     img_B = self.t1(mid_B.transpose(0, 1), mid_B1, mid_B1) + img_B
    #     img_A = self.ffn(self.ln(img_A))
    #     img_B = self.ffn(self.ln(img_B))

    def dwconv(self, img):
        img = img.transpose(1, 2)
        img_5 = self.dwconv_5x5(img)
        img_7 = self.dwconv_7x7(img)
        img_9 = self.dwconv_9x9(img)
        img579 = torch.cat([img_5, img_7, img_9], 1)
        # img579 = img_5 + img_7 + img_9
        img = self.dwconv_1x1(img579) + img
        return img.transpose(1, 2)

    def add_pos_embedding(self, x):
        if len(x.shape) == 3:
            bs, c, dim = x.shape  # 1,256,768
            x = x.transpose(1, 2).view(bs, dim, 16, 16)
        batch, dim, h, w = x.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                   embed_h.unsqueeze(1).repeat(1, w, 1)],
                                  dim=-1)
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        x = x + pos_embedding
        return x.view(bs, dim, -1).transpose(1, 2)


    def forward(self, img_A, img_B):
        bs, c, dim = img_A.shape  # 1,256,768

        img_A1 = self.add_pos_embedding(img_A)
        img_B1 = self.add_pos_embedding(img_B)

        # img_A = self.dwconv(img_A)
        # img_B = self.dwconv(img_B)
        # a, b = self.dif_encode(img_A1, img_B1)

        a, b = self.dif_encode(img_A1, img_B1)
        a = self.dwconv(a.view(bs, dim, -1).transpose(1, 2))
        b = self.dwconv(b.view(bs, dim, -1).transpose(1, 2))
        # dif_A = self.dblock_master((a-b).transpose(1, 2)).transpose(1, 2)
        # dif_B = self.dblock_master((b-a).transpose(1, 2)).transpose(1, 2)
        # a = a.transpose(0, 1)
        # b = b.transpose(0, 1)
        dif_A = a-b
        dif_B = b-a
        a = a.transpose(0, 1)
        b = b.transpose(0, 1)
        # dif_A = self.dblock_master(a-b).view(bs, dim, -1).transpose(1, 2)
        # dif_B = self.dblock_master(b-a).view(bs, dim, -1).transpose(1, 2)
        # a = a.view(bs, dim, -1).transpose(1, 2).transpose(0, 1)
        # b = b.view(bs, dim, -1).transpose(1, 2).transpose(0, 1)
        dif_A = self.ln1(self.mha(dif_A.transpose(0, 1), a, a)[0].transpose(0, 1) + dif_A)
        dif_B = self.ln1(self.mha(dif_B.transpose(0, 1), b, b)[0].transpose(0, 1) + dif_B)
        dif_A = self.ln2(self.fforward(dif_A) + dif_A)
        dif_B = self.ln2(self.fforward(dif_B) + dif_B)
        # img_A = self.dwconv(img_A)
        # img_B = self.dwconv(img_B)

        img_A = self.sa(img_A) + img_A
        img_B = self.sa(img_B) + img_B
        mid_A = self.ln(img_A.transpose(0, 1))  # 256,bs,768
        mid_B = self.ln(img_B.transpose(0, 1))
        img_A = self.mha(mid_A, mid_A, mid_A)[0].transpose(0, 1) + img_A
        img_B = self.mha(mid_B, mid_B, mid_B)[0].transpose(0, 1) + img_B

        mid_A1 = self.ffn(self.ln1(img_A))
        mid_B1 = self.ffn(self.ln1(img_B))

        mid_A1 = self.sa(mid_A1) + mid_A1
        mid_B1 = self.sa(mid_B1) + mid_B1
        img_A = self.t1(mid_A.transpose(0, 1), mid_A1, mid_A1) + img_A
        img_B = self.t1(mid_B.transpose(0, 1), mid_B1, mid_B1) + img_B
        img_A = self.ffn(self.ln(img_A)) + dif_B
        img_B = self.ffn(self.ln(img_B)) + dif_A

        feat = torch.cat([img_A, img_B], 1)
        return feat


if __name__ == '__main__':
    # test
    img_A = torch.randn(16, 49, 768).cuda()
    img_B = torch.randn(16, 49, 768).cuda()
    encoder = AttentiveEncoder(n_layers=3, feature_size=(7, 7, 768), heads=8).cuda()
    feat_cap = encoder(img_A, img_B)
    print(feat_cap.shape)
    print(feat_cap)
    print('Done')
