#!/usr/bin/env python
# coding=utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import argparse
import json, random, cv2
from tqdm import tqdm
from torch import nn
from data.LEVIR_CC.LEVIRCC import LEVIRCCDataset
import torch.nn.functional as F
from RSCaMa.model.model_encoder_attMamba import Encoder, AttentiveEncoder, Transformer, SpatialAttention
from RSCaMa.model.model_decoder import DecoderTransformer
from VideoX.SeqTrack.lib.models.seqtrack.vit import vit_base_patch16
from model.iml_vit_model_git import iml_vit_model
from grad_cam import GradCAM
import matplotlib.pyplot as plt
import numpy as np
from RSCaMa.utils_tool.utils import *
from model.model_encoder import Encoder as Resnet101
from typing import List, Optional
from torch import Tensor, reshape, stack
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)
import clip
import re
torch.autograd.set_detect_anomaly(True)


def word2clip(word_voc, clip_token):
    clip_vocab = torch.zeros(len(word_voc), 77)
    clip_vocab = {}
    num_list = []
    for i in range(len(word_voc)):
        key = next((key for key, value in word_voc.items() if value == i), None)
        key = key.replace('<', '').replace('>', '')
        if key == 'intersect':
            print()
        key_encode = clip_token(key)
        a = key_encode[:, 2]
        if torch.eq(key_encode[:, 2], 49407):
            key_encode_ = key_encode[:, 1].item()
        elif torch.eq(key_encode[:, 3], 49407):
            key_encode_ = key_encode[:, 1].item() + key_encode[:, 2].item()
        elif torch.eq(key_encode[:, 4], 49407):
            key_encode_ = key_encode[:, 1].item() + key_encode[:, 2].item() + key_encode[:, 3].item()
        elif torch.eq(key_encode[:, 5], 49407):
            key_encode_ = key_encode[:, 1].item() + key_encode[:, 2].item() + key_encode[:, 3].item() + key_encode[:, 4].item()
        if key_encode_<49407:
            while key_encode_ in num_list:
                key_encode_ += 1
            clip_vocab[i] = key_encode_
            num_list.append(key_encode_)
        else:
            key_encode_ = int(key_encode_/2)
            assert key_encode_<49407
            while key_encode_ in num_list:
                key_encode_ += 1
            clip_vocab[i] = key_encode_
    return clip_vocab

class PixelwiseLinear(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)

class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))

        # Mixing:
        return self._convmix(mixed)

class Feat_aug(nn.Module):
    """use the grouped convolution to make a sort of attention"""

    def __init__(self, dim=512, channel = 768):
        super().__init__()
        self.dim = dim
        self.conv13 = nn.Conv1d(dim // 2, dim // 2, 3, 1, 1, bias=True, groups=dim // 2)
        self.conv15 = nn.Conv1d(dim // 2, dim // 2, 5, 1, 2, bias=True, groups=dim // 2)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(dim)

        self.conv1 = nn.Conv1d(dim, dim, 1, 1, bias=True, groups=dim)
        self.conv3 = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim // 2)
        self.conv5 = nn.Conv1d(dim, dim, 5, 1, 2, bias=True, groups=dim // 2)
        self.bn = nn.BatchNorm1d(dim)
        self.LN_norm = nn.LayerNorm(channel)

    def forward(self, x):
        _x = x

        # x11, x12 = x[:, :self.dim // 2, :], x[:, self.dim // 2:, :]
        # x11 = self.conv15(self.conv13(x11))
        # x12 = self.conv15(self.conv13(x12))
        # x = torch.cat([x11, x12], dim=1)
        # x = self.act1(self.bn1(x)) + _x

        x = self.bn(self.conv1(x))
        x = self.bn(self.conv3(x))
        x = self.bn(self.conv1(x))
        x = self.LN_norm(_x + x)
        return x


class CNN_ViT(nn.Module):
    def __init__(self, inplanes=256, embed_dim=768):
        super().__init__()
        self.inplanes = inplanes
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(8 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        dim = 256
        self.dwconv11 = nn.Conv1d(dim*4*4, dim*4, 1, 1, bias=True, groups=dim)
        self.dwconv12 = nn.Conv1d(dim*4, dim, 1, 1, bias=True, groups=dim)
        self.dwconv13 = nn.Conv1d(dim//2, dim//2, 3, 1, 1, bias=True, groups=dim//2)
        self.dwconv15 = nn.Conv1d(dim//2, dim//2, 5, 1, 2, bias=True, groups=dim//2)

        self.dwconv21 = nn.Conv1d(dim*4, dim*2, 1, 1, bias=True, groups=dim)
        self.dwconv22 = nn.Conv1d(dim*2, dim, 1, 1, bias=True, groups=dim)
        self.dwconv23 = nn.Conv1d(dim//2, dim//2, 3, 1, 1, bias=True, groups=dim//2)
        self.dwconv25 = nn.Conv1d(dim//2, dim//2, 5, 1, 2, bias=True, groups=dim//2)

        self.dwconv33 = nn.Conv1d(dim//2, dim//2, 3, 1, 1, bias=True, groups=dim//2)
        self.dwconv35 = nn.Conv1d(dim//2, dim//2, 5, 1, 2, bias=True, groups=dim//2)

        self.dwconv41 = nn.Conv1d(dim//4, dim//2, 1, 1, bias=True, groups=dim//4)
        self.dwconv42 = nn.Conv1d(dim//2, dim, 1, 1, bias=True, groups=dim//2)
        self.dwconv43 = nn.Conv1d(dim//2, dim//2, 3, 1, 1, bias=True, groups=dim//2)
        self.dwconv45 = nn.Conv1d(dim//2, dim//2, 5, 1, 2, bias=True, groups=dim//2)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(dim)

        self.actx = nn.GELU()
        self.fcx = nn.Linear(dim*4, dim*4)
        self.drop = nn.Dropout(0.)
        self.h_feat = 16
        self.w_feat = 16
        self.attention = nn.MultiheadAttention(dim*4, 8, dropout=0.)
        self.norm_layer = nn.LayerNorm(dim*4)
        self.ffn = ConvFFN()
        init_values = 0.
        self.gamma = nn.Parameter(init_values * torch.ones((768)), requires_grad=True)

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2.permute(0,2,3,1) + self.level_embed[0]
        c3 = c3.permute(0,2,3,1) + self.level_embed[1]
        c4 = c4.permute(0,2,3,1) + self.level_embed[2]
        return c2.permute(0,3,1,2), c3.permute(0,3,1,2), c4.permute(0,3,1,2)

    def forward(self, x, y):
        c1, c2, c3, c4 = x
        c1 = self.fc1(c1)       # 256,64,64 --> 768,64,64
        c2 = self.fc2(c2)       # 512,32,32 --> 768,32,32
        c3 = self.fc3(c3)       # 1024,16,16 --> 768,16,16
        c4 = self.fc4(c4)       # 2048,8,8 --> 768,8,8
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        bs, dim, _, _ = c2.shape
        c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s     4096,768
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s     1024,768
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s    256,768
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s    64,768

        _, dim1, _ = c1.shape
        c1 = self.dwconv11(c1)  # BxCxHxW
        c1 = self.dwconv12(c1)
        x11, x12 = c1[:, :self.inplanes // 2, :], c1[:, self.inplanes // 2:, :]
        x11 = self.dwconv13(x11)  # BxCxHxW
        x12 = self.dwconv15(x12)
        c1 = torch.cat([x11, x12], dim=1)
        c1 = self.act1(self.bn1(c1))

        _, dim2, _ = c2.shape
        c2 = self.dwconv21(c2)  # BxCxHxW
        c2 = self.dwconv22(c2)
        x11, x12 = c2[:, :self.inplanes // 2, :], c2[:, self.inplanes // 2:, :]
        x11 = self.dwconv23(x11)  # BxCxHxW
        x12 = self.dwconv25(x12)
        c2 = torch.cat([x11, x12], dim=1)
        c2 = self.act1(self.bn1(c2))

        _, dim3, _ = c3.shape
        x11, x12 = c3[:, :self.inplanes // 2, :], c3[:, self.inplanes // 2:, :]
        x11 = self.dwconv33(x11)  # BxCxHxW
        x12 = self.dwconv35(x12)
        c3 = torch.cat([x11, x12], dim=1)
        c3 = self.act1(self.bn1(c3))

        _, dim4, _ = c4.shape
        c4 = self.dwconv41(c4)  # BxCxHxW
        c4 = self.dwconv42(c4)
        x11, x12 = c4[:, :self.inplanes // 2, :], c4[:, self.inplanes // 2:, :]
        x11 = self.dwconv43(x11)  # BxCxHxW
        x12 = self.dwconv45(x12)
        c4 = torch.cat([x11, x12], dim=1)
        c4 = self.act1(self.bn1(c4))

        x = torch.cat([c1, c2, c3, c4], 1)
        x = self.actx(x.permute(0,2,1))
        x = self.fcx(x)
        c = self.drop(x)
        c = c + self.ffn(self.norm_layer(c))
        c = c.permute(0,2,1)
        c_select1, c_select2, c_select3, c_select4 = c[:, :256, :], c[:, 256:256*2, :], c[:, 256*2:256*3, :], c[:, 256*3:, :]
        y = y + self.gamma * (c_select1 + c_select2 + c_select3 + c_select4)
        return y
# _84898_2205
class ConvFFN(nn.Module):
    def __init__(self, dim=768, channel=1024, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(channel, channel)
        self.dwconv_3 = nn.Conv1d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv_1 = nn.Conv1d(dim, dim, 1)
        self.act_ = act_layer()
        self.fc_2 = nn.Linear(channel, channel)
        self.drop_ = nn.Dropout(drop)

    def forward(self, x):
        # x = x.permute(0,2,1)
        x = self.fc_1(x)
        x = self.dwconv_3(x)
        # x = self.dwconv_1(x)
        x = self.act_(x)
        x = self.drop_(x)
        x = self.fc_2(x)
        x = self.drop_(x)
        return x

class CNN_ViT_ceshizhong(nn.Module):
    def __init__(self, inplanes=256, embed_dim=768):
        super().__init__()
        self.inplanes = inplanes
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(8 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))

        dim = 256
        self.dwconv11 = nn.Conv1d(dim * 4 * 4, dim * 4, 1, 1, bias=True, groups=dim)
        self.dwconv12 = nn.Conv1d(dim * 4, dim, 1, 1, bias=True, groups=dim)
        self.dwconv13 = nn.Conv1d(dim // 2, dim // 2, 3, 1, 1, bias=True, groups=dim // 2)
        self.dwconv15 = nn.Conv1d(dim // 2, dim // 2, 5, 1, 2, bias=True, groups=dim // 2)

        self.dwconv21 = nn.Conv1d(dim * 4, dim * 2, 1, 1, bias=True, groups=dim)
        self.dwconv22 = nn.Conv1d(dim * 2, dim, 1, 1, bias=True, groups=dim)
        self.dwconv23 = nn.Conv1d(dim // 2, dim // 2, 3, 1, 1, bias=True, groups=dim // 2)
        self.dwconv25 = nn.Conv1d(dim // 2, dim // 2, 5, 1, 2, bias=True, groups=dim // 2)

        self.dwconv33 = nn.Conv1d(dim // 2, dim // 2, 3, 1, 1, bias=True, groups=dim // 2)
        self.dwconv35 = nn.Conv1d(dim // 2, dim // 2, 5, 1, 2, bias=True, groups=dim // 2)

        self.dwconv41 = nn.Conv1d(dim // 4, dim // 2, 1, 1, bias=True, groups=dim // 4)
        self.dwconv42 = nn.Conv1d(dim // 2, dim, 1, 1, bias=True, groups=dim // 2)
        self.dwconv43 = nn.Conv1d(dim // 2, dim // 2, 3, 1, 1, bias=True, groups=dim // 2)
        self.dwconv45 = nn.Conv1d(dim // 2, dim // 2, 5, 1, 2, bias=True, groups=dim // 2)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(dim)

        self.actx = nn.GELU()
        self.fcx = nn.Linear(dim * 4, dim * 4)
        self.drop = nn.Dropout(0.1)
        self.h_feat = 16
        self.w_feat = 16
        self.attention = nn.MultiheadAttention(768, 8, dropout=0.1)
        self.norm_layer = nn.LayerNorm(dim * 4)
        self.ffn = ConvFFN()
        init_values = 0.
        self.gamma = nn.Parameter(init_values * torch.ones((768)), requires_grad=True)

        self.liner_proj = nn.Linear(768*4, 768)
        self.ln = nn.LayerNorm(768)
        # Feed-forward network (2-layer MLP)
        self.out_ffn = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.SiLU(),
            nn.Linear(768*2, 768)
        )
        # self.sa = SpatialAttention()
        # from DTCDSCN import SEBasicBlock, CDNet_model
        # self.dif_model = CDNet_model(SEBasicBlock, [3, 4, 6, 3])

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2.permute(0,2,3,1) + self.level_embed[0]
        c3 = c3.permute(0,2,3,1) + self.level_embed[1]
        c4 = c4.permute(0,2,3,1) + self.level_embed[2]
        return c2.permute(0,3,1,2), c3.permute(0,3,1,2), c4.permute(0,3,1,2)

    def forward(self, x, y):
        # imgA = self.dif_model(x, y)
        # return imgA
        c1, c2, c3, c4 = x
        c1 = self.fc1(c1)       # 256,64,64 --> 768,64,64
        c2 = self.fc2(c2)       # 512,32,32 --> 768,32,32
        c3 = self.fc3(c3)       # 1024,16,16 --> 768,16,16
        c4 = self.fc4(c4)       # 2048,8,8 --> 768,8,8
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        bs, dim, _, _ = c2.shape
        c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s     4096,768
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s     1024,768
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s    256,768
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s    64,768

        _, dim1, _ = c1.shape
        c1 = self.dwconv11(c1)  # BxCxHxW
        c1 = self.dwconv12(c1)
        x11, x12 = c1[:, :self.inplanes // 2, :], c1[:, self.inplanes // 2:, :]
        x11 = self.dwconv13(x11)  # BxCxHxW
        x12 = self.dwconv15(x12)
        c1 = torch.cat([x11, x12], dim=1)
        c1 = self.act1(self.bn1(c1))

        _, dim2, _ = c2.shape
        c2 = self.dwconv21(c2)  # BxCxHxW
        c2 = self.dwconv22(c2)
        x11, x12 = c2[:, :self.inplanes // 2, :], c2[:, self.inplanes // 2:, :]
        x11 = self.dwconv23(x11)  # BxCxHxW
        x12 = self.dwconv25(x12)
        c2 = torch.cat([x11, x12], dim=1)
        c2 = self.act1(self.bn1(c2))

        _, dim3, _ = c3.shape
        x11, x12 = c3[:, :self.inplanes // 2, :], c3[:, self.inplanes // 2:, :]
        x11 = self.dwconv33(x11)  # BxCxHxW
        x12 = self.dwconv35(x12)
        c3 = torch.cat([x11, x12], dim=1)
        c3 = self.act1(self.bn1(c3))

        _, dim4, _ = c4.shape
        c4 = self.dwconv41(c4)  # BxCxHxW
        c4 = self.dwconv42(c4)
        x11, x12 = c4[:, :self.inplanes // 2, :], c4[:, self.inplanes // 2:, :]
        x11 = self.dwconv43(x11)  # BxCxHxW
        x12 = self.dwconv45(x12)
        c4 = torch.cat([x11, x12], dim=1)
        c4 = self.act1(self.bn1(c4))

        x = torch.cat([c1, c2, c3, c4], 1)
        x = self.actx(x.permute(0,2,1))
        x = self.fcx(x)
        c = self.drop(x)
        c = c + self.ffn(self.norm_layer(c))
        c = c.permute(0,2,1)

        c_select1, c_select2, c_select3, c_select4 = c[:, :256, :], c[:, 256:256*2, :], c[:, 256*2:256*3, :], c[:, 256*3:, :]

        # y = c_select1 + c_select2 + c_select3 + c_select4
        y = y + self.gamma * (c_select1 + c_select2 + c_select3 + c_select4)
        return y



class Trainer(object):
    def __init__(self, args):
        """
        Training and validation.
        """
        self.args = args
        random_str = str(random.randint(1, 10000))
        name = args.decoder_type + '_' + time_file_str()
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath)==False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)

        self.best_bleu4 = 0.6  # BLEU-4 score right now
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
            self.clip_vocab = word2clip(self.word_vocab, clip.tokenize)
            self.reversed_dict = {v: k for k, v in self.clip_vocab.items()}

        self.build_model()

        # Loss function
        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_cap_cls = torch.nn.CrossEntropyLoss().cuda()

        # Custom dataloaders
        if args.data_name == 'LEVIR_CC':
            self.train_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'train', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
            self.val_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'val', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
            self.test_loader = data.DataLoader(
                LEVIRCCDataset(args.network, args.data_folder, args.list_path, 'test', args.token_folder, self.word_vocab, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

        self.l_resizeA = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
        self.l_resizeB = torch.nn.Upsample(size = (256, 256), mode ='bilinear', align_corners = True)
        self.index_i = 0
        self.hist = np.zeros((args.num_epochs*2 * len(self.train_loader), 5))
        # Epochs

        self.best_model_path = None
        self.best_epoch = 0

    def build_model(self):
        args = self.args
        # Initialize / load checkpoint

        self.encoder = vit_base_patch16(pretrained=True, pretrain_type='mae',
                                                       search_size=256, template_size=256,
                                                       search_number=1, template_number=1,
                                                       drop_path_rate=0.1,
                                                       use_checkpoint=False)
        # self.feataug = Feat_aug()
        self.resnet101 = Resnet101('resnet101', fine_tune=True)
        self.cnn_vit = CNN_ViT()
        self.encoder_trans = AttentiveEncoder(n_layers=args.n_layers, feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                              heads=args.n_heads, dropout=0.3)
        self.decoder = DecoderTransformer(decoder_type=args.decoder_type,
                                          embed_dim=args.embed_dim,
                                          vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                          word_vocab=self.word_vocab, n_head=args.n_heads,
                                          n_layers=args.decoder_n_layers, dropout=args.dropout)
        checkpoint = torch.load('/workstation/wrj/Chg2Cap/models_ckpt/transformer_decoder_2024-11-27-16-15-57/LEVIR_CC_bts_16_epo12_Bleu4_64120.pth')
        self.encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
        self.resnet101.load_state_dict(checkpoint['resnet101_dict'], strict=True)
        self.cnn_vit.load_state_dict(checkpoint['cnn_vit_dict'], strict=True)
        self.decoder.load_state_dict(checkpoint['decoder_dict'], strict=False)
        # set optimizer
        self.encoder_optimizer = torch.optim.Adam(params=self.encoder.parameters(),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.resnet101_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.resnet101.parameters()),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.cnn_vit_optimizer = torch.optim.Adam(params=self.cnn_vit.parameters(),
                                                  lr=args.encoder_lr) if args.fine_tune_encoder else None
        # self.feataug_optimizer = torch.optim.Adam(params=self.feataug.parameters(),
        #                                           lr=args.encoder_lr) if args.fine_tune_encoder else None
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr)

        # Move to GPU, if available
        self.encoder = self.encoder.cuda()
        self.resnet101 = self.resnet101.cuda()
        self.cnn_vit = self.cnn_vit.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()
        self.encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5,
                                                                    gamma=1.0) if args.fine_tune_encoder else None
        self.resnet101_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.resnet101_optimizer, step_size=5,
                                                                    gamma=1.0)
        self.cnn_vit_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.cnn_vit_optimizer, step_size=5,
                                                                    gamma=1.0)
        # self.feataug_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.feataug_optimizer, step_size=5,
        #                                                             gamma=1.0)
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_trans_optimizer, step_size=5,
                                                                          gamma=1.0)
        self.decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5,gamma=1.0)

    def training(self, args, epoch):
        self.encoder.train()
        self.resnet101.train()
        self.cnn_vit.train()
        # self.feataug.train()
        self.encoder_trans.train()
        self.decoder.train()
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.resnet101_optimizer.zero_grad()
        self.cnn_vit_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
            self.encoder_trans_optimizer.zero_grad()

        for id, batch_data in enumerate(self.train_loader):
            start_time = time.time()
            accum_steps = 64//args.train_batchsize

            imgA = batch_data['imgA']
            imgB = batch_data['imgB']
            token = batch_data['token']
            # encode_ys = batch_data['tokens_encode_ys']
            token_len = batch_data['token_len']
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            token = token.cuda()
            token_len = token_len.cuda()
            # Forward prop.
            feat = self.encoder((imgA, imgB))
            # feat1 = feat[:, :feat.shape[1] // 2, :]
            # feat2 = feat[:, feat.shape[1] // 2:, :]
            # feat = self.encoder_trans(feat1, feat2)
            # self.pltshow(feat)
            imgA_list, imgB_list = self.resnet101(imgA, imgB)
            feat1 = feat[:, :feat.shape[1] // 2, :]
            feat2 = feat[:, feat.shape[1] // 2:, :]
            feat1 = self.cnn_vit(imgA_list, feat1)
            feat2 = self.cnn_vit(imgB_list, feat2)
            # feat = self.encoder_trans(feat1, feat2)
            feat = torch.cat([feat1, feat2], 1)
            # feat = torch.cat([feat, feat_cnn], 1)
            # feat = self.feataug(feat)
            # feat = feat * feat_cnn
            # feat = self.feataug(feat)
            # feat = feat_vit + feat_cnn
            # feat = self.encoder_trans(feat1, feat2)         # 64,49,768、、、16,256,768
            scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat, token, token_len) #2,42,501   2,42
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data  #16,501
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data  #16
            # Calculate loss
            feat.retain_grad()
            feat1.retain_grad()
            loss = self.criterion_cap(scores, targets)

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_value_(self.resnet101.parameters(), args.grad_clip)
                    torch.nn.utils.clip_grad_value_(self.cnn_vit.parameters(), args.grad_clip)
            # Back prop.
            loss = loss / accum_steps
            loss.backward()
            cam_map = self.grad_cam(feat)
            self.visualize_grad_cam(cam_map, 512, batch_data)
            cam_map = self.grad_cam(feat1)
            self.visualize_grad_cam(cam_map, 512, batch_data)
            # Update weights
            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()
                    self.resnet101_optimizer.step()
                    self.cnn_vit_optimizer.step()

                # Adjust learning rate
                self.decoder_lr_scheduler.step()
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    self.encoder_lr_scheduler.step()
                    self.resnet101_lr_scheduler.step()
                    self.cnn_vit_lr_scheduler.step()

                self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()
                    self.resnet101_optimizer.zero_grad()
                    self.cnn_vit_optimizer.zero_grad()

            # Keep track of metrics
            self.hist[self.index_i, 0] = time.time() - start_time #batch_time
            self.hist[self.index_i, 1] = loss.item()  # train_loss
            self.hist[self.index_i, 2] = accuracy_v0(scores, targets, 1) # top5

            self.index_i += 1
            # Print status
            if self.index_i % args.print_freq == 0:
                print_log('Training Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Cap_loss: {4:.5f}\t'
                    'Text_Top-5 Acc: {5:.3f}'
                    .format(epoch, id, len(self.train_loader),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,0])*args.print_freq,
                                         np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,1]),
                                        np.mean(self.hist[self.index_i-args.print_freq:self.index_i-1,2])
                                ), self.log)

    # One epoch's validation
    def validation(self, epoch):
        # self.validation_test()
        word_vocab = self.word_vocab
        self.decoder.eval()
        self.encoder_trans.eval()
        self.resnet101.eval()
        self.cnn_vit.eval()
        # self.feataug.eval()
        if self.encoder is not None:
            self.encoder.eval()

        val_start_time = time.time()
        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        with torch.no_grad():
            # Batches
            for ind, batch_data in enumerate(
                    tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):

                imgA = batch_data['imgA']
                imgB = batch_data['imgB']
                token_all = batch_data['token_all']
                token_all_len = batch_data['token_all_len']
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                token_all = token_all.squeeze(0).cuda()
                # Forward prop.
                feat = self.encoder((imgA, imgB))
                feat1 = feat[:, :feat.shape[1] // 2, :]
                feat2 = feat[:, feat.shape[1] // 2:, :]
                feat = self.encoder_trans(feat1, feat2)

                seq = self.decoder.sample(feat, k=1)

                # for captioning
                except_tokens = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
                img_token = token_all.tolist()
                img_tokens = list(map(lambda c: [w for w in c if w not in except_tokens],
                        img_token))  # remove <start> and pads
                references.append(img_tokens)

                pred_seq = [w for w in seq if w not in except_tokens]
                hypotheses.append(pred_seq)
                assert len(references) == len(hypotheses)

                if ind % self.args.print_freq == 0:
                    pred_caption = ""
                    for i in pred_seq:
                        pred_caption += (list(word_vocab.keys())[i]) + " "
                    ref_caption = ""
                    for i in img_tokens:
                        for j in i:
                            ref_caption += (list(word_vocab.keys())[j]) + " "
                        ref_caption += ".    "
            val_time = time.time() - val_start_time
            # Fast test during the training
            # Calculate evaluation scores
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']
            Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']
            Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']
            Rouge = score_dict['ROUGE_L']
            Cider = score_dict['CIDEr']
            print_log('Captioning_Validation:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t' 
                'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
                .format(val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider), self.log)

        # Check if there was an improvement
        if Bleu_4 > self.best_bleu4:
            self.best_bleu4 = max(Bleu_4, self.best_bleu4)

            # save_checkpoint
            print('Save Model')
            state = {'encoder_dict': self.encoder.state_dict(),
                     'encoder_trans_dict': self.encoder_trans.state_dict(),
                     'resnet101_dict': self.resnet101.state_dict(),
                     # 'feataug_dict': self.feataug.state_dict(),
                     'cnn_vit_dict': self.cnn_vit.state_dict(),
                     'decoder_dict': self.decoder.state_dict()
                     }
            metric = f'Bleu4_{round(100000 * self.best_bleu4)}'
            model_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_epo{epoch}_{metric}.pth'
            # if epoch > 1:
            torch.save(state, os.path.join(self.args.savepath, model_name.replace('/','-')))
            # save a txt file
            text_path = os.path.join(self.args.savepath, model_name.replace('/','-'))
            with open(text_path.replace('.pth', '.txt'), 'w') as f:
                f.write('Bleu_1: ' + str(Bleu_1) + '\t')
                f.write('Bleu_2: ' + str(Bleu_2) + '\t')
                f.write('Bleu_3: ' + str(Bleu_3) + '\t')
                f.write('Bleu_4: ' + str(Bleu_4) + '\t')
                f.write('Meteor: ' + str(Meteor) + '\t')
                f.write('Rouge: ' + str(Rouge) + '\t')
                f.write('Cider: ' + str(Cider) + '\t')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Changes_to_Captions')

    # Data parametersLEVIR_CC----------new_Levir
    parser.add_argument('--sys', default='win', choices=('linux'), help='system')
    parser.add_argument('--data_folder', default='/workstation/Chg2Cap/data/LEVIR_CC/images',help='folder with data files')
    parser.add_argument('--list_path', default='/workstation/Chg2Cap/data/LEVIR_CC/', help='path of the data lists')
    parser.add_argument('--token_folder', default='/workstation/Chg2Cap/data/LEVIR_CC/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=42, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_CC",help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=2, help='gpu id in the training.')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    # Training parameters
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=1, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=80, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--workers', type=int, default=16, help='for data-loading; right now, o                           nly 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value of.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--decoder_type', default='transformer_decoder', help='mamba or gpt or transformer_decoder')
    # Validation
    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default="/workstation1/wrj/Chg2cap/models_ckpt/")
    # backbone parameters
    parser.add_argument('--network', default='CLIP-ViT-B/32', help=' define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=768, help='the dim of extracted features of backbone ')
    parser.add_argument('--feat_size', type=int, default=16, help='size of extracted features of backbone')
    # Model parameters
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=768, help='embedding dimension')
    args = parser.parse_args()


    if args.network == 'CLIP-RN50':
        clip_emb_dim = 1024
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN101':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 2048, 7
    elif args.network == 'CLIP-RN50x4':
        clip_emb_dim = 640
        args.encoder_dim, args.feat_size = 2560, 9
    elif args.network == 'CLIP-RN50x16':
        clip_emb_dim = 768
        args.encoder_dim, args.feat_size = 3072, 12
    elif args.network == 'CLIP-ViT-B/16' or args.network == 'CLIP-ViT-L/16':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 14
    elif args.network == 'CLIP-ViT-B/32' or args.network == 'CLIP-ViT-L/32':
        clip_emb_dim = 512
        args.encoder_dim, args.feat_size = 768, 16   # 768, 7
    elif args.network == 'segformer-mit_b1':
        args.encoder_dim, args.feat_size = 512, 8

    args.embed_dim = args.encoder_dim

    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
        trainer.training(trainer.args, epoch)
        trainer.validation(epoch)
        # trainer.validation_test(epoch)

