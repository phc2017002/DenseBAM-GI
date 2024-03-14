import math
from typing import Tuple

#import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops.einops import rearrange
from torch import FloatTensor, LongTensor

#from .pos_enc import ImgPosEnc


# DenseNet-B
class _Bottleneck(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_Bottleneck, self).__init__()
        interChannels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(n_channels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(
            interChannels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class _SingleLayer(nn.Module):
    def __init__(self, n_channels: int, growth_rate: int, use_dropout: bool):
        super(_SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class _Transition(nn.Module):
    def __init__(self, n_channels: int, n_out_channels: int, use_dropout: bool):
        super(_Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_out_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class _ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class _SpatialAttention(nn.Module):
    def __init__(self):
        super(_SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

num_features1 = 288

num_features2 = 256

num_features3 = 1024

class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int,
        num_layers: int,
        reduction: float = 0.5,
        bottleneck: bool = True,
        use_dropout: bool = True,
    ):
        super(DenseNet, self).__init__()
        #n_dense_blocks = num_layers
        n_dense_blocks_1 = 6
        n_dense_blocks_2 = 12
        n_dense_blocks_3 = 24

        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            2, n_channels, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.dense1 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks_1, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks_1 * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks_2, bottleneck, use_dropout
        )
        n_channels += n_dense_blocks_2 * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = _Transition(n_channels, n_out_channels, use_dropout)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(
            n_channels, growth_rate, n_dense_blocks_3, bottleneck, use_dropout
        )

        self.out_channels = n_channels + n_dense_blocks_3* growth_rate
        self.post_norm = nn.BatchNorm2d(self.out_channels)

        self.channel_attention1 = _ChannelAttention(num_features1)
        self.spatial_attention1 = _SpatialAttention()

        self.channel_attention2 = _ChannelAttention(num_features2)
        self.spatial_attention2 = _SpatialAttention()

        self.channel_attention3 = _ChannelAttention(num_features3)
        self.spatial_attention3 = _SpatialAttention()

    @staticmethod
    def _make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(_Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(_SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)
    


    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        
        '''
        channel_attention = self.channel_attention1(out)
        out = out * channel_attention

        spatial_attention = self.spatial_attention1(out)
        out = out * spatial_attention
        '''

        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)

        
        channel_attention = self.channel_attention2(out)
        out = out * channel_attention

        spatial_attention = self.spatial_attention2(out)
        out = out * spatial_attention
        

        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.post_norm(out)

        #print('out.shape-----',out.shape)
        
        channel_attention = self.channel_attention3(out)
        out = out * channel_attention

        spatial_attention = self.spatial_attention3(out)
        out = out * spatial_attention
        

        return out
