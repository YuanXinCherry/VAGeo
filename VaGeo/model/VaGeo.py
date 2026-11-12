# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .darknet import *
import torchvision.models as models


class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.base_model = models.resnet18(weights=None)
        # your path
        state_dict = torch.load('/mnt/data/cjr/VaGeo/VaGeo/saved_models/resnet18-f37072fd.pth')
        self.base_model.load_state_dict(state_dict)
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        # print(('x1', x.shape), flush=True)
        x = self.base_model.layer2(x)
        # print(('x2', x.shape), flush=True)
        x = self.base_model.layer3(x)
        # print(('x3', x.shape), flush=True)
        x = self.base_model.layer4(x)
        # print(('x4', x.shape), flush=True)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output shape will be [B, C, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        avg_out = self.global_avg_pool(x).view(B, C)  # [B, C]
        avg_out = self.fc(avg_out)  # [B, C]
        avg_out = avg_out.view(B, C, 1, 1)  # [B, C, 1, 1]
        return x * avg_out  # [B, C, H, W]


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # Convolutional layer that generates the spatial attention map
        # The convolution takes the two concatenated channels as input and produces one output channel.
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # [B, 1, H, W]
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute the channel-wise mean and maximum of the input feature map.
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenate the mean and maximum feature maps.
        out = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # Generate the spatial attention map with a convolutional layer.
        out = self.conv1(out)  # [B, 1, H, W]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.sigmoid(out)  # [B, 1, H, W]

        # Generate spatial attention weights with the sigmoid function.
        return x * out  # [B, C, H, W]


class CrossViewFusionModule(nn.Module):
    def __init__(self):
        super(CrossViewFusionModule, self).__init__()

    # normlized global_query:B, D
    # normlized value: B, D, H, W
    def forward(self, global_query, value):
        global_query = F.normalize(global_query, p=2, dim=-1)
        value = F.normalize(value, p=2, dim=1)

        B, D, W, H = value.shape
        new_value = value.permute(0, 2, 3, 1).view(B, W * H, D)
        score = torch.bmm(global_query.view(B, 1, D), new_value.transpose(1, 2))
        score = score.view(B, W * H)
        with torch.no_grad():
            score_np = score.clone().detach().cpu().numpy()
            max_score, min_score = score_np.max(axis=1), score_np.min(axis=1)

        attn = Variable(torch.zeros(B, H * W).cuda())
        for ii in range(B):
            attn[ii, :] = (score[ii] - min_score[ii]) / (max_score[ii] - min_score[ii])

        attn = attn.view(B, 1, W, H)
        context = attn * value
        return context, attn

class VaGeo(nn.Module):
    def __init__(self, emb_size=512, leaky=True):
        super(VaGeo, self).__init__()
        ## Visual model
        self.query_resnet = MyResnet()

        self.reference_darknet = Darknet(config_path='./model/yolov3_rs.cfg')
        self.reference_darknet.load_weights('./saved_models/yolov3.weights')

        use_instnorm = False

        self.combine_clickptns_conv = ConvBatchNormReLU(4, 3, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm)
        self.crossview_fusionmodule = CrossViewFusionModule()

        self.query_visudim = 512
        self.reference_visudim = 512

        self.query_mapping_visu = ConvBatchNormReLU(self.query_visudim, emb_size, 1, 1, 0, 1, leaky=leaky,
                                                    instance=use_instnorm)
        self.reference_mapping_visu = ConvBatchNormReLU(self.reference_visudim, emb_size, 1, 1, 0, 1, leaky=leaky,
                                                        instance=use_instnorm)

        self.channel_attention = ChannelAttention(512)
        self.spatial_attention = SpatialAttention(512)

        ## output head
        self.fcn_out = torch.nn.Sequential(
            ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky, instance=use_instnorm),
            nn.Conv2d(emb_size // 2, 9 * 5, kernel_size=1))

    def forward(self, query_imgs, reference_imgs, mat_clickptns):
        mat_clickptns = mat_clickptns.unsqueeze(1)

        query_imgs = self.combine_clickptns_conv(torch.cat((query_imgs, mat_clickptns), dim=1))
        query_fvisu = self.query_resnet(query_imgs)

        reference_raw_fvisu = self.reference_darknet(reference_imgs)
        reference_fvisu = reference_raw_fvisu[1]

        query_fvisu = self.query_mapping_visu(query_fvisu)
        reference_fvisu = self.reference_mapping_visu(reference_fvisu)

        query_fvisu = self.channel_attention(query_fvisu)
        query_fvisu = self.spatial_attention(query_fvisu)

        B, D, Hquery, Wquery = query_fvisu.shape
        B, D, Hreference, Wreference = reference_fvisu.shape

        # cross-view fusion
        query_gvisu = torch.mean(query_fvisu.view(B, D, Hquery * Wquery), dim=2, keepdims=False).view(B, D)
        fused_features, attn_score = self.crossview_fusionmodule(query_gvisu, reference_fvisu)
        attn_score = attn_score.squeeze(1)

        outbox = self.fcn_out(fused_features)
        return outbox, attn_score