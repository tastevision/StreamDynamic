#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# class SpeedDetector(nn.Module):
#     """
#     速度检测器
#     """
#     def __init__(self, target_size):
#         """
#         由于帧间差在很小的尺寸下就可以体现出速度，这里考虑先对输入图像进行resize
#         """
#         super().__init__()
#         self.target_size = target_size
#         self.resize = transforms.Resize(list(target_size))
#         # 定义一个简单的二维分类模型
#         self.extractor = nn.Sequential(
#             nn.Conv2d(3, 8, 1, bias=True),  # (16, w - 2, h - 2)
#             nn.AvgPool2d(2),      # (16, (w - 2) // 2, (h - 2) // 2)
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, 1, 1, bias=True),  # (4, ((w - 2) // 2) - 2, ((h - 2) // 2) - 2)
#             nn.AvgPool2d(2),      # (4, (((w - 2) // 2) - 2) // 2, (((h - 2) // 2) - 2) // 2)
#             nn.Flatten(1),
#         )
#         # 这里存在可以改进的地方，目前这个维数是不可自动调整的
#         self.head = nn.Sequential(
#             nn.Linear(625, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         """
#         这里的x包含了当前帧和过去帧，返回一个评分
#         """
#         """
#         [batchsize, channel, height, width]
#         x[0][:,:-3,...]   : 当前帧(t)
#         x[1][:,:3,...]    : t - 1
#         x[1][:,3:6,...]   : t - 2
#         x[1][:,6:9,...]   : t - 3
#         x[1][:,9:12,...]  : t - 4
#         ...
#         """
#         current_frame = x[0][:,:-3,...]  # frame t
#         history_frame = x[1][:,:3,...] # history frame
#         if current_frame.size() != history_frame.size():
#             frame_diff = self.resize(history_frame)
#         else:
#             # 用x计算出帧间差
#             frame_diff = self.resize(current_frame - history_frame)
#         features = self.extractor(frame_diff)
#         scores = self.head(features)
#         # 之后如果直接分类，那么整个batch中的每一个sample都有一个速度评分，
#         # 这样的话，单个sample就存在频繁切换分支的情况，所以可以考虑直接同一
#         # 个batch输出一个单独的速度评分
#         return torch.mean(scores)

class SpeedDetector(nn.Module):
    """
    速度检测器
    """
    def __init__(self, target_size):
        """
        由于帧间差在很小的尺寸下就可以体现出速度，这里考虑先对输入图像进行resize
        """
        super().__init__()
        self.target_size = target_size
        self.resize = transforms.Resize(list(target_size))

    def forward(self, x):
        """
        这里的x包含了当前帧和过去帧，返回一个评分
        在这个版本中，直接不学习其中的特征，只通过帧间差完成评分
        """
        """
        [batchsize, channel, height, width]
        x[0][:,:-3,...]   : 当前帧(t)
        x[1][:,:3,...]    : t - 1
        x[1][:,3:6,...]   : t - 2
        x[1][:,6:9,...]   : t - 3
        x[1][:,9:12,...]  : t - 4
        ...
        """
        current_frame = x[0][:,:-3,...]  # frame t
        history_frame = x[1][:,:3,...] # history frame
        if current_frame.size() != history_frame.size():
            current_frame = x[0]
        current_frame = current_frame / 255.0
        history_frame = history_frame / 255.0
        
        # frame_diff = self.resize(history_frame)
        # if current_frame.size() != history_frame.size():
        #     frame_diff = self.resize(history_frame)
        # else:
        #     # 用x计算出帧间差
        #     frame_diff = self.resize(((current_frame - history_frame) + 1.0) / 2.0)
        #
        # # 去掉过大和过小的值，否则其均值会没有变化
        # factor = 0.2
        # frame_diff[frame_diff < factor] = 0
        # frame_diff[frame_diff > (1 - factor)] = 0
        # frame_diff = (frame_diff - factor) / (1 - 2 * factor)

        frame_diff = torch.abs(torch.mean(current_frame, dim=(1, 2, 3)) - torch.mean(history_frame, dim=(1, 2, 3))) * 50
        return frame_diff.max()
