#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeedDetector(nn.Module):
    """
    速度检测器
    """
    def __init__(self):
        super().__init__()
        # 定义一个简单的二维分类模型

    def forward(self, x):
        """
        这里的x包含了当前帧和过去帧，返回一个评分
        """
        # 
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

        # # 保存一下图像，确定得到的图像是没有问题的
        # img1 = np.transpose(x[0][0,:-3,...].detach().cpu().numpy().astype(np.uint8), (1, 2, 0))
        # img2 = np.transpose(x[1][0,:3,...].detach().cpu().numpy().astype(np.uint8), (1, 2, 0))
        # cv.imwrite("/tmp/img1.jpg", img1)
        # cv.imwrite("/tmp/img2.jpg", img2)

        frame_diff = current_frame - history_frame

        # 之后如果直接分类，那么整个batch中的每一个sample都有一个速度评分，这样的话，单个sample就存在频繁切换分支的情况，所以可以考虑直接同一个batch输出一个单独的速度评分
        
        # 用x计算出帧间差
        return 0.8
