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
        pass

    def forward(self, x):
        """
        这里的x包含了当前帧和过去帧，返回一个评分
        """
        return 0.2
