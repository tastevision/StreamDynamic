#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.

import torch
import torch.nn as nn
import numpy as np

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn_long import DFPPAFPNLONG
from exps.model.dfp_pafpn_short import DFPPAFPNSHORT

from yolox.models.network_blocks import BaseConv


class YOLOXLONGSHORTV3ODDIL(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(
        self, 
        speed_detector: nn.Module,  # 环境速度检测器
        long_backbone=None, 
        short_backbone=None, 
        backbone_neck=None,
        head=None, 
        backbone_t=None,
        head_t=None,
        merge_form="add", 
        in_channels=[256, 512, 1024], 
        width=1.0, 
        act="silu",
        with_short_cut=False,
        long_cfg=None,
        coef_cfg=None,
        dil_loc="head",
        # still_teacher=True,
    ):
        """Summary
        
        Args:
            long_backbone (None, optional): Description
            short_backbone (None, optional): Description
            head (None, optional): Description
            merge_form (str, optional): "add" or "concat" or "pure_concat" or "long_fusion"
            in_channels (list, optional): Description
        """
        super().__init__()
        if short_backbone is None:
            short_backbone = DFPPAFPNSHORT()
        if head is None:
            head = TALHead(20)

        self.dil_loc = dil_loc 
        self.speed_detector = speed_detector
        self.long_backbone = long_backbone
        self.short_backbone = short_backbone
        self.backbone = backbone_neck
        self.head = head
        self.backbone_t = backbone_t # teacher model
        self.head_t = head_t
        self.merge_form = merge_form
        self.in_channels = in_channels
        self.with_short_cut = with_short_cut
        self._freeze_teacher_model()
        self._set_eval_teacher_model()
        # self.dil_loss = nn.MSELoss(reduction='sum')
        # coef_cfg = coef_cfg if coef_cfg is not None else dict()
        # self.dil_loss_coef = coef_cfg.get("dil_loss_coef", 1.)
        # self.det_loss_coef = coef_cfg.get("det_loss_coef", 1.)
        # self.reg_coef = coef_cfg.get("reg_coef", 1.)
        # self.cls_coef = coef_cfg.get("cls_coef", 1.)
        # self.obj_coef = coef_cfg.get("obj_coef", 1.)
        self.long_cfg = long_cfg

        if merge_form == "concat":
            self.jian2 = BaseConv(
                in_channels=int(in_channels[0] * width),
                out_channels=int(in_channels[0] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            )

            self.jian1 = BaseConv(
                in_channels=int(in_channels[1] * width),
                out_channels=int(in_channels[1] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            )

            self.jian0 = BaseConv(
                in_channels=int(in_channels[2] * width),
                out_channels=int(in_channels[2] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            )
        elif merge_form == "long_fusion":
            self.jian2 = [BaseConv(
                in_channels=sum([x[0][0] * (i + 1) for x in self.long_cfg["out_channels"]]),
                out_channels=int(in_channels[0] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            ) for i in range(self.long_cfg["frame_num"])]

            self.jian1 = [BaseConv(
                in_channels=sum([x[0][1] * (i + 1) for x in self.long_cfg["out_channels"]]),
                out_channels=int(in_channels[1] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            ) for i in range(self.long_cfg["frame_num"])]

            self.jian0 = [BaseConv(
                in_channels=sum([x[0][2] * (i + 1) for x in self.long_cfg["out_channels"]]),
                out_channels=int(in_channels[2] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            ) for i in range(self.long_cfg["frame_num"])]

    def compute_speed_score(self, x): # 计算速度评分
        speed_score = self.speed_detector(x) # 通过x，即多帧的结果计算得到速度评分
        return speed_score

    def forward(self, x, targets=None, buffer=None, mode='off_pipe', N_frames=None, train_router=False):
        # fpn output content features of [dark3, dark4, dark5]
        """
        branch_num: 指定走哪条分支
        train_router: 是否训练router
        """
        assert mode in ['off_pipe', 'on_pipe']
        outputs = dict()

        if self.training and not train_router: # 训练时，随机选择一条路径进行训练
            # 在这里需要解冻主结构，以便避免主结构无法更新
            assert N_frames is not None
            N = N_frames
        elif train_router:
            speed_score = self.compute_speed_score(x) # 一个向量，长度和branch的数量一致
            return speed_score
        else: # 测试时，按照speed_detector给出的结果进行测试
            speed_score = self.compute_speed_score(x) # 一个向量，长度和branch的数量一致
            N = int(speed_score.argmin(dim=1)[0])
            # N = int(len(self.long_cfg) * speed_score) # 选择第几条分支
            # if N >= len(self.long_cfg):
            #     N = len(self.long_cfg) - 1

        if mode == 'off_pipe':
            if self.training:
                # import pdb; pdb.set_trace()
                # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
                short_fpn_outs, rurrent_pan_outs = self.short_backbone(x[0][:,:-3,...], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone)
                fpn_outs_t = self.backbone_t(x[0][:,-3:,...], buffer=buffer, mode='off_pipe')
                long_fpn_outs = self.long_backbone(x[1][:,:(N+1)*3,...], N + 1, buffer=buffer, mode='off_pipe', backbone_neck=self.backbone) if self.long_backbone is not None else None,
            else:
                short_fpn_outs, rurrent_pan_outs = self.short_backbone(x[0], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone)
                long_fpn_outs = self.long_backbone(x[1][:,:(N+1)*3,...], N + 1, buffer=buffer, mode='off_pipe', backbone_neck=self.backbone) if self.long_backbone is not None else None

            if not self.with_short_cut:
                if self.long_backbone is None:
                    fpn_outs = short_fpn_outs
                else:
                    if self.merge_form == "add":
                        fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                    elif self.merge_form == "concat":
                        if isinstance(long_fpn_outs[0], tuple):
                            fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0][0])], dim=1)
                            fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[0][1])], dim=1)
                            fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[0][2])], dim=1)
                        else:
                            fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == "pure_concat":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == "long_fusion":
                        if isinstance(long_fpn_outs[0], tuple):
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0][0])], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[0][1])], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[0][2])], dim=1)
                        else:
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    else:
                        raise Exception(f'merge_form must be in ["add", "concat"]')
            else:
                if self.long_backbone is None:
                    fpn_outs = [x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)]
                else:
                    if self.merge_form == "add":
                        fpn_outs = [x + y + z for x, y, z in zip(short_fpn_outs, long_fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "concat":
                        if isinstance(long_fpn_outs[0], tuple):
                            fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0][0])], dim=1)
                            fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[0][1])], dim=1)
                            fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[0][2])], dim=1)
                        else:
                            fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "pure_concat":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "long_fusion":
                        if isinstance(long_fpn_outs[0], tuple):
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0][0])], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[0][1])], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[0][2])], dim=1)
                        else:
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    else:
                        raise Exception(f'merge_form must be in ["add", "concat"]')

            if self.training:
                assert targets is not None
                # (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), reg_outputs, obj_outputs, cls_outputs = self.head(
                #     fpn_outs, targets, x
                # )
                bbox_preds_t, obj_preds_t, cls_preds_t = self.head_t(fpn_outs_t)
                
                if self.dil_loc == "head":
                    knowledge = (
                        bbox_preds_t, 
                        obj_preds_t, 
                        cls_preds_t
                    )
                    (
                        loss, 
                        iou_loss, 
                        conf_loss, 
                        cls_loss, 
                        l1_loss, 
                        reg_dil_loss,
                        obj_dil_loss,
                        cls_dil_loss,
                        loss_dil_hint,
                        num_fg
                    )   = self.head(
                        fpn_outs, 
                        targets, 
                        x,
                        knowledge=knowledge
                    )

                    losses = {
                        "total_loss": loss,
                        # "det_loss": loss,
                        "iou_loss": iou_loss,
                        "l1_loss": l1_loss,
                        "conf_loss": conf_loss,
                        "cls_loss": cls_loss,
                        # "dil_loss": dil_loss,
                        # "neck_dil_loss":neck_dil_loss,
                        "reg_dil_loss": reg_dil_loss,
                        "cls_dil_loss": cls_dil_loss,
                        "obj_dil_loss": obj_dil_loss,
                        "loss_dil_hint":loss_dil_hint,
                        "num_fg": num_fg,
                    }

                    outputs.update(losses)

            else:
                outputs = self.head(fpn_outs)

            return outputs
        elif mode == 'on_pipe':
            # fpn_outs, buffer_ = self.backbone(x,  buffer=buffer, mode='on_pipe')
            buffer_ = None
            fpn_outs = self.backbone(x)
            outputs = self.head(fpn_outs)
            
            return outputs, buffer_

    def _freeze_teacher_model(self):
        for name, param in self.backbone_t.named_parameters():
            param.requires_grad = False
        for name, param in self.head_t.named_parameters():
            param.requires_grad = False

    def _set_eval_teacher_model(self):
        self.backbone_t.eval()
        self.head_t.eval()

    def _get_tensors_numel(self, tensors):
        num = 0
        for t in tensors:
            num += torch.numel(t)
        return num


