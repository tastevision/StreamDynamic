#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) DAMO Academy, Alibaba Group and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torch.nn import functional as F
import numpy as np
from torch import nn

# from yolox.data import DataPrefetcher
from .longshort_data_prefetcher import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    # ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from .ema import ModelEMA


class Trainer:
    def __init__(self, exp: Exp, args, branch_num):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        self.branch_num = branch_num

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.speed_router_scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # 速度判断器的损失，需要在这里定义，在训练过程中生成必要的监督信息
        self.speed_lossfn = nn.KLDivLoss(reduction="batchmean", log_target=True)

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        self.ignore_keys = ["backbone_t", "head_t"]

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        """
        在这里，根据self.epoch判断，训练2轮主结构(此时冻结speed_router)，训练1轮speed_router(此时冻结主结构)
        """
        if self.epoch == 0:
            logger.info("训练主结构，分支判断由随机数给出，保证基础的收敛状态")
            self.model.module.freeze_speed_detector()
            for self.iter in range(self.max_iter):
                self.before_iter()
                self.train_one_iter_mode_1()
                self.after_iter()
            self.model.module.unfreeze_speed_detector()
        elif self.epoch % 2 == 0 and self.epoch != 0:
            logger.info("训练speed_router，冻结主结构")
            self.model.module.freeze_main_model()
            for self.iter in range(self.max_iter):
                self.before_iter()
                self.train_one_iter_mode_2()
                self.after_iter_2()
            self.model.module.unfreeze_main_model()
        else:
            logger.info("训练主结构，分支判断由speed_router给出，同时冻结speed_router")
            self.model.module.freeze_speed_detector()
            for self.iter in range(self.max_iter):
                self.before_iter()
                self.train_one_iter_mode_3()
                self.after_iter()
            self.model.module.unfreeze_speed_detector()

    def train_one_iter_mode_1(self):
        """
        训练主结构，分支判断由随机数给出，保证基础的收敛状态
        """
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        # inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        # targets.requires_grad = False
        inps = [inps[0].to(self.data_type), inps[1].to(self.data_type)]
        targets = (targets[0].to(self.data_type), targets[1].to(self.data_type))
        targets[0].requires_grad = False
        targets[1].requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        N = np.random.randint(self.branch_num)

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, N_frames=N)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def train_one_iter_mode_2(self):
        """
        训练speed_router，冻结主结构
        """
        inps, targets = self.prefetcher.next()
        # inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        # targets.requires_grad = False
        inps = [inps[0].to(self.data_type), inps[1].to(self.data_type)]
        targets = (targets[0].to(self.data_type), targets[1].to(self.data_type))
        targets[0].requires_grad = False
        targets[1].requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)

        branch_compute_time = [1e2 for _ in range(self.branch_num)]
        branch_total_loss = [1e10 for _ in range(self.branch_num)]

        for i in range(self.branch_num):
            beg = time.time()

            # 计算各个分支的损失
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model(inps, targets, N_frames=i)
            loss = outputs["total_loss"]
            branch_total_loss[i] = loss
            end = time.time()

            # 伪更新，用于抑制报错
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 计算各个分支计算的耗时
            branch_compute_time[i] = end - beg

        # 计算出训练速度判断器需要的监督信息
        branch_compute_time = torch.tensor(branch_compute_time).to(self.device)
        branch_total_loss = torch.tensor(branch_total_loss).to(self.device)
        speed_supervision = F.softmax(branch_total_loss * branch_compute_time, dim=0) # 同时在计算速度和损失上达到最小的那个分支，被视为最合适的分支

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            speed_score = self.model.module.compute_speed_score(inps)
            speed_supervision = speed_supervision.repeat(speed_score.size()[0], 1)
            speed_loss = self.speed_lossfn(speed_score, speed_supervision)
        self.speed_router_optimizer.zero_grad()
        self.speed_router_scaler.scale(speed_loss).backward()
        self.speed_router_scaler.step(self.speed_router_optimizer)
        self.speed_router_scaler.update()
        logger.info(f"speed detector loss: f{speed_loss}")


    def train_one_iter_mode_3(self):
        """
        训练主结构，分支判断由speed_router给出，同时冻结speed_router
        """
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        # inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        # targets.requires_grad = False
        inps = [inps[0].to(self.data_type), inps[1].to(self.data_type)]
        targets = (targets[0].to(self.data_type), targets[1].to(self.data_type))
        targets[0].requires_grad = False
        targets[1].requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            # 用model内的speed_router给出分支数
            speed_score = self.model.module.compute_speed_score(inps)
            N = int(speed_score.argmin(dim=1)[0])
            outputs = self.model(inps, targets, N_frames=N)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # 需要在这里特别地将model.long_backbone加载到gpu上
        for m in model.jian0:
            m.to(self.device)
        for m in model.jian1:
            m.to(self.device)
        for m in model.jian2:
            m.to(self.device)
        model.speed_detector.to(self.device)
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size, ignore_keys=self.ignore_keys)
        self.speed_router_optimizer = torch.optim.Adam(model.speed_detector.parameters())

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998, ignore_keys=self.ignore_keys)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.eval_batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                wandb_params = dict()
                for k, v in zip(self.args.opts[0::2], self.args.opts[1::2]):
                    if k.startswith("wandb-"):
                        wandb_params.update({k.lstrip("wandb-"): v})
                self.wandb_logger = WandbLogger(config=vars(self.exp), **wandb_params)
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "wandb":
                    self.wandb_logger.log_metrics({k: v.latest for k, v in loss_meter.items()})
                    self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    def after_iter_2(self):
        """
        `after_iter` contains two parts of logic:
            * reset setting of resize
        """
        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt
            teacher_ckpt_file = self.args.teacher_ckpt
            teacher_ckpt = torch.load(teacher_ckpt_file, map_location=self.device)["model"]

            ckpt = torch.load(ckpt_file, map_location=self.device)
            for k, v in teacher_ckpt.items():
                if "head." in k:
                    k_new = f'head_t.{k[len("head."):]}'
                elif "backbone." in k:
                    k_new = f'backbone_t.{k[len("backbone."):]}'
                else:
                    raise Exception("Load teacher ckpt error.")
                ckpt["model"][k_new] = v
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                ckpt = self.update_ckpt(ckpt)
                teacher_ckpt_file = self.args.teacher_ckpt
                teacher_ckpt = torch.load(teacher_ckpt_file, map_location=self.device)["model"]

                for k, v in teacher_ckpt.items():
                    if "head." in k:
                        k_new = f'head_t.{k[len("head."):]}'
                    elif "backbone." in k:
                        k_new = f'backbone_t.{k[len("backbone."):]}'
                    else:
                        raise Exception("Load teacher ckpt error.")
                    ckpt[k_new] = v
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            ap50_95, ap50, summary = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "epoch": self.epoch + 1,
                })
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            save_state_dict = dict()
            for k, v in save_model.state_dict().items():
                ig_flag = False
                for ig_k in self.ignore_keys:
                    if ig_k in k:
                        ig_flag = True
                        break
                if ig_flag:
                    continue
                save_state_dict[k] = v
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(self.file_name, ckpt_name, update_best_ckpt)

    def update_ckpt(self, ckpt):

        res_ckpt = ckpt.copy()

        for k, v in ckpt.items():
            if k.startswith("backbone"):
                res_ckpt[f'short_{k}'] = v
                res_ckpt[f'long_{k}'] = v
            if k.startswith("neck"):
                res_ckpt[f'backbone.{k}'] = v

        return res_ckpt
