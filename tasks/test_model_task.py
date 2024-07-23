#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: test_model_task.py
@Author: Jiyufei
@Date: 2024/07/14
"""
import os
import shutil
from typing import Union

import torch
import json
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import TaskType, LoadConfigResultDate
from common import WaymoDataset
from net_works import BackBone
from tasks import BaseTask
from utils import MathUtil, VisualizeUtil
from data.longtail_dataload import longtailtrajDataset

class TestModelTask(BaseTask):

    # (Your existing __init__ and execute methods)
    def __init__(self):
        super(TestModelTask, self).__init__()
        self.task_type = TaskType.Test_MODEl
        self.device = torch.device("cpu")
        self.multi_gpus = False
        self.gpu_ids = list()
    def execute(self,result_info: LoadConfigResultDate,current_time):
        train_model_config = result_info.train_model_config
         # 初始化device
        if train_model_config.use_gpu:
            if train_model_config.gpu_ids:
                self.device = torch.device(f"cuda:{train_model_config.gpu_ids[0]}")
                self.multi_gpus = True
                self.gpu_ids = train_model_config.gpu_ids
            else:
                self.device = torch.device('cuda')
        # 初始化dataloader
        # waymo_dataset = WaymoDataset(
        #     train_dir, train_model_config.his_step, train_model_config.max_pred_num,
        #     train_model_config.max_other_num, train_model_config.max_traffic_light,
        #     train_model_config.max_lane_num, train_model_config.max_point_num
        # )
        # data_loader = DataLoader(
        #     waymo_dataset,
        #     shuffle=False,
        #     batch_size=train_model_config.batch_size,
        #     num_workers=train_model_config.num_works,
        #     pin_memory=True,
        #     drop_last=False
        # )

        # ours Dataset
        with open('data/test_date_all.json', 'r', encoding='utf-8') as file:  
            # 使用json.load()方法解析JSON数据  
            train_data = json.load(file) 

        long_tail_dataset = longtailtrajDataset(train_data)
        data_loader = DataLoader(
            long_tail_dataset,
            shuffle=False,
            batch_size=train_model_config.batch_size,
            num_workers=train_model_config.num_works,
            pin_memory=True,
            drop_last=False
        )
        model = self.init_model(result_info).to(self.device)
      
      
        epoch_step = len(long_tail_dataset) // train_model_config.batch_size
        if epoch_step == 0:
            raise ValueError("dataset is too small, epoch_step = 0")
      
        total_diffusion_loss = 0
        total_traj_loss = 0
        total_confidence_loss = 0
        total_loss = 0
        num_batches = len(data_loader)

        with torch.no_grad():  # 在验证期间不需要计算梯度
            gt = {}
            pred = {}
            for iteration,data in enumerate(data_loader):
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(self.device).to(torch.float32)
                diffusion_loss, traj_loss, confidence_loss,min_loss_traj,pred_traj,pred_traj_confidence = model(data)
                diffusion_loss = diffusion_loss.mean()
                traj_loss = traj_loss.mean()
                confidence_loss = confidence_loss.mean()
                total_loss += (diffusion_loss + traj_loss + confidence_loss).item()
                total_diffusion_loss += diffusion_loss.item()
                total_traj_loss += traj_loss.item()
                total_confidence_loss += confidence_loss.item()

                gt[iteration] = data['predicted_future_traj'].detach().cpu().numpy().tolist()
                pred[iteration] = {'pred_traj':pred_traj.detach().cpu().numpy().tolist(),
                                   'confidence':pred_traj_confidence.detach().cpu().numpy().tolist()}

        avg_loss = total_loss / num_batches
        avg_diffusion_loss = total_diffusion_loss / num_batches
        avg_traj_loss = total_traj_loss / num_batches
        avg_confidence_loss = total_confidence_loss / num_batches

        output_file = os.path.join('output/reulsts',current_time)
        os.makedirs(output_file)
        with open(os.path.join(output_file,'test_gt.json'), 'w') as f:
            json.dump(gt, f, indent=4)
            
        with open(os.path.join(output_file,'test_pred.json'), 'w') as f:
            json.dump(pred, f, indent=4)   
            


        print(
            'total_loss',avg_loss,
            'diffusion_loss', avg_diffusion_loss,
            'traj_loss', avg_traj_loss,
            'confidence_loss', avg_confidence_loss
        )
    

    def init_model(self, result_info: LoadConfigResultDate) -> Union[BackBone, nn.DataParallel]:
        train_model_config = result_info.train_model_config
        task_config = result_info.task_config
        # 初始化diffusion的betas
        if train_model_config.schedule == "cosine":
            betas = MathUtil.generate_cosine_schedule(train_model_config.time_steps)
        else:
            schedule_low = 1e-4
            schedule_high = 0.008
            betas = MathUtil.generate_linear_schedule(
                train_model_config.time_steps,
                schedule_low * 1000 / train_model_config.time_steps,
                schedule_high * 1000 / train_model_config.time_steps,
            )
        model = BackBone(betas) # BackBone在net_works中定义
        # 预训练模型参数
        if task_config.pre_train_model:
            pre_train_model_path = task_config.pre_train_model
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pre_train_model_path)
            # 模型参数赋值
            new_model_dict = dict()
            for key in model_dict.keys():
                if ("module." + key) in pretrained_dict:
                    new_model_dict[key] = pretrained_dict["module." + key]
                elif key in pretrained_dict:
                    new_model_dict[key] = pretrained_dict[key]
                else:
                    print("key: ", key, ", not in pretrained")
            model.load_state_dict(new_model_dict)
            result_info.task_logger.logger.info("load pre_train_model success")
        model = model.to(self.device)
        if self.multi_gpus:
            model = nn.DataParallel(model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])
        return model
