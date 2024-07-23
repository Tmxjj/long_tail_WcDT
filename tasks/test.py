# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tester class of Trajectron++ framework.

    Some of the functions are based on the evaluation code under
    thirdparty/Trajectron_plus_plus/experiments/pedestrians/evaluate.py
    of the following repository:
    https://github.com/StanfordASL/Trajectron-plus-plus,
    see LICENSE under thirdparty/Trajectron_plus_plus/LICENSE for usage."""
import copy
import json
import logging
import os
import pathlib
import random
import time

import dill
import joblib
import model.dataset as dataset
import model.model_registrar
import numpy as np
import tensorboardX
import torch
import tqdm
import tensorflow as tf
from common import TaskType, LoadConfigResultDate
from net_works import BackBone
from tasks import BaseTask
from utils import DataUtil, MathUtil, MapUtil
from metrics.eval_forecasting import get_ade, get_fde, get_displacement_errors_and_miss_rate
# from common import utilities 


from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.utils.sim_agents import visualizations, submission_specs




class Tester(object):
    def __init__(self):
        super(Tester, self).__init__()
        self.task_type = TaskType.EVAL_MODEL
        # self.load_model = self.load_pretrain_model(result_info)


    def execute(self, result_info: LoadConfigResultDate):
        self.run(result_info)

    @staticmethod
    def load_pretrain_model(result_info: LoadConfigResultDate) -> BackBone:
        betas = MathUtil.generate_linear_schedule(result_info.train_model_config.time_steps)
        model = BackBone(betas).eval()
        device = torch.device("cpu")
        pretrained_dict = torch.load(MODEL_PATH, map_location=device)
        model_dict = model.state_dict()
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
        print("load_pretrain_model success")
        return model
    
    def predict(self, data_dict, model):
        predict_traj = model(data_dict)[-1]
        predicted_traj_mask = data_dict['predicted_traj_mask'][0]
        predicted_future_traj = data_dict['predicted_future_traj'][0]
        predicted_his_traj = data_dict['predicted_his_traj'][0]
        predicted_num = 0
        for i in range(predicted_traj_mask.shape[0]):
            if int(predicted_traj_mask[i]) == 1:
                predicted_num += 1
        generate_traj = predict_traj[:predicted_num]
        predicted_future_traj = predicted_future_traj[:predicted_num]
        predicted_his_traj = predicted_his_traj[:predicted_num]
        real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].detach().numpy()
        real_yaw = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, 2].detach().numpy()
        model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].detach().numpy()
        model_yaw = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, 2].detach().numpy()
        return model_output, real_traj, real_yaw, model_yaw
    
    def run(self, result_info: LoadConfigResultDate, dataset_path, number_of_runs, without_neighbours=False):
        model = self.load_pretrain_model(result_info)
        filenames = os.listdir('/home/liupei/data/waymo/val_set/')
        ROOT_PATH = "/home/liupei/data/waymo/val_set/"
        for i in range(len(filenames)):
            filenames[i] = os.path.join(ROOT_PATH, filenames[i])
        match_filenames = tf.io.matching_files(filenames[:3])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data")
        dataset_iterator = dataset.as_numpy_iterator()
        
        results = {}
        for index, scenario_bytes in enumerate(dataset_iterator):
            results[index] = {"ade": []}
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)
            if len(data_dict) == 0:
                continue

            predict_traj = model(data_dict)[-1]

            predicted_traj_mask = data_dict['predicted_traj_mask'][0]
            predicted_future_traj = data_dict['predicted_future_traj'][0]
            predicted_his_traj = data_dict['predicted_his_traj'][0]
            predicted_num = 0

            
            for i in range(predicted_traj_mask.shape[0]):
                if int(predicted_traj_mask[i]) == 1:
                    predicted_num += 1
            generate_traj = predict_traj[:predicted_num]
            predicted_future_traj = predicted_future_traj[:predicted_num]
            predicted_his_traj = predicted_his_traj[:predicted_num]
            real_traj = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, :2].detach().numpy()
            real_yaw = torch.cat((predicted_his_traj, predicted_future_traj), dim=1)[:, :, 2].detach().numpy()
            model_output = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, :2].detach().numpy()
            model_yaw = torch.cat((predicted_his_traj, generate_traj), dim=1)[:, :, 2].detach().numpy()
            tmp = []
            for i in range(model_output.shape[0]):
                # res = get_ade(model_output[i, :, :], real_traj[i, :, :])
                res = get_fde(model_output[i, :, :], real_traj[i, :, :])
                tmp.append(res)
            results[index]["ade"].append(np.mean(tmp))
            print('********* ade = {}'.format(tmp[0]))
        return results

   