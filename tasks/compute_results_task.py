import os.path
import shutil
from typing import Any, Dict

import sys
sys.path.append('/home/liupei/code/WcDT/')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from waymo_open_dataset.utils.sim_agents import visualizations, submission_specs

from common import TaskType, LoadConfigResultDate
from net_works import BackBone
from tasks import BaseTask
from utils import DataUtil, MathUtil, MapUtil

# from metrics import MR
# from metrics import minADE
# from metrics import minAHE
# from metrics import minFDE
# from metrics import minFHE

RESULT_DIR = r"/home/liupei/code/WcDT/output/image"
DATA_SET_PATH = r"/home/liupei/data/waymo/val_set/validation_interactive.tfrecord-00000-of-00150"
MODEL_PATH = r"/home/liupei/code/WcDT/output/models/model_epoch152.pth"

result_info = LoadConfigResultDate()

import math 
def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2) #2代表两列xy
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0] #预测单条轨迹的行数==每个轨迹包含轨迹点的个数
    ade = float(  #单条轨迹中所有轨迹点坐标与真值的欧氏距离的平均值
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade
class ComputeResultsTask():
    # def __init__(self):
        # self.minADE = minADE(max_guesses=6)
        # self.minAHE = minAHE(max_guesses=6)
        # self.minFDE = minFDE(max_guesses=6)
        # self.minFHE = minFHE(max_guesses=6)
        # self.MR = MR(max_guesses=6)
    
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
    
    def compute_result(self, result_info: LoadConfigResultDate):
        model = self.load_pretrain_model(result_info)
        match_filenames = tf.io.matching_files([DATA_SET_PATH])
        dataset = tf.data.TFRecordDataset(match_filenames, name="train_data").take(100)
        dataset_iterator = dataset.as_numpy_iterator()

        for index, scenario_bytes in enumerate(dataset_iterator):
            scenario = scenario_pb2.Scenario.FromString(scenario_bytes)
            data_dict = DataUtil.transform_data_to_input(scenario, result_info)
            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(torch.float32).unsqueeze(dim=0)

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

            ade = get_ade(generate_traj, predicted_future_traj)

if __name__ == "__main__":
    Compute = ComputeResultsTask()
    Compute.compute_result(result_info)
    pass





