import argparse
import logging
import os
import numpy as np
import random
import torch

import test
from common import TaskType, LoadConfigResultDate
from tasks import BaseTask

import shapley_values
from main import TaskFactory


def scale_and_convert_to_relative_trajectory(trajectory, data_scale):
    return (trajectory - trajectory[:, :, :1, :]) * data_scale
def get_initial_position_scaled(trajectory):
    return trajectory[:, :, 7, :].copy() / 1000
def get_data(data, data_scale, device):
    trajectory = np.stack(data)
    initial_pos = get_initial_position_scaled(trajectory)
    trajectory = scale_and_convert_to_relative_trajectory(trajectory, data_scale)
    return (
        torch.DoubleTensor(trajectory).to(device),
        torch.DoubleTensor(initial_pos).to(device),
        torch.ones(trajectory.shape[0], trajectory.shape[1], trajectory.shape[1]).to(device),
    )

class ShapleyValuesTask(BaseTask):
    def __init__(self):
        super(ShapleyValuesTask, self).__init__()
        self.task_type = TaskType.SHAPLEY_VALUES

    def execute(self, result_info: LoadConfigResultDate):
        # logger = result_info.task_logger.get_logger()
        # logging.basicConfig(level=logging.INFO)
        # self.initialize_device_and_seed(0)
        load_config_result = TaskFactory.init_config()
        tester = test.Tester(load_config_result)
        shapley_values_estimator = shapley_values.ShapleyValues(
            tester, self.get_indices_no_mask, self.get_random_neighbor_no_mask
        )

        dataset_path = result_info.task_config.data_dir
        scene_index = result_info.task_config.scene_index
        result = shapley_values_estimator.run(dataset_path, scene_index, get_data)



    def get_indices_no_mask(batch_size, _mask, index, device):
        valid_indices = torch.tensor([x for x in range(batch_size)])
        neighbors_indices = valid_indices[valid_indices != index]
        return torch.cat([torch.tensor([index]), neighbors_indices], dim=0).to(device)

    
    def get_random_neighbor_no_mask(trajectories, initial_positions, scene_index):
        random_scene_index = random.choice(
            [x for x in range(trajectories.shape[0]) if x != scene_index]
        )
        random_batch_trajectory = trajectories[random_scene_index]
        random_batch_initial_pos = initial_positions[random_scene_index]
        random_player_index = random.choice([x for x in range(random_batch_trajectory.shape[0])])
        return (
            random_batch_trajectory[random_player_index : random_player_index + 1],
            random_batch_initial_pos[random_player_index : random_player_index + 1],
        )

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model full path", type=str)
    parser.add_argument("checkpoint", help="checkpoint number", type=int)
    parser.add_argument("data", help="full path to data file", type=str)
    parser.add_argument("device", help="gpu or cpu", type=str)
    parser.add_argument("node_type", help="node type", type=str)
    parser.add_argument(
        "metric", choices=shapley_values.METRICS.values(), type=shapley_values.create_metric
    )
    parser.add_argument(
        "variant", choices=shapley_values.VARIANTS.values(), type=shapley_values.create_variant
    )
    parser.add_argument("scene_index", help="scene index", type=int)
    parser.add_argument("output_path", help="result directory", type=str)
    parser.add_argument(
        "--random_node_types", help="list of random node types", type=str, nargs="+"
    )
    parser.add_argument("--store_visualization", action="store_true")
    return parser.parse_args()

def initialize_device_and_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_device_and_seed(0)
    parameters = parse_arguments()

    load_config_result = TaskFactory.init_config()
    tester = test.Tester(load_config_result)
    shapley_values_estimator = shapley_values.ShapleyValues(
        tester, parameters.node_type, parameters.random_node_types, parameters.metric
    )

    get_replacement_trajectory = lambda edge_type: parameters.variant(
        tester.environment.scenes,
        shapley_values_estimator.data_parameters,
        tester.environment,
        edge_type,
    )
    test_scene = tester.environment.scenes[parameters.scene_index]
    result = shapley_values_estimator.run(
        test_scene, get_replacement_trajectory, store_visualization=parameters.store_visualization
    )
    # with open(output_file_path, "wb") as file_writer:
    #     dill.dump(result, file_writer)
