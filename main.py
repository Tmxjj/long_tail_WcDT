#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: main.py
@Author: YangChen
@Date: 2023/12/20
"""
from typing import List
from datetime import datetime
from common import TaskType, LoadConfigResultDate, TaskLogger
from tasks import BaseTask
from tasks import (LoadConfigTask, DataPreprocessTask, DataSplitTask,
                   DataCountTask, TrainModelTask, ShowResultsTask,
                   TestModelTask)


class TaskFactory:
    # 初始化任务对象
    @staticmethod
    def init_config() -> LoadConfigResultDate:
        result_info = LoadConfigResultDate()
        load_config_task = LoadConfigTask()
        load_config_task.execute(result_info)
        return result_info
    # 确认任务类型
    @staticmethod
    def init_tasks(task_type_list: List[TaskType]) -> List[BaseTask]:
        task_list = list()
        for task_type in task_type_list:
            if task_type == TaskType.DATA_PREPROCESS:
                task_list.append(DataPreprocessTask())
            elif task_type == TaskType.DATA_SPLIT:
                task_list.append(DataSplitTask())
            elif task_type == TaskType.DATA_COUNT:
                task_list.append(DataCountTask())
            elif task_type == TaskType.TRAIN_MODEL:
                task_list.append(TrainModelTask())
            elif task_type == TaskType.SHOW_RESULTS:
                task_list.append(ShowResultsTask())
            elif task_type == TaskType.Test_MODEl:
                task_list.append(TestModelTask())
        return task_list

    # 执行各任务
def execute_tasks():
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    load_config_result = TaskFactory.init_config()
    task_list = TaskFactory.init_tasks(load_config_result.task_config.task_list)
    task_logger: TaskLogger = load_config_result.task_logger
    for task in task_list:
        task_logger.logger.info(f"task type {task.task_type.value} start")
        task.execute(load_config_result,current_time,task_logger) if task.task_type.value == 'TRAIN_MODEL' else task.execute(load_config_result,current_time) 
        task_logger.logger.info(f"task type {task.task_type.value} success")


if __name__ == "__main__":
    execute_tasks()
