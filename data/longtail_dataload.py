import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
import numpy as np
import re
import warnings
import json

def extact_data(data_frame):
    data_list = []
    # print(data_frame.index.tolist())
    for idx in tqdm(data_frame.index.tolist(),desc="Processing"):

        # 读取特征
        trajectory_data = str2np(data_frame.loc[idx, 'gt']).reshape(2,125).T # (125,2)
        velocity_x = str2np(data_frame.loc[idx, 'velocity_x_list'])  #(124)

        velocity_y = str2np(data_frame.loc[idx, 'velocity_y_list']) # (124)
        velocity = np.column_stack([velocity_x,velocity_y])[:75, :]
  
        rttc = str2np_rttc(data_frame.loc[idx, 'rttc'][1:-1])[:75, :] # (125,6)
        nearby_vehicle_distance =  str2np_dist(data_frame.loc[idx,'nearest_vehicles(1/dx, 1/dy)' ][1:-1])[:75, :] # (125,12)

        traffic_density = np.full((75, 1), data_frame.loc[idx, 'K'])
        traffic_flow = np.full((75, 1), data_frame.loc[idx, 'Q'])
        traffic_speed = np.full((75, 1), data_frame.loc[idx, 'V'])

        predicted_his_traj = trajectory_data[ :75, :]
        predicted_his_pos = trajectory_data[75, :].reshape(1,2)
        predicted_his_traj_delt = predicted_his_traj[1:] - predicted_his_traj[:-1]

        predicted_future_traj = trajectory_data[ 75: ,:]
        # print(predicted_future_traj.shape)
        # print(velocity.shape)
        # print(rttc.shape)
        # print(nearby_vehicle_distance.shape)
        # print(predicted_his_pos.shape)
        # print(predicted_his_traj_delt.shape)
        # print(predicted_future_traj.shape)
        
        # print(traffic_density.shape)
        # print(traffic_flow.shape)
        # print(traffic_speed.shape)
        # 转换为浮点数张量
        sample = {
            'predicted_his_traj':predicted_his_traj.tolist(),
            'velocity': velocity.tolist(),
            'rttc': rttc.tolist(),
            'nearby_vehicle_distance': nearby_vehicle_distance.tolist(),
            'predicted_his_pos':predicted_his_pos.tolist(),
            'predicted_his_traj_delt':predicted_his_traj_delt.tolist(),
            'predicted_future_traj':predicted_future_traj.tolist(),
            'traffic_density':traffic_density.tolist(),
            'traffic_flow':traffic_flow.tolist(),
            'traffic_speed':traffic_speed.tolist()

        }
        data_list.append(sample)
    return data_list



def str2np(data_str):
    # 去掉多余字符
    data_str = data_str.replace('[', '').replace(']', '').replace('\n', '').replace('(','').replace(')','')
    # 将字符串分割为单个元素
    data_list = [item for item in data_str.split() if item]

    #转换为NumPy数组
    data_array = np.array([float(item.replace(',', '')) for item in data_list])
    return data_array
def str2np_dist(dist_str):
    # 使用正则表达式提取方括号内的内容
    matches = re.findall(r'\[(.*?)\]', dist_str)
    dist_list = []
  
    for i, match in enumerate(matches):
        dist = str2np(match)
        if dist.shape[0] != 2:
            dist = dist.reshape(2,125).T
        else:
            dist =  np.zeros((125,2))
        if dist.shape[0] != 125:
            if dist.shape[0]!=1:
                # 输出一个简单的警告
                warnings.warn("The dimensionality of rttc error" ,dist.shape[0])
           
        dist_list.append(dist)
    
          # 在第二维度上堆叠
    # 创建一个形状为 (125, 12) 的零数
    if len(dist_list)!= 0:
        stacked = np.hstack(dist_list)
      
        # 将堆叠后的数据放入结果数组
        
    return stacked
def str2np_rttc(dist_str):
    # 使用正则表达式提取方括号内的内容
    matches = re.findall(r'\[(.*?)\]', dist_str)
    dist_list = []
  
    for i, match in enumerate(matches):
        rttc = str2np(match)
        if rttc.shape[0] != 125:
            if rttc.shape[0]!=1:
                # 输出一个简单的警告
                warnings.warn("The dimensionality of rttc error" ,rttc.shape[0])
            rttc = np.zeros((125))
        dist_list.append(rttc)
          # 在第二维度上堆叠

    if len(dist_list)!= 0:
        stacked = np.column_stack(dist_list)
      
       
    return stacked
        

# 自定义数据集类
class longtailtrajDataset(Dataset):
    def __init__(self, data):
        self.samples = data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        numpy_sample= {key: np.array(value) for key, value in sample.items()}
        return numpy_sample
        
       


if __name__ == "__main__":
# 读取CSV文件
    csv_file = 'data/long_tail_all.csv'
    data_frame = pd.read_csv(csv_file)
        # 划分训练集和测试集
    train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=42)

    train_data = extact_data(train_df)
    test_date = extact_data(test_df)

    # 保存为JSON文件
    with open('data/train_data_all.json', 'w') as f:
        json.dump(train_data, f, indent=4)
        
    with open('data/test_date_all.json', 'w') as f:
        json.dump(test_date, f, indent=4)   
    
    print('已保存')








    # 划分训练集和测试集
    # train_df, test_df = train_test_split(data_frame, test_size=0.2, random_state=42)

    # # 保存训练集和测试集为pkl文件
    # train_df.to_pickle('train_data.pkl')
    # test_df.to_pickle('test_data.pkl')
    # print('已保存')
    # 加载pkl文件
    # train_data = pd.read_pickle('train_data.pkl')
    # test_data = pd.read_pickle('test_data.pkl')

    # 创建数据集和DataLoader
    # train_dataset = longtailtrajDataset(train_data)
    # test_dataset = longtailtrajDataset(test_data)

    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # # 测试DataLoader
    # for i, data in enumerate(train_dataloader):
    #     print(i, data)
    #     break  # 只打印第一个批次


