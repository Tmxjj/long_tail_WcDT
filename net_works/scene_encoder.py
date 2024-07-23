#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: scene_encoder.py
@Author: YangChen
@Date: 2023/12/27
"""
import torch
from torch import nn

from net_works.transformer import TransformerCrossAttention, TransformerSelfAttention


class OtherFeatureFormer(nn.Module):
    def __init__(
            self, block_num: int, input_dim: int, conditional_dim: int,
            head_dim: int = 64, num_heads: int = 8
    ):
        super(OtherFeatureFormer, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(block_num):
            self.blocks.append(TransformerCrossAttention(input_dim, conditional_dim, head_dim, num_heads))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input_x, other_feature):
        for block in self.blocks:
            input_x = block(input_x, other_feature)
        return self.norm(input_x)


class SelfFeatureFormer(nn.Module):
    def __init__(self, block_num: int, input_dim: int, head_dim: int = 64, num_heads: int = 8):
        super(SelfFeatureFormer, self).__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(block_num):
            self.blocks.append(TransformerSelfAttention(input_dim, head_dim, num_heads))
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, input_x):
        for block in self.blocks:
            input_x = block(input_x)
        return self.norm(input_x)


class SceneEncoder(nn.Module):
    def __init__(
            self, dim: int = 256, embedding_dim: int = 32,
            his_step: int = 4, other_agent_depth: int = 4,
            map_feature_depth: int = 4, traffic_light_depth: int = 2,
            self_attention_depth: int = 4
    ):
        super(SceneEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.pos_embedding = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        self.feature_embedding = nn.Sequential(
            nn.Linear(7, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        linear_input_dim = (his_step - 1) * 5 + embedding_dim + embedding_dim
        self.linear_input = nn.Sequential(
            nn.Linear(linear_input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.other_agent_former = OtherFeatureFormer(block_num=other_agent_depth, input_dim=dim, conditional_dim=114)
        self.map_former = OtherFeatureFormer(block_num=map_feature_depth, input_dim=dim, conditional_dim=embedding_dim)
        self.traffic_light_former = OtherFeatureFormer(block_num=traffic_light_depth, input_dim=dim,
                                                       conditional_dim=43)
        self.fusion_block = SelfFeatureFormer(block_num=self_attention_depth, input_dim=dim, num_heads=16)

    def forward(
            self, noise, lane_list,
            other_his_traj_delt, other_his_pos, other_feature,
            predicted_his_traj_delt, predicted_his_pos, predicted_feature,
            traffic_light, traffic_light_pos
    ):
        '''
        noise：噪声张量，用于扰动输入数据。
        lane_list：车道列表，包含车道的位置信息。

        other_his_traj_delt：其他车辆的历史轨迹增量。
        other_his_pos：其他车辆的历史位置。
        other_feature：其他车辆的属性特征。

        predicted_his_traj_delt：预测的历史轨迹增量。
        predicted_his_pos：预测的历史位置。
        predicted_feature：预测车辆的属性特征。
        
        traffic_light：交通信号灯信息。
        traffic_light_pos：交通信号灯位置
        '''
        batch_size, obs_seq_len = noise.shape[0], noise.shape[1]
        # batch, obs_num(8), his_step, 3
        x = predicted_his_traj_delt + (noise * 0.001)
        # x = torch.flatten(x, start_dim=2)
        # other_his_traj_delt = torch.flatten(other_his_traj_delt, start_dim=2)
       
        # 对各个位置进行位置编码
        # lane_list = self.pos_embedding(lane_list)
        # lane_list = lane_list.view(batch_size, -1, self.embedding_dim)
        # traffic_light_pos = self.pos_embedding(traffic_light_pos)
        # other_his_pos = self.pos_embedding(other_his_pos)
        predicted_his_pos = self.pos_embedding(predicted_his_pos)
        
        # 对属性进行编码
        # other_feature = self.feature_embedding(other_feature)
        predicted_feature = self.feature_embedding(predicted_feature)
        # 组合输入信息
        x = torch.cat((x, predicted_his_pos, predicted_feature), dim=-1)
        # batch, obs_num(15), 256
        x = self.linear_input(x)

        # other agent former
        other_obs_feature = torch.cat((other_his_traj_delt, other_his_pos, other_feature), dim=-1)
        x = self.other_agent_former(x, other_obs_feature)
        # map_point_transformer
        x = self.map_former(x, lane_list)
        # traffic_light_transformer
        traffic_light = torch.cat((traffic_light, traffic_light_pos), dim=-1)
        x = self.traffic_light_former(x, traffic_light)
        x = self.fusion_block(x)
        return x


class SceneEncoder_longtail(nn.Module):
    def __init__(
            self, dim: int = 256, embedding_dim: int = 32,
            his_step: int = 4, other_agent_depth: int = 2,
            map_feature_depth: int = 4, traffic_light_depth: int = 2,
            self_attention_depth: int = 4
    ):
        super(SceneEncoder_longtail, self).__init__()
        self.embedding_dim = embedding_dim
        self.pos_embedding = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        self.feature_embedding_spe = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        self.feature_embedding_rttc = nn.Sequential(
            nn.Linear(6, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )

        self.feature_embedding_nearleat_veh = nn.Sequential(
            nn.Linear(12, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, embedding_dim)
        )
        linear_input_dim = 2 + embedding_dim + embedding_dim + embedding_dim
        self.linear_input = nn.Sequential(
            nn.Linear(linear_input_dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.other_agent_former = OtherFeatureFormer(block_num=other_agent_depth, input_dim=dim, conditional_dim=32)
        # self.map_former = OtherFeatureFormer(block_num=map_feature_depth, input_dim=dim, conditional_dim=embedding_dim)
        # self.traffic_light_former = OtherFeatureFormer(block_num=traffic_light_depth, input_dim=dim,
                                                    #    conditional_dim=43)
        self.fusion_block = SelfFeatureFormer(block_num=self_attention_depth, input_dim=dim, num_heads=16)

    def forward(
            self,
            noise, 
            other_his_dist, 
            predicted_his_traj_delt, 
            predicted_his_pos,
            predicted_his_rttc,
            predicted_feature
    ):
        '''
        noise：噪声张量，用于扰动输入数据。
        other_his_dist:周围车辆距离，
        predicted_his_traj_delt：预测的历史轨迹增量。
        predicted_his_pos：预测的位置。
        predicted_his_rttc：预测车辆危险性
        predicted_feature：预测车辆速度
        '''
        # batch,  his_step -1 , 2
        x = predicted_his_traj_delt + (noise * 0.001)
        # x = torch.flatten(x, start_dim=2)
        mean_x = torch.mean(x, axis=1, keepdims=True)
        # 在第二个维度上连接平均值列
        x = torch.cat((x, mean_x), axis=1)
        # 对各个位置进行位置编码
        predicted_his_pos = self.pos_embedding(predicted_his_pos.repeat(1, 75, 1))
        # 对属性进行embdeding

        predicted_velocity = self.feature_embedding_spe(predicted_feature)
        predicted_his_rttc = self.feature_embedding_rttc(predicted_his_rttc)
        other_his_dist = self.feature_embedding_nearleat_veh(other_his_dist)

        # 组合输入信息
        x = torch.cat((x, predicted_his_pos, predicted_velocity,predicted_his_rttc), dim=-1)
        # batch, obs_num(15), 256
        x = self.linear_input(x)

        # other agent former
        other_obs_feature = other_his_dist
        x = self.other_agent_former(x, other_obs_feature)
        x = self.fusion_block(x)
        return x
