#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project :back_to_the_realm
@File    :model.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


### 没想到能修改什么,除了初始的权重的偏置,ReLU函数ma...
class Model(nn.Module):
    #  状态维度,动作维度,是否添加softmax层(?)
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        # 3层卷积层的定义
        cnn_layer1 = [
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
        ]
        # 定义池化层
        max_pool = [nn.MaxPool2d(kernel_size=(2, 2))]
        # 组合
        self.cnn_layer = cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool
        # 序列化
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        # 同上,3个全连接层的定义
        fc_layer1 = [nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True)]
        fc_layer2 = [nn.Linear(256, 128), nn.ReLU(inplace=True)]
        fc_layer3 = [nn.Linear(128, np.prod(action_shape))]

        self.fc_layers = fc_layer1 + fc_layer2

        if action_shape:
            self.fc_layers += fc_layer3
        if softmax:
            self.fc_layers += [nn.Softmax(dim=-1)]

        # 序列化  `
        self.model = nn.Sequential(*self.fc_layers)

        self.apply(self.init_weights)

    # 权重从N(0,2/n)的正态分布中随机获得,偏置为0  
    def init_weights(self, m):
        # 卷积层处理
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                # 初始化为0了?
                nn.init.constant_(m.bias, 0)
        # 全连接层处理
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        # 用卷积层处理图像
        feature_maps = self.cnn_model(feature_maps)
        feature_maps = feature_maps.view(feature_maps.shape[0], -1)
        # 用全连接层输出
        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)
        logits = self.model(concat_feature)
        return logits, state
