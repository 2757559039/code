#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from target_dqn.model.model import Model
from target_dqn.feature.definition import (
    ActData,
    printer,
)
import numpy as np
from copy import deepcopy
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from target_dqn.config import Config



@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        # 动作维度=移动方向+闪现方向
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        # 移动维度
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        # 闪现维度
        self.talent_direction = Config.DIM_OF_TALENT
        # 观测维度?
        self.obs_shape = Config.DIM_OF_OBSERVATION
        # 贪心概率
        self.epsilon = Config.EPSILON
        # 探索因子?
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        # 目标网络更新频率
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        # 特征分割???
        self.obs_split = Config.DESC_OBS_SPLIT
        # 折现率
        self._gamma = Config.GAMMA
        # 学习率
        self.lr = Config.START_LR

        # 设备?不用管
        self.device = device
        # 创建神经网络,state_shape为输入层维度,action_shape为输出层维度
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
         # 设备?不用管
        self.model.to(self.device)
        # 创建了一个梯度下降的优化器
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        # 创建目标网络
        self.target_model = deepcopy(self.model)
        # 训练次数
        self.train_step = 0
        # 预测数量?
        self.predict_count = 0
        ###不用管
        self.last_report_monitor_time = 0
        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    # 将输入的data转化为Pytorch可以接收的数据:理解为格式化,用就完啦
    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                data,
                device=self.device,
                dtype=torch.float32,
            )

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        # 数据帧堆叠?
        batch = len(list_obs_data)

        # 获得...数据?
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        # 合法动作标志
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        # 预处理
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )
        model = self.model
        model.eval()
        # 探索因子, 初始epsilon为0.5，我们希望epsilon随着预测步数越来越小，直到0.1为止
        self.epsilon = max(0.1, 0.5 - self.predict_count / self.egp)
        printer.print(f"这里是agent部分{self.epsilon}")


        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                # 生成一系列动作,维度为帧堆和动作维度
                random_action = np.random.rand(batch, self.act_shape)
                # 张量化,就是做一个处理,张量n维度数组
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                # 如果动作不合法(闪现在cd),那么不执行该操作
                random_action = random_action.masked_fill(~legal_act, 0)
                # 将动作从堆栈中提取出来
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    # 调用上面的处理方法,将vec张化
                    self.__convert_to_tensor(feature_vec),
                    # 将帧堆栈数加入到map中
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                # _不用管,logits是?动作概率?
                logits, _ = model(feature, state=None)
                # 屏蔽非法操作(闪现cd),最小化非法操作
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                # 返回最高分动作?
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()
        # 将动作格式化
        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        # 预测数量+1
        self.predict_count += 1
        # 封装为ActData
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    @predict_wrapper
    # 不用管,封装给开悟的?
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        # 获得帧堆栈队列
        t_data = list_sample_data
        # 获取数量
        batch = len(t_data)

        # [b, d]
        # 从堆栈中获取特征和动作组
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        # 合法动作校验
        _batch_obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        _batch_obs_legal = (
            torch.cat(
                (
                    _batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    _batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        # 动作后环节信息中的特征  
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done == 1 else 1 for frame in t_data]), device=self.device)

        #特征合并,应该这么说嘛?
        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        model = getattr(self, "target_model")
        model.eval()
        # 生成目标Q
        with torch.no_grad():
            q, h = model(_batch_feature, state=None)
            q = q.masked_fill(~_batch_obs_legal, float(torch.min(q)))
            q_max = q.max(dim=1).values.detach()

        target_q = rew + self._gamma * q_max * not_done

        # 清空梯度
        self.optim.zero_grad()

        # 生成预测Q
        model = getattr(self, "model")
        model.train()
        logits, h = model(batch_feature, state=None)

        # 计算损失
        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss.backward()
        # 梯度优化
        self.optim.step()

        self.train_step += 1

        # 更新target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": 0,
                "diy_2": 0,
                "diy_3": 0,
                "diy_4": 0,
                "diy_5": 0,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(
            torch.load(model_file_path, map_location=self.device),
        )

        self.logger.info(f"load model {model_file_path} successfully")

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
