#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# tager
"""
@Project :back_to_the_realm
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import time
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached

from target_dqn.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
from conf.usr_conf import usr_conf_check

def convert_to_str_2d(array_2d):
    str_array_2d = []
    for row in array_2d:
        str_row = []
        for item in row:
            if isinstance(item, int):
                str_row.append(str(item))
            else:
                str_row.append(item)
        str_array_2d.append(str_row)
    return str_array_2d

def print_128x128_array(array):
    ###行为数组预处理
    for i in range(128):
        for j in range(128):
            if array[i][j] == 0:
                array[i][j] = ' '

    # start_value = usr_conf["diy"]["start"]
    # end_value = usr_conf["diy"]["end"]
    # array[start_value[1]][start_value[0]]="起点"
    # if(win_mark):
    #     array[end_value[1]][end_value[0]]="获胜"
    # else:
    #     array[end_value[1]][end_value[0]]="终点"

    # 先创建一个包含所有索引的集合，以提高查找效率
    # chest_positions_set = {tuple(row) for row in over_time_chest_location}

    # # 然后遍历 chest_location 并根据索引更新 array
    # for i in chest_location:
    #     # 检查索引 i 是否存在于 chest_positions_set 中
    #     if tuple(i) in chest_positions_set:
    #         array[i[1]][i[0]] = "宝"  # 假设 i 是 [行索引, 列索引]
    #     else:
    #         array[i[1]][i[0]] = "开"

    array=[row.copy() for row in reversed(array)]
    array=convert_to_str_2d(array)

    with open("guiji.txt", 'a') as file:  # 打开文件准备写入
        # file.write("第"+str(episode)+"次训练,epsilon:"+str(epsilon)+",已经走步数:"+str(cont)+",宝箱总数:"+str(len(chest_location))+",已获得宝箱数量:"+str(len(chest_location)-len(over_time_chest_location))+",总得分:"+str(total_score)+"\n")
        for i, row in enumerate(array):
            if i == 0 or i == len(array) - 1:  # 打印（写入）顶部和底部的边框
                # 写入顶部和底部边框
                file.write('+' + '-' * (len(array[0]) * 4 - 1) + '+' + '\n')
            else:
                row_output = ['|']  # 每行开始添加'|'
                for j, val in enumerate(row):
                    if j == len(row) - 1:  # 每行末尾添加'|'
                        row_output.append('|')
                    else:
                        row_output.append(f"{val:4s} ")
                # 将当前行写入文件，使用join避免逐个字符写入的低效
                file.write(''.join(row_output) + '|\n')
    array=[[0 for _ in range(128)] for _ in range(128)]

@attached
def workflow(envs, agents, logger=None, monitor=None):

    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    # 帧数阈值,堆栈
    g_data_truncat = 256
    last_save_model_time = 0

    # 用户自定义的游戏启动配置
    usr_conf = {
        "diy": {
            "start": 2,
            "end": 1,
            "treasure_id": [3,4, 5, 6, 7, 8, 9,10,11,12,13,14,15],
            "treasure_random": 0,
            "talent_type": 1,
            # "treasure_num": 8,
            "max_step": 2000,
        }
    }

    # usr_conf_check会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = usr_conf_check(usr_conf, logger)
    if not valid:
        logger.error(f"usr_conf_check return False, please check")
        return

    for epoch in range(epoch_num):
        # 累计奖励
        epoch_total_rew = 0
        # 数据长度,用于计算平局回报的
        data_length = 0
        # 接收 训练局数 环境 智能体 帧数 配置 日志记录器
        # 训练
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger):
            # 做统计
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            # 执行训练
            agent.learn(g_data)
            g_data.clear()
            # g_data是?

        # 平均奖励
        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # 保存model文件,不用管
        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


# 接收 局数 环境 智能体 帧数阈值 配置 日志记录器
def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger):
    for episode in range(n_episode):
        array_128x128 = [[0 for _ in range(128)] for _ in range(128)]
        collector = list()

        # 重置游戏, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)
        env_info = None

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
        agent.load_model(id="latest")

        # 特征处理
        # 将初始数据转化为定义数据的预处理
        # obs包括信息:
        obs_data = observation_process(obs)

        done = False
        # 步数
        step = 0
        # 这个是?
        bump_cnt = 0

        while not done:

            # Agent 进行推理, 
            # act_data为定义的动作数据
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # ActData 解包成动作
            act = action_process(act_data)

            # 与环境交互, 执行动作, 
            # 帧数 下一步环境信息 即时奖励 结束 超时? 环境信息
            frame_no, _obs, score, terminated, truncated, _env_info = env.step(act)
            if _obs is None:
                break
            step += 1
            print(step)
            array_128x128[_obs.feature.grid_pos.z][_obs.feature.grid_pos.x]+=1
            # 特征处理
            # ?
            _obs_data = observation_process(_obs, _env_info)

            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0
            # Calculate reward
            # 计算 reward
            if env_info is None:
                reward = 0
            else:
                # is_bump?
                reward, is_bump = reward_shaping(
                    frame_no,
                    score,
                    terminated,
                    truncated,
                    obs,
                    _obs,
                    env_info,
                    _env_info,
                )
                treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]
                treasures_num = treasure_dists.count(1.0)
                # Wall bump behavior statistics
                # 撞墙行为统计
                bump_cnt += is_bump

            # 判断游戏结束, 并更新胜利次数
            if truncated:
                logger.info(
                    f"truncated is True, so this episode {episode} timeout, \
                        collected treasures: {treasures_num  - 7}"
                )
            elif terminated:
                logger.info(
                    f"terminated is True, so this episode {episode} reach the end, \
                        collected treasures: {treasures_num  - 7}"
                )
            done = terminated or truncated

            # 构造游戏帧，为构造样本做准备
            # 用于训练
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                rew=reward,
                done=done,
                ret=reward,
            )

            collector.append(frame)

            # 如果游戏帧数达到阈值，则进行样本处理，将样本送去训练
            if len(collector) % g_data_truncat == 0:
                # ???
                collector = sample_process(collector)
                yield collector

            # 如果游戏结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector
                break
            # print(step)
            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            env_info = _env_info
        print_128x128_array(array_128x128)
            