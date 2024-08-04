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
    printer,
)
from conf.usr_conf import usr_conf_check
from target_dqn.config import WeiZhiHuoQu
from target_dqn.config import CustomPrinter
from target_dqn.config import guijiGeneration
@attached
def workflow(envs, agents, logger=None, monitor=None):

    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 20
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
        printer.print(f"这是第{epoch}轮")
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
    guiji=guijiGeneration()
    for episode in range(n_episode):
        printer.print(f"这是第{episode}个周期")
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
        # 首帧obs,用于获取起点
        first_frame_obs =obs
        done = False
        # 步数
        step = 0
        # 撞墙次数
        bump_cnt = 0
        ###自定义标记
        # 获胜标记
        terminatedCode=0
        # 获胜次数
        terminatedNum=0
        # 使用技能次数
        usedSkillNum=0
        # 技能使用前后位置, 用列表存储多次技能信息
        usedSkillStar=[]
        usedSkillEnd=[]
        # buff获得次数
        buffNum=0
        # 初始宝箱
        treasure_info_list=[ (1 if treasure.grid_distance<1 else 0) for i,treasure in enumerate(obs.feature.treasure_pos) if i>1]
        # 训练结束时宝箱
        _treasure_info_list=treasure_info_list
        ans=0
        while not done:
            printer.print(f"这是第{ans}步")
            ans+=1

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
          
            # 特征处理
            # ?
            _obs_data = observation_process(_obs, _env_info)

            # with open("test.txt", 'a') as file:
                # file.write(str(obs)+'\n')  
                # file.write(str(obs_data)+'\n')             
                # file.write('+++++++++++++++++++++++++++++++++++++++++++++\n')

            step += 1
            # print(step)
            if(act>7):
                usedSkillNum+=1
                usedSkillStar.append([obs.feature.grid_pos.x,obs.feature.grid_pos.z])
                usedSkillEnd.append([_obs.feature.grid_pos.x,_obs.feature.grid_pos.z])
            # 抽象hhh,获取拿到buff的判定

            # 当前buff不存在,上一帧buff存在,从而表示拿到了buff
            if _obs.feature.buff_pos.grid_distance==1 and obs.feature.buff_pos.grid_distance!=1:
                    buffNum+=1

            array_128x128[_obs.feature.grid_pos.z][_obs.feature.grid_pos.x]+=1

            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0

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
                terminatedCode=1
                terminatedNum+=1
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

        _treasure_info_list=[ (1 if treasure.grid_distance<1 else 0) for i,treasure in enumerate(obs.feature.treasure_pos) if i>1]
        have_treasure_list=[treasure ^ _treasure for treasure,_treasure in zip(treasure_info_list, _treasure_info_list)]
        first_frame_pos =first_frame_obs.feature.grid_pos
        x=first_frame_pos.x
        z=first_frame_pos.z
        array_128x128[z][x]="起点"
        with open("guiji.txt", 'a') as file:
            file.write('第' + str(episode+1)+'轮\n')
            file.write('本轮数据:\n')
            file.write('结果:'+'胜利' if terminatedCode else '失败'+'\n')
            file.write("胜率:"+str((terminatedNum/(episode+1))*100)+'%\n')
            file.write('步数:'+str(step)+'\n')
            file.write('撞墙次数:'+str(bump_cnt)+',撞墙率:'+str(bump_cnt/step))
            file.write('闪现次数:'+str(usedSkillNum)+'\n')
            file.write('闪现位置:\n')
            file.write('闪现起点:'+str(usedSkillStar)+'\n')
            file.write('闪现终点:'+str(usedSkillEnd)+'\n')
            file.write('获取buff次数:'+str(buffNum)+'\n')
            file.write('宝箱信息:'+str(treasure_info_list)+'\n')
            file.write('已得宝箱:'+str(have_treasure_list)+'\n')
            file.write('未得信息:'+str([treasure - h_treasure for treasure,h_treasure in zip(treasure_info_list, have_treasure_list)])+'\n')
            if(terminatedCode):
                file.write('最终得分:'+str((2000-step+2)*0.2+150+sum(have_treasure_list)*100)+',分\n')
            else:
                file.write('最终得分:'+str(0)+'分\n')
            
        
        guiji.print_128x128_array(array_128x128)


