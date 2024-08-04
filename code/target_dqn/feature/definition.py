#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.back_to_the_realm.target_dqn.feature_process import (
    one_hot_encoding,
    read_relative_position,
    bump,
)
from util.printHelp.CustomPrinter import CustomPrinter
from util.printHelp.WeiZhiHuoQu import WeiZhiHuoQu
from target_dqn.config import WeiZhiHuoQu
from target_dqn.config import CustomPrinter

output_filename = 'output_definition.txt'
printer = CustomPrinter(output_filename)

output_filename = 'weizhi.txt'
weizhi = CustomPrinter(output_filename)

# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    # 特征
    feature=None,
    # 合法动作标志
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    # 移动方向
    move_dir=None,
    # 是否使用召唤师技能?
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    # 标志
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


### 重点
def reward_shaping(
    frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info
):
    reward = 0
    ### 解包信息
    # 获取当前智能体的位置坐标
    pos = _env_info.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    ###获得当前智能体的网格坐标
    grid_pos=_obs.feature.grid_pos

    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    end_dist = _obs.feature.end_pos.grid_distance
    buff_dist = _obs.feature.buff_pos.grid_distance
    treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]

    # 获取智能体上一帧的位置
    prev_pos = env_info.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = obs.feature.end_pos.grid_distance
    prev_buff_dist = obs.feature.buff_pos.grid_distance
    prev_treasure_dists = [pos.grid_distance for pos in obs.feature.treasure_pos]

    # 获取buff的状态
    buff_availability = 0
    for organ in env_info.frame_state.organs:
        if organ.sub_type == 2:
            buff_availability = 1
            
    prev_buff_availability = 0
    for organ in env_info.frame_state.organs:
        if organ.sub_type == 2 and organ.status==1:
            buff_availability = 1

    # 获取智能体的加速状态
    prev_speed_up = env_info.frame_state.heroes[0].speed_up
    speed_up = _env_info.frame_state.heroes[0].speed_up
    


        #####{
    # 假设_env_info和_obs是已经定义好的变量，包含了所需的环境信息
    # 以下是模拟状态信息的打印
    printer.print(f"\n---------我是分割符---------\n")
    printer.print(f"这里是奖励函数")
    # 打印智能体当前位置坐标
    printer.print(f"当前智能体位置坐标: X = {curr_pos_x}, Z = {curr_pos_z}")

    # 打印当前智能体相对于终点、buff和宝箱的栅格化距离
    printer.print(f"当前智能体距离终点栅格化距离: {end_dist}")
    printer.print(f"当前智能体距离buff栅格化距离: {buff_dist}")
    printer.print(f"当前智能体距离宝箱栅格化距离: {treasure_dists}")

    # 打印智能体上一帧的位置坐标
    printer.print(f"上一帧智能体位置坐标: X = {prev_pos_x}, Z = {prev_pos_z}")

    # 打印智能体上一帧相对于终点、buff和宝箱的栅格化距离
    printer.print(f"上一帧智能体距离终点栅格化距离: {prev_end_dist}")
    printer.print(f"上一帧智能体距离buff栅格化距离: {prev_buff_dist}")
    printer.print(f"上一帧智能体距离宝箱栅格化距离: {prev_treasure_dists}")

    # 打印buff的状态
    printer.print(f"当前buff状态: {'有' if buff_availability else '无'}")

    # 打印智能体的加速状态
    printer.print(f"当前智能体加速状态: {'加速' if speed_up else '未加速'}")

    # 打印智能体的动作选择
    # printer.print(f"当前智能体的动作选择: {act}")
    printer.print(f"\n---------我是分割符---------\n")


    #暴力获取位置(如果出现问题,就尝试注释这三行代码)
    wz = WeiZhiHuoQu('data.json')
    wz.update_data(curr_pos_x, curr_pos_z, prev_buff_availability, buff_availability,prev_treasure_dists,treasure_dists,terminated)
    wz.save_to_json()
    ####}

    """
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # 奖励1.1 向终点靠近的奖励
    # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
    if prev_end_dist != 1:
        reward_end_dist += 1 if end_dist < prev_end_dist else 0

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        reward_win += 1

    """
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)

    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    location_list=[]
    if prev_treasure_dists.count(1.0) < treasure_dists.count(1.0):
        with open('location.txt', 'a') as file:
            print(str(grid_pos)+"\n", file=file)
        reward_treasure = 1

    """
    奖励3. 与buff相关的奖励
    """
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = 0

    # 奖励3.2 获得buff的奖励
    reward_buff = 0
    # if prev_buff_dist.count(1.0)<buff_dist.count(1.0):
    #     print("buff的坐标为:X,"+str(curr_pos_x)+"Z,"+str(curr_pos_z)+"\n")
    #     reward_buff=1

    """
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # 奖励4.1 撞墙闪现的惩罚 (TODO)

    # 奖励4.2 正常闪现的奖励 (TODO)

    # 奖励4.3 超级闪现的奖励 (TODO)

    """
    奖励5. 关于快速通关的奖励
    """
    reward_step = 1

    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)

    # 奖励5.2 重复探索的惩罚
    reward_memory = 0
    memory_map = obs.feature.memory_map

    # 奖励5.3 撞墙的惩罚
    reward_bump = 0

    # 判断是否撞墙
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    """
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    REWARD_CONFIG = {
        "reward_end_dist": "1",
        "reward_win": "1",
        "reward_buff_dist": "1",
        "reward_buff": "1",
        "reward_treasure_dists": "1",
        "reward_treasure": "1",
        "reward_flicker": "1",
        "reward_step": "-1",
        "reward_bump": "-1",
        "reward_memory": "-1",
    }

    reward = [
        reward_end_dist * float(REWARD_CONFIG["reward_end_dist"]),
        reward_win * float(REWARD_CONFIG["reward_win"]),
        reward_buff_dist * float(REWARD_CONFIG["reward_buff_dist"]),
        reward_buff * float(REWARD_CONFIG["reward_buff"]),
        reward_treasure_dist * float(REWARD_CONFIG["reward_treasure_dists"]),
        reward_treasure * float(REWARD_CONFIG["reward_treasure"]),
        reward_flicker * float(REWARD_CONFIG["reward_flicker"]),
        reward_step * float(REWARD_CONFIG["reward_step"]),
        reward_bump * float(REWARD_CONFIG["reward_bump"]),
        reward_memory * float(REWARD_CONFIG["reward_memory"]),
    ]

    return sum(reward), is_bump


@attached
### 重点
def observation_process(raw_obs, env_info=None):
    """
    该函数是特征处理的重要函数, 主要负责：
        - 解析原始数据里的信息
        - 解析预处理后的特征数据
        - 对特征进行处理, 并返回处理后的特征向量
        - 特征的拼接
        - 合法动作的标注
    函数的输入：
        - raw_obs: 预处理后的特征数据
        - env_info: 游戏返回的环境信息
    函数的输出：
        - observation: 特征向量
        - legal_action: 合法动作的标注
    """
    feature, legal_act = [], []

    # 对预处理后的特征数据按照协议进行解包
    norm_pos = raw_obs.feature.norm_pos
    grid_pos = raw_obs.feature.grid_pos
    start_pos = raw_obs.feature.start_pos
    end_pos = raw_obs.feature.end_pos
    buff_pos = raw_obs.feature.buff_pos
    treasure_poss = raw_obs.feature.treasure_pos
    obstacle_map = list(raw_obs.feature.obstacle_map)
    memory_map = list(raw_obs.feature.memory_map)
    treasure_map = list(raw_obs.feature.treasure_map)
    end_map = list(raw_obs.feature.end_map)

    # 特征处理1：当前位置的one-hot编码
    one_hot_pos = one_hot_encoding(grid_pos)

    # 特征处理2：归一化位置
    norm_pos = [norm_pos.x, norm_pos.z]

    # 特征处理3：当前位置相对终点点位的信息
    end_pos_features = read_relative_position(end_pos)

    # 特征处理4: 当前位置相对宝箱位置的信息
    treasure_poss_features = []
    for treasure_pos in treasure_poss:
        treasure_poss_features = treasure_poss_features + list(
            read_relative_position(treasure_pos)
        )

    # 特征处理5：buff是否可收集
    buff_availability = 1
    if env_info:
        for organ in env_info.frame_state.organs:
            if organ.sub_type == 2:
                buff_availability = organ.status


    # 特征处理6：闪现技能是否可使用
    talent_availability = 1
    if env_info:
        talent_availability = env_info.frame_state.heroes[0].talent.status



    # 特征拼接：将所有需要的特征进行拼接作为向量特征 (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    feature = (
        # 智能体位置
        norm_pos
        # one-hot智能体绘制
        + one_hot_pos
        # 终点相对位置
        + end_pos_features
        # 宝箱相对位置
        + treasure_poss_features
        # 宝箱可否收集,技能是否可以使用
        + [buff_availability, talent_availability]
        ###....
        + obstacle_map
        + end_map
        + treasure_map
        + memory_map
    )

    # 合法动作
    legal_act = list(raw_obs.legal_act)

    return ObsData(feature=feature, legal_act=legal_act)


### 可以不用管
@attached
def action_process(act_data):
    result = act_data.move_dir
    result += act_data.use_talent * 8
    return result


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> 
###自定义数据结构所需处理映射
@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:10808],
        _obs=s_data[10808:21616],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
