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

from kaiwu_agent.back_to_the_realm.dqn.feature_process import (
    one_hot_encoding,
    read_relative_position,
    bump,
)

import json
#这里是打印
import os
import sys
class CustomPrinter:

    def __init__(self, filename):
        self.filename = filename

    def clear_text_file(filename):
        with open(filename, 'w') as file:
            pass  # 无需写入任何内容，文件已被清空
    def print_map(self, map_data, map_name, fuhao0=0, fuhao1=1):
        """
        将给定的一维数组或51x51的地图数据格式化并打印为51x51的网格。
        :param map_data: 表示地图的一维列表或二维列表。
        :param map_name: 地图名称，用于打印和文件记录。
        """
        # 检查map_data是否为长度为2601的一维列表
        if isinstance(map_data, list) and len(map_data) == 2601:
            # 转换一维数组为51x51的二维数组
            if map_name != "记忆地图":
                map_data = [fuhao1 if x == 1 else fuhao0 if x == 0 else x for x in map_data]
            map_data = [map_data[i * 51:(i + 1) * 51] for i in range(51)]
        # 检查数据是否为51x51
        if isinstance(map_data, list) and len(map_data) == 51 and all(len(row) == 51 for row in map_data):
            # 格式化地图数据为字符串，并添加边界
            border = '==='
            map_str = '\n'.join(
                border +
                ''.join(str(cell) for cell in row) +  # 网格内容
                border + '\n' +  # 行边界
                border  # 行末边界
                for row in map_data
            )
            
        else:
            print(f"错误：{map_name} 不是51x51的地图或不是长度为2601的一维数组。")
            return
        
        # 将网格数据写入文件
        with open(self.filename, 'a', encoding='utf-8') as file:
            file.write(f"{map_name}:\n")
            file.write(map_str + '\n')
            file.write("=" * 107 + '\n')  # 文件中的分隔线
    def print_obs_features(self, raw_obs):
        
        """
        打印观察特征中的地图数据。
        :param raw_obs: 包含特征的对象。
        """
        norm_pos = raw_obs.feature.norm_pos
        grid_pos = raw_obs.feature.grid_pos
        start_pos = raw_obs.feature.start_pos
        end_pos = raw_obs.feature.end_pos
        buff_pos = raw_obs.feature.buff_pos
        treasure_poss = list(raw_obs.feature.treasure_pos)

        feature_str = f"norm_pos在这里: \n{norm_pos}\n" \
                      f"grid_pos在这里: \n{grid_pos}\n" \
                      f"start_pos在这里: \n{start_pos}\n" \
                      f"end_pos在这里: \n{end_pos}\n" \
                      f"buff_pos在这里: \n{buff_pos}\n" \
        
        self.print(feature_str)
        self.print("\n".join([f"第{index + 1}个treasure_poss在这里:\n {pos}" for index, pos in enumerate(treasure_poss)]))
        # 获取特征中的地图数据
        obstacle_map = list(raw_obs.feature.obstacle_map)
        memory_map = list(raw_obs.feature.memory_map)
        treasure_map = list(raw_obs.feature.treasure_map)
        end_map = list(raw_obs.feature.end_map)

        # 打印每张地图
        self.print_map(obstacle_map, "视野中的障碍物地图","0","#")
        self.print_map(memory_map, "记忆地图")
        self.print_map(treasure_map, "视野中的宝藏地图","0","#")
        self.print_map(end_map, "视野中的终点地图","0","#")
    def print(self, *args, **kwargs):
        """
        自定义的print函数，支持格式化字符串并保存到文件。
        Also supports printing a 51x51 array in a visually formatted way.
        """

        # 格式化字符串
        if args and kwargs:
            message = args[0].format(*args[1:], **kwargs)
        else:
            message = ' '.join(str(arg) for arg in args)
        
        # 将消息保存到文本文件中
        with open(self.filename, 'a', encoding='utf-8') as file:
            file.write(message + '\n')
            file.write('-' * 52 + '\n')  # 添加分隔线，51个字符加1个换行符


output_filename = 'output_definition.txt'
printer = CustomPrinter(output_filename)


output_filename = 'weizhi.txt'
weizhi = CustomPrinter(output_filename)


class WeiZhiHuoQu:
    def __init__(self, filename):
        self.filename = filename
        data = self.load_from_json()  # 从JSON文件加载数据
        self.list=data.get('list', []) 
        self.index = data.get('index',[])

    def load_from_json(self):
        # 从JSON文件中加载数据
        with open(self.filename, 'r') as f:
            data = json.load(f)
            return data # 默认返回空列表

    def update_data(self, curr_pos_x, curr_pos_z, prev_buff_availability, buff_availability,prev_treasure_dists,treasure_dists,terminated):
        # 更新数据的函数
        for i in range(3,16):  #假设有16个判断
            list_t=self.list[i-1]
            if prev_treasure_dists[i-1] != 1 and treasure_dists[i-1] == 1 and (sum(treasure_dists) != sum(prev_treasure_dists)):
                if list_t:  # 如果当前维度的列表不为空
                    list_t.append((curr_pos_x,curr_pos_z))
                else:
                    list_t = [(curr_pos_x, curr_pos_z)]  # 初始化列表
                self.index[i-1] = len(list_t)  # 更新索引
                self.list[i-1]=list_t
                printer.print(f"收集了第{i-1}号宝箱的位置")

    def save_to_json(self):
        # 将数据保存到JSON文件
        with open(self.filename, 'w') as f:
            json.dump({'list': self.list, 'index': self.index}, f, indent=4)


# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def reward_shaping(
    frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info,act
):
    reward = 0
    #这里没有解包宝箱的方位信息

    grid_pos_x=obs.feature.grid_pos.x
    grid_pos_z=obs.feature.grid_pos.z

    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _env_info.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    # Get the grid-based distance of the current agent's position relative to the end point, buff, and treasure chest
    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    end_dist = _obs.feature.end_pos.grid_distance
    buff_dist = _obs.feature.buff_pos.grid_distance
    treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = env_info.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the grid-based distance of the agent's position from the previous
    # frame relative to the end point, buff, and treasure chest
    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = obs.feature.end_pos.grid_distance
    prev_buff_dist = obs.feature.buff_pos.grid_distance
    prev_treasure_dists = [pos.grid_distance for pos in obs.feature.treasure_pos]


    # Get the status of the buff
    # 获取buff的状态
    buff_availability = 0
    for organ in _env_info.frame_state.organs:
        if organ.sub_type == 2 and organ.status==1:
            buff_availability = 1

    prev_buff_availability = 0
    for organ in env_info.frame_state.organs:
        if organ.sub_type == 2 and organ.status==1:
            buff_availability = 1
    
    # Get the acceleration status of the agent
    # 获取智能体的加速状态
    prev_speed_up = env_info.frame_state.heroes[0].speed_up
    speed_up = _env_info.frame_state.heroes[0].speed_up

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
    printer.print(f"当前智能体的动作选择: {act}")
    printer.print(f"\n---------我是分割符---------\n")


    #暴力获取位置(如果出现问题,就尝试注释这三行代码)
    wz = WeiZhiHuoQu('data.json')
    wz.update_data(grid_pos_x, grid_pos_z, prev_buff_availability, buff_availability,prev_treasure_dists,treasure_dists,terminated)
    wz.save_to_json()
    
    """
        Reward 1. Reward related to the end point
        奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励

    # Boundary handling: At the first frame, prev_end_dist is initialized to 1,
    # and no reward is calculated at this time
    # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
    if prev_end_dist != 1:
        reward_end_dist += 1 if end_dist < prev_end_dist else 0

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        reward_win += 1

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    if prev_treasure_dists.count(1.0) < treasure_dists.count(1.0):
        reward_treasure = 1
        printer.print("拿到宝箱了")

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = 0

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励
    reward_buff = 0

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # Reward 4.1 Penalty for flickering into the wall (TODO)
    # 奖励4.1 撞墙闪现的惩罚 (TODO)

    # Reward 4.2 Reward for normal flickering (TODO)
    # 奖励4.2 正常闪现的奖励 (TODO)

    # Reward 4.3 Reward for super flickering (TODO)
    # 奖励4.3 超级闪现的奖励 (TODO)

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 1
    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # (TODO: Give penalty after collecting all the treasure chests, encourage full collection)
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0
    memory_map = obs.feature.memory_map

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0

    # Determine whether it bumps into the wall
    # 判断是否撞墙
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
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
def observation_process(raw_obs, env_info=None):
    """
    This function is an important feature processing function, mainly responsible for:
        - Parsing information in the raw data
        - Parsing preprocessed feature data
        - Processing the features and returning the processed feature vector
        - Concatenation of features
        - Annotation of legal actions
    Function inputs:
        - raw_obs: Preprocessed feature data
        - env_info: Environment information returned by the game
    Function outputs:
        - observation: Feature vector
        - legal_action: Annotation of legal actions

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

    # Unpack the preprocessed feature data according to the protocol
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

    # Feature processing 1: One-hot encoding of the current position
    # 特征处理1：当前位置的one-hot编码
    one_hot_pos = one_hot_encoding(grid_pos)

    # Feature processing 2: Normalized position
    # 特征处理2：归一化位置
    norm_pos = [norm_pos.x, norm_pos.z]

    # Feature processing 3: Information about the current position relative to the end point
    # 特征处理3：当前位置相对终点点位的信息
    end_pos_features = read_relative_position(end_pos)

    # Feature processing 4: Information about the current position relative to the treasure position
    # 特征处理4: 当前位置相对宝箱位置的信息
    treasure_poss_features = []
    for treasure_pos in treasure_poss:
        treasure_poss_features = treasure_poss_features + list(
            read_relative_position(treasure_pos)
        )

    # Feature processing 5: Whether the buff is collectable
    # 特征处理5：buff是否可收集
    buff_availability = 1
    if env_info:
        for organ in env_info.frame_state.organs:
            if organ.sub_type == 2:
                buff_availability = organ.status

    # Feature processing 6: Whether the flash skill can be used
    # 特征处理6：闪现技能是否可使用
    talent_availability = 1
    if env_info:
        talent_availability = env_info.frame_state.heroes[0].talent.status

    # Feature concatenation:
    # Concatenate all necessary features as vector features (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    # 特征拼接：将所有需要的特征进行拼接作为向量特征 (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    feature = (
        norm_pos
        + one_hot_pos
        + end_pos_features
        + treasure_poss_features
        + [buff_availability, talent_availability]
        + obstacle_map
        + end_map
        + treasure_map
        + memory_map
    )

    # Legal actions
    # 合法动作
    legal_act = list(raw_obs.legal_act)

    return ObsData(feature=feature, legal_act=legal_act)


@attached
def action_process(act_data):
    result = act_data.move_dir
    result += act_data.use_talent * 8
    return result


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
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



'''
    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励
    
    # Boundary handling: At the first frame, prev_end_dist is initialized to 1,
    # and no reward is calculated at this time
    # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
    if prev_end_dist != 1:
        reward_end_dist += 1 if end_dist < prev_end_dist else 0

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        reward_win += 1

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱-采用了贪心策略)
    #获取当前距离最近的宝箱的下标,然后对比两者的距离大小
    min_index = np.argmin(treasure_dists)
    if treasure_dists[min_index]<prev_treasure_dists[min_index]:
        reward_treasure_dist += 1

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    
    if prev_treasure_dists.count(1.0) < treasure_dists.count(1.0):
        reward_treasure += 1
        printer.print("拿到宝箱了")

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = 0
    if buff_availability and(prev_buff_dist>buff_dist):
        reward_buff_dist += 1

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励
    reward_buff = 0
    if prev_speed_up==0 and speed_up==1:
       reward_buff += 1 


    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # Reward 4.1 Penalty for flickering into the wall (TODO)
    # 奖励4.1 撞墙闪现的惩罚 (TODO)

    # Reward 4.2 Reward for normal flickering (TODO)
    # 奖励4.2 正常闪现的奖励 (TODO)

    # Reward 4.3 Reward for super flickering (TODO)
    # 奖励4.3 超级闪现的奖励 (TODO)

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 0
    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # (TODO: Give penalty after collecting all the treasure chests, encourage full collection)
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)
    if sum(prev_treasure_dists)==15 and sum(treasure_dists)==15 and (end_dist > prev_end_dist):
        reward_step += 1
    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0
    memory_map = obs.feature.memory_map

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0

    # Determine whether it bumps into the wall
    # 判断是否撞墙
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
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
'''
