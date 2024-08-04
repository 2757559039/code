#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :config.py
@Author  :kaiwu
@Date    :2023/7/1 10:37

"""
import json
# Configuration
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中dqn的维度是21624, target_dqn的维度是21624
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    ### 样本维度?
    SAMPLE_DIM = 21624

    # observation的维度，注意在我们的示例代码中原特征维度是10808，这里是经过CNN处理之后的维度与原始向量特征拼接后的维度
    ### 观测维度?
    DIM_OF_OBSERVATION = 4096 + 404

    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # 描述如何进行特征分割，示例代码中的特征处理成向量+特征图，以下配置描述了两者的维度
    # pos_float + pos_onehot + organ + cd&talent, obstacle_map, treasure_map, end_map, location_memory
    #         2 +   128*2    +  9*17 +     2,     51*51*4
    # ???
    DESC_OBS_SPLIT = [404, (4, 51, 51)]  # sum = 10808

    # 以下是一些算法配置

    # target网络的更新频率
    TARGET_UPDATE_FREQ = 1000

    # 探索因子, epsilon的计算见上面注释中的函数
    ### ?
    EPSILON_GREEDY_PROBABILITY = 300000

    # RL中的回报折扣GAMMA
    GAMMA = 0.9

    # epsilon
    EPSILON = 0.9

    # 初始的学习率
    START_LR = 1e-3

    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
#配置类
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

            if prev_treasure_dists[i-1] != 1 and treasure_dists[i-1] == 1 and sum(treasure_dists)!=15 and sum(prev_treasure_dists)!=15:
                if self.list[i-1]:  # 如果当前维度的列表不为空
                    self.list[i-1].append((curr_pos_x,curr_pos_z))
                else:
                    self.list[i-1] = [(curr_pos_x, curr_pos_z)]  # 初始化列表
                self.index[i-1] += 1  # 更新索引

    def save_to_json(self):
        # 将数据保存到JSON文件
        with open(self.filename, 'w') as f:
            json.dump({'list': self.list, 'index': self.index}, f, indent=4)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

class guijiGeneration:
    def __init__(self,array=None):
        self.array=array
    def convert_to_str_2d(self,array_2d):
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

    
    def print_128x128_array(self,array):
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

        array = [row.copy() for row in reversed(array)]
        array = self.convert_to_str_2d(array)

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
        array = [[0 for _ in range(128)] for _ in range(128)]

#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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

