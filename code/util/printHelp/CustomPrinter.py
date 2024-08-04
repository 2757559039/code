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