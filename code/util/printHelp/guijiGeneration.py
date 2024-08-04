#!/usr/bin/env python3
# -*- coding:utf-8 -*-

class guijiGeneration:
    def __init__(self,array=None):
        array=self.array

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