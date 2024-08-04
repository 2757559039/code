#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import json

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