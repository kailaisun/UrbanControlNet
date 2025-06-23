import cv2
import glob
import numpy as np
import os
import pandas as pd
import re
from torch.utils.data import Dataset
from itertools import compress
import torch
import json
# from annotator.hed import HEDdetector
# from annotator.util import HWC3

# from decimal import Decimal

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, data_dir,paths=False):

        self.data = []
        self.paths = paths
        with open(data_dir, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)





    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        dem_filename = item['dem']
        prompt = item['prompt']

        # 读取图像和提示图像
        target = cv2.imread(target_filename)
        source = cv2.imread(source_filename)
        # print(source_filename)
        dem = cv2.imread(dem_filename)


        # 如果提示图像是4通道，转换为3通道
        if source.shape[2] == 4:
            trans_mask = source[:, :, 3] == 0
            source[trans_mask] = [255, 255, 255, 255]
            source = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)

        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)
        dem = cv2.resize(dem, (512, 512), interpolation=cv2.INTER_AREA)

        # 转换颜色空间
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        dem = cv2.cvtColor(dem, cv2.COLOR_BGR2RGB)

        # 归一化
        source = source.astype(np.float32) / 255.0
        dem = dem.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0


        source = np.concatenate([source, dem], axis=2)  # add more hint images

        # 转换为张量并移动到设备
        # to(device)
        target = torch.tensor(target)  #.to(device)
        source = torch.tensor(source)#.to(device)
        # print(target.shape,source.shape)

        if self.paths==True:
            return target_filename, prompt,source_filename,dem_filename

        return dict(jpg=target, txt=prompt, hint=source)


# 示例使用
if __name__ == "__main__":
    data_dir = './urban_data/hundredcities/train.json.json'

    # 初始化数据集
    dataset = MyDataset(data_dir)
    print(f"Dataset size: {len(dataset)}")

    # 如果数据集为空，提示用户检查
    if len(dataset) == 0:
        print("Dataset is empty. Please check if the file names in the directories match.")
    else:
        sample = dataset[0]
        print(sample["txt"])
