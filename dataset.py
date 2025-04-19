# -*- coding: utf-8 -*-
#!/usr/bin/env python
from torch.utils import data
import torch
import logging
import h5py
import numpy as np
import random
import time
import torch.nn.functional as F
from natsort import natsorted

class DataSet(data.Dataset):
    def __init__(self, radar_list, load_count, start_index=0, test=False):
        self.radar_list = radar_list
        self.load_count = min(load_count, len(radar_list)) 
        self.test = test
        self.start_index = start_index
        if self.start_index >= len(self.radar_list):
            raise ValueError("start_index is out of range for radar_list")

    def __getitem__(self, index):
        if self.test:
            index = index + self.start_index
            if index >= len(self.radar_list):
                raise IndexError("Index out of range for radar_list")
        
        with h5py.File(self.radar_list[index], 'r') as fhandle:
            radar_data = fhandle[u'data'][1::2].astype(np.float32)
            target_reg = fhandle[u'label'][4::5].astype(np.float32)

        radar_data_tensor = torch.from_numpy(radar_data).float().unsqueeze(0)
        target_reg_tensor = torch.from_numpy(target_reg).float().unsqueeze(0)
        
        # Resize radar_data from 270x350 to 280x360 using bilinear interpolation
        radar_data_resized = F.interpolate(radar_data_tensor, size=(280, 360), mode='bilinear', align_corners=False).squeeze(0)
        # Resize target_reg from 270x350 to 280x360 using bilinear interpolation
        target_reg_resized = F.interpolate(target_reg_tensor, size=(280, 360), mode='bilinear', align_corners=False).squeeze(0)

        return radar_data_resized, target_reg_resized

    def __len__(self):
        return self.load_count

if __name__ == '__main__':
    # dataset list
    # /media/data8T/wy/two_stream/5-4/dataset_path/path_train_radar.txt
    with open('/home/wkl/Pixel/dataset_path/path_train_radar.txt', 'r') as fhandle:
        radar_dataset_list = fhandle.read().split('\n')
    radar_dataset_list.pop()

    print('radar_dataset_list num :%d' % len(radar_dataset_list))

    train_dataset = DataSet(radar_dataset_list, load_count=1055, start_index=0, test=True)

    print(len(train_dataset))

    trainloader = data.DataLoader(train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  drop_last=False)

    for ii, (input2, label_reg) in enumerate(trainloader):
        print(ii)
        print(input2.size())
        print(label_reg.size())
        print("------")