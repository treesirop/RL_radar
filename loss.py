# -*- coding: utf-8 -*-
# !/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import math

from config import DefaultConfigure
opt = DefaultConfigure()
class RegLoss(nn.Module):
    def __init__(self, task_num=4, v=[math.log(0.5)], 
                 mse_weight=0.4, mae_weight=0.6, NORMAL_LOSS_GLOBAL_SCALE=0.00005, 
                 width=280, height=360):
        super(RegLoss, self).__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.width = width
        self.height = height
        self.eps = 1e-10
        self.loss_func_reg_1 = nn.MSELoss(reduction='none')
        self.loss_func_reg_2 = nn.L1Loss(reduction='none')
        
        # 创建权重图
        weight = torch.zeros(self.width, self.height, dtype=torch.float)
        for i in range(24):
            weight[i:self.width-i, i:self.height-i].add_(1)
        weight_map = torch.nn.functional.softmax(weight.view(-1)).view(self.width, self.height)
        self.weight_map = nn.Parameter(weight_map, requires_grad=False)
        
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.tensor(v))       
        # 确保result目录存在
        import os
        os.makedirs("./result", exist_ok=True)
        self.logs = open("./result/log_vars.txt", "w")
        
        # 时间点权重
        self.time_weights = {
            'default': [1.0, 1.0, 1.0, 1.0],  # 默认权重
            '30': [1.0, 0.5, 0.3, 0.2],       # 侧重30分钟
            '60': [0.5, 1.0, 0.5, 0.3],       # 侧重60分钟
            '90': [0.3, 0.5, 1.0, 0.5],       # 侧重90分钟
            '120': [0.2, 0.3, 0.5, 1.0]       # 侧重120分钟
        }
        
    def _norm(self, ele):
        return (ele + 10) / 85.0
        
    def forward(self, output_reg, label_reg, focus_time='default'):
        """
        计算损失函数
        
        参数:
            output_reg: 回归输出
            label_reg: 回归标签
            focus_time: 侧重的时间点，可选值为'default', '30', '60', '90', '120'
        """
        # 获取时间权重
        weights = self.time_weights[focus_time]
        
        # 计算平衡权重
        balancing_weights = (1, 2, 5, 10, 40)   
        weights_tensor = torch.ones_like(output_reg) * balancing_weights[0]   
        thresholds = [self._norm(ele) for ele in balancing_weights[:-1]]  
        for i, threshold in enumerate(thresholds):       
            weights_tensor = weights_tensor + (balancing_weights[i + 1] - balancing_weights[i]) * (label_reg >= threshold).float() 
        
        # 计算MSE和MAE损失
        mse_losses = []
        mae_losses = []
        
        for i in range(4):  # 4个时间点
            # MSE损失
            if i == 0:
                mseerror = self.weight_map * self.loss_func_reg_1(output_reg[:,i,:,:], label_reg[:,i,:,:])
            else:
                mseerror = self.weight_map * weights_tensor[:,i,:,:] * self.loss_func_reg_1(output_reg[:,i,:,:], label_reg[:,i,:,:])
            mse_losses.append(mseerror.sum(dim=(1, 2)))
            
            # MAE损失
            if i == 0:
                maeerror = self.weight_map * self.loss_func_reg_2(output_reg[:,i,:,:], label_reg[:,i,:,:])
            else:
                maeerror = self.weight_map * weights_tensor[:,i,:,:] * self.loss_func_reg_2(output_reg[:,i,:,:], label_reg[:,i,:,:])
            mae_losses.append(maeerror.sum(dim=(1, 2)))
        
        # 计算均值
        u_values = []
        for mse, mae in zip(mse_losses, mae_losses):
            u_values.append(torch.mean(self.mse_weight * mse + self.mae_weight * mae))
        
        # 计算sigma
        sigmas = []
        for mse, mae, u in zip(mse_losses, mae_losses, u_values):
            sigma = (sgn(mse + mae - u) + sgn(abs(mse + mae - u))) / 2
            sigmas.append(sigma.cuda())
        
        # 计算回归损失
        reg_losses = []
        for sigma, mse, mae in zip(sigmas, mse_losses, mae_losses):
            reg_losses.append(sigma * (self.mse_weight * mse + self.mae_weight * mae))
        
        # 应用时间权重
        loss_reg = sum(w * reg_loss for w, reg_loss in zip(weights, reg_losses))
        loss_reg = loss_reg.unsqueeze(1)
        
        # 计算精度
        precision = torch.exp(-self.log_vars[0])
        loss = torch.sum(precision * loss_reg + abs(self.log_vars[0]), -1)
        
        # 记录日志
        precisions = [torch.exp(-self.log_vars[i]) for i in range(len(self.log_vars))]
        log_str = "    ".join([str(p) for p in precisions])
        self.logs.write(log_str + "\n")
        self.logs.flush()
        
        return torch.mean(loss)


# 针对特定时间点的损失函数类
class RegLoss30(RegLoss):
    def forward(self, output_reg,  label_reg):
        return super().forward(output_reg,  label_reg, focus_time='30')


class RegLoss60(RegLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='60')


class RegLoss90(RegLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='90')


class RegLoss120(RegLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='120')


def sgn(x):
    y = torch.ones(x.shape)
    y[x < torch.zeros(x.shape).cuda()] = -1
    y[x == torch.zeros(x.shape).cuda()] = 0
    return y

class MyLoss(nn.Module):
    def __init__(self, task_num = 2, v = [0, math.log(0.5)], mse_weight=0.4, mae_weight=0.6, NORMAL_LOSS_GLOBAL_SCALE=0.00005):
        super(MyLoss, self).__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.loss_func_reg_1 = nn.MSELoss(reduction='none')
        self.loss_func_reg_2 = nn.L1Loss(reduction='none')
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.tensor(v))
        
        # 时间点权重
        self.time_weights = {
            'default': [1.0, 1.0, 1.0, 1.0],  # 默认权重
            '30': [1.0, 0.5, 0.3, 0.2],       # 侧重30分钟
            '60': [0.5, 1.0, 0.5, 0.3],       # 侧重60分钟
            '90': [0.3, 0.5, 1.0, 0.5],       # 侧重90分钟
            '120': [0.2, 0.3, 0.5, 1.0]       # 侧重120分钟
        }
        
    def forward(self, output_reg, label_reg, focus_time='default'):
        # 获取时间权重
        weights = self.time_weights[focus_time]
        
        balancing_weights = opt.balancing_weights     
        weights_tensor = torch.ones_like(output_reg) * balancing_weights[0]   
        thresholds = [self._norm(ele) for ele in opt.thresholds]  
        for i, threshold in enumerate(thresholds):       
            weights_tensor = weights_tensor + (balancing_weights[i + 1] - balancing_weights[i]) * (label_reg >= threshold).float()         
        
        # 计算损失
        loss_reg_1 = self.loss_func_reg_1(output_reg, label_reg)      
        loss_reg_1 = (loss_reg_1 * weights_tensor).sum(dim=(1, 2, 3))         
        loss_reg_2 = self.loss_func_reg_2(output_reg, label_reg)
        loss_reg_2 = (loss_reg_2 * weights_tensor).sum(dim=(1, 2, 3))
        
        # 应用时间权重
        loss_reg = self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*loss_reg_1 + self.mae_weight*loss_reg_2)
        loss_reg = loss_reg * torch.tensor(weights, device=loss_reg.device).view(1, -1, 1, 1)
        loss_reg = loss_reg.sum(dim=1).unsqueeze(1)   #(b, 1)
        
        precision2 = torch.exp(-self.log_vars[1])
        loss = torch.sum(0.5 * precision2 * loss_reg + abs(self.log_vars[1]), -1)
        loss = torch.mean(loss)
        return loss
        
    def _norm(self, ele):
        return ele / 80.0 


class MyLoss30(MyLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='30')

class MyLoss60(MyLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='60')

class MyLoss90(MyLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='90')

class MyLoss120(MyLoss):
    def forward(self, output_reg, label_reg):
        return super().forward(output_reg, label_reg, focus_time='120')