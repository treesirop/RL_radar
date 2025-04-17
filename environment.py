# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CSI import calc_CSI_reg

class OptimizedRadarEnvironment:
    def __init__(self, dataset, input_seq_len=10, output_seq_len=4, thresholds=(25, 35, 45, 55)):
        """1
        Optimized radar environment with better rewards and error handling
        
        Args:
            dataset: The radar dataset
            input_seq_len: Input sequence length
            output_seq_len: Output sequence length 
            thresholds: Rainfall intensity thresholds for evaluation
        """
        self.dataset = dataset
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.current_index = 0
        self.max_index = len(dataset)
        
        # Multiple thresholds for better evaluation
        self.thresholds = thresholds
        
    def reset(self):
        """Reset the environment to a new starting point"""
        # Randomize starting point for better exploration
        self.current_index = np.random.randint(0, self.max_index)
        self.episode_steps = 0
        
        input_seq, _ = self.dataset[self.current_index]
        return input_seq
    
    def step(self, action):
        """Execute action (prediction) and return next state, reward, done, info"""
        # Get ground truth for the current sample
        _, true_future = self.dataset[self.current_index]
        
        # Ensure action is detached and moved to CPU before numpy conversion
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()  # Changed from squeeze() to detach()
        if isinstance(true_future, torch.Tensor):
            true_future = true_future.detach().cpu().numpy()
        
        # Handle single sample case
        if action.ndim == 2:  # [H,W]
            action = action[np.newaxis, np.newaxis, ...]  # Add batch and time dims
        elif action.ndim == 3:  # [T,H,W]
            action = action[np.newaxis, ...]  # Add batch dim
            
        # Calculate MSE for each time step
        mse = np.mean([(pred_t - true_t)**2 for t, (pred_t, true_t) in enumerate(zip(action[0], true_future))])
        max_mse = 100.0  # 根据数据特性设定的最大可能MSE
        norm_mse = mse / max_mse  # 归一化到[0,1]
        mse_reward = -norm_mse  # [-1, 0]
        
        # 2. CSI奖励（鼓励预测与实际的重叠）
        csi_scores = []
        for t in range(len(true_future)):
            csi, _, _ = calc_CSI_reg(action[0,t], true_future[t], threshold=35)
            csi_scores.append(csi)
        csi_reward = np.mean(csi_scores)  # [0,1]
        
        # 3. 时序一致性奖励（鼓励预测序列平滑）
        temporal_penalty = 0
        if len(action[0]) > 1:
            diffs = np.diff(action[0], axis=0)
            temporal_penalty = -0.1 * np.mean(np.abs(diffs))  # 小幅惩罚剧烈变化
        
        # 4. 极端事件奖励（特别关注强降水）
        extreme_reward = 0
        for t in range(len(true_future)):
            true_extreme = (true_future[t] > 45).sum()
            pred_extreme = (action[0,t] > 45).sum()
            extreme_reward += 1.0 - min(1.0, abs(true_extreme - pred_extreme) / (true_extreme + 1e-6))
        extreme_reward /= len(true_future)
        
        # 组合奖励（可调整权重）
        reward = (
            0.4 * mse_reward + 
            0.3 * csi_reward + 
            0.1 * temporal_penalty + 
            0.2 * extreme_reward
        )
        
        # Advance to next sample
        self.current_index = (self.current_index + 1) % self.max_index
        self.episode_steps += 1
        
        done = (self.episode_steps >= 100)
        next_state = self.dataset[self.current_index][0] if not done else None
        
        return next_state, reward, done, {}
            