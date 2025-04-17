# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CSI import calc_CSI_reg

class OptimizedRadarEnvironment:
    def __init__(self, dataset, input_seq_len=10, output_seq_len=4, thresholds=(25, 35, 45, 55)):
        """
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
        
        # Metric weights for reward calculation
        self.metric_weights = {
            'csi': 0.4,  # Critical Success Index
            'pod': 0.3,  # Probability of Detection
            'far': 0.3   # False Alarm Rate
        }
        
    def reset(self):
        """Reset the environment to a new starting point"""
        # Randomize starting point for better exploration
        self.current_index = np.random.randint(0, self.max_index)
        self.episode_steps = 0
        
        try:
            input_seq, _ = self.dataset[self.current_index]
            return input_seq
        except Exception as e:
            print(f"Error during environment reset: {e}")
            # Fallback to a safe starting point
            self.current_index = 0
            input_seq, _ = self.dataset[0]
            return input_seq
    
    def step(self, action):
        """
        Execute action (prediction) and return next state, reward, done, info
        
        Args:
            action: Predicted future sequence
        """
        try:
            # Get ground truth for the current sample
            _, true_future = self.dataset[self.current_index]
            
            # Calculate rewards and metrics for each time step
            rewards = []
            metrics = []
            
            for t in range(min(self.output_seq_len, len(action))):
                # Ensure prediction and ground truth are numpy arrays with matching shapes
                # Extract prediction for current time step t
                pred_t = action[:, t].cpu().numpy() if isinstance(action, torch.Tensor) else action[t]
                true_t = true_future[t].cpu().numpy() if isinstance(true_future[t], torch.Tensor) else true_future[t]
                
                # Reshape tensors if necessary to match dimensions
                if len(pred_t.shape) > 2:  # Handle any extra dimensions
                    pred_t = pred_t.squeeze()
                if len(true_t.shape) > 2:
                    true_t = true_t.squeeze()
                
                # Ensure both arrays are 2D
                if len(pred_t.shape) == 1:
                    pred_t = pred_t.reshape(-1, 1)
                if len(true_t.shape) == 1:
                    true_t = true_t.reshape(-1, 1)
                
                # Evaluate at primary threshold (35 dBZ)
                try:
                    csi, pod, far = calc_CSI_reg(pred_t, true_t, threshold=35.0)
                except ValueError as e:
                    print(f"Shape mismatch - pred: {pred_t.shape}, true: {true_t.shape}")
                    raise e
                
                # Calculate weighted reward
                reward_t = (
                    self.metric_weights['csi'] * csi + 
                    self.metric_weights['pod'] * pod + 
                    self.metric_weights['far'] * (1 - far)  # Lower FAR is better
                )
                
                # Apply time-based weighting (earlier predictions more important)
                time_weight = 1.0 / (1.0 + 0.1 * t)
                reward_t *= time_weight
                
                rewards.append(reward_t)
                metrics.append({'CSI': csi, 'POD': pod, 'FAR': far})
            
            # Advance to next sample
            self.current_index = (self.current_index + 1) % self.max_index
            self.episode_steps += 1
            
            # Episode terminates after one complete epoch or max steps
            done = (self.current_index == 0) or (self.episode_steps >= 100)
            
            # Get next state if not done
            if not done:
                next_input, _ = self.dataset[self.current_index]
                next_state = next_input
            else:
                next_state = None
            
            # Total reward is sum of all time step rewards
            total_reward = sum(rewards) if rewards else 0.0
            
            return next_state, total_reward, done, {
                'rewards_per_step': rewards,
                'metrics': metrics,
                'episode_steps': self.episode_steps
            }
        
        except Exception as e:
            print(f"Error in environment step: {str(e)}")
            # Provide informative error and safe fallback
            return self.reset(), -1.0, True, {
                "error": str(e),
                "error_type": type(e).__name__,
                "current_index": self.current_index
            }