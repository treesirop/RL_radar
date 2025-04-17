import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import time
from typing import List

class OptimizedTemporalModelSelectorAgent(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, num_models=4, input_size=(280, 360)):
        super(OptimizedTemporalModelSelectorAgent, self).__init__()
        
        # Feature extractor for single-channel inputs
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Always expect 1 channel
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature size
        h, w = input_size
        h_out = h // 8
        w_out = w // 8
        self.feature_size = hidden_dim * h_out * w_out
        
        # Rest of the network is the same...
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_models)
            ) for _ in range(4)
        ])
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """Simple forward pass that treats each frame separately"""
        original_shape = x.shape
        batch_size = x.size(0)
        
        # Handle case where input is [batch_size, seq_len, height, width]
        if len(original_shape) == 4 and original_shape[1] > 1:
            # Reshape to [batch_size * seq_len, 1, height, width]
            seq_len = original_shape[1]
            x = x.view(batch_size * seq_len, 1, original_shape[2], original_shape[3])
            
            # Process through feature extractor
            features = self.feature_extractor(x)
            
            # Reshape back to account for sequence
            features = features.view(batch_size, seq_len, -1)
            
            # Average across sequence dimension
            features = features.mean(dim=1)
            
        # Handle standard case [batch_size, channels, height, width]
        elif len(original_shape) == 4 and original_shape[1] == 1:
            # Just process directly
            features = self.feature_extractor(x)
            features = features.view(batch_size, -1)
        
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
        
        # Process through FC layers
        # fc_features = self.fc(features)
        if features.size(0) == 1 and self.training:
            # During training with single sample, temporarily use eval mode for BatchNorm
            self.fc.eval()
            with torch.no_grad():
                fc_features = self.fc(features)
            self.fc.train()
        else:
            fc_features = self.fc(features)
        # Generate model selection probabilities
        model_probs = []
        for policy_head in self.policy_heads:
            logits = policy_head(fc_features)
            probs = F.softmax(logits, dim=1)
            model_probs.append(probs)
        
        # Stack time step outputs
        model_probs = torch.stack(model_probs, dim=1)
        
        # State value
        state_value = self.value_head(fc_features)
        
        return model_probs, state_value