import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionLayer


class OptimizedTemporalModelSelectorAgent(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, num_models=4, input_size=(280, 360)):
        super(OptimizedTemporalModelSelectorAgent,self).__init__()

        # Feature extractor for single-channel inputs
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # Always expect 1 channel
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
            ) for _ in range(4) # Assuming 4 time steps
        ])

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def extract_features(self, x):
        """封装特征提取流程（含形状处理）"""
        original_shape = x.shape
        batch_size = x.size(0)
        
        # 处理序列输入的情况
        if len(original_shape) == 4 and original_shape[1] > 1:
            seq_len = original_shape[1]
            x_reshaped = x.view(batch_size * seq_len, 1, original_shape[2], original_shape[3])
            features = self.feature_extractor(x_reshaped)
            features = features.view(batch_size, seq_len, -1).mean(dim=1)
        # 处理单帧输入的情况
        elif len(original_shape) == 4 and original_shape[1] == 1:
            x_reshaped = x.squeeze(1)
            features = self.feature_extractor(x_reshaped)
            features = features.view(batch_size, -1)
        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")
            
        # 处理BatchNorm的特殊情况
        if features.size(0) == 1 and self.training:
            self.fc.eval()
            with torch.no_grad():
                fc_features = self.fc(features)
            self.fc.train()
        else:
            fc_features = self.fc(features)
            
        return fc_features

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
        # Need to handle BatchNorm1d correctly during training if batch_size=1
        if features.size(0) == 1 and self.training:
            # Use eval mode for BatchNorm layers if batch size is 1
            self.fc.eval()
            with torch.no_grad():
                 fc_features = self.fc(features)
            self.fc.train() # Set back to train mode
        else:
            fc_features = self.fc(features)

        # Generate model weights
        model_weights = []
        for policy_head in self.policy_heads:
            logits = policy_head(fc_features)
            # Apply softmax to get weights summing to 1 for each time step
            weights = F.softmax(logits, dim=1)
            model_weights.append(weights)

        # Stack time step outputs: shape [batch_size, 4, num_models]
        model_weights = torch.stack(model_weights, dim=1) # <--- Renamed variable

        # State value
        state_value = self.value_head(fc_features)

        return model_weights, state_value
    
    

class DynamicFusionAgent(OptimizedTemporalModelSelectorAgent):
    def __init__(self, input_channels=1, hidden_dim=64, num_models=4, input_size=(280, 360)):
        super().__init__(input_channels, hidden_dim, num_models, input_size)
        
        # Prediction encoder
        self.prediction_encoder = nn.Sequential(
            nn.Conv3d(in_channels=num_models,
                      out_channels=16,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        self.combined_fc_in = hidden_dim + 32
        
        # Combined FC layers with LayerNorm instead of BatchNorm
        self.combined_fc = nn.Sequential(
            nn.Linear(self.combined_fc_in, 512),
            nn.LayerNorm(512),  # Using LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, model_predictions):
        B, T, M, H, W = model_predictions.shape
        
        # Extract original features
        original_features = super().extract_features(x)
        
        # Process predictions
        preds_5d = model_predictions.permute(0, 2, 1, 3, 4)  # [B, M, T, H, W]
        pred_features = self.prediction_encoder(preds_5d)
        pred_features = pred_features.view(B, 32, T).mean(dim=2)  # [B, 32]
        
        # Feature fusion
        combined = torch.cat([original_features, pred_features], dim=1)
        
        # Through FC layers
        fc_out = self.combined_fc(combined)
        
        # Generate weights
        model_weights = []
        for head in self.policy_heads:
            logits = head(fc_out)
            weights = F.softmax(logits, dim=1)
            model_weights.append(weights)
        model_weights = torch.stack(model_weights, dim=1)
        
        # Value estimation
        state_value = self.value_head(fc_out)
        
        return model_weights, state_value
    
class SpatioTemporalAttentionFusionAgent(OptimizedTemporalModelSelectorAgent):
    def __init__(self, input_channels=1, hidden_dim=64, num_models=4, input_size=(280, 360)):
        super().__init__(input_channels, hidden_dim, num_models, input_size)
        
        # 预测编码器
        self.prediction_encoder = nn.Sequential(
            nn.Conv3d(in_channels=num_models,
                      out_channels=16,
                      kernel_size=(3, 3, 3),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # 时间注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # 空间注意力 (修复输出通道数)
        self.spatial_attention = AttentionLayer(
            input_channels=hidden_dim + 32,
            output_channels=hidden_dim + 32  # 设置为与输入相同的维度
        )
        
        # 最终FC层
        self.final_fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, model_predictions):
        B, T, M, H, W = model_predictions.shape
        
        # 提取原始特征
        original_features = super().extract_features(x)  # [B, hidden_dim]
        
        # 处理预测
        preds_5d = model_predictions.permute(0, 2, 1, 3, 4)  # [B, M, T, H, W]
        pred_features = self.prediction_encoder(preds_5d)
        pred_features = pred_features.view(B, 32, T).mean(dim=2)  # [B, 32]
        
        # 拼接特征
        combined = torch.cat([original_features, pred_features], dim=1)  # [B, hidden_dim+32]
        
        # 应用时间注意力
        query = original_features.unsqueeze(0)  # [1, B, hidden_dim]
        key = value = original_features.unsqueeze(0)  # [1, B, hidden_dim]
        
        attended_temporal, _ = self.temporal_attention(query, key, value)
        attended_temporal = attended_temporal.squeeze(0)  # [B, hidden_dim]
        
        # 应用空间注意力
        attended_spatial = combined.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_dim+32, 1, 1]
        attended_spatial = self.spatial_attention(attended_spatial)  # 输出 [B, hidden_dim+32, 1, 1]
        
        # 融合两种注意力结果
        attended_spatial = attended_spatial.view(B, -1)  # [B, hidden_dim+32]
        fused = torch.cat([attended_temporal, attended_spatial[:, -32:]], dim=1)  # [B, hidden_dim+32]
        
        # 应用最终FC
        fc_out = self.final_fc(fused[:, :64])  # 只取前hidden_dim维
        
        # 生成权重
        model_weights = []
        for head in self.policy_heads:
            logits = head(fc_out)
            weights = F.softmax(logits, dim=1)
            model_weights.append(weights)
        model_weights = torch.stack(model_weights, dim=1)
        # 在agent的forward函数中添加：
        model_weights = torch.clamp(model_weights, max=0.8)  # 限制单一模型权重≤0.8
        model_weights = model_weights / model_weights.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 价值估计
        state_value = self.value_head(fc_out)
        
        return model_weights, state_value