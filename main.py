# -*- coding: utf-8 -*-
# !/usr/bin/python
from matplotlib import colors
from tqdm import tqdm
from CSI import calc_CSI_reg
from models.SimVP.simvp_model import SimVP_Model
from models.ConvLSTM.ConvLSTM import ConvLSTM
from models.CubiodTransformer.cuboid_transformer import CuboidTransformerModel
from models.ForcastNet.afnonet import AFNONet
from loss import Loss
from config import DefaultConfigure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse

# 导入项目中的其他模块
from dataset import DataSet
from model import Net
from agent import SpatioTemporalAttentionFusionAgent
from environment import OptimizedRadarEnvironment
from train_utils import train_rl_agent_optimized

opt = DefaultConfigure()
# 设置随机种子以确保结果可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 加载数据集
def load_data(batch_size=opt.batch_size, val_batch_size=opt.batch_size, data_root=None, num_workers=opt.num_workers):
    # 读取训练数据列表
    with open(data_root, 'r') as fhandle:
        train_radar_list = fhandle.read().split('\n')
    if train_radar_list[-1] == '':
        train_radar_list.pop()
    
    # 读取测试数据列表
    with open(opt.radar_test_data_root, 'r') as fhandle:
        test_radar_list = fhandle.read().split('\n')
    if test_radar_list[-1] == '':
        test_radar_list.pop()
    
    # 创建数据集
    train_dataset = DataSet(train_radar_list, len(train_radar_list))# len(train_radar_list)
    test_dataset = DataSet(test_radar_list, 374, start_index=0, test=True) #374, start_index=0,
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, train_dataset, None, None


def load_test_data(opt, batch_size=None):
    """只加载测试数据"""
    if batch_size is None:
        batch_size = opt.batch_size
    
    # 读取测试数据列表
    with open(opt.radar_test_data_root, 'r') as fhandle:
        test_radar_list = fhandle.read().split('\n')
    if test_radar_list[-1] == '':
        test_radar_list.pop()
    
    # 创建测试数据集
    test_dataset = DataSet(test_radar_list, len(test_radar_list)-374,start_index=374, test=True)
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    
    return test_loader

def train_models():
    """训练四个基础预测模型"""
    # 设置随机种子
    setup_seed(42)
    
    # 加载数据
    train_loader, test_loader, train_dataset, _, _ = load_data(
        batch_size=opt.batch_size, 
        val_batch_size=opt.batch_size,
        data_root=opt.radar_train_data_root, 
        num_workers=opt.num_workers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建四个模型实例
    models = {
        '30': Net().to(device),
        '60': Net().to(device),
        '90': Net().to(device),
        '120': Net().to(device)
    }
    
    # 为每个模型设置优化器
    optimizers = {
        time: torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        for time, model in models.items()
    }
    schedulers = {
        time: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
        for time, optimizer in optimizers.items()
    }
    
    # 创建保存模型的目录
    model_path = "./model_path"
    os.makedirs(model_path, exist_ok=True)
    
    # 初始化最佳验证损失
    best_val_loss = {time: float('inf') for time in models.keys()}
    
    # 训练循环
    epochs = 10
    for epoch in range(epochs):
        for time, model in models.items():
            model.train()
            epoch_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizers[time].zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                
                # 使用特定时间点的损失函数
                 # Ensure loss function is on same device
                if time == '30':
                    loss_func = Loss().to(device)
                elif time == '60':
                    loss_func = Loss().to(device)
                elif time == '90':
                    loss_func = Loss().to(device)
                elif time == '120':
                    loss_func = Loss().to(device)
                
                # 计算损失
                loss = loss_func(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizers[time].step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Model {time} - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    mse_loss = F.mse_loss(outputs, targets)
                    mae_loss = F.l1_loss(outputs, targets)
                    loss = 0.4 * mse_loss + 0.6 * mae_loss
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            print(f"Model {time} - Epoch {epoch}, Train Loss: {epoch_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
            
            schedulers[time].step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss[time]:
                best_val_loss[time] = val_loss
                torch.save(model.state_dict(), os.path.join(model_path, f"regression_model_{time}_best.pth"))
                print(f"Saved new best model for time {time} with val loss {val_loss:.6f}")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存所有模型
    # for time, model in models.items():
    #     torch.save(model.state_dict(), os.path.join(model_path, f"regression_model_{time}.pth"))
    
    print("基础模型训练完成!")
    
def train_models_with_different_batch():
    """训练四个基础预测模型，每个模型使用不同的batch size"""
    # 设置随机种子
    setup_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义每个模型对应的batch size
    batch_sizes = {
        '30': 4,    # 模型30使用batch size 32
        '60': 2,    # 模型60使用batch size 64
        '90': 2,   # 模型90使用batch size 128
        '120': 1   # 模型120使用batch size 256
    }
    
    # 创建四个模型实例
    models = {
        '30': SimVP_Model(in_shape=(5,1,280,360)).to(device),   
        '60': CuboidTransformerModel(input_shape=[5,280,360,1],target_shape=[4,280, 360,1]).to(device),  
        '90': ConvLSTM().to(device),  
        '120': AFNONet(img_size=(280, 360), patch_size=(4,4), in_chans=5, out_chans=4).to(device) 
    }
    
    # 为每个模型设置优化器
    optimizers = {
        time: torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        for time, model in models.items()
    }
    
    schedulers = {
        time: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
        for time, optimizer in optimizers.items()
    }
    
    # 创建保存模型的目录
    model_path = "./model_path"
    os.makedirs(model_path, exist_ok=True)
    
    # 初始化最佳验证损失
    best_val_loss = {time: float('inf') for time in models.keys()}
    
    # 训练循环
    epochs = 10
    for time, model in models.items():
        for epoch in range(epochs):
            # 为当前模型加载对应batch size的数据
            train_loader, test_loader,_ , _, _ = load_data(
                batch_size=batch_sizes[time], 
                val_batch_size=batch_sizes[time],  # 验证集使用相同batch size
                data_root=opt.radar_train_data_root, 
                num_workers=opt.num_workers
            )
            
            model.train()
            epoch_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizers[time].zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                
                # 使用特定时间点的损失函数
                loss_func = Loss().to(device)
                
                # 计算损失
                loss = loss_func(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizers[time].step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Model {time} (bs={batch_sizes[time]}) - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    mse_loss = F.mse_loss(outputs, targets)
                    mae_loss = F.l1_loss(outputs, targets)
                    loss = 0.4 * mse_loss + 0.6 * mae_loss
                    val_loss += loss.item()
            
            val_loss /= len(test_loader)
            print(f"Model {time} (bs={batch_sizes[time]}) - Epoch {epoch}, Train Loss: {epoch_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
            
            schedulers[time].step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss[time]:
                best_val_loss[time] = val_loss
                torch.save(model.state_dict(), os.path.join(model_path, f"regression_model_{time}_best.pth"))
                print(f"Saved new best model for time {time} with val loss {val_loss:.6f}")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print("基础模型训练完成!")

def train_agent():
    """Train the reinforcement learning agent with improved selection logic"""
    # Load data
    _, _, train_dataset, _, _ = load_data(
        batch_size=opt.agent_batch_size, 
        val_batch_size=opt.agent_batch_size,
        data_root=opt.radar_train_data_root, 
        num_workers=opt.num_workers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models and load weights
    models = {
        '30': SimVP_Model(in_shape=(5,1,280,360)).to(device),   
        '60': CuboidTransformerModel(input_shape=[5,280,360,1],target_shape=[4,280, 360,1]).to(device),  
        '90': ConvLSTM().to(device),  
        '120': AFNONet(img_size=(280, 360), patch_size=(4,4), in_chans=5, out_chans=4).to(device) 
    }
    
    model_path = "./model_path"
    for time, model in models.items():
        model.load_state_dict(torch.load(os.path.join(model_path, f"regression_model_{time}_best.pth")))
        model.eval()  # Set to evaluation mode
    
    # Initialize the optimized agent
    input_size = (280, 360)  # Match your actual input dimensions
    agent = SpatioTemporalAttentionFusionAgent(
        input_channels=1, 
        hidden_dim=64, 
        num_models=len(models),
        input_size=input_size
    ).to(device)
    
    # Use Adam with weight decay to prevent overfitting
    agent_optimizer = torch.optim.Adam(
        agent.parameters(), 
        lr=0.001,
        weight_decay=1e-5
    )
    
    # Initialize the optimized environment
    env = OptimizedRadarEnvironment(
        dataset=train_dataset,
        thresholds=(25, 35, 45, 55)  # Multiple thresholds for better evaluation
    )
    
    print("Starting agent training with improved selection logic...")
    
    # Train agent with optimized function
    trained_agent, rewards, metrics = train_rl_agent_optimized(
        env=env,
        agent=agent,
        models=models,
        optimizer=agent_optimizer,
        num_episodes=opt.agent_epochs,
        # batch_size=opt.agent_batch_size,
        # early_stopping_patience=10
    )
    
    print("Agent training complete!")
    
    # Save training results
    with open('result.txt', 'w') as f:
        f.write("Agent Training Results:\n")
        f.write(f"Final Average Reward: {np.mean(rewards[-10:]):.4f}\n\n")
        
        f.write("Final Metrics (Last 10 Episodes):\n")
        avg_csi = np.mean([m['csi'] for m in metrics[-10:]])
        avg_pod = np.mean([m['pod'] for m in metrics[-10:]])
        avg_far = np.mean([m['far'] for m in metrics[-10:]])
        
        f.write(f"CSI: {avg_csi:.4f}\n")
        f.write(f"POD: {avg_pod:.4f}\n")
        f.write(f"FAR: {avg_far:.4f}\n")
    
    return trained_agent

def denormalize(image):
    """将归一化的图像数据转换回原始尺度"""
    return image * 85 - 10

def evaluate_system_with_metrics():
    """评估系统性能并可视化结果，同时计算每个时间步的详细指标"""
    # 加载配置
    opt = DefaultConfigure()
    
    # 加载测试数据
    test_loader = load_test_data(opt)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载训练好的模型
    models = {
        '30': SimVP_Model(in_shape=(5,1,280,360)).to(device),   
        '60': CuboidTransformerModel(input_shape=[5,280,360,1],target_shape=[4,280, 360,1]).to(device),  
        '90': ConvLSTM().to(device),  
        '120': AFNONet(img_size=(280, 360), patch_size=(4,4), in_chans=5, out_chans=4).to(device) 
    }
    class ParallelPredictor(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models.values())
                
            def forward(self, x):
                return torch.stack([model(x) for model in self.models], dim=2)
        
    parallel_predictor = ParallelPredictor(models).eval().to(device)
    model_path = "./model_path"
    for time, model in models.items():
        model.load_state_dict(torch.load(os.path.join(model_path, f"regression_model_{time}_best.pth"), map_location=device))
        model.eval()
        print(f"Loaded model for time {time} from {os.path.join(model_path, f'regression_model_{time}_best.pth')}")
    
    # 加载训练好的智能体
    input_size = (280, 360)  # Match your actual input dimensions
    agent = SpatioTemporalAttentionFusionAgent(
        input_channels=1, 
        hidden_dim=64, 
        num_models=len(models),
        input_size=input_size
    ).to(device)
    agent_path = os.path.join('rl_model', 'best_agent_weighted.pth')
    agent.load_state_dict(torch.load(agent_path, map_location=device)['model_state_dict'])
    agent.eval()
    print(f"Loaded agent from {agent_path}")
    
    # 定义时间步
    time_steps = ['30', '60', '90', '120']
    
    # 初始化结果结构
    results = {
        'agent': {
            'by_time': {t: {'csi': [], 'pod': [], 'far': []} for t in time_steps},
            'average': {'csi': [], 'pod': [], 'far': []}
        },
        'models': {
            model_time: {
                'by_time': {t: {'csi': [], 'pod': [], 'far': []} for t in time_steps},
                'average': {'csi': [], 'pod': [], 'far': []}
            } for model_time in models.keys()
        }
    }
    
    # 设置绘图参数
    cdict = ['whitesmoke', 'dodgerblue', 'limegreen', 'green', 'darkgreen', 'yellow', 'goldenrod', 'orange', 'red',
             'darkred']
    clevs = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    my_map = colors.ListedColormap(cdict)
    norm = colors.BoundaryNorm(clevs, len(clevs) - 1)
    
    # 创建结果保存目录
    os.makedirs('result', exist_ok=True)
    
    # 开始评估
    processed_batches = 0
    batch_limit = None  # 设置为None处理所有批次，或者设置一个数字限制批次数
    
    # 设置评估的阈值
    thresholds = [25, 35, 45, 55]
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch_limit is not None and batch_idx >= batch_limit:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            all_model_preds = []
            model_preds = parallel_predictor(inputs)
            # 智能体生成权重并融合预测
            model_probs, _ = agent(inputs, model_preds)  # model_probs形状: [batch_size, 4, num_models]
            print(f"Batch {batch_idx} agent weights (first sample):")
            print(model_probs[0])  # 查看第一个样本的权重分布
            # 获取所有模型的预测 (形状: [batch_size, 4, num_models, H, W])
            all_model_preds = torch.stack([model(inputs) for model in models.values()], dim=2)

            # 加权融合 (广播权重到空间维度)
            agent_predictions = torch.sum(
                model_probs.unsqueeze(-1).unsqueeze(-1) * all_model_preds, 
                dim=2  # 沿num_models维度求和
            )  # 输出形状: [batch_size, 4, H, W]
            
            # 计算每个时间步的指标
            for threshold in thresholds:
                # 智能体指标
                for t in range(min(4, targets.size(1))):
                    # 提取该时间步的预测和真实值
                    agent_time_preds = agent_predictions[:, t].cpu().numpy()
                    time_targets = targets[:, t].cpu().numpy()
                    
                    # 反归一化
                    denorm_agent_preds = denormalize(agent_time_preds)
                    denorm_time_targets = denormalize(time_targets)
                    
                    # 计算指标
                    metrics = calc_CSI_reg(denorm_agent_preds, denorm_time_targets, threshold=threshold)
                    
                    # 存储该时间步的指标
                    time_step = time_steps[t]
                    results['agent']['by_time'][time_step]['csi'].append(float(metrics[0]))
                    results['agent']['by_time'][time_step]['pod'].append(float(metrics[1]))
                    results['agent']['by_time'][time_step]['far'].append(float(metrics[2]))
                
                # 计算智能体的整体指标（所有时间步）
                denorm_agent_preds_all = denormalize(agent_predictions.cpu().numpy())
                denorm_targets_all = denormalize(targets.cpu().numpy())
                metrics_all = calc_CSI_reg(denorm_agent_preds_all, denorm_targets_all, threshold=threshold)
                
                results['agent']['average']['csi'].append(float(metrics_all[0]))
                results['agent']['average']['pod'].append(float(metrics_all[1]))
                results['agent']['average']['far'].append(float(metrics_all[2]))
                
                # 计算每个模型在每个时间步的指标
                for model_time, model in models.items():
                    model_predictions = model(inputs)
                    
                    # 计算每个时间步的指标
                    for t in range(min(4, targets.size(1))):
                        # 提取该时间步的预测和真实值
                        model_time_preds = model_predictions[:, t].cpu().numpy()
                        time_targets = targets[:, t].cpu().numpy()
                        
                        # 反归一化
                        denorm_model_preds = denormalize(model_time_preds)
                        denorm_time_targets = denormalize(time_targets)
                        
                        # 计算指标
                        metrics = calc_CSI_reg(denorm_model_preds, denorm_time_targets, threshold=threshold)
                        
                        # 存储该时间步的指标
                        time_step = time_steps[t]
                        results['models'][model_time]['by_time'][time_step]['csi'].append(float(metrics[0]))
                        results['models'][model_time]['by_time'][time_step]['pod'].append(float(metrics[1]))
                        results['models'][model_time]['by_time'][time_step]['far'].append(float(metrics[2]))
                    
                    # 计算模型的整体指标（所有时间步）
                    denorm_model_preds_all = denormalize(model_predictions.cpu().numpy())
                    metrics_all = calc_CSI_reg(denorm_model_preds_all, denorm_targets_all, threshold=threshold)
                    
                    results['models'][model_time]['average']['csi'].append(float(metrics_all[0]))
                    results['models'][model_time]['average']['pod'].append(float(metrics_all[1]))
                    results['models'][model_time]['average']['far'].append(float(metrics_all[2]))
            
            # 每隔一定批次生成可视化结果
            if batch_idx % 10 == 0:  # 每10个批次生成一次图像
                print(f"正在生成第{batch_idx}批次的预测结果图...")
                # 在evaluate_system_with_metrics中添加：
               
                # 创建多行子图布局
                int_time_steps = [30, 60, 90, 120]
                num_models = len(models)
                rows = 2 + num_models  # 真实值(1行) + 智能体(1行) + 每个模型一行
                cols = len(int_time_steps) # 每行4个时间步

                plt.figure(figsize=(24, 6 * rows))# 调整画布尺寸

                # 第一行：真实值的四个时间步
                for t in range(len(int_time_steps)):
                    plt.subplot(rows, cols, 1 + t)
                    true_frame = denormalize(targets[0, t]).cpu().numpy()
                    plt.title(f'Ground Truth t+{int_time_steps[t]}min')
                    img = plt.imshow(true_frame, cmap=my_map, norm=norm)
                    plt.colorbar(img, fraction=0.046, pad=0.04)
                    plt.axis('off')

                # 第二行：智能体预测的四个时间步
                for t in range(len(int_time_steps)):
                    plt.subplot(rows, cols, cols + 1 + t)
                    pred_frame = denormalize(agent_predictions[0, t]).cpu().numpy()
                    plt.title(f'Agent t+{int_time_steps[t]}min')
                    img = plt.imshow(pred_frame, cmap=my_map, norm=norm)
                    plt.colorbar(img, fraction=0.046, pad=0.04)
                    plt.axis('off')

                # 后续行：各专家模型的预测结果（每行4个）
                for model_idx, (model_time, model) in enumerate(models.items()):
                    with torch.no_grad():
                        preds = model(inputs).cpu()
                    
                    # 获取该模型对应的时间步索引映射
                    time_step_mapping = {
                        30: 0,   # 所有模型预测的第一个时间步对应30分钟
                        60: 1,   # 第二个时间步对应60分钟
                        90: 2,   # 第三个时间步对应90分钟
                        120: 3   # 第四个时间步对应120分钟
                    }
                    
                    for col_idx, t in enumerate(int_time_steps):
                        subplot_pos = (2 + model_idx) * cols + col_idx + 1
                        if subplot_pos > rows * cols:
                            continue
                        
                        plt.subplot(rows, cols, subplot_pos)
                        
                        # 获取正确的时间步索引
                        pred_step = time_step_mapping[t]
                        pred_step = max(0, min(pred_step, preds.shape[1]-1))
                        
                        pred_frame = denormalize(preds[0, pred_step]).numpy()
                        plt.title(f"{model_time}min Model\nt+{t}min")
                        img = plt.imshow(pred_frame, cmap=my_map, norm=norm)
                        plt.colorbar(img, fraction=0.046, pad=0.04)
                        plt.axis('off')
                    
                plt.tight_layout()
                plt.savefig(f'result/comparison_batch{batch_idx}_time.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"第{batch_idx}批次的预测结果图已保存至result/comparison_batch{batch_idx}_time.png")
            
            processed_batches += 1
    
    # 计算平均指标（所有批次的平均值）
    print("\n计算平均指标...")
    average_results = {
        'agent': {
            'by_time': {t: {'csi': 0.0, 'pod': 0.0, 'far': 0.0} for t in time_steps},
            'average': {'csi': 0.0, 'pod': 0.0, 'far': 0.0}
        },
        'models': {
            model_time: {
                'by_time': {t: {'csi': 0.0, 'pod': 0.0, 'far': 0.0} for t in time_steps},
                'average': {'csi': 0.0, 'pod': 0.0, 'far': 0.0}
            } for model_time in models.keys()
        }
    }
    
    # 计算智能体的平均指标
    for time_step in time_steps:
        average_results['agent']['by_time'][time_step]['csi'] = np.mean(results['agent']['by_time'][time_step]['csi'])
        average_results['agent']['by_time'][time_step]['pod'] = np.mean(results['agent']['by_time'][time_step]['pod'])
        average_results['agent']['by_time'][time_step]['far'] = np.mean(results['agent']['by_time'][time_step]['far'])
    
    average_results['agent']['average']['csi'] = np.mean(results['agent']['average']['csi'])
    average_results['agent']['average']['pod'] = np.mean(results['agent']['average']['pod'])
    average_results['agent']['average']['far'] = np.mean(results['agent']['average']['far'])
    
    # 计算各模型的平均指标
    for model_time in models.keys():
        for time_step in time_steps:
            average_results['models'][model_time]['by_time'][time_step]['csi'] = np.mean(results['models'][model_time]['by_time'][time_step]['csi'])
            average_results['models'][model_time]['by_time'][time_step]['pod'] = np.mean(results['models'][model_time]['by_time'][time_step]['pod'])
            average_results['models'][model_time]['by_time'][time_step]['far'] = np.mean(results['models'][model_time]['by_time'][time_step]['far'])
        
        average_results['models'][model_time]['average']['csi'] = np.mean(results['models'][model_time]['average']['csi'])
        average_results['models'][model_time]['average']['pod'] = np.mean(results['models'][model_time]['average']['pod'])
        average_results['models'][model_time]['average']['far'] = np.mean(results['models'][model_time]['average']['far'])
    
    # 保存评估结果
    with open('result/detailed_metrics.txt', 'w') as f:
        f.write("===== 系统评估详细结果 =====\n")
        f.write(f"处理批次数: {processed_batches}\n\n")
        
        f.write("----- 智能体性能指标 -----\n")
        for time_step in time_steps:
            metrics = average_results['agent']['by_time'][time_step]
            f.write(f"\n时间步 t+{time_step}min:\n")
            f.write(f"CSI: {metrics['csi']:.4f}\n")
            f.write(f"POD: {metrics['pod']:.4f}\n")
            f.write(f"FAR: {metrics['far']:.4f}\n")
        
        f.write("\n所有时间步平均:\n")
        avg_metrics = average_results['agent']['average']
        f.write(f"CSI: {avg_metrics['csi']:.4f}\n")
        f.write(f"POD: {avg_metrics['pod']:.4f}\n")
        f.write(f"FAR: {avg_metrics['far']:.4f}\n")
        
        f.write("\n----- 各模型性能指标 -----\n")
        for model_time in models.keys():
            f.write(f"\n模型 {model_time}:\n")
            
            for time_step in time_steps:
                metrics = average_results['models'][model_time]['by_time'][time_step]
                f.write(f"  时间步 t+{time_step}min:\n")
                f.write(f"  CSI: {metrics['csi']:.4f}\n")
                f.write(f"  POD: {metrics['pod']:.4f}\n")
                f.write(f"  FAR: {metrics['far']:.4f}\n")
            
            avg_metrics = average_results['models'][model_time]['average']
            f.write(f"\n  所有时间步平均:\n")
            f.write(f"  CSI: {avg_metrics['csi']:.4f}\n")
            f.write(f"  POD: {avg_metrics['pod']:.4f}\n")
            f.write(f"  FAR: {avg_metrics['far']:.4f}\n")
    
    # 打印评估结果
    print("\n===== 评估结果摘要 =====")
    print("\n----- 智能体性能指标 -----")
    for time_step in time_steps:
        metrics = average_results['agent']['by_time'][time_step]
        print(f"\n时间步 t+{time_step}min:")
        print(f"CSI: {metrics['csi']:.4f}")
        print(f"POD: {metrics['pod']:.4f}")
        print(f"FAR: {metrics['far']:.4f}")
    
    print("\n所有时间步平均:")
    avg_metrics = average_results['agent']['average']
    print(f"CSI: {avg_metrics['csi']:.4f}")
    print(f"POD: {avg_metrics['pod']:.4f}")
    print(f"FAR: {avg_metrics['far']:.4f}")
    
    print("\n----- 各模型性能指标 -----")
    for model_time in models.keys():
        print(f"\n模型 {model_time}:")
        
        for time_step in time_steps:
            metrics = average_results['models'][model_time]['by_time'][time_step]
            print(f"  时间步 t+{time_step}min:")
            print(f"  CSI: {metrics['csi']:.4f}")
            print(f"  POD: {metrics['pod']:.4f}")
            print(f"  FAR: {metrics['far']:.4f}")
        
        avg_metrics = average_results['models'][model_time]['average']
        print(f"\n  所有时间步平均:")
        print(f"  CSI: {avg_metrics['csi']:.4f}")
        print(f"  POD: {avg_metrics['pod']:.4f}")
        print(f"  FAR: {avg_metrics['far']:.4f}")
    
    print(f"\n详细结果已保存至 result/detailed_metrics.txt")
    
    # 绘制性能指标对比图
    # 每个时间步的CSI对比
    plt.figure(figsize=(15, 20))
    
    # 为每个时间步分别绘制CSI, POD, FAR的对比图
    for i, time_step in enumerate(time_steps):
        # CSI对比
        plt.subplot(4, 3, i*3 + 1)
        plt.title(f'CSI Comparison (t+{time_step}min)')
        csi_data = [results['agent']['by_time'][time_step]['csi']]
        csi_labels = ['Agent']
        
        for model_time in models.keys():
            csi_data.append(results['models'][model_time]['by_time'][time_step]['csi'])
            csi_labels.append(f'{model_time}')
        
        plt.boxplot(csi_data, labels=csi_labels)
        plt.grid(True)
        
        # POD对比
        plt.subplot(4, 3, i*3 + 2)
        plt.title(f'POD Comparison (t+{time_step}min)')
        pod_data = [results['agent']['by_time'][time_step]['pod']]
        pod_labels = ['Agent']
        
        for model_time in models.keys():
            pod_data.append(results['models'][model_time]['by_time'][time_step]['pod'])
            pod_labels.append(f'{model_time}')
        
        plt.boxplot(pod_data, labels=pod_labels)
        plt.grid(True)
        
        # FAR对比
        plt.subplot(4, 3, i*3 + 3)
        plt.title(f'FAR Comparison (t+{time_step}min)')
        far_data = [results['agent']['by_time'][time_step]['far']]
        far_labels = ['Agent']
        
        for model_time in models.keys():
            far_data.append(results['models'][model_time]['by_time'][time_step]['far'])
            far_labels.append(f'{model_time}')
        
        plt.boxplot(far_data, labels=far_labels)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('result/metrics_by_timestep.png', dpi=150)
    plt.close()
    
    # 绘制整体性能对比图
    plt.figure(figsize=(15, 5))
    
    # CSI整体对比
    plt.subplot(1, 3, 1)
    plt.title('Overall CSI Comparison')
    csi_data = [results['agent']['average']['csi']]
    csi_labels = ['Agent']
    
    for model_time in models.keys():
        csi_data.append(results['models'][model_time]['average']['csi'])
        csi_labels.append(f'{model_time}')
    
    plt.boxplot(csi_data, labels=csi_labels)
    plt.grid(True)
    
    # POD整体对比
    plt.subplot(1, 3, 2)
    plt.title('Overall POD Comparison')
    pod_data = [results['agent']['average']['pod']]
    pod_labels = ['Agent']
    
    for model_time in models.keys():
        pod_data.append(results['models'][model_time]['average']['pod'])
        pod_labels.append(f'{model_time}')
    
    plt.boxplot(pod_data, labels=pod_labels)
    plt.grid(True)
    
    # FAR整体对比
    plt.subplot(1, 3, 3)
    plt.title('Overall FAR Comparison')
    far_data = [results['agent']['average']['far']]
    far_labels = ['Agent']
    
    for model_time in models.keys():
        far_data.append(results['models'][model_time]['average']['far'])
        far_labels.append(f'{model_time}')
    
    plt.boxplot(far_data, labels=far_labels)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('result/overall_metrics_comparison.png', dpi=150)
    plt.close()
    
    print(f"评估结果图表已保存至 result/metrics_by_timestep.png 和 result/overall_metrics_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='雷达回波预测系统')
    parser.add_argument('--train_models', action='store_true', help='训练基础预测模型')
    parser.add_argument('--train_agent', action='store_true', help='训练强化学习智能体')
    parser.add_argument('--eval', action='store_true', help='评估系统性能')
    parser.add_argument('--all', action='store_true', help='按顺序执行所有训练和评估步骤')
    parser.add_argument('--train_eval', action='store_true', help='按顺序执行所有训练和评估步骤')
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(42)
    
    if args.all:
        train_models_with_different_batch()
        train_agent()
        evaluate_system_with_metrics()
    elif args.train_eval:
        train_agent()
        evaluate_system_with_metrics()
    elif args.train_models:
        train_models_with_different_batch()
    elif args.train_agent:
        train_agent()
    elif args.eval:
        evaluate_system_with_metrics()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()