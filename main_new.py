# -*- coding: utf-8 -*-
# !/usr/bin/python
from matplotlib import colors
from CSI import calc_CSI_reg
from loss import RegLoss120, RegLoss30, RegLoss60, RegLoss90
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
from agent import OptimizedTemporalModelSelectorAgent
from environment import OptimizedRadarEnvironment
from train_utils import train_rl_agent_optimized, train_rl_agent_with_metrics

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
    test_dataset = DataSet(test_radar_list, len(test_radar_list), test=True) #len(test_radar_list)
    
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
                    loss_func = RegLoss30().to(device)
                elif time == '60':
                    loss_func = RegLoss60().to(device)
                elif time == '90':
                    loss_func = RegLoss90().to(device)
                elif time == '120':
                    loss_func = RegLoss120().to(device)
                
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

def train_agent():
    """Train the reinforcement learning agent with improved selection logic"""
    # Load data
    _, _, train_dataset, _, _ = load_data(
        batch_size=opt.batch_size, 
        val_batch_size=opt.batch_size,
        data_root=opt.radar_train_data_root, 
        num_workers=opt.num_workers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models and load weights
    models = {
        '30': Net().to(device),
        '60': Net().to(device),
        '90': Net().to(device),
        '120': Net().to(device)
    }
    
    model_path = "./model_path"
    for time, model in models.items():
        model.load_state_dict(torch.load(os.path.join(model_path, f"regression_model_{time}_best.pth")))
        model.eval()  # Set to evaluation mode
    
    # Initialize the optimized agent
    input_size = (280, 360)  # Match your actual input dimensions
    agent = OptimizedTemporalModelSelectorAgent(
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
        batch_size=opt.agent_batch_size,
        early_stopping_patience=10
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

def evaluate_system():
    """评估整个系统性能并可视化结果"""
    # 加载测试数据
    _, test_loader, _, _, _ = load_data(
        batch_size=opt.batch_size, 
        val_batch_size=opt.batch_size,
        data_root=opt.radar_test_data_root, 
        num_workers=opt.num_workers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练好的模型
    models = {
        '30': Net().to(device),
        '60': Net().to(device),
        '90': Net().to(device),
        '120': Net().to(device)
    }
    
    model_path = "./model_path"
    for time, model in models.items():
        model.load_state_dict(torch.load(os.path.join(model_path, f"regression_model_{time}_best.pth")))
        model.eval()
    
    # 加载训练好的智能体
    agent = OptimizedTemporalModelSelectorAgent(
        input_channels=1, 
        hidden_dim=64, 
        num_models=len(models),
        input_size=(280,360)
    ).to(device)
    agent.load_state_dict(torch.load(os.path.join('rl_model', 'best_agent.pth'))['model_state_dict'])
    agent.eval()
    
    # 评估结果
    results = {
        'agent': {'csi': [], 'pod': [], 'far': []},
        'models': {time: {'csi': [], 'pod': [], 'far': []} for time in models.keys()}
    }
    
    # 创建结果保存目录
    os.makedirs('result', exist_ok=True)
    cdict = ['whitesmoke', 'dodgerblue', 'limegreen', 'green', 'darkgreen', 'yellow', 'goldenrod', 'orange', 'red',
             'darkred']
    clevs = [0, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    my_map = colors.ListedColormap(cdict)
    norm = colors.BoundaryNorm(clevs, len(clevs) - 1)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 智能体选择模型
            model_probs, _ = agent(inputs)
            final_predictions = []
            for t in range(min(4, model_probs.size(1))):
                time_probs = model_probs[0, t]
                dist = torch.distributions.Categorical(time_probs)
                action = dist.sample()
                model_idx = action.item() % len(models)
                selected_time = list(models.keys())[model_idx]
                selected_prediction = models[selected_time](inputs)[:, t]
                final_predictions.append(selected_prediction)
            
            agent_predictions = torch.stack(final_predictions, dim=1)
            
            # 计算智能体性能指标
            metrics = calc_CSI_reg(agent_predictions.cpu().numpy(), targets.cpu().numpy(), threshold=35)
            results['agent']['csi'].append(metrics[0])  # CSI
            results['agent']['pod'].append(metrics[1])  # POD
            results['agent']['far'].append(metrics[2])  # FAR
            
            # 计算各个模型的性能指标
            for time, model in models.items():
                predictions = model(inputs)
                metrics = calc_CSI_reg(predictions.cpu().numpy(), targets.cpu().numpy(), threshold=35)
                results['models'][time]['csi'].append(metrics[0])
                results['models'][time]['pod'].append(metrics[1])
                results['models'][time]['far'].append(metrics[2])
            
            if batch_idx % 1 == 0:  # 每1个batch绘制一次
                print(f"正在生成第{batch_idx}批次的预测结果图...")
                # 创建多行子图布局
                time_steps = [30, 60, 90, 120]
                num_models = len(models)
                rows = 2 + num_models  # 真实值(1行) + 智能体(1行) + 每个模型一行
                cols = len(time_steps) # 每行4个时间步

                plt.figure(figsize=(24, 6 * rows))# 调整画布尺寸

                # 第一行：真实值的四个时间步
                for t in range(len(time_steps)):
                    plt.subplot(rows, cols, 1 + t)
                    true_frame = denormalize(targets[0, t]).cpu().numpy()
                    plt.title(f'Ground Truth t+{t+1}')
                    img = plt.imshow(true_frame, cmap=my_map, norm=norm)
                    plt.colorbar(img, fraction=0.046, pad=0.04)
                    plt.axis('off')

                # 第二行：智能体预测的四个时间步
                for t in range(len(time_steps)):
                    plt.subplot(rows, cols, cols + 1 + t)
                    pred_frame = agent_predictions[0, t].cpu().numpy()
                    plt.title(f'Agent t+{t+1}')
                    img = plt.imshow(pred_frame, cmap=my_map, norm=norm)
                    plt.colorbar(img, fraction=0.046, pad=0.04)
                    plt.axis('off')

                # 后续行：各专家模型的预测结果（每行4个）
                for model_idx, (model_time, model) in enumerate(models.items()):
                    with torch.no_grad():
                        preds = model(inputs).cpu()
                    
                    # 获取该模型预测的时间步索引映射
                    time_mapping = {t: t//int(model_time) - 1 for t in time_steps}
                    
                    for col_idx, t in enumerate(time_steps):
                        # 计算子图位置
                        subplot_pos = (2 + model_idx) * cols + col_idx + 1
                        
                        # 跳过超出总子图数的情况
                        if subplot_pos > rows * cols:
                            continue
                        
                        plt.subplot(rows, cols, subplot_pos)
                        
                        # 处理索引越界
                        pred_step = time_mapping[t]
                        pred_step = max(0, min(pred_step, preds.shape[1]-1))
                        
                        # 只绘制该模型对应的时间步
                        if t % int(model_time) == 0:
                            pred_frame = denormalize(preds[0, pred_step]).numpy()
                            plt.title(f"{model_time}min Model\nt+{t}min")
                            img = plt.imshow(pred_frame, cmap=my_map, norm=norm)
                            plt.colorbar(img, fraction=0.046, pad=0.04)
                            plt.axis('off')
                    
                plt.tight_layout()
                plt.savefig(f'result/comparison_batch{batch_idx}_time.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"第{batch_idx}批次的预测结果图已保存至result/comparison_batch{batch_idx}_time.png")
                
    # 保存评估结果
    with open('result.txt', 'w') as f:
        f.write("\n系统评估结果:\n")
        f.write("\n智能体性能:\n")
        f.write(f"CSI: {np.mean(results['agent']['csi']):.4f}\n")
        f.write(f"POD: {np.mean(results['agent']['pod']):.4f}\n")
        f.write(f"FAR: {np.mean(results['agent']['far']):.4f}\n")
        
        f.write("\n各模型性能:\n")
        for time in models.keys():
            f.write(f"\n模型 {time}:\n")
            f.write(f"CSI: {np.mean(results['models'][time]['csi']):.4f}\n")
            f.write(f"POD: {np.mean(results['models'][time]['pod']):.4f}\n")
            f.write(f"FAR: {np.mean(results['models'][time]['far']):.4f}\n")
    
    # 绘制性能指标对比图
    plt.figure(figsize=(15, 5))
    
    # CSI对比
    plt.subplot(1, 3, 1)
    plt.title('CSI Comparison')
    plt.boxplot([results['agent']['csi']] + [results['models'][time]['csi'] for time in models.keys()],
                labels=['Agent'] + list(models.keys()))
    plt.grid(True)
    
    # POD对比
    plt.subplot(1, 3, 2)
    plt.title('POD Comparison')
    plt.boxplot([results['agent']['pod']] + [results['models'][time]['pod'] for time in models.keys()],
                labels=['Agent'] + list(models.keys()))
    plt.grid(True)
    
    # FAR对比
    plt.subplot(1, 3, 3)
    plt.title('FAR Comparison')
    plt.boxplot([results['agent']['far']] + [results['models'][time]['far'] for time in models.keys()],
                labels=['Agent'] + list(models.keys()))
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('result/metrics_comparison.png')
    plt.close()
    
def denormalize(image):
    return image * 85 - 10

def main():
    parser = argparse.ArgumentParser(description='雷达回波预测系统')
    parser.add_argument('--train_models', action='store_true', help='训练基础预测模型')
    parser.add_argument('--train_agent', action='store_true', help='训练强化学习智能体')
    parser.add_argument('--eval', action='store_true', help='评估系统性能')
    parser.add_argument('--all', action='store_true', help='按顺序执行所有训练和评估步骤')
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(42)
    
    if args.all:
        train_models()
        train_agent()
        evaluate_system()
    elif args.train_models:
        train_models()
    elif args.train_agent:
        train_agent()
    elif args.eval:
        evaluate_system()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()