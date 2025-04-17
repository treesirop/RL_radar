# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CSI import calc_CSI_reg

def evaluate_rl_system_with_metrics(test_loader, agent, models, threshold=35.0):
    """
    使用CSI、POD和FAR评估RL系统，同时展示单独模型和agent调度的性能差异
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.eval()
    for model in models.values():
        model.eval()
    
    # Agent调度结果统计
    agent_csi = 0.0
    agent_pod = 0.0
    agent_far = 0.0
    
    # 单独模型结果统计
    model_metrics = {time_key: {'csi': 0.0, 'pod': 0.0, 'far': 0.0} for time_key in models.keys()}
    model_counts = {time_key: 0 for time_key in models.keys()}
    
    count = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 获取模型选择概率
            model_probs, _ = agent(inputs)
            # 让每个模型预测所有未来帧
            all_model_predictions = {}
            for time_key, model in models.items():
                pred = model(inputs)
                all_model_predictions[time_key] = pred
                
                # 计算单独模型的指标
                for b in range(inputs.size(0)):
                    for t in range(pred.size(1)):
                        pred_t = pred[b, t].cpu().numpy()
                        true_t = targets[b, t].cpu().numpy()
                        csi, pod, far = calc_CSI_reg(pred_t, true_t, threshold)
                        
                        model_metrics[time_key]['csi'] += csi
                        model_metrics[time_key]['pod'] += pod
                        model_metrics[time_key]['far'] += far
                        model_counts[time_key] += 1
            
            # 选择每个时间步最佳的模型
            final_predictions = []
            for t in range(min(4, model_probs.size(1))):
                time_probs = model_probs[:, t]
                selected_models = torch.argmax(time_probs, dim=1)
                
                # 使用选定模型的预测
                batch_predictions = []
                for b in range(inputs.size(0)):
                    model_idx = selected_models[b].item() % len(models)
                    selected_time = list(models.keys())[model_idx]
                    pred = all_model_predictions[selected_time][b:b+1, t]
                    batch_predictions.append(pred)
                
                time_predictions = torch.cat(batch_predictions, dim=0)
                final_predictions.append(time_predictions)
            
            del model_probs
            # 合并预测结果
            predictions = torch.stack(final_predictions, dim=1)
            
            # 计算评估指标
            for b in range(inputs.size(0)):
                for t in range(predictions.size(1)):
                    pred = predictions[b, t].cpu().numpy()
                    true = targets[b, t].cpu().numpy()
                    
                    csi, pod, far = calc_CSI_reg(pred, true, threshold)
                    
                    agent_csi += csi
                    agent_pod += pod
                    agent_far += far
    
    # 计算平均值
    avg_agent_csi = agent_csi / count if count > 0 else 0
    avg_agent_pod = agent_pod / count if count > 0 else 0
    avg_agent_far = agent_far / count if count > 0 else 0
    
    # 计算单独模型的平均指标
    model_results = {}
    for time_key in models.keys():
        model_results[time_key] = {
            'csi': model_metrics[time_key]['csi'] / model_counts[time_key] if model_counts[time_key] > 0 else 0,
            'pod': model_metrics[time_key]['pod'] / model_counts[time_key] if model_counts[time_key] > 0 else 0,
            'far': model_metrics[time_key]['far'] / model_counts[time_key] if model_counts[time_key] > 0 else 0
        }
    
    # 将结果写入文件
    with open('result.txt', 'w') as f:
        f.write('评估结果 (阈值=35dBz)\n\n')
        f.write('Agent调度性能:\n')
        f.write(f'CSI: {avg_agent_csi:.4f}, POD: {avg_agent_pod:.4f}, FAR: {avg_agent_far:.4f}\n\n')
        
        f.write('单独模型性能:\n')
        for time_key, metrics in model_results.items():
            f.write(f'{time_key}: CSI={metrics["csi"]:.4f}, POD={metrics["pod"]:.4f}, FAR={metrics["far"]:.4f}\n')
    
    return {
        'agent': {'csi': avg_agent_csi, 'pod': avg_agent_pod, 'far': avg_agent_far},
        'models': model_results
    }


import os

def train_rl_agent_with_metrics(env, agent, models, optimizer, num_episodes=200, batch_size=32):
    """
    使用CSI、POD和FAR指标训练RL智能体，并保存最佳模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    for model in models.values():
        model.to(device)
        model.eval()
    
    episode_rewards = []
    metrics_history = []
    
    # 添加最佳模型保存相关变量
    best_performance = float('-inf')
    os.makedirs('rl_model', exist_ok=True)
    best_model_path = os.path.join('rl_model', 'best_agent.pth')
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        episode_reward = 0
        
        while True:
            # 1. 获取动作概率和状态值
            model_probs, state_value = agent(state.unsqueeze(0))
            
            # 2. 逐个模型获取预测，而不是一次性全部获取
            final_preds = []
            selected_models = []
            
            for t in range(4):
                time_probs = model_probs[0, t]
                dist = torch.distributions.Categorical(time_probs)
                action = dist.sample()
                
                selected_idx = action.item()
                selected_model = list(models.keys())[selected_idx]
                
                # 显存优化：只计算需要的模型预测
                with torch.no_grad():
                    pred = models[selected_model](state.unsqueeze(0))[:, t]
                    final_preds.append(pred)
                
                selected_models.append(selected_model)
            
            # 3. 合并预测结果
            predictions = torch.stack(final_preds, dim=1)
            
            # 4. 环境交互
            next_state, reward, done, _ = env.step(predictions)
            
            # 5. 计算并累积奖励
            episode_reward += reward * 100.0  # 保持原有奖励缩放
            
            # 6. 显存优化：及时删除中间变量
            del model_probs, state_value, predictions
            torch.cuda.empty_cache()
            
            if done:
                break
                
            # 7. 更新状态
            state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        
        # 记录episode结果
        episode_rewards.append(episode_reward)
        
        current_performance = episode_reward
        
        # 如果当前性能超过历史最佳，保存模型
        if current_performance > best_performance:
            best_performance = current_performance
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
            }, best_model_path)
            print(f"\nSaved best model at episode {episode+1} with performance {best_performance:.4f}")
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.4f}")
    
    # 训练结束后加载最佳模型
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model with performance {checkpoint['performance']:.4f}")
    
    return agent, episode_rewards, metrics_history
# 使用生成器避免同时保存所有模型预测
def get_model_prediction(model, inputs):
    with torch.no_grad():
        return model(inputs)

def train_rl_agent_optimized(env, agent, models, optimizer, num_episodes=10, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.train().to(device)
    
    # 关键修改1：禁用不需要的梯度计算
    for model in models.values():
        model.eval().requires_grad_(False).to(device)
    
    # 关键修改2：配置CUDA优化参数
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    best_performance = float('-inf')
    os.makedirs('rl_model', exist_ok=True)
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        episode_reward = 0
        
        while True:
            # 关键修改3：强制清空梯度缓存
            optimizer.zero_grad(set_to_none=True)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                # 仅计算必要的前向传播
                model_probs, state_value = agent(state.unsqueeze(0))
                
                # 逐个时间步处理避免显存累积
                predictions = []
                for t in range(4):
                    # 采样动作
                    with torch.no_grad():
                        action = torch.argmax(model_probs[0, t]).item()
                    
                    # 仅计算被选中的模型
                    selected_model = list(models.values())[action]
                    pred = selected_model(state.unsqueeze(0))[:, t]
                    predictions.append(pred)
                
                predictions = torch.stack(predictions, dim=1)
                
                # 环境交互
                next_state, reward, done, _ = env.step(predictions.detach())
                reward_tensor = torch.tensor(reward, device=device).float()
                
                # 计算损失
                advantage = reward_tensor - state_value
                loss = -torch.log(model_probs[0, :, action] + 1e-6) * advantage.detach()
                loss = loss.mean() + F.mse_loss(state_value, reward_tensor)
            
            # 梯度缩放与更新
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            episode_reward += reward
            state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            
            if done:
                break
        
        # 每episode清理显存
        torch.cuda.empty_cache()
        
        # 模型保存逻辑保持不变
        if episode_reward > best_performance:
            best_performance = episode_reward
            torch.save(agent.state_dict(), f"rl_model/best_agent_ep{episode}.pth")
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    return agent

def train_rl_agent_optimized(env, agent, models, optimizer, num_episodes=10, 
                           batch_size=4, gamma=0.99, early_stopping_patience=20):
    """修正后的训练函数，确保参数一致性"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    # 显存优化
    torch.cuda.empty_cache()
    
    # 模型设置
    for model in models.values():
        model.to(device).eval()
    
    # 训练记录
    episode_rewards = []
    best_performance = float('-inf')
    os.makedirs('rl_model', exist_ok=True)
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        episode_reward = 0
        
        while True:
            # 获取动作概率
            model_probs, state_value = agent(state.unsqueeze(0))
            
            # 选择动作并预测
            predictions = []
            for t in range(4):
                # 采样动作
                action = torch.argmax(model_probs[0, t]).item()
                pred = models[list(models.keys())[action]](state.unsqueeze(0))[:, t]
                predictions.append(pred)
            
            predictions = torch.stack(predictions, dim=1)
            
            # 环境交互
            next_state, reward, done, _ = env.step(predictions.detach())
            episode_reward += reward
            
            if done:
                break
                
            state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        
        # 记录和保存
        episode_rewards.append(episode_reward)
        if episode_reward > best_performance:
            best_performance = episode_reward
            torch.save(agent.state_dict(), f"rl_model/best_agent.pth")
        
        print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    return agent, episode_rewards, []