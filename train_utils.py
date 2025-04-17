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
                    count += 1
    
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
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
        episode_metrics = {'csi': [], 'pod': [], 'far': []}
        
        while True:
            # 获取动作概率和状态值
            model_probs, state_value = agent(state.unsqueeze(0))
            
            # 让每个模型预测所有未来帧
            all_model_predictions = {}
            for time_key, model in models.items():
                with torch.no_grad():
                    pred = model(state.unsqueeze(0))
                    all_model_predictions[time_key] = pred.detach()
            
            # 选择动作并生成预测
            final_predictions = []
            for t in range(min(4, model_probs.size(1))):
                time_probs = model_probs[0, t]
                dist = torch.distributions.Categorical(time_probs)
                action = dist.sample()
                
                # 使用选定模型的预测
                model_idx = action.item() % len(models)
                selected_time = list(models.keys())[model_idx]
                selected_prediction = all_model_predictions[selected_time][:, t]
                final_predictions.append(selected_prediction)
            
            # 合并预测结果
            predictions = torch.stack(final_predictions, dim=1)
            
            # 环境步进
            next_state, reward, done, info = env.step(predictions)
            
            # 记录奖励和指标
            episode_reward += reward
            if 'metrics' in info:
                for metric in info['metrics']:
                    episode_metrics['csi'].append(metric['CSI'])
                    episode_metrics['pod'].append(metric['POD'])
                    episode_metrics['far'].append(metric['FAR'])
            
            if done:
                break
            
            state = torch.FloatTensor(next_state).to(device)
        
        # 记录episode结果
        episode_rewards.append(episode_reward)
        metrics_history.append({
            'csi': np.mean(episode_metrics['csi']),
            'pod': np.mean(episode_metrics['pod']),
            'far': np.mean(episode_metrics['far'])
        })
        
        # 计算当前性能（使用加权组合指标）
        current_metrics = metrics_history[-1]
        current_performance = (
            0.4 * current_metrics['csi'] +
            0.4 * current_metrics['pod'] +
            0.2 * (1 - current_metrics['far'])  # FAR越低越好
        )
        
        # 如果当前性能超过历史最佳，保存模型
        if current_performance > best_performance:
            best_performance = current_performance
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'performance': best_performance,
                'metrics': current_metrics
            }, best_model_path)
            print(f"\nSaved best model at episode {episode+1} with performance {best_performance:.4f}")
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_metrics = {
                'csi': np.mean([m['csi'] for m in metrics_history[-10:]]),
                'pod': np.mean([m['pod'] for m in metrics_history[-10:]]),
                'far': np.mean([m['far'] for m in metrics_history[-10:]])
            }
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.4f}")
            print(f"Avg Metrics - CSI: {avg_metrics['csi']:.4f}, POD: {avg_metrics['pod']:.4f}, FAR: {avg_metrics['far']:.4f}")
    
    # 训练结束后加载最佳模型
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model with performance {checkpoint['performance']:.4f}")
    
    return agent, episode_rewards, metrics_history

def train_rl_agent_optimized(env, agent, models, optimizer, num_episodes=200, 
                            batch_size=32, gamma=0.99, early_stopping_patience=20):
    """
    Optimized training function with better stabilization and early stopping
    
    Args:
        env: The environment
        agent: The agent model
        models: Dictionary of prediction models
        optimizer: Optimizer for the agent
        num_episodes: Maximum number of episodes
        batch_size: Batch size for updates
        gamma: Discount factor
        early_stopping_patience: Episodes without improvement before stopping
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    
    # Move all models to device and set to evaluation mode
    for model_name, model in models.items():
        model.to(device)
        model.eval()
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
    
    # Track training progress
    episode_rewards = []
    metrics_history = []
    
    # Early stopping variables
    best_performance = float('-inf')
    no_improvement_count = 0
    
    # Ensure model save directory exists
    os.makedirs('rl_model', exist_ok=True)
    best_model_path = os.path.join('rl_model', 'best_agent.pth')
    
    # Map from index to model key
    time_keys = list(models.keys())
    
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
        episode_metrics = {'csi': [], 'pod': [], 'far': []}
        
        while True:
            # Get action probabilities and state value
            model_probs, state_value = agent(state.unsqueeze(0))
            
            # Get predictions from all models
            all_model_predictions = {}
            for time_key, model in models.items():
                with torch.no_grad():
                    pred = model(state.unsqueeze(0))
                    all_model_predictions[time_key] = pred.detach()
            
            # Select actions and generate predictions
            selected_actions = []
            final_predictions = []
            
            for t in range(min(4, model_probs.size(1))):
                time_probs = model_probs[0, t]
                
                # During training, sample from distribution
                if agent.training:
                    dist = torch.distributions.Categorical(time_probs)
                    action = dist.sample()
                # During evaluation, take best action
                else:
                    action = torch.argmax(time_probs)
                    
                selected_actions.append(action)
                
                # Use selected model's prediction
                model_idx = action.item() % len(time_keys)  # Safe indexing
                selected_time = time_keys[model_idx]
                selected_prediction = all_model_predictions[selected_time][:, t]
                final_predictions.append(selected_prediction)
            
            # Combine predictions
            predictions = torch.stack(final_predictions, dim=1)
            
            # Environment step
            next_state, reward, done, info = env.step(predictions)
            
            # Track reward and metrics
            episode_reward += reward
            if 'metrics' in info:
                for metric in info['metrics']:
                    episode_metrics['csi'].append(metric['CSI'])
                    episode_metrics['pod'].append(metric['POD'])
                    episode_metrics['far'].append(metric['FAR'])
            
            if done:
                break
            
            # Move to next state
            state = torch.FloatTensor(next_state).to(device)
        
        # Record episode results
        episode_rewards.append(episode_reward)
        avg_metrics = {
            'csi': np.mean(episode_metrics['csi']) if episode_metrics['csi'] else 0,
            'pod': np.mean(episode_metrics['pod']) if episode_metrics['pod'] else 0,
            'far': np.mean(episode_metrics['far']) if episode_metrics['far'] else 0
        }
        metrics_history.append(avg_metrics)
        
        # Calculate overall performance
        current_performance = (
            0.4 * avg_metrics['csi'] +
            0.4 * avg_metrics['pod'] + 
            0.2 * (1 - avg_metrics['far'])
        )
        
        # Update learning rate
        scheduler.step(current_performance)
        
        # Save best model and check for early stopping
        if current_performance > best_performance:
            best_performance = current_performance
            torch.save({
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode,
                'performance': best_performance,
                'metrics': avg_metrics
            }, best_model_path)
            print(f"\nSaved best model at episode {episode+1} with performance {best_performance:.4f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {episode+1} episodes")
                break
        
        # Print progress
        if (episode + 1) % 1 == 0:  # 每个episode都显示
            avg_reward = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else episode_rewards[-1]
            avg_csi = np.mean([m['csi'] for m in metrics_history[-5:]]) if len(metrics_history) >= 5 else metrics_history[-1]['csi']
            avg_pod = np.mean([m['pod'] for m in metrics_history[-5:]]) if len(metrics_history) >= 5 else metrics_history[-1]['pod']
            avg_far = np.mean([m['far'] for m in metrics_history[-5:]]) if len(metrics_history) >= 5 else metrics_history[-1]['far']
            
            print(f"\n=== Episode {episode+1}/{num_episodes} ===")
            print(f"Reward: {episode_reward:.4f} (Avg last 5: {avg_reward:.4f})")
            print(f"Metrics - CSI: {avg_metrics['csi']:.4f}, POD: {avg_metrics['pod']:.4f}, FAR: {avg_metrics['far']:.4f}")
            print(f"Avg Metrics (last 5) - CSI: {avg_csi:.4f}, POD: {avg_pod:.4f}, FAR: {avg_far:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Best performance: {best_performance:.4f}")
            
            # 显示模型选择分布
            action_counts = {time_key: 0 for time_key in time_keys}
            for action in selected_actions:
                model_idx = action.item() % len(time_keys)
                selected_time = time_keys[model_idx]
                action_counts[selected_time] += 1
            print("Model selection distribution:")
            for model_name, count in action_counts.items():
                print(f"{model_name}: {count}")
            print("-" * 50)
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.4f}")
            print(f"Avg Metrics - CSI: {avg_csi:.4f}, POD: {avg_pod:.4f}, FAR: {avg_far:.4f}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model at the end of training
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model with performance {checkpoint['performance']:.4f}")
    
    return agent, episode_rewards, metrics_history