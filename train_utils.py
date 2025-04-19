# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CSI import calc_CSI_reg # Make sure this function can handle tensors directly or convert them
from environment import OptimizedRadarEnvironment # Import Environment to potentially get num_models later if needed

# Helper function to calculate weighted prediction
def calculate_weighted_prediction(models, state, model_weights, device):
    """Calculates the weighted average prediction."""
    batch_size, num_timesteps, num_models = model_weights.shape
    # Ensure state has batch dim
    if state.dim() == 3: # If state is [seq, H, W]
        state = state.unsqueeze(0) # Add batch dim -> [1, seq, H, W]
    elif state.dim() == 4 and state.shape[0] != batch_size : # Mismatched batch size
         state = state.repeat(batch_size, 1, 1, 1) # Repeat state if needed (shouldn't happen with batch_size=1 usually)

    weighted_predictions = []
    with torch.no_grad(): # Base models should not be trained here
        all_model_preds = []
        for model_key in models.keys():
             # Ensure model is on the correct device
             model = models[model_key].to(device)
             # Get prediction for all 4 timesteps: shape [batch_size, 4]
             pred = model(state) # [B, 4, H, W]
             all_model_preds.append(pred) # Add model pred to list

        # Stack model predictions: [B, 4, num_models, H, W]
        all_model_preds = torch.stack(all_model_preds, dim=2)

        # Perform weighted sum: model_weights[batch, t, m] * all_model_preds[batch, t, m, h, w]
        # Result shape: [batch, 4, H, W]
        weighted_preds = torch.sum(model_weights.unsqueeze(-1).unsqueeze(-1) * all_model_preds, dim=2)
    return weighted_preds


def evaluate_rl_system_with_metrics(test_loader, agent, models, threshold=35.0):
    """
    Evaluates the RL system using weighted combination and CSI/POD/FAR metrics.
    Compares agent performance with individual model performance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.eval().to(device)
    num_models = len(models)
    class ParallelPredictor(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models.values())
                
            def forward(self, x):
                return torch.stack([model(x) for model in self.models], dim=2)
        
    parallel_predictor = ParallelPredictor(models).eval().to(device)
    # Ensure models are on the correct device and in eval mode
    for model in models.values():
        model.eval().to(device)

    # Agent stats
    total_agent_csi = 0.0
    total_agent_pod = 0.0 # Assuming calc_CSI_reg returns (csi, pod, far)
    total_agent_far = 0.0

    # Individual model stats
    model_metrics = {key: {'csi': 0.0, 'pod': 0.0, 'far': 0.0} for key in models.keys()}
    model_counts = {key: 0 for key in models.keys()}

    total_samples = 0 # Count total prediction samples (batch_size * num_timesteps)

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device) # Shape [batch, 4]
            batch_size = inputs.size(0)

            # 1. Get weights from Agent
            model_preds = parallel_predictor(inputs)
            model_weights, _ = agent(inputs,model_preds) # Shape [batch, 4, num_models]

            # 2. Calculate weighted prediction
            weighted_preds = calculate_weighted_prediction(models, inputs, model_weights, device) # Shape [batch, 4]

            # 3. Calculate agent metrics
            # Iterate through batch and timesteps
            for b in range(batch_size):
                for t in range(weighted_preds.size(1)): # Should be 4 timesteps
                    pred_t = weighted_preds[b, t]
                    target_t = targets[b, t]
                    # Assuming calc_CSI_reg handles tensors or numpy arrays
                    # Convert if necessary: pred_t.cpu().numpy(), target_t.cpu().numpy()
                    csi, pod, far = calc_CSI_reg(pred_t, target_t, threshold) # Modify calc_CSI_reg if needed
                    total_agent_csi += csi
                    total_agent_pod += pod
                    total_agent_far += far
                    total_samples += 1

            # 4. Calculate individual model metrics (as before)
            for model_key, model in models.items():
                model_pred = model(inputs) # Shape [batch, 4]
                for b in range(batch_size):
                    for t in range(model_pred.size(1)):
                        pred_t = model_pred[b, t]
                        target_t = targets[b, t]
                        csi, pod, far = calc_CSI_reg(pred_t, target_t, threshold)
                        model_metrics[model_key]['csi'] += csi
                        model_metrics[model_key]['pod'] += pod
                        model_metrics[model_key]['far'] += far
                        model_counts[model_key] += 1

    # Calculate average metrics
    avg_agent_csi = total_agent_csi / total_samples if total_samples > 0 else 0
    avg_agent_pod = total_agent_pod / total_samples if total_samples > 0 else 0
    avg_agent_far = total_agent_far / total_samples if total_samples > 0 else 0

    model_results = {}
    for key in models.keys():
        count = model_counts[key]
        model_results[key] = {
            'csi': model_metrics[key]['csi'] / count if count > 0 else 0,
            'pod': model_metrics[key]['pod'] / count if count > 0 else 0,
            'far': model_metrics[key]['far'] / count if count > 0 else 0
        }

    # Print and write results (optional)
    print("--- Evaluation Results ---")
    print(f"Agent Weighted Performance: CSI={avg_agent_csi:.4f}, POD={avg_agent_pod:.4f}, FAR={avg_agent_far:.4f}")
    print("Individual Model Performance:")
    for key, metrics in model_results.items():
        print(f"  Model {key}: CSI={metrics['csi']:.4f}, POD={metrics['pod']:.4f}, FAR={metrics['far']:.4f}")
    print("--------------------------")

    # # Optional: Write results to file
    # with open('result/evaluation_weighted.txt', 'w') as f:
    #     f.write(f'Evaluation Results (Threshold={threshold}dBz)


    #     f.write('Agent Weighted Performance:

    #     f.write(f'CSI: {avg_agent_csi:.4f}, POD: {avg_agent_pod:.4f}, FAR: {avg_agent_far:.4f}


    #     f.write('Individual Model Performance:

    #     for key, metrics in model_results.items():
    #         f.write(f'  Model {key}: CSI={metrics["csi"]:.4f}, POD={metrics["pod"]:.4f}, FAR={metrics["far"]:.4f}


    return {
        'agent': {'csi': avg_agent_csi, 'pod': avg_agent_pod, 'far': avg_agent_far},
        'models': model_results
    }


import os
# Use the second definition and adapt it
def train_rl_agent_optimized(env, agent, models, optimizer, num_episodes=10, gamma=0.99): # Removed unused batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.train().to(device)
    num_models = len(models)
    class ParallelPredictor(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models.values())
                
            def forward(self, x):
                return torch.stack([model(x) for model in self.models], dim=2)  # [B, T, M, H, W]
        
    parallel_predictor = ParallelPredictor(models).eval().to(device)
    # Base models in eval mode and no gradient calculation needed
    for model in models.values():
        model.eval().requires_grad_(False).to(device)

    # Optional: CUDA optimizations
    torch.backends.cudnn.benchmark = True # Can speed up if input sizes don't vary much
    # torch.backends.cuda.matmul.allow_tf32 = True # TF32 can speed up on Ampere GPUs

    best_performance = float('-inf')
    os.makedirs('rl_model', exist_ok=True)
    best_model_path = os.path.join('rl_model', 'best_agent_weighted.pth')
    episode_rewards = []
    log_vars_path = "result/log_vars.txt" # Define path for logging variables

    for episode in range(num_episodes):
        state = env.reset() # Get initial state from environment
        state = torch.as_tensor(state, dtype=torch.float32, device=device) # Shape [seq, H, W]
        episode_reward = 0
        log_data = [] # To store variables for this episode

        done = False
        while not done:
            # Ensure state has batch dimension for agent
            # Define state_batch based on the current state
            if state.dim() == 3: # If state is [seq, H, W]
                state_batch = state.unsqueeze(0) # Add batch dim -> [1, seq, H, W]
            elif state.dim() == 4: # If state is already [B, seq, H, W]
                state_batch = state # Assume it's correct
            else:
                # Handle unexpected state dimensions
                raise ValueError(f"Unexpected state dimension: {state.dim()}. Expected 3 or 4.")

            optimizer.zero_grad(set_to_none=True)
            model_preds = parallel_predictor(state_batch)

            # 1. Get model weights and state value from Agent using state_batch
            model_weights, state_value = agent(state_batch, model_preds) # weights: [1, 4, num_models], value: [1, 1]

            # 2. Calculate weighted prediction using state_batch
            weighted_preds = calculate_weighted_prediction(models, state_batch, model_weights, device) # Shape [1, 4, H, W]

            # 3. Environment Step
            action_for_env = weighted_preds.detach().squeeze(0) # Shape [4, H, W] if batch size is 1
            next_state, reward, done, _ = env.step(action_for_env)
            reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32)
            episode_reward += reward

            # 4. Calculate Critic Target (target_value)
            with torch.no_grad():
                if done:
                    target_value = reward_tensor
                else:
                    next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device)
                    # Define next_state_batch based on next_state_tensor
                    if next_state_tensor.dim() == 3:
                         next_state_batch = next_state_tensor.unsqueeze(0)
                    elif next_state_tensor.dim() == 4:
                         next_state_batch = next_state_tensor
                    else:
                         raise ValueError(f"Unexpected next_state dimension: {next_state_tensor.dim()}. Expected 3 or 4.")
                    next_model_preds = parallel_predictor(next_state_batch)
                    _, next_state_value = agent(next_state_batch,next_model_preds) # Get value of next state
                    target_value = reward_tensor + gamma * next_state_value.squeeze()

            # 5. Calculate Loss (Actor-Critic)
            # ... (Loss calculation remains the same) ...
            critic_loss = F.mse_loss(state_value.squeeze(), target_value)
            advantage = (target_value - state_value).detach()
            entropy = -torch.mean(torch.sum(model_weights * torch.log(model_weights + 1e-9), dim=-1))
            log_weights = torch.log(model_weights + 1e-9)
            actor_loss = - (advantage * log_weights).mean() + 0.5 * entropy
            loss = actor_loss + critic_loss
            # 在计算loss之后添加反向传播和优化步骤
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # Update state ONLY if not done
            if not done:
                 state = next_state_tensor # Update the main state variable for the next loop iteration

            # Clean up memory
            del model_weights, state_value, weighted_preds, advantage, loss, actor_loss, critic_loss
            # Only delete next_state_value if it was created
            if not done:
                 del next_state_value
            if device == 'cuda':
                 torch.cuda.empty_cache()

        # --- End of Episode ---
        episode_rewards.append(episode_reward)

        # Save log data for the episode
        # Clear the file at the start of training or handle appending carefully
        mode = "w" if episode == 0 else "a" 
        with open(log_vars_path, mode) as f: # Append mode ('a') or Write mode ('w')
             if episode == 0:
                  f.write("[") # Start of JSON list
             else:
                 f.seek(f.tell() - 1, os.SEEK_SET) # Go back one char to overwrite trailing bracket or comma
                 f.write(",") # Add comma before new entry

             for i, entry in enumerate(log_data):
                 f.write(str(entry).replace("'", '"')) # Basic JSON-like format
                 if i < len(log_data) - 1:
                     f.write(",")
             f.write("]") # End of JSON list for now


        # Save best model based on episode reward
        if episode_reward > best_performance:
            best_performance = episode_reward
            # Save model state and optimizer state
            torch.save({
                 'model_state_dict': agent.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'episode': episode,
                 'performance': best_performance,
             }, best_model_path)
            print(f"Episode {episode+1}: New best model saved with reward {best_performance:.4f}")

        # Print progress
        if (episode + 1) % 1 == 0: # Print every episode
            avg_reward = np.mean(episode_rewards[-10:]) # Avg reward of last 10 episodes
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.4f}, Avg Reward (last 10): {avg_reward:.4f}")


    # Load best model after training
    if os.path.exists(best_model_path):
         print(f"Training finished. Loading best model from {best_model_path}")
         checkpoint = torch.load(best_model_path)
         agent.load_state_dict(checkpoint['model_state_dict'])
         print(f"Loaded best model from episode {checkpoint['episode']+1} with performance {checkpoint['performance']:.4f}")
    else:
         print("Training finished. No best model was saved.")

    # Finalize JSON log file
    if os.path.exists(log_vars_path):
        with open(log_vars_path, 'rb+') as f:
            try:
                f.seek(-1, os.SEEK_END)
                if f.read(1) == b',':
                    f.seek(-1, os.SEEK_CUR)
                    f.write(b']')
                elif f.read(1) != b']': # Make sure it doesn't add extra ]
                    f.write(b']')
            except OSError: # Handle empty file case if needed
                f.seek(0)
                f.write(b'[]')


    return agent, episode_rewards, [] # Return empty list for metrics history for now
