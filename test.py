# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Radar Echo Prediction System - Metrics Test Script
This script evaluates the trained models and agent using only metrics (CSI, POD, FAR)
without generating visualizations.
"""

import torch
import numpy as np
import os
import argparse
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
from config import DefaultConfigure
from dataset import DataSet
from model import Net
from agent import OptimizedTemporalModelSelectorAgent
from CSI import calc_CSI_reg

def denormalize(image):
    """Convert normalized data back to original scale"""
    return image * 85 - 10

def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_test_data(opt, batch_size=None):
    """Load only the test dataset"""
    if batch_size is None:
        batch_size = opt.batch_size
    
    # Read test data list
    with open(opt.radar_test_data_root, 'r') as fhandle:
        test_radar_list = fhandle.read().split('\n')
    if test_radar_list[-1] == '':
        test_radar_list.pop()
    
    # Create test dataset
    test_dataset = DataSet(test_radar_list, len(test_radar_list), test=True)
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    
    return test_loader

# Add this utility function to convert NumPy types to Python native types
def convert_numpy_to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_numpy_to_python_types(k): convert_numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_to_python_types(i) for i in obj]
    else:
        return obj

def evaluate_metrics(test_loader, models, agent, device, thresholds=[35], batch_limit=None):
    """
    Evaluate system performance using metrics only (no visualizations)
    
    Args:
        test_loader: DataLoader for test data
        models: Dict of trained prediction models
        agent: Trained selection agent
        device: Computation device (CPU/GPU)
        thresholds: List of thresholds for metrics calculation
        batch_limit: Optional limit on number of batches to process (for testing)
    
    Returns:
        results: Dict containing performance metrics
    """
    # Initialize results structure
    results = {
        'thresholds': {},
        'overall': {
            'agent': {'csi': 0, 'pod': 0, 'far': 0},
            'models': {time: {'csi': 0, 'pod': 0, 'far': 0} for time in models.keys()}
        }
    }
    
    # Initialize results for each threshold
    for threshold in thresholds:
        results['thresholds'][str(threshold)] = {
            'agent': {'csi': [], 'pod': [], 'far': []},
            'models': {time: {'csi': [], 'pod': [], 'far': []} for time in models.keys()}
        }
    
    # Set models and agent to evaluation mode
    for model in models.values():
        model.eval()
    agent.eval()
    
    # Process batches
    processed_batches = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch_limit is not None and batch_idx >= batch_limit:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Get agent's model selection
            model_probs, _ = agent(inputs)
            final_predictions = []
            
            for t in range(min(4, model_probs.size(1))):
                time_probs = model_probs[:, t]  # Get probabilities for all samples in batch
                
                # Apply selection for each sample in batch
                selected_predictions = []
                for b in range(batch_size):
                    # Select model for this sample
                    dist = torch.distributions.Categorical(time_probs[b])
                    action = dist.sample()
                    model_idx = action.item() % len(models)
                    selected_time = list(models.keys())[model_idx]
                    
                    # Get prediction from selected model
                    model_prediction = models[selected_time](inputs[b:b+1])
                    selected_predictions.append(model_prediction[:, t])
                
                # Stack predictions for this time step
                time_predictions = torch.cat(selected_predictions, dim=0)
                final_predictions.append(time_predictions)
            
            # Stack all time steps
            agent_predictions = torch.stack(final_predictions, dim=1)
            
            # Calculate metrics for each threshold
            for threshold in thresholds:
                # Denormalize predictions and targets before calculating metrics
                denorm_agent_preds = denormalize(agent_predictions.cpu().numpy())
                denorm_targets = denormalize(targets.cpu().numpy())
                
                # Calculate agent metrics
                metrics = calc_CSI_reg(denorm_agent_preds, denorm_targets, threshold=threshold)
                
                results['thresholds'][str(threshold)]['agent']['csi'].append(float(metrics[0]))  # CSI
                results['thresholds'][str(threshold)]['agent']['pod'].append(float(metrics[1]))  # POD
                results['thresholds'][str(threshold)]['agent']['far'].append(float(metrics[2]))  # FAR
                
                # Calculate metrics for each model
                for time, model in models.items():
                    predictions = model(inputs)
                    denorm_preds = denormalize(predictions.cpu().numpy())
                    
                    metrics = calc_CSI_reg(denorm_preds, denorm_targets, threshold=threshold)
                    
                    results['thresholds'][str(threshold)]['models'][time]['csi'].append(float(metrics[0]))
                    results['thresholds'][str(threshold)]['models'][time]['pod'].append(float(metrics[1]))
                    results['thresholds'][str(threshold)]['models'][time]['far'].append(float(metrics[2]))
            
            processed_batches += 1
    
    # Calculate overall metrics (average across batches)
    for threshold in thresholds:
        threshold_str = str(threshold)
        # Agent metrics
        results['overall']['agent']['csi'] += float(np.mean(results['thresholds'][threshold_str]['agent']['csi'])) / len(thresholds)
        results['overall']['agent']['pod'] += float(np.mean(results['thresholds'][threshold_str]['agent']['pod'])) / len(thresholds)
        results['overall']['agent']['far'] += float(np.mean(results['thresholds'][threshold_str]['agent']['far'])) / len(thresholds)
        
        # Models metrics
        for time in models.keys():
            results['overall']['models'][time]['csi'] += float(np.mean(results['thresholds'][threshold_str]['models'][time]['csi'])) / len(thresholds)
            results['overall']['models'][time]['pod'] += float(np.mean(results['thresholds'][threshold_str]['models'][time]['pod'])) / len(thresholds)
            results['overall']['models'][time]['far'] += float(np.mean(results['thresholds'][threshold_str]['models'][time]['far'])) / len(thresholds)
    
    # Add metadata
    results['metadata'] = {
        'processed_batches': processed_batches,
        'total_samples': total_samples
    }
    
    # Convert all NumPy types to Python native types for JSON serialization
    results = convert_numpy_to_python_types(results)
    
    return results

def load_models_and_agent(device, model_path="./model_path", agent_path="./rl_model/best_agent.pth"):
    """Load trained models and agent"""
    # Create models
    models = {
        '30': Net().to(device),
        '60': Net().to(device),
        '90': Net().to(device),
        '120': Net().to(device)
    }
    
    # Load model weights
    for time, model in models.items():
        model_file = os.path.join(model_path, f"regression_model_{time}_best.pth")
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location=device))
            print(f"Loaded model for time {time} from {model_file}")
        else:
            print(f"Warning: Model file {model_file} not found!")
    
    # Create and load agent
    agent = OptimizedTemporalModelSelectorAgent(
        input_channels=1, 
        hidden_dim=64, 
        num_models=len(models),
        input_size=(280, 360)
    ).to(device)
    
    if os.path.exists(agent_path):
        agent.load_state_dict(torch.load(agent_path, map_location=device))
        print(f"Loaded agent from {agent_path}")
    else:
        print(f"Warning: Agent file {agent_path} not found!")
    
    return models, agent

def main():
    parser = argparse.ArgumentParser(description='Radar Echo Prediction System - Metrics Test')
    parser.add_argument('--model_path', type=str, default='./model_path', help='Path to trained models')
    parser.add_argument('--agent_path', type=str, default='./rl_model/best_agent.pth', help='Path to trained agent')
    parser.add_argument('--batch_size', type=int, default=8, help='Test batch size')
    parser.add_argument('--batch_limit', type=int, default=None, help='Limit number of batches (for quick testing)')
    parser.add_argument('--thresholds', type=str, default='35', help='Comma-separated list of thresholds for metrics')
    parser.add_argument('--output', type=str, default='metrics_results.json', help='Output file for results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
    args = parser.parse_args()
    
    # Set random seed
    setup_seed(42)
    
    # Parse thresholds
    thresholds = [int(t) for t in args.thresholds.split(',')]
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load configuration
    opt = DefaultConfigure()
    
    # Load test data
    test_loader = load_test_data(opt, batch_size=args.batch_size)
    
    # Load models and agent
    models, agent = load_models_and_agent(device, args.model_path, args.agent_path)
    
    # Run evaluation
    print(f"Starting evaluation with thresholds: {thresholds}")
    results = evaluate_metrics(
        test_loader=test_loader,
        models=models,
        agent=agent,
        device=device,
        thresholds=thresholds,
        batch_limit=args.batch_limit
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary results
    print("\n===== EVALUATION RESULTS =====")
    print(f"Processed {results['metadata']['processed_batches']} batches, {results['metadata']['total_samples']} samples")
    
    print("\n----- OVERALL METRICS (AVERAGED ACROSS THRESHOLDS) -----")
    print("\nAgent Performance:")
    print(f"CSI: {results['overall']['agent']['csi']:.4f}")
    print(f"POD: {results['overall']['agent']['pod']:.4f}")
    print(f"FAR: {results['overall']['agent']['far']:.4f}")
    
    print("\nIndividual Models Performance:")
    for time, metrics in results['overall']['models'].items():
        print(f"\nModel {time}:")
        print(f"CSI: {metrics['csi']:.4f}")
        print(f"POD: {metrics['pod']:.4f}")
        print(f"FAR: {metrics['far']:.4f}")
    
    print(f"\nDetailed results saved to {args.output}")

if __name__ == "__main__":
    main()
