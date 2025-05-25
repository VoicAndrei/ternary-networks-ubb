#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import itertools
import json
import os
from train import LeNet5, validate
import argparse


def quick_train_and_validate(mode, lr, batch_size, rho=1e-3, epochs=1, max_batches=200):
    """Quick training and validation for hyperparameter tuning"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=False, transform=transform)
    val_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = LeNet5(mode=mode, rho=rho).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Quick training
    model.train()
    total_loss = 0
    batch_count = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # Add ADMM penalty if applicable
            if mode == 'admm':
                admm_loss = 0
                for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
                    if hasattr(layer, 'weight') and hasattr(layer, 'ternary_weight'):
                        residual = layer.weight - layer.ternary_weight + layer.dual_variable
                        admm_loss += layer.rho / 2 * torch.sum(residual ** 2)
                loss += admm_loss
            
            loss.backward()
            optimizer.step()
            
            if mode == 'admm':
                model.admm_update()
            
            total_loss += loss.item()
            batch_count += 1
    
    # Validation
    val_accuracy = validate(model, device, val_loader)
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    
    return val_accuracy, avg_loss


def tune_hyperparameters(mode, search_space, output_file):
    """Systematic hyperparameter tuning"""
    print(f"Tuning hyperparameters for {mode.upper()}")
    print(f"Search space: {search_space}")
    
    results = []
    best_accuracy = 0
    best_params = None
    
    # Generate all parameter combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        print(f"\nTesting: {params}")
        
        try:
            if mode == 'admm':
                accuracy, loss = quick_train_and_validate(
                    mode, params['lr'], params['batch_size'], params['rho']
                )
            else:
                accuracy, loss = quick_train_and_validate(
                    mode, params['lr'], params['batch_size']
                )
            
            result = {
                'mode': mode,
                'params': params,
                'accuracy': accuracy,
                'loss': loss
            }
            results.append(result)
            
            print(f"Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                print(f"New best accuracy: {best_accuracy:.2f}%")
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump({
            'mode': mode,
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'all_results': results
        }, f, indent=2)
    
    print(f"\nBest parameters for {mode.upper()}: {best_params}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Results saved to: {output_file}")
    
    return best_params, best_accuracy


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Ternary Networks')
    parser.add_argument('--mode', type=str, choices=['fp', 'twn', 'ttq', 'admm', 'all'], default='all',
                       help='Quantization mode to tune')
    parser.add_argument('--output_dir', type=str, default='./tuning_results',
                       help='Directory to save tuning results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define search spaces for each method
    search_spaces = {
        'fp': {
            'lr': [0.001, 0.002, 0.0005],
            'batch_size': [32, 64, 128]
        },
        'twn': {
            'lr': [0.001, 0.002, 0.0005, 0.0002],
            'batch_size': [32, 64, 128]
        },
        'ttq': {
            'lr': [0.0001, 0.0005, 0.001, 0.0002],
            'batch_size': [32, 64, 128]
        },
        'admm': {
            'lr': [0.001, 0.0005, 0.002],
            'batch_size': [64, 128],
            'rho': [1e-6, 1e-5, 1e-4, 1e-7]
        }
    }
    
    modes_to_tune = [args.mode] if args.mode != 'all' else ['fp', 'twn', 'ttq', 'admm']
    
    all_best_params = {}
    
    for mode in modes_to_tune:
        if mode in search_spaces:
            output_file = os.path.join(args.output_dir, f'{mode}_tuning.json')
            best_params, best_accuracy = tune_hyperparameters(mode, search_spaces[mode], output_file)
            all_best_params[mode] = {'params': best_params, 'accuracy': best_accuracy}
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'tuning_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_best_params, f, indent=2)
    
    print(f"\n=== TUNING SUMMARY ===")
    for mode, result in all_best_params.items():
        print(f"{mode.upper()}: {result['params']} -> {result['accuracy']:.2f}%")
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main() 