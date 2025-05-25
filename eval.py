import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from train import LeNet5, TernaryConv2d, TTQConv2d, TernaryLinear, TTQLinear, ADMMConv2d, ADMMLinear


def count_nonzero_weights(model):
    """Count total number of non-zero weights in the model"""
    total_nonzero = 0
    total_weights = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only count weight parameters, not biases
            weights = param.data
            nonzero_count = torch.count_nonzero(weights).item()
            total_count = weights.numel()
            
            total_nonzero += nonzero_count
            total_weights += total_count
            
            print(f"{name}: {nonzero_count}/{total_count} ({100*nonzero_count/total_count:.1f}% non-zero)")
    
    print(f"\nTotal non-zero weights: {total_nonzero}/{total_weights} ({100*total_nonzero/total_weights:.1f}%)")
    return total_nonzero


def evaluate_model(model, device, test_loader):
    """Evaluate model on test set and return accuracy"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate LeNet-5 with Ternary Quantization')
    parser.add_argument('--mode', type=str, choices=['fp', 'twn', 'ttq', 'admm'], required=True,
                       help='Quantization mode: fp (full precision), twn (TWN), ttq (TTQ), admm (ADMM)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (default: ./checkpoints/{mode}.pth)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Evaluation mode: {args.mode}')
    
    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_path = f'./checkpoints/{args.mode}.pth'
    else:
        checkpoint_path = args.checkpoint
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint file {checkpoint_path} not found!')
        return
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    model = LeNet5(mode=args.mode).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Load checkpoint
    print(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Evaluate model
    print(f'\nEvaluating {args.mode.upper()} model on MNIST test set...')
    test_accuracy = evaluate_model(model, device, test_loader)
    
    # Count non-zero weights
    print(f'\nCounting non-zero weights in {args.mode.upper()} model...')
    total_nonzero = count_nonzero_weights(model)
    
    # Summary
    print(f'\n=== EVALUATION SUMMARY ===')
    print(f'Mode: {args.mode.upper()}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Total Non-zero Weights: {total_nonzero}')
    print(f'Checkpoint: {checkpoint_path}')


if __name__ == '__main__':
    main() 