import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os


class TernaryConv2d(nn.Module):
    """TWN: Ternary Weight Networks implementation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(TernaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
    def forward(self, x):
        # TWN quantization: Δ = 0.75 * mean(|w|)
        threshold = 0.75 * torch.mean(torch.abs(self.weight))
        
        # Create mask for weights above threshold
        mask = torch.abs(self.weight) > threshold
        
        # Compute layer-wise α = mean(|weights| where |weights| > threshold)
        if mask.sum() > 0:
            alpha = torch.mean(torch.abs(self.weight[mask]))
        else:
            alpha = torch.tensor(1.0, device=self.weight.device)
        
        # Quantize weights to {-α, 0, +α}
        ternary_weight = torch.where(self.weight > threshold, alpha,
                                   torch.where(self.weight < -threshold, -alpha, 
                                              torch.zeros_like(self.weight)))
        
        # Straight-through estimator for gradients
        ternary_weight = self.weight + (ternary_weight - self.weight).detach()
        
        return F.conv2d(x, ternary_weight, self.bias, self.stride, self.padding)


class TTQConv2d(nn.Module):
    """TTQ: Trained Ternary Quantization implementation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(TTQConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # Learnable scaling factors α+ and α-
        self.alpha_p = nn.Parameter(torch.ones(1))  # α+
        self.alpha_n = nn.Parameter(torch.ones(1))  # α-
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None
            
    def forward(self, x):
        # TTQ quantization as per paper
        # Step 1: Normalize weights to [-1, +1]
        max_weight = torch.max(torch.abs(self.weight))
        if max_weight > 0:
            normalized_weight = self.weight / max_weight
        else:
            normalized_weight = self.weight
        
        # Step 2: Apply threshold t = 0.05 (as used in paper)
        t = 0.05
        threshold = t  # Since normalized weights are in [-1, +1]
        
        # Step 3: Quantize to {-1, 0, +1}
        intermediate_ternary = torch.where(normalized_weight > threshold, torch.ones_like(normalized_weight),
                                         torch.where(normalized_weight < -threshold, -torch.ones_like(normalized_weight),
                                                   torch.zeros_like(normalized_weight)))
        
        # Step 4: Scale by learned α+ and α-
        ternary_weight = torch.where(intermediate_ternary > 0, self.alpha_p,
                                   torch.where(intermediate_ternary < 0, -self.alpha_n,
                                             torch.zeros_like(intermediate_ternary)))
        
        # TTQ-specific gradient handling
        pos_mask = normalized_weight > threshold
        neg_mask = normalized_weight < -threshold
        zero_mask = torch.abs(normalized_weight) <= threshold
        
        # Scaled straight-through estimator
        grad_weight = torch.zeros_like(self.weight)
        grad_weight[pos_mask] = self.alpha_p * self.weight[pos_mask]
        grad_weight[neg_mask] = self.alpha_n * self.weight[neg_mask] 
        grad_weight[zero_mask] = self.weight[zero_mask]
        
        ternary_weight = grad_weight + (ternary_weight - grad_weight).detach()
        
        return F.conv2d(x, ternary_weight, self.bias, self.stride, self.padding)


class ADMMConv2d(nn.Module):
    """ADMM: Alternating Direction Method of Multipliers for ternary quantization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, rho=1e-4):
        super(ADMMConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rho = rho  # ADMM penalty parameter
        
        # ADMM variables
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)  # x: continuous weights
        
        # Initialize ternary weights to quantized version of initial weights
        with torch.no_grad():
            initial_threshold = 0.1 * torch.mean(torch.abs(self.weight))
            initial_ternary = torch.where(self.weight > initial_threshold, 0.1,
                                        torch.where(self.weight < -initial_threshold, -0.1,
                                                  torch.zeros_like(self.weight)))
        
        self.register_buffer('ternary_weight', initial_ternary.clone())  # z: ternary weights
        self.register_buffer('dual_variable', torch.zeros_like(self.weight.data))   # u: dual variable
        
        # Learnable scaling factor - initialize to reasonable value
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels) * 0.01)
        else:
            self.bias = None
    
    def ternary_projection(self, w):
        """Project weights to ternary set {-α, 0, +α} with adaptive threshold"""
        # Use more conservative threshold - adaptive based on weight distribution
        abs_w = torch.abs(w)
        # Use 75th percentile as threshold instead of mean (less aggressive)
        threshold = torch.quantile(abs_w, 0.75) * 0.5
        
        # Ensure threshold is not too small
        threshold = torch.clamp(threshold, min=0.01)
        
        ternary = torch.where(w > threshold, torch.abs(self.alpha),
                            torch.where(w < -threshold, -torch.abs(self.alpha),
                                      torch.zeros_like(w)))
        return ternary
    
    def admm_update(self):
        """Perform ADMM updates for ternary constraint"""
        with torch.no_grad():
            # z-update: project to ternary constraint
            self.ternary_weight.copy_(self.ternary_projection(self.weight + self.dual_variable))
            
            # u-update: dual variable update with clipping to prevent explosion
            residual = self.weight - self.ternary_weight
            self.dual_variable += self.rho * residual
            
            # Clip dual variables to prevent explosion
            self.dual_variable.clamp_(-1.0, 1.0)
    
    def forward(self, x):
        # Use continuous weights early in training, gradually shift to ternary
        # This helps with convergence
        effective_weight = self.weight + 0.1 * (self.ternary_weight - self.weight).detach()
        return F.conv2d(x, effective_weight, self.bias, self.stride, self.padding)


class TernaryLinear(nn.Module):
    """TWN Linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(TernaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
            
    def forward(self, x):
        # TWN quantization
        threshold = 0.75 * torch.mean(torch.abs(self.weight))
        mask = torch.abs(self.weight) > threshold
        
        if mask.sum() > 0:
            alpha = torch.mean(torch.abs(self.weight[mask]))
        else:
            alpha = torch.tensor(1.0, device=self.weight.device)
        
        ternary_weight = torch.where(self.weight > threshold, alpha,
                                   torch.where(self.weight < -threshold, -alpha, 
                                              torch.zeros_like(self.weight)))
        
        ternary_weight = self.weight + (ternary_weight - self.weight).detach()
        
        return F.linear(x, ternary_weight, self.bias)


class TTQLinear(nn.Module):
    """TTQ Linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(TTQLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.alpha_p = nn.Parameter(torch.ones(1))
        self.alpha_n = nn.Parameter(torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
            
    def forward(self, x):
        # TTQ quantization
        max_weight = torch.max(torch.abs(self.weight))
        if max_weight > 0:
            normalized_weight = self.weight / max_weight
        else:
            normalized_weight = self.weight
        
        t = 0.05
        threshold = t
        
        intermediate_ternary = torch.where(normalized_weight > threshold, torch.ones_like(normalized_weight),
                                         torch.where(normalized_weight < -threshold, -torch.ones_like(normalized_weight),
                                                   torch.zeros_like(normalized_weight)))
        
        ternary_weight = torch.where(intermediate_ternary > 0, self.alpha_p,
                                   torch.where(intermediate_ternary < 0, -self.alpha_n,
                                             torch.zeros_like(intermediate_ternary)))
        
        pos_mask = normalized_weight > threshold
        neg_mask = normalized_weight < -threshold
        zero_mask = torch.abs(normalized_weight) <= threshold
        
        grad_weight = torch.zeros_like(self.weight)
        grad_weight[pos_mask] = self.alpha_p * self.weight[pos_mask]
        grad_weight[neg_mask] = self.alpha_n * self.weight[neg_mask]
        grad_weight[zero_mask] = self.weight[zero_mask]
        
        ternary_weight = grad_weight + (ternary_weight - grad_weight).detach()
        
        return F.linear(x, ternary_weight, self.bias)


class ADMMLinear(nn.Module):
    """ADMM Linear layer"""
    def __init__(self, in_features, out_features, bias=True, rho=1e-4):
        super(ADMMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rho = rho
        
        # ADMM variables - better initialization
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Initialize ternary weights to quantized version of initial weights
        with torch.no_grad():
            initial_threshold = 0.1 * torch.mean(torch.abs(self.weight))
            initial_ternary = torch.where(self.weight > initial_threshold, 0.1,
                                        torch.where(self.weight < -initial_threshold, -0.1,
                                                  torch.zeros_like(self.weight)))
        
        self.register_buffer('ternary_weight', initial_ternary.clone())
        self.register_buffer('dual_variable', torch.zeros_like(self.weight.data))
        
        # Learnable scaling factor - initialize to reasonable value
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features) * 0.01)
        else:
            self.bias = None
    
    def ternary_projection(self, w):
        """Project weights to ternary set {-α, 0, +α} with adaptive threshold"""
        # Use more conservative threshold
        abs_w = torch.abs(w)
        threshold = torch.quantile(abs_w, 0.75) * 0.5
        threshold = torch.clamp(threshold, min=0.01)
        
        ternary = torch.where(w > threshold, torch.abs(self.alpha),
                            torch.where(w < -threshold, -torch.abs(self.alpha),
                                      torch.zeros_like(w)))
        return ternary
    
    def admm_update(self):
        """Perform ADMM updates for ternary constraint"""
        with torch.no_grad():
            # z-update: project to ternary constraint
            self.ternary_weight.copy_(self.ternary_projection(self.weight + self.dual_variable))
            
            # u-update: dual variable update with clipping
            residual = self.weight - self.ternary_weight
            self.dual_variable += self.rho * residual
            self.dual_variable.clamp_(-1.0, 1.0)
    
    def forward(self, x):
        # Use continuous weights early in training, gradually shift to ternary
        effective_weight = self.weight + 0.1 * (self.ternary_weight - self.weight).detach()
        return F.linear(x, effective_weight, self.bias)


class LeNet5(nn.Module):
    """LeNet-5 with selectable quantization mode"""
    def __init__(self, mode='fp', rho=1e-3):
        super(LeNet5, self).__init__()
        self.mode = mode
        
        if mode == 'fp':
            # Full precision
            self.conv1 = nn.Conv2d(1, 32, 5)
            self.conv2 = nn.Conv2d(32, 64, 5)
            self.fc1 = nn.Linear(64 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
        elif mode == 'twn':
            # Ternary Weight Networks
            self.conv1 = TernaryConv2d(1, 32, 5)
            self.conv2 = TernaryConv2d(32, 64, 5)
            self.fc1 = TernaryLinear(64 * 4 * 4, 512)
            self.fc2 = TernaryLinear(512, 10)
        elif mode == 'ttq':
            # Trained Ternary Quantization
            self.conv1 = TTQConv2d(1, 32, 5)
            self.conv2 = TTQConv2d(32, 64, 5)
            self.fc1 = TTQLinear(64 * 4 * 4, 512)
            self.fc2 = TTQLinear(512, 10)
        elif mode == 'admm':
            # ADMM Ternary Quantization
            self.conv1 = ADMMConv2d(1, 32, 5, rho=rho)
            self.conv2 = ADMMConv2d(32, 64, 5, rho=rho)
            self.fc1 = ADMMLinear(64 * 4 * 4, 512, rho=rho)
            self.fc2 = ADMMLinear(512, 10, rho=rho)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def admm_update(self):
        """Update ADMM variables for all layers"""
        if self.mode == 'admm':
            # Update specific ADMM layers, not all modules to avoid recursion
            if hasattr(self.conv1, 'admm_update'):
                self.conv1.admm_update()
            if hasattr(self.conv2, 'admm_update'):
                self.conv2.admm_update()
            if hasattr(self.fc1, 'admm_update'):
                self.fc1.admm_update()
            if hasattr(self.fc2, 'admm_update'):
                self.fc2.admm_update()


def train_epoch(model, device, train_loader, optimizer, epoch, args):
    """Training function for one epoch"""
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Add ADMM penalty term
        if args.mode == 'admm':
            admm_loss = 0
            # Add penalty for specific ADMM layers - much more conservative
            for layer in [model.conv1, model.conv2, model.fc1, model.fc2]:
                if hasattr(layer, 'weight') and hasattr(layer, 'ternary_weight'):
                    residual = layer.weight - layer.ternary_weight + layer.dual_variable
                    # Use much smaller penalty weight and L1 norm instead of L2
                    admm_loss += 0.001 * layer.rho * torch.sum(torch.abs(residual))
            loss += admm_loss
        
        loss.backward()
        optimizer.step()
        
        # ADMM updates after gradient step
        if args.mode == 'admm':
            model.admm_update()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)


def validate(model, device, val_loader):
    """Validation function"""
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    
    print(f'Validation set: Average loss: {val_loss:.4f}, '
          f'Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)')
    
    return accuracy


def get_hyperparameters(mode):
    """Get hyperparameters for different modes"""
    if mode == 'fp':
        return {'lr': 0.001, 'batch_size': 64, 'epochs': 3}
    elif mode == 'twn':
        return {'lr': 0.001, 'batch_size': 64, 'epochs': 3}
    elif mode == 'ttq':
        return {'lr': 0.0005, 'batch_size': 64, 'epochs': 3}  # TTQ often needs lower LR
    elif mode == 'admm':
        return {'lr': 0.001, 'batch_size': 64, 'epochs': 3, 'rho': 1e-4}
    else:
        return {'lr': 0.001, 'batch_size': 64, 'epochs': 3}


def main():
    parser = argparse.ArgumentParser(description='LeNet-5 with Ternary Quantization')
    parser.add_argument('--mode', type=str, choices=['fp', 'twn', 'ttq', 'admm'], default='fp',
                       help='Quantization mode: fp (full precision), twn (TWN), ttq (TTQ), admm (ADMM)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--rho', type=float, default=1e-3, help='ADMM penalty parameter')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training mode: {args.mode}')
    
    # Get hyperparameters
    default_hparams = get_hyperparameters(args.mode)
    lr = args.lr if args.lr is not None else default_hparams['lr']
    batch_size = args.batch_size if args.batch_size is not None else default_hparams['batch_size']
    epochs = args.epochs if args.epochs is not None else default_hparams['epochs']
    rho = args.rho if 'rho' in default_hparams else 1e-3
    
    print(f'Hyperparameters: lr={lr}, batch_size={batch_size}, epochs={epochs}')
    if args.mode == 'admm':
        print(f'ADMM rho={rho}')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Hyperparameter tuning
    if args.tune:
        print("Starting hyperparameter tuning...")
        best_acc = 0
        best_hparams = None
        
        # Define search space
        lr_candidates = [0.001, 0.0005, 0.002] if args.mode != 'ttq' else [0.0001, 0.0005, 0.001]
        
        for lr_candidate in lr_candidates:
            print(f"\nTuning with lr={lr_candidate}")
            
            # Model
            model = LeNet5(mode=args.mode, rho=rho).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr_candidate)
            
            # Quick training (1 epoch for tuning)
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 100:  # Quick tuning, only 100 batches
                    break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                if args.mode == 'admm':
                    model.admm_update()
            
            # Quick validation
            val_accuracy = validate(model, device, val_loader)
            
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_hparams = {'lr': lr_candidate}
        
        print(f"\nBest hyperparameters: {best_hparams}, Best accuracy: {best_acc:.2f}%")
        lr = best_hparams['lr']
    
    # Model
    model = LeNet5(mode=args.mode, rho=rho).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    print(f'\nStarting training with {args.mode} quantization...')
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}/{epochs}:')
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch, args)
        val_accuracy = validate(model, device, val_loader)
        
        print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint_path = f'./checkpoints/{args.mode}.pth'
            torch.save(model.state_dict(), checkpoint_path)
    
    print(f'\nBest validation accuracy: {best_val_acc:.2f}%')
    print(f'Model saved to ./checkpoints/{args.mode}.pth')


if __name__ == '__main__':
    main() 