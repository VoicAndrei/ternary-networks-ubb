#!/usr/bin/env python3

import torch
import numpy as np
from train import LeNet5

def calculate_deployment_size(checkpoint_path, mode):
    """Calculate the actual deployment size if properly compressed"""
    
    # Load model
    model = LeNet5(mode=mode)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    total_original_bits = 0
    total_compressed_bits = 0
    
    print(f"\n=== {mode.upper()} Deployment Compression Analysis ===")
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            original_bits = param.numel() * 32  # 32-bit floats
            
            if mode == 'fp':
                # Full precision - no compression
                compressed_bits = original_bits
                compression_ratio = 1.0
            else:
                # Ternary: each weight can be stored in 2 bits {00, 01, 10} = {-α, 0, +α}
                compressed_bits = param.numel() * 2
                compression_ratio = 32 / 2  # 16x compression
            
            total_original_bits += original_bits
            total_compressed_bits += compressed_bits
            
            print(f"{name}: {param.numel():,} weights")
            print(f"  Original: {original_bits/8/1024:.1f} KB (32-bit)")
            print(f"  Compressed: {compressed_bits/8/1024:.1f} KB (2-bit)")
            print(f"  Compression: {compression_ratio:.1f}x")
            print()
    
    # Add alpha parameters (still need full precision)
    if mode in ['twn', 'ttq', 'admm']:
        alpha_count = 4 if mode == 'ttq' else 4  # One per layer
        alpha_bits = alpha_count * 32
        total_compressed_bits += alpha_bits
        print(f"Alpha parameters: {alpha_bits/8/1024:.3f} KB (still 32-bit)")
    
    print(f"TOTAL ORIGINAL: {total_original_bits/8/1024:.1f} KB")
    print(f"TOTAL COMPRESSED: {total_compressed_bits/8/1024:.1f} KB")
    print(f"OVERALL COMPRESSION: {total_original_bits/total_compressed_bits:.1f}x")
    
    return total_compressed_bits/8/1024  # Return in KB

def main():
    """Compare deployment sizes of all methods"""
    
    methods = {
        'fp': 'checkpoints/fp.pth',
        'twn': 'checkpoints/twn.pth', 
        'ttq': 'checkpoints/ttq.pth',
        'admm': 'checkpoints/admm.pth'
    }
    
    results = {}
    
    for mode, checkpoint in methods.items():
        try:
            size_kb = calculate_deployment_size(checkpoint, mode)
            results[mode] = size_kb
        except Exception as e:
            print(f"Error processing {mode}: {e}")
    
    print("\n" + "="*60)
    print("DEPLOYMENT SIZE COMPARISON")
    print("="*60)
    
    fp_size = results.get('fp', 0)
    
    for mode, size_kb in results.items():
        compression = fp_size / size_kb if size_kb > 0 else 1
        print(f"{mode.upper():4s}: {size_kb:6.1f} KB ({compression:4.1f}x smaller than FP)")
    
    print("\n🎯 KEY INSIGHT:")
    print("Training files are large because they store optimization states.")
    print("Deployment models would be ~16x smaller due to 2-bit quantization!")

if __name__ == '__main__':
    main() 