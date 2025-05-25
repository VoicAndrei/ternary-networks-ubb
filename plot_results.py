import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_results():
    """Read results.csv and create bar chart of accuracy comparison with compression info"""
    
    try:
        # Read CSV file
        df = pd.read_csv('results.csv')
        print("Results data:")
        print(df)
        print()
        
        # Extract data
        models = df['Mode'].values
        accuracy = df['Test_Accuracy'].values
        
        # Calculate theoretical compression ratios
        # All ternary methods have 16x theoretical compression vs FP
        compression_ratios = [1.0 if model == 'FP' else 16.0 for model in models]
        
        # Create figure with two subplots - increased size and height for better spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Subplot 1: Accuracy comparison
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars1 = ax1.bar(models, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracy):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Test Accuracy Comparison\n(Higher is Better)', fontsize=14, fontweight='bold', pad=25)
        ax1.set_ylim(90, 100)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xlabel('Quantization Method', fontsize=13, fontweight='bold')
        
        # Subplot 2: Compression ratio
        bars2 = ax2.bar(models, compression_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, comp in zip(bars2, compression_ratios):
            height = bar.get_height()
            label = f'{comp:.0f}×' if comp > 1 else 'Baseline'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax2.set_ylabel('Theoretical Compression Ratio', fontsize=13, fontweight='bold')
        ax2.set_title('Model Compression\n(Higher is Better)', fontsize=14, fontweight='bold', pad=25)
        ax2.set_ylim(0, 18)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlabel('Quantization Method', fontsize=13, fontweight='bold')
        
        # Add overall title with much more space
        fig.suptitle('Ternary Network Quantization: Accuracy vs Compression Trade-off', 
                    fontsize=16, fontweight='bold', y=0.93)
        
        # Adjust layout with much more space at top
        plt.tight_layout()
        plt.subplots_adjust(top=0.80)
        
        # Save plot
        plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Results comparison chart saved as 'results_comparison.png'")
        
        # Display summary statistics
        print("\nDetailed Results Summary:")
        print("=" * 60)
        baseline_acc = accuracy[models == 'FP'][0] if 'FP' in models else accuracy[0]
        
        for model, acc, comp in zip(models, accuracy, compression_ratios):
            acc_drop = acc - baseline_acc if model != 'FP' else 0
            status = "Baseline" if model == 'FP' else f"{acc_drop:+.2f}%"
            print(f"{model:4s}: {acc:5.2f}% accuracy | {comp:2.0f}× compression | {status}")
        
        # Key insights
        print("\n🎯 Key Insights:")
        print(f"• ADMM achieves near-baseline accuracy ({accuracy[models == 'ADMM'][0]:.2f}% vs {baseline_acc:.2f}%)")
        print(f"• All ternary methods provide 16× theoretical compression")
        print(f"• ADMM shows only {accuracy[models == 'ADMM'][0] - baseline_acc:.2f}% accuracy drop")
        
    except FileNotFoundError:
        print("Error: results.csv not found!")
        print("Please run the evaluation commands first to generate results.csv")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    plot_results() 