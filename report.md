# Neural Network Compression: A Comparative Study of Ternary Weight Quantization Methods

## 1. Problem

Deep neural networks have achieved remarkable performance across various computer vision tasks, but their deployment on resource-constrained devices remains challenging due to significant storage and computational requirements [1]. The large number of parameters in state-of-the-art models creates bottlenecks for mobile and edge computing applications where memory bandwidth, energy consumption, and storage capacity are severely limited. Ternary weight quantization offers a promising solution by constraining weights to only three discrete values {-α, 0, +α}, providing a balance between the extreme compression of binary networks and the expressiveness of full-precision models.

## 2. Algorithms

This study implements and compares three prominent ternary quantization methods: Ternary Weight Networks (TWN), Trained Ternary Quantization (TTQ), and Alternating Direction Method of Multipliers (ADMM). TWN uses a symmetric quantization scheme where weights are mapped to {-α, 0, +α} based on a threshold Δ = 0.75 × mean(|W|), with a single learnable scaling factor α per layer computed as the mean of weights exceeding the threshold [2]. TTQ introduces asymmetric quantization with separate learnable scaling factors α⁺ and α⁻ for positive and negative weights respectively, allowing the model to learn both the ternary values and ternary assignments through gradient descent [3]. ADMM employs a constrained optimization approach using dual variables to enforce ternary constraints while maintaining training stability through augmented Lagrangian methods [4]. The key differences lie in TTQ's asymmetric scaling capability, TWN's simplicity, and ADMM's principled optimization framework for handling quantization constraints.

## 3. Experiments

We conducted experiments using LeNet-5 architecture on the MNIST dataset to evaluate the effectiveness of different quantization methods. Four models were trained: a full-precision (FP) baseline, TWN-quantized, TTQ-quantized, and ADMM-quantized networks, all using optimized training configurations determined through systematic hyperparameter tuning. The ADMM method required careful parameter tuning with ρ=1e-5, lr=0.002, and batch_size=128, while TWN and TTQ used lr=0.001 with batch_size=64. All models trained for 3 epochs using Adam optimizer. The experimental setup ensures fair comparison by maintaining the same network architecture (32-64 convolutional channels, 512 hidden units) across all methods, with performance evaluation focusing on classification accuracy on the MNIST test set and model compression analysis.

## 4. Results

The experimental results demonstrate varying trade-offs between accuracy and model compression across the four quantization methods. The full-precision baseline achieved 98.96% test accuracy with 581,408 total parameters, serving as the upper bound for comparison. Surprisingly, ADMM quantization achieved the best performance with 98.67% accuracy, representing only a 0.29% accuracy drop while theoretically providing 16× compression through 2-bit weight encoding. TWN quantization resulted in 95.46% accuracy (-3.50% vs baseline), while TTQ achieved 94.60% accuracy (-4.36% vs baseline). All ternary methods maintain the same parameter count during training (581,408 weights) but would achieve significant compression (from 2.2MB to 142KB) in deployment scenarios through proper 2-bit weight encoding.

## 5. Discussion

The comparative analysis reveals important insights about the accuracy-compression trade-off in ternary neural networks. ADMM demonstrated superior performance by achieving the best balance between accuracy retention and model compression, likely due to its principled constrained optimization approach that provides better gradient flow through dual variables and more stable convergence compared to direct quantization methods. The results highlight that while TTQ's asymmetric quantization provides additional flexibility compared to TWN's symmetric approach, both methods suffer from more aggressive accuracy degradation than ADMM's sophisticated optimization framework. A critical limitation of our implementation is that actual compression benefits are only realized at deployment - our training checkpoints store full optimization states rather than compressed weights. Future work should investigate deployment-optimized implementations, scalability to larger datasets and architectures, and hardware-specific optimizations that could further improve the efficiency gains of ternary quantization in practical scenarios.

---

## References

[1] Neural Network Compression: Ternary Networks Project Handout, Universitatea Babeș-Bolyai Cluj, 2024.

[2] F. Li, B. Liu, X. Wang, B. Zhang, and J. Yan, "Ternary Weight Networks," *arXiv preprint arXiv:1605.04711*, 2016.

[3] C. Zhu, S. Han, H. Mao, and W. J. Dally, "Trained Ternary Quantization," in *International Conference on Learning Representations (ICLR)*, 2017.

[4] C. Leng, Z. Dou, H. Li, S. Zhu, and R. Jin, "Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM," in *The AAAI Conference on Artificial Intelligence*, 2018. 