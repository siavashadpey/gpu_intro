# 

This repository includes code written for the course [CS179 - GPU programming](courses.cms.caltech.edu/cs179/).

## Concepts 
The concepts taught in this course are:

1. GPU hardware and its abstractions (e.g. with grid, blocks, threads, and warps).
2. GPU memory (e.g. registers, shared mem, local mem, global mem, L1/L2/L3 caches, etc.) and their characteristics.
3. Best practices w.r.t. memory optimization (e.g. memory calescing, bank conflicts, register spilling).
4. Thread divergence and latency hiding via occupancy/thread-level parallelism (TLP) and instruction-level parallelism (ILP), and streaming parallelism.
5. Introduction to CUDA libraries such as cuBLAS, cuFFT, and cuDNN.

## Code
Code written for this course includes:

1. Lab 1 - Small kernel convolution.
2. Lab 2 - Matrix transposing. (concepts: memory coalescing, avoiding bank conflicts, ILP, atomic operations).
3. Lab 3 - Reduction and FFT. (concepts: writing a reduction algorithm, use of cuBLAS and cuFFT).
4. Lab 5 - Convolutional Neural Networks. (concepts: writing a CNN for MNIST handwritten digit classication, use of cuBLAS and cuDNN).

Additional code:

1. Tiled matrix multiplication.
