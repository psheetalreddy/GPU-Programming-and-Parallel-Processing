# GPU Programming and Parallel Processing with CUDA

## Overview
This project explores parallel computing using **NVIDIA CUDA** to understand how GPU-based execution improves performance compared to CPU-based execution. It implements and analyzes multiple parallel algorithms using custom CUDA kernels and compares their behavior, efficiency, and overheads.

The focus is on learning GPU programming fundamentals, memory management, and performance trade-offs in heterogeneous systems.

---

## Objectives
- Understand the CUDA programming model (kernels, threads, blocks, grids)  
- Implement basic parallel algorithms on the GPU  
- Compare CPU vs GPU execution performance  
- Analyze memory transfer overhead and kernel efficiency  

---

## Implemented Modules
### 1. Vector / Array Addition
- Parallel vector addition using CUDA kernels  
- Each thread computes one element independently  
- Compared execution with sequential CPU implementation

### 2. Standard Deviation using Asynchronous GPU Computing
- Implemented standard deviation calculation using asynchronous CUDA execution
- Utilized CUDA streams to overlap computation and memory transfers
- Demonstrated how asynchronous execution improves throughput compared to synchronous execution
- Highlighted reduction-based computation patterns on the GPU 

### 2. Image Processing: RGB → Grayscale Conversion
- Converted RGB images to grayscale using GPU parallelism  
- Each thread processes one pixel  
- Demonstrates data-parallel workloads well suited for GPUs

### 3. Matrix Multiplication (Tiled Implementation)
- Implemented matrix multiplication using shared memory tiling  
- Reduced global memory access by reusing shared memory  
- Demonstrated performance improvement over naive implementation  

---

## Technologies Used
- **Language:** C / C++  
- **Framework:** NVIDIA CUDA  
- **Tools:** NVCC, T4 GPU, Windows (CUDA-enabled system)  

---

## CUDA Concepts Covered
- CUDA kernels and kernel launches
- Thread hierarchy (grid, block, thread)
- Global and shared memory
- Asynchronous execution using CUDA streams
- Overlapping computation and memory transfers
- Host–device memory transfer (cudaMemcpy, cudaMemcpyAsync)
- Synchronization mechanisms (cudaDeviceSynchronize, cudaStreamSynchronize)
- Reduction operations
- Kernel launch configuration  

---

## Performance Analysis
- Compared CPU vs GPU execution for numerical and image-processing tasks
- Observed performance gains for large input sizes
- Identified memory transfer overhead as a limiting factor in synchronous execution
- Demonstrated improved efficiency using asynchronous computation with CUDA streams
- Showed that overlapping data transfer and computation can reduce total execution time 

---

## Project Structure
```
├── Addition of two arrays/
│   ├── add.cu
│   ├── 1.AdditionOf2Arrays.ipynb
├── Standard deviation calculation/
│   ├── stdDeviation.cu
│   ├── AsynchronousComputing.ipynb
├── RGB to Greyscale/
│   ├── rgb_to_greyScale.ipynb
│   ├── RGBToGrey.cu
│   ├── input.jpg
│   ├── output.jpg
├── Tiled Multiplication/
│   ├── TileMultiplication.ipynb
│   ├── tile_multiply.cu
│   ├── tile_multiply_result.txt
├── README.md
```

---

## How to Compile and Run
### Prerequisites
- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- NVCC compiler available  

### Compile
```bash
nvcc vector_add.cu -o vector_add
nvcc async_std_deviation.cu -o async_std_dev
nvcc rgb_to_grayscale.cu -o rgb_to_grayscale
nvcc matrix_multiplication.cu -o matrix_multiplication
```

### Run
```bash
./vector_add
./async_std_dev
./rgb_to_grayscale
./matrix_multiplication
```

---

## Limitations
- Optimizations are basic and focused on learning  
- Performance evaluation is hardware-dependent   

---

## Learning Outcomes
- Hands-on experience with GPU programming using CUDA  
- Understood when GPU acceleration is beneficial
- Understood asynchronous GPU execution and CUDA streams  
- Learned memory hierarchy and synchronization concepts  
- Built confidence in writing and debugging CUDA kernels  
