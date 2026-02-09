#include <stdio.h>
#include <cuda_runtime.h>

// This program simulates tile multiplication using CUDA.

// Define matrix dimensions and tile size
#define M 1024
#define N 1024
#define K 1024
#define TILE_SIZE 32

__global__ void tileMultiplyKernel(float* A, float* B, float* C) {
    // Shared memory for tiles of matrices A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate the thread's global row and column
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    float accumulator = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {

        // Load tiles from global memory to shared memory
        if (globalRow < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[globalRow * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros if outside matrix bounds
        }

        if (tile * TILE_SIZE + threadIdx.y < K && globalCol < N) {
             Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + globalCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f; // Pad with zeros if outside matrix bounds
        }

        // Synchronize threads to ensure tiles are loaded before computation
        __syncthreads();

        // Perform matrix multiplication on tiles in shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            accumulator += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize threads to ensure shared memory is not overwritten prematurely
        __syncthreads();
    }

    // Accumulate results in the C matrix
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = accumulator;
    }
}

int main() {
    // Declare host pointers
    float *h_A, *h_B, *h_C;

    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    // Initialize matrices A and B with sample data
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)(i % 10); // Simple initialization
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)(i % 5); // Simple initialization
    }

    // Declare device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    if (d_A == NULL || d_B == NULL || d_C == NULL) {
        fprintf(stderr, "Failed to allocate device memory!\n");
        return 1;
    }

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    tileMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Check for kernel launch errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Write the result to a file
    FILE *outFile = fopen("/content/drive/MyDrive/VITProjects/GPU_Programming/tile_multiply_result.txt", "w");
    if (outFile == NULL) {
        fprintf(stderr, "Failed to open output file!\n");
        return 1;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            fprintf(outFile, "%f ", h_C[i * N + j]);
        }
        fprintf(outFile, "\n");
    }

    fclose(outFile);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Matrix multiplication completed successfully!\n");
    printf("Result written to tile_multiply_result.txt\n");

    return 0;
}
