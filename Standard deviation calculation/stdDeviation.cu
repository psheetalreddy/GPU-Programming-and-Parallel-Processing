
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute SUM of elements
__global__ void ComputeSum(float *data, int N, float *sum) {
    float partial = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        partial += data[i];
    }
    printf("[Kernel SUM] Thread %d partial = %f\n", threadIdx.x, partial);
    atomicAdd(sum, partial);
}

// Kernel to compute SUM of squared diffs from mean
__global__ void ComputeSumSqDiff(float *data, int N, float *mean, float *sumsq) {
    float partial = 0.0f;
    float m = *mean;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float diff = data[i] - m;
        partial += diff * diff;
    }
    printf("[Kernel SUMSQ] Thread %d partial = %f\n", threadIdx.x, partial);
    atomicAdd(sumsq, partial);
}

int main() {
    const int N = 4;
    float data[N] = {1.0, 2.0, 3.0, 4.0};

    // Device memory
    float *d_data, *d_sum, *d_sumsq, *d_mean;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sumsq, sizeof(float));
    cudaMalloc(&d_mean, sizeof(float));

    // Create streams and event
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEvent_t meanComputed;
    cudaEventCreate(&meanComputed);

    // Copy input
    cudaMemcpyAsync(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream1);
    cudaMemsetAsync(d_sumsq, 0, sizeof(float), stream2);

    // --- Step 1: Compute SUM (on stream1) ---
    ComputeSum<<<1, 4, 0, stream1>>>(d_data, N, d_sum);
    cudaEventRecord(meanComputed, stream1);   // mark completion of ComputeSum

    // Wait for meanComputed before launching Step 2
    cudaStreamWaitEvent(stream2, meanComputed, 0);

    // Copy sum back asynchronously
    float sum;
    cudaMemcpyAsync(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream1);

    // Synchronize stream1 so we know sum is ready
    cudaStreamSynchronize(stream1);

    // Compute mean on host
    float mean = sum / N;
    cudaMemcpyAsync(d_mean, &mean, sizeof(float), cudaMemcpyHostToDevice, stream2);

    // --- Step 2: Compute squared diffs (on stream2) ---
    ComputeSumSqDiff<<<1, 4, 0, stream2>>>(d_data, N, d_mean, d_sumsq);

    // Copy results back
    float sumsq, stddev;
    cudaMemcpyAsync(&sumsq, d_sumsq, sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // Synchronize stream2 (wait for all GPU work to finish)
    cudaStreamSynchronize(stream2);

    stddev = sqrt(sumsq / N);

    // Final result
    printf("mean = %f, stddev = %f\n", mean, stddev);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_sumsq);
    cudaFree(d_mean);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(meanComputed);

    return 0;
}


