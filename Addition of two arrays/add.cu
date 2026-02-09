
#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int N = 10;
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    (cudaMalloc(&d_a, N * sizeof(int)), "alloc d_a");
    (cudaMalloc(&d_b, N * sizeof(int)), "alloc d_b");
    (cudaMalloc(&d_c, N * sizeof(int)), "alloc d_c");

    (cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice), "copy a to d_a");
    (cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice), "copy b to d_b");

    add<<<1, N>>>(d_a, d_b, d_c, N);

    (cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost), "copy d_c to c");

    printf("Result:");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

