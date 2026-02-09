#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// CUDA kernel for RGB to Grayscale conversion
__global__ void rgb_to_grayscale(unsigned char *input, unsigned char *output, 
                                  int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * channels;
        
        unsigned char r = input[rgb_idx];
        unsigned char g = input[rgb_idx + 1];
        unsigned char b = input[rgb_idx + 2];
        
        // Weighted average for grayscale: 0.299*R + 0.587*G + 0.114*B
        output[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }
    
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    
    // Load image
    int width, height, channels;
    unsigned char *h_input = stbi_load(input_file, &width, &height, &channels, 3);
    
    if (!h_input) {
        fprintf(stderr, "Error loading image: %s\n", input_file);
        return 1;
    }
    
    printf("Image loaded: %dx%d with %d channels\n", width, height, channels);
    
    // Allocate host memory for output
    size_t gray_size = width * height * sizeof(unsigned char);
    size_t rgb_size = width * height * channels * sizeof(unsigned char);
    unsigned char *h_output = (unsigned char*)malloc(gray_size);
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, rgb_size));
    CUDA_CHECK(cudaMalloc(&d_output, gray_size));
    
    // Copy input image to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, rgb_size, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    rgb_to_grayscale<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, gray_size, cudaMemcpyDeviceToHost));
    
    // Save grayscale image
    if (stbi_write_png(output_file, width, height, 1, h_output, width)) {
        printf("Grayscale image saved to: %s\n", output_file);
    } else {
        fprintf(stderr, "Error saving image: %s\n", output_file);
    }
    
    // Cleanup
    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
