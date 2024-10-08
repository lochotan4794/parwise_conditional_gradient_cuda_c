#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "BPCG.h"

#define checkCudaErr(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__device__ void ArgMinWraper(volatile  float* s_max_values, volatile unsigned int* s_argmax) {
    if (s_max_values[threadIdx.x] > s_max_values[threadIdx.x + 32]) {
        s_max_values[threadIdx.x] = s_max_values[threadIdx.x + 32];
        s_argmax[threadIdx.x] = s_argmax[threadIdx.x + 32];
    }
    if (s_max_values[threadIdx.x] > s_max_values[threadIdx.x + 16]) {
        s_max_values[threadIdx.x] = s_max_values[threadIdx.x + 16];
        s_argmax[threadIdx.x] = s_argmax[threadIdx.x + 16];
    }
    if (s_max_values[threadIdx.x] > s_max_values[threadIdx.x + 8]) {
        s_max_values[threadIdx.x] = s_max_values[threadIdx.x + 8];
        s_argmax[threadIdx.x] = s_argmax[threadIdx.x + 8];
    }
    if (s_max_values[threadIdx.x] > s_max_values[threadIdx.x + 4]) {
        s_max_values[threadIdx.x] = s_max_values[threadIdx.x + 4];
        s_argmax[threadIdx.x] = s_argmax[threadIdx.x + 4];
    }
    if (s_max_values[threadIdx.x] > s_max_values[threadIdx.x + 2]) {
        s_max_values[threadIdx.x] = s_max_values[threadIdx.x + 2];
        s_argmax[threadIdx.x] = s_argmax[threadIdx.x + 2];
    }
    if (s_max_values[threadIdx.x] > s_max_values[threadIdx.x + 1]) {
        s_max_values[threadIdx.x] = s_max_values[threadIdx.x + 1];
        s_argmax[threadIdx.x] = s_argmax[threadIdx.x + 1];
    }
}

__global__ void KernelGetMinAndCreateIndex(const unsigned int array_size, float* d_scores, float* d_index) {
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int border = array_size >> 1;
    extern __shared__ float s_max_values[];

    if (tidx > border)
        return;

    unsigned int* s_argmax = (unsigned int*)&s_max_values[blockDim.x];
    unsigned int max_id = tidx;
    float max_value = d_scores[max_id];
    unsigned compare_idx = border + tidx;
    if (compare_idx < array_size && max_value > d_scores[compare_idx]) {
        max_id = compare_idx;
        max_value = d_scores[max_id];
    }

    s_max_values[threadIdx.x] = max_value;
    s_argmax[threadIdx.x] = max_id;

    for (border = blockDim.x >> 1; border > 32; border >>= 1) {
        if (threadIdx.x > border)
            return;
        compare_idx = threadIdx.x + border;
        __syncthreads();
        if (compare_idx < blockDim.x && max_value > s_max_values[compare_idx]) {
            max_value = s_max_values[compare_idx];
            max_id = s_argmax[compare_idx];
        }

        s_max_values[threadIdx.x] = max_value;
        s_argmax[threadIdx.x] = max_id;
    }

    if (threadIdx.x < 32)
        ArgMinWraper(s_max_values, s_argmax);

    if (threadIdx.x == 0) {
        d_scores[blockIdx.x] = s_max_values[0];
        d_index[blockIdx.x] = s_argmax[0];
    }
}

__global__ void KernelGetArgMin(const unsigned int array_size, float* d_scores, float* d_index) {
    unsigned int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int border = array_size >> 1;
    extern __shared__ float s_max_values[];
    if (tidx > border)
        return;
    unsigned int* s_argmax = (unsigned int*)&s_max_values[blockDim.x];
    float max_value = d_scores[tidx];
    unsigned int max_id = d_index[tidx];
    unsigned compare_idx = border + tidx;

    if (compare_idx < array_size && max_value > d_scores[compare_idx]) {
        max_id = d_index[compare_idx];
        max_value = d_scores[compare_idx];
    }

    s_max_values[threadIdx.x] = max_value;
    s_argmax[threadIdx.x] = max_id;

    for (border = blockDim.x >> 1; border > 32; border >>= 1) {
        if (threadIdx.x > border)
            return;
        compare_idx = border + threadIdx.x;
        __syncthreads();
        if (compare_idx < blockDim.x && max_value > s_max_values[compare_idx]) {
            max_value = s_max_values[compare_idx];
            max_id = s_argmax[compare_idx];
        }
        s_max_values[threadIdx.x] = max_value;
        s_argmax[threadIdx.x] = max_id;
    }

    if (threadIdx.x < 32)
        ArgMinWraper(s_max_values, s_argmax);

    if (threadIdx.x == 0) {
        // printf("Second Stage:\tTid: %d\tBlock-id:%d\tblockDim: %d\tmax_value: %d\tmax_id: %d\n\n",tidx, blockIdx.x, blockDim.x, max_value, max_id);
        d_scores[blockIdx.x] = s_max_values[0];
        d_index[blockIdx.x] = s_argmax[0];
    }
}


void GPUGetMinArgMin(unsigned int array_size, float* d_scores, float* d_index, float& max_score, unsigned int& argmax,
    const int block_size = 1024) {
    int grid_size = std::ceil(array_size / float(block_size) / 2);
    const unsigned int s_mem_size = block_size * (sizeof(float) + sizeof(unsigned int));
    float h_index;
    // cudaMalloc((void **)&d_index, sizeof(unsigned int) * grid_size);
    KernelGetMinAndCreateIndex << <grid_size, block_size, s_mem_size >> > (array_size, d_scores, d_index);
    // cudaDeviceSynchronize(); checkCudaErr(cudaGetLastError());
    while (grid_size > 1) {
        array_size = grid_size;
        grid_size = std::ceil(array_size / float(block_size) / 2);
        KernelGetArgMin << <grid_size, block_size, s_mem_size >> > (array_size, d_scores, d_index);
        // cudaDeviceSynchronize(); checkCudaErr(cudaGetLastError());
    }
    cudaMemcpy(&max_score, d_scores, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_index, d_index, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //Directly copy unsinged int from device to host returs weird value.
    argmax = h_index;
    // cudaFree(d_scores);
    // cudaFree(d_index);
}