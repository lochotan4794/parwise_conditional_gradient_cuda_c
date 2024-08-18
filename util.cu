#include <cublas_v2.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include "BPCG.h"

__device__ volatile float blk_vals[MAX_BLOCKS];
__device__ volatile int   blk_idxs[MAX_BLOCKS];
__device__ int   blk_num = 0;

__global__ void max_idx_kernel(float* data, int dsize, int* result) {

    __shared__ volatile float   vals[nTPB];
    __shared__ volatile int idxs[nTPB];
    __shared__ volatile int last_block;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    last_block = 0;
    float   my_val = FLOAT_MIN;
    int my_idx = -1;
    // sweep from global memory
    while (idx < dsize) {
        if (data[idx] > my_val) { my_val = data[idx]; my_idx = idx; }
        idx += blockDim.x * gridDim.x;
    }
    // populate shared memory
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x] = my_idx;
    __syncthreads();
    // sweep in shared memory
    for (int i = (nTPB >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i)
            if (vals[threadIdx.x] < vals[threadIdx.x + i]) { vals[threadIdx.x] = vals[threadIdx.x + i]; idxs[threadIdx.x] = idxs[threadIdx.x + i]; }
        __syncthreads();
    }
    // perform block-level reduction
    if (!threadIdx.x) {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x] = idxs[0];
        if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
            last_block = 1;
    }
    __syncthreads();
    if (last_block) {
        idx = threadIdx.x;
        my_val = FLOAT_MIN;
        my_idx = -1;
        while (idx < gridDim.x) {
            if (blk_vals[idx] > my_val) { my_val = blk_vals[idx]; my_idx = blk_idxs[idx]; }
            idx += blockDim.x;
        }
        // populate shared memory
        vals[threadIdx.x] = my_val;
        idxs[threadIdx.x] = my_idx;
        __syncthreads();
        // sweep in shared memory
        for (int i = (nTPB >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i)
                if (vals[threadIdx.x] < vals[threadIdx.x + i]) { vals[threadIdx.x] = vals[threadIdx.x + i]; idxs[threadIdx.x] = idxs[threadIdx.x + i]; }
            __syncthreads();
        }
        if (!threadIdx.x)
            *result = idxs[0];
    }
}

__global__ void min_idx_kernel(float* data, int dsize, int* result) {

    __shared__ volatile float   vals[nTPB];
    __shared__ volatile int idxs[nTPB];
    __shared__ volatile int last_block;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    last_block = 0;
    float   my_val = FLOAT_MAX;
    int my_idx = -1;
    // sweep from global memory
    while (idx < dsize) {
        if (data[idx] < my_val) { my_val = data[idx]; my_idx = idx; }
        idx += blockDim.x * gridDim.x;
    }
    // populate shared memory
    vals[threadIdx.x] = my_val;
    idxs[threadIdx.x] = my_idx;
    __syncthreads();
    // sweep in shared memory
    for (int i = (nTPB >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i)
            if (vals[threadIdx.x] > vals[threadIdx.x + i]) { vals[threadIdx.x] = vals[threadIdx.x + i]; idxs[threadIdx.x] = idxs[threadIdx.x + i]; }
        __syncthreads();
    }
    // perform block-level reduction
    if (!threadIdx.x) {
        blk_vals[blockIdx.x] = vals[0];
        blk_idxs[blockIdx.x] = idxs[0];
        if (atomicAdd(&blk_num, 1) == gridDim.x - 1) // then I am the last block
            last_block = 1;
    }
    __syncthreads();
    if (last_block) {
        idx = threadIdx.x;
        my_val = FLOAT_MAX;
        my_idx = -1;
        while (idx < gridDim.x) {
            if (blk_vals[idx] < my_val) { my_val = blk_vals[idx]; my_idx = blk_idxs[idx]; 
            }
            idx += blockDim.x;
        }
        // populate shared memory
        vals[threadIdx.x] = my_val;
        idxs[threadIdx.x] = my_idx;
        __syncthreads();
        // sweep in shared memory
        for (int i = (nTPB >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i)
                if (vals[threadIdx.x] > vals[threadIdx.x + i]) { vals[threadIdx.x] = vals[threadIdx.x + i]; idxs[threadIdx.x] = idxs[threadIdx.x + i]; }
            __syncthreads();
        }
        if (!threadIdx.x)
            *result = idxs[0];
    }
}


int main1() {

    int nrelements = DSIZE;
    float* d_vector, *k_vector;
    float h_vector[DSIZE] = { 0 };
    for (int i = 0; i < DSIZE; i++) h_vector[i] = -rand() / (float) RAND_MAX;
    h_vector[10] = 0;  // create definite max element
    cudaMalloc(&d_vector, DSIZE * sizeof(float));
    cudaMalloc(&k_vector, DSIZE * sizeof(float));
    cudaMemcpy(d_vector, h_vector, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    h_vector[11] = 0;

    cudaMemcpy(k_vector, h_vector, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    int max_index = 0;
    max_index = 0;
    int* d_max_index;
    cudaMalloc(&d_max_index, sizeof(int));
    min_idx_kernel << <min(MAX_KERNEL_BLOCKS, ((DSIZE + nTPB - 1) / nTPB)), nTPB >> > (d_vector, DSIZE, d_max_index);
//    vecsub << <1, DSIZE >> > (d_vector, k_vector, k_vector);
    cudaMemcpy(&max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);

    printf("min index %d \n", max_index);

    return 0;
}

