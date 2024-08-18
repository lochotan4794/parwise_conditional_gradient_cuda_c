#pragma once
#ifndef _H_BPCG
#define _H_BPCG

#include <assert.h>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#define MAX_CACHE_SIZE 5
#define THREAD_PER_BLOCK 100


#define DSIZE 100
// nTPB should be a power-of-2
#define nTPB 64
#define MAX_KERNEL_BLOCKS 100
#define MAX_BLOCKS ((DSIZE/nTPB)+1)
#define MIN(a,b) ((a>b)?b:a)
#define FLOAT_MIN -1.0f
#define FLOAT_MAX  1e10f


//void mean(float**, float*, int, int);
//void std_data(float**, float*, int, int);
void normalize(float**, float**, int, int );

void GPUGetMaxArgMax(unsigned int array_size, float* d_scores, float* d_index, float& max_score, unsigned int& argmax, const int block_size);
void GPUGetMinArgMin(unsigned int, float*, float*, float&, unsigned int&, const int);

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif


// float gauss_kernel(int numCoords, int numObjs, float *objects, int objectId1, int objectId2, int N);
// void primal_function(float *x, float *prob, int numObjs, int numCoords, float **objects, float *primal, int N);
// void gradient_function(int numCoords, int numObjs, float **objects, float mu, float *x, float *gradient, int N );
// void update_x(cuda_cache *cache, float *x_t, float step);

void bpcg_optimizer(int, float*, float*, float*, float**, float*, int, int, int);

float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);

// __host__ __device__ inline static void VecAdd(float* A, float* B, float* C);
// __host__ __device__ inline static void VecMul(float* A, float* B, float* C);
// __host__ __device__ inline static void VecSub(float* A, float* B, float* C);
// __host__ __device__ inline static void VecInner(float* A, float* B, float *C);
// __host__ __device__ inline static void VecNorm(float* A, int n, float *res);
// __host__ __device__ inline static void VectorSubWithScale(float* A, float* B, float* C, float scale);

// void set_alpha(int id, cuda_cache *cache, float alpha);
// void add_cache(int id, cuda_cache *cache, float alpha, float *v);
// int in_cache(cuda_cache cache, int id);

#endif