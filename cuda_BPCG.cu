
#include <stdio.h>
#include <stdlib.h>
#include "BPCG.h"

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void compute_extreme_point(float *grad, float *extreme_point)// [numCoords][numObjs]
                                                     
{
    int imin = 0;
    float min_val = grad[0];
    int i = threadIdx.x;

    if (grad[i] < min_val) 
    {
        min_val = grad[i];
        imin = i;
    }
    extreme_point[imin] = 1;
}


__host__ __device__ void away_step(float *grad, float *extreme_point)// [numCoords][numObjs]                 
{
    int i = threadIdx.x;
    int imax = 0;
    float max_val = grad[0];
    if (grad[i] > max_val) 
    {
        max_val = grad[i];
        imax = i;
    }
    extreme_point[imax] = 1;
}

__host__ __device__ float step_size(int numCoords, float *grad, float *x_t, float **objects, float gama_max)// [numCoords][numObjs]
                                                     
{
    float tau = 1.5;

    float mu = 0.5;

    float M = mu * L;

    float n_dt;

    float f_new;

    VectorNorm<<1,N>>(dt, n_dt);

    n_dt = n_dt * n_dt;

    gama = min((g_t / (M * n_dt)), gama_max);

    primal_function(x_t, numCoords, objects, &f_new)

    Q_t = f_new - gama * g_t + 0.5 * M * n_dt * gama**2;

    VectorSubWithScale<<<1, N>>>(x_t, d_t, x_t, gama);

    primal_function(x_t, numCoords, objects, &f_new)

    while f_new > Q_t:

        M = tau * M;

        gama = min((g_t / (M * n_dt)), gama_max);

        primal_function(x_t, numCoords, objects, &f_new);

        Q_t = fnew - gama * g_t + 0.5 * M * n_dt * gama**2;

        VectorSubWithScale<<<1, N>>>(x_t, d_t, x_t, gama);
        
        primal_function(x_t, numCoords, objects, &f_new)

    return gama;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static float gauss_kernel(int numCoords,
                                                     int numObjs,
                                                     float *objects, // [numCoords][numObjs]
                                                     int objectId1,
                                                     int objectId2)
{
    int i;
    float ans = 0.0;
    euclid_dist_2<<<1, N>>>(numCoords, numObjs, objectId1, objectId2, objects, &ans);
    ans = (1 + sqrt(3) * ans) * exp(-sqrt(3) * ans);
    return (ans);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__global__ inline static void euclid_dist_2(int numObjs,
                                                      int objectId1,
                                                      int objectId2,
                                                      float *objects, float *res) // [numCoords][numObjs]
{
    int i = threadIdx.x;
    float ans;
    ans += (objects[numObjs * i + objectId1] - objects[numObjs * i + objectId2]) *
               (objects[numObjs * i + objectId1] - objects[numObjs * i + objectId2]);
    &res = sqrt(ans);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static float primal_function(
    float *x,
    int numCoords,
    float **objects,
    float *primal)

{
    int i;
    int k;
    int j;

    float v;
    VecSub<<1, N>>(x, mu, &v)

    for (k = 0; k < numObjs; k++)
    {
        float kerSum = 0;
        for (j = 0; j < numObjs; j++)
        {   
            kerSum += gauss_kernel(numCoords, numObjs, *objects, k, j);
        }
        &primal += kerSum * v[k];
    }
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static void gradient_function(int numCoords,
                                                         int numObjs,
                                                         int numClusters,
                                                         float *objects,
                                                         float *mu,
                                                         float *x, 
                                                         float *gradient )
{
    int i;
    int k;
    int j;

    float v;
    VecSub<<1, N>>(x, mu, &v)

    for (k = 0; k < numObjs; k++)
    {
        float kerSum = 0;
        for (j = 0; j < numObjs; j++)
        {   
            kerSum += gauss_kernel(numCoords, numObjs, *objects, k, j);
        }
        gradient[k] = kerSum * v[k];
    }
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void local_linear_minimization_oracle(int numCoords, int numObjs, float *gradient, float **cache, float *res)// [numCoords][numObjs]
                                                     
{
    int i;
    int imin = 0;
    float min_val = 0;
    for (i = 1; i < numObjs; i++)
    {
        float innerVal = 0;
        vectorInner<<<1, N>>>(cache[i], gradient, &innerVal)
        if (innerVal < min_val)
        {
            imin = i;
            min_val = innerVal;
        }
    }
    for (i = 1; i < numObjs; i++) res[i] = 0;
    res[imin] = 1;
}

__host__ __device__ void update_x(cuda_cache *cache, float *x_t, float step)
{
    int i = threadIdx.x;
    int j = 0;
    for(j=0;j<cache.len;j++)
    {
        x_t[i] = x_t[i] - cache.alpha[j] * cache[i][j];
    }
}

__host__ __device__ inline static void bpcg_optimizer(float* x0, int maxIter, cuda_cache cache, int dim, float* x_t, float **objects)
{
    int L = 1;
    int i = 0;
    int number_drop;
    size_t size = dim * sizeof(float);
    int vertex_added = 0;
    // Allocate input vectors h_A and h_B in host memory
    float* gradient = (float*)malloc(size);
    float* d_FW = (float*)malloc(size);
    float* d_LFW = (float*)malloc(size);
    float* d_AW = (float*)malloc(size);
    float* direction = (float*)malloc(size);

    float* d_grad;
    float* dd_FW;
    float* dd_LFW;
    float* dd_AW;
    float d_direction;
    float primal;

    cudaMalloc(&d_gradient, size);
    cudaMalloc(&dd_FW, size);
    cudaMalloc(&dd_LFW, size);
    cudaMalloc(&dd_AW, size);
    cudaMalloc(&d_direction, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_grad, gradient, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_FW, dd_FW, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_LFW, dd_LFW, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_AW, dd_AW, size, cudaMemcpyHostToDevice);
    cudaMemcpy(direction, d_direction, size, cudaMemcpyHostToDevice);

    while (i < maxIter)
    {
        /* code */
        gradient_function(numCoords, numObjs, objects, mu, x_t,  d_grad );

        cudaMemset(dd_FW, 0, dim);

        cudaMemset(dd_LFW, 0, dim);

        cudaMemset(dd_AW, 0, dim);
	
        int id_LFW  = local_linear_minimization_oracle(numCoords, d_grad, extreme_point);

        int id_AW  = away_step( numCoords, d_grad, extreme_point);

        int id_FW  = compute_extreme_point(numCoords, d_grad, extreme_point);

        dd_FW[id_FW] = 1;

        dd_AW[id_AW] = 1;

        dd_LFW[id_LFW] = 1;

        float dual_gap;

        float away_gap;

        VectorSub<<<1, N>>>(dd_FW, dd_AW, direction);

        VectorInner<<<1, N>>>(direction, d_grad, &dual_gap);

        VectorSub<<<1, N>>>(dd_LFW, dd_AW, direction);

        VectorInner<<<1, N>>>(direction, d_grad, &away_gap);

        if (away_gap > dual_gap)
        {

            VectorSub<<<1, N>>>(dd_LFW, dd_AW, direction);

            float d = alpha_t[id_A];

            float gap = VectorInner<<<1, N>>>(d_t,  d_grad);

            float step = step_size(dim, grad, x_t, objects, gama_max);
           
            if (step < d)
            {
                float a_FW = cache.alpha[id_FW] + step;
                set_alpha(id_FW, cache, a_FW);
                float a_FW = cache.alpha[id_FW] - step;
                set_alpha(id_A, cache, a_FW);
            }
            else
            {

                float a_FW = cache.alpha[id_FW] + d;

                set_alpha(id_FW, cache, a_FW);

                set_alpha(id_A, cache, 0);

                number_drop = number_drop + 1;
            }

        } 
        else 
        {

            VectorSub<<<1, N>>>(dd_FW, dd_AW, direction);

            VectorInner<<<1, N>>>(direction, d_grad, gap);

            float step = step_size(dim, grad, x_t, objects, gama_max);

            update_alpha(id_W, cache, step);

            vertex_added = 1;

        }

        update_x<<<1, N>>>(cuda_cache *cache, float *x_t, float step);

        primal_function(x_t, numCoords, objects, &primal);

        i = i + 1;

        printf("%x \n", primal);
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
}
