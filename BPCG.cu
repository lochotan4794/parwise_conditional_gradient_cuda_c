

#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include "BPCG.h"

#include "util.cuh"

#include "cuda.h"



void primal_function(int, int, float*, float*, float*, float*, int);

void gradient_function(int, int, float**, float, float*, float*, int);

__host__ __device__

float euclid_dist_2(int numCoords, int numObjs, float* objects, int objectId1, int objectId2);





// In this case the number of GPU threads is smaller than the number of elements in the domain:

//  every iterates over multple elements to ensure than the entire domain is covered

__global__ void sum_vectors(const float* v1, const float* v2, float* out, size_t num_elements) {

    // compute current thread id

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over vector: grid can be smaller than vector, it is therefore

    // required that each thread iterate over more than one vector element

    while (xIndex < num_elements) {

        out[xIndex] = v1[xIndex] + v2[xIndex];

        xIndex += gridDim.x * blockDim.x;

    }

}



// In this case the number of GPU threads is smaller than the number of elements in the domain:

//  every iterates over multple elements to ensure than the entire domain is covered

__global__ void substr_vectors_with_scale(const float* v1, const float* v2, float* out, float* scale, size_t num_elements) {

    // compute current thread id

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over vector: grid can be smaller than vector, it is therefore

    // required that each thread iterate over more than one vector element

    while (xIndex < num_elements) {

        out[xIndex] = v1[xIndex] - v2[xIndex] * (*scale);

        xIndex += gridDim.x * blockDim.x;

    }

    __syncthreads();


}



// In this case the number of GPU threads is smaller than the number of elements in the domain:

//  every iterates over multple elements to ensure than the entire domain is covered

__global__ void substr_vectors(const float* v1, const float* v2, float* out, size_t num_elements) {

    // compute current thread id

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over vector: grid can be smaller than vector, it is therefore

    // required that each thread iterate over more than one vector element

    while (xIndex < num_elements) {

        out[xIndex] = v1[xIndex] - v2[xIndex];

        xIndex += gridDim.x * blockDim.x;

    }

    __syncthreads();
}

#define imin(a,b) (a<b?a:b)

const int N = DSIZE;
const int threadsPerBlock = 256; const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c, int N) {

    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();
    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


void VecInner(float* a, float* b, float* c, int size)
{

    float * dev_partial_c;

    float * partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaMalloc((void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    dot<<<blocksPerGrid, threadsPerBlock >>> (a, b, dev_partial_c, size);


    cudaDeviceSynchronize(); checkLastCudaError();

    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    *c = 0;

    for (int i = 0; i < blocksPerGrid; i++) {
        *c += partial_c[i];
    }


}

//__global__ void VecInner(float* a, float* b, float* c, int size)
//
//{
//
//    __shared__ float temp[DSIZE];
//
//
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//    temp[threadIdx.x] = (index < DSIZE) ? a[index] * b[index] : 0;
//
//
//    __syncthreads();
//
//
//    if (0 == threadIdx.x) {
//
//
//
//        float sum = 0;
//
//
//
//        for (int i = 0; i < DSIZE; i++)
//
//        {
//
//            sum += temp[i];
//
//            //printf("%f \n", temp[i]);
//
//        }
//
//
//
//        *c = 0;
//
//        //*c = sum
//        (void)atomicAdd(c, sum);
//        //*c = *c + sum;
//
//
//
//    }
//
//}





__global__  void VecNorm(float* a, int n, float* res)

{

    __shared__ float product[DSIZE];



    int i = threadIdx.x;



    if (i < DSIZE)

    {

        product[i] = a[i] * a[i];

    }



    __syncthreads();



    if (threadIdx.x == 0)

    {

        int* sum;



        *sum = 0;



        for (int k = 0; k < DSIZE; k++)

        {

            atomicAdd(sum, product[k]);

        }

        *res = sqrtf(*sum);

    }

}



__global__  void VecSquaredNorm(float* a, int n, float* res)

{

    __shared__ float product[DSIZE];

    int i = threadIdx.x;

    if (i < DSIZE)

    {

        product[i] = a[i] * a[i];

    }



    __syncthreads();



    if (threadIdx.x == 0)

    {

        float sum = 0;



        for (int k = 0; k < DSIZE; k++)

        {

            atomicAdd(&sum, product[k]);

        }

        *res = sum;

    }

}



void compute_extreme_point(float* d_vector, int* active, float* d, int* d_max_index)

{

    const unsigned int array_size = DSIZE;

    float* h_scores = new float[array_size];

    for (int i = 0; i < array_size; i++)

        h_scores[i] = std::rand();

    float max_score = 0;

    unsigned int argmax = 0;

    //std::cout << "CPU Max and Argmax: " << max_score << "\t" << argmax << "\ttakes" << "" << "sec\n\n";

    const unsigned int block_size = 64;

    float* d_scores, * d_index;

    int grid_size = std::ceil(array_size / float(block_size) / 2);



    //cudaMallc occupies most of the time consuming!

    cudaMalloc((void**)&d_scores, sizeof(float) * array_size);

    cudaMalloc((void**)&d_index, sizeof(unsigned int) * grid_size);

    cudaMemcpy(d_scores, h_scores, sizeof(float) * array_size, cudaMemcpyHostToDevice);

    //begin = std::chrono::steady_clock::now();

    GPUGetMinArgMin(array_size, d_vector, d_index, max_score, argmax, block_size);

    //end = std::chrono::steady_clock::now();

    cudaMemcpy(d_max_index, &argmax, sizeof(int), cudaMemcpyHostToDevice);



    float res[DSIZE] = { 0 };



    res[argmax] = 1.0;



    checkCuda(cudaMemcpy(d, res, DSIZE * sizeof(float), cudaMemcpyHostToDevice));



}



float step_size(int numCoords, int numObjs, int blocksPerGrid, int threadsPerBlock, float* probs, float* grad, float* x_t, float* const d_t, float g_t, float* objects, float* gama_max, float* dv, int dim) // [numCoords][numObjs]
{

    float tau = 1.5;

    float mu = 0.5;

    float L = 1;

    float M = mu * L;

    float* f_new = (float*)malloc(sizeof(float));

    *f_new = 0;

    float* norm = (float*)malloc(sizeof(float));

    *norm = 0;

    float* d_norm;

    float* d_fnew;

    float* d_gama;

    float v[DSIZE] = { 0 };

    float x_inter[DSIZE] = { 0 };

    float* d_x_inter;



    checkCuda(cudaMalloc(&d_norm, sizeof(float)));

    checkCuda(cudaMalloc(&d_x_inter, DSIZE * sizeof(float)));



    VecSquaredNorm << <blocksPerGrid, threadsPerBlock >> > (d_t, dim, d_norm);



    cudaDeviceSynchronize(); checkLastCudaError();



    checkCuda(cudaMemcpy(norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost));



    float gama = 0;



    gama = MIN(g_t / (M * (*norm)), *gama_max);



    checkCuda(cudaMalloc(&d_fnew, sizeof(float)));



    checkCuda(cudaMemcpy(d_fnew, f_new, sizeof(float), cudaMemcpyHostToDevice));



    checkCuda(cudaMalloc(&d_gama, sizeof(float)));



    float* zero = 0;


#if DEBUG

    checkCuda(cudaMemcpy(v, dv, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));


#endif



    primal_function(numCoords, numObjs, objects, dv, grad, f_new, dim);



    cudaDeviceSynchronize(); checkLastCudaError();



    float f_start = *f_new;



    float Q_t = f_start - gama * g_t + 0.5 * M * (*norm) * gama * gama;



    checkCuda(cudaMemcpy(d_gama, &gama, sizeof(float), cudaMemcpyHostToDevice));



    substr_vectors_with_scale << <blocksPerGrid, threadsPerBlock >> > (x_t, d_t, d_x_inter, d_gama, dim);

    cudaDeviceSynchronize(); checkLastCudaError();



    substr_vectors << <blocksPerGrid, threadsPerBlock >> > (d_x_inter, probs, dv, dim);

    cudaDeviceSynchronize(); checkLastCudaError();



    primal_function(numCoords, numObjs, objects, dv, grad, f_new, dim);

    cudaDeviceSynchronize(); checkLastCudaError();


    int loop = 0;


    while (*f_new > Q_t)

    {



        M = tau * M;



        gama = MIN(g_t / (M * (*norm)), *gama_max);



        checkCuda(cudaMemcpy(d_gama, &gama, sizeof(float), cudaMemcpyHostToDevice));



        Q_t = f_start - gama * g_t + 0.5 * M * *norm * gama * gama;



        float h_x[DSIZE] = { 0 };

        float h_d[DSIZE] = { 0 };





        substr_vectors_with_scale << <blocksPerGrid, threadsPerBlock >> > (x_t, d_t, d_x_inter, d_gama, dim);

        cudaDeviceSynchronize(); checkLastCudaError();





        checkCuda(cudaMemcpy(h_x, d_x_inter, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));

        checkCuda(cudaMemcpy(h_d, d_t, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));



        substr_vectors << <blocksPerGrid, threadsPerBlock >> > (d_x_inter, probs, dv, dim);

        cudaDeviceSynchronize(); checkLastCudaError();



        primal_function(numCoords, numObjs, objects, dv, grad, f_new, dim);



        cudaDeviceSynchronize(); checkLastCudaError();



        checkCuda(cudaMemcpy(d_t, h_d, DSIZE * sizeof(float), cudaMemcpyHostToDevice));


        loop = loop + 1;

        if (loop % 100 == 0)
        {
            printf("%d \n", loop);
        }


    }





    //free(norm);

    //free(f_new);

    //cudaFree(d_fnew);

    //cudaFree(d_norm);



    //printf("gama done ! %f \n", gama);

    printf("gama ! %.9f \n", gama);



    return gama;

}



__host__ __device__ inline static

float euclid_dist_2(int    numCoords,



    int    numObjs,



    float* objects,     // [numCoords][numObjs]



    int    objectId1,



    int    objectId2)



{



    int t;



    float ans = 0.0;



    for (t = 0; t < numCoords; t++) {



        ans += (objects[numObjs * t + objectId1] - objects[numObjs * t + objectId2]) *



            (objects[numObjs * t + objectId1] - objects[numObjs * t + objectId2]);

    }



    return ans;



}



__global__ void hibertspace(int numCoords, int numObjs, float* objects, float* kernel, float* v, int id)

{

    int idx = blockDim.x * blockIdx.x + threadIdx.x;



    __shared__ float ans[DSIZE];



    float tmp = euclid_dist_2(numCoords, numObjs, objects, idx, id);



    ans[idx] = (1 + sqrtf(3) * tmp) * exp(-sqrtf(3) * tmp);



    __syncthreads();



    if (threadIdx.x == 0)

    {

        for (int k = 0; k < DSIZE; k++)

        {

//            printf("dis %d %f", k, ans[k]);

            atomicAdd(&kernel[id], ans[k] * v[k]);
            //kernel[id] = kernel[id] + ans[k] * v[k];

        }

   //     printf("\n");

    }



}



void primal_function(int numCoords, int numObjs, float* d_objects, float* v, float* d_grad, float* primal, int dim)

{



    //float* kernel = (float*)malloc(dim * sizeof(float));

    float kernel[DSIZE] = { 0 };

    float* d_kernel;

    *primal = 0;



    cudaMalloc(&d_kernel, dim * sizeof(float));



    float h_objects[3000] = { 0 };



    //cudaMalloc(&d_objects, numObjs * numCoords * sizeof(float));

    float hv[DSIZE] = { 0 };

    cudaMemcpy(hv, v, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_objects, d_objects, numObjs * numCoords * sizeof(float), cudaMemcpyDeviceToHost);


    //*primal = 0;



    for (int i = 0; i < numObjs; i++) {



        hibertspace << <1, numObjs >> > (numCoords, numObjs, d_objects, d_kernel, v, i);



        cudaDeviceSynchronize(); checkLastCudaError();


        cudaMemcpy(kernel, d_kernel, dim * sizeof(float), cudaMemcpyDeviceToHost);



        *primal = *primal + kernel[i] * hv[i];



    }

}



void gradient_function(int numCoords, int numObjs, float* d_objects, float* v, float* gradient, int dim)

{



    float kernel[DSIZE] = { 0 };



    //float* d_objects;



   // cudaMalloc(&d_objects, numObjs * numCoords * sizeof(float));



   // cudaMemcpy(d_objects, objects, numObjs * numCoords * sizeof(float), cudaMemcpyHostToDevice);



    float* d_kernel;



    cudaMalloc(&d_kernel, dim * sizeof(float));


#if DEBUG

    float hv[DSIZE] = { 0 };

    cudaMemcpy(hv, v, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

#endif



    for (int i = 0; i < numObjs; i++) {



        hibertspace << <1, numObjs >> > (numCoords, numObjs, d_objects, d_kernel, v, i);



        cudaDeviceSynchronize(); checkLastCudaError();



        cudaMemcpy(kernel, d_kernel, dim * sizeof(float), cudaMemcpyDeviceToHost);



        gradient[i] = kernel[i];



    }



    //cudaFree(d_kernel);

}





__global__

void Inner_from_cache(float* A, float* B, float* C, int dim, int cacheIdx)

{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dim)

    {

        C[i] = C[i] + A[cacheIdx * dim + i] * B[i];

        __syncthreads();

    }

}



__global__ void copy_from_cache_to_ptr(float* __restrict__ output, float* __restrict__ input, int idx, int N)

{

    int i = blockIdx.x * blockDim.x + threadIdx.x; //i < N; i += blockDim.x * gridDim.x)

    output[i] = input[idx * DSIZE + i];

    __syncthreads();



}





void local_away_vertex(float* cache, int* active, float* alpha, float* x, float* res, int len, int dim, int* minIdx)

{

    float* inner = (float*)malloc(len * sizeof(float));

    float* d_inner;



    cudaMalloc(&d_inner, len * sizeof(float));

    int idxs[DSIZE] = { 0 };





    int l = 0;



    for (int i = 0; i < dim; i++)

    {

        if (active[i] > 0)

        {

            inner[l] = x[i];

            idxs[l] = i;

            l = l + 1;

        }

    }



    cudaMemcpy(d_inner, inner, len * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize(); checkLastCudaError();



    int maxId = 0;

    int maxV = inner[0];



    for (int i = 0; i < l; i++)

    {

        if (inner[i] > maxV)

        {

            maxV = inner[i];

            maxId = i;

        }

    }



    int mIdx = idxs[maxId];



    cudaMemcpy(minIdx, &mIdx, sizeof(int), cudaMemcpyHostToDevice);



    float ret[DSIZE] = { 0 };



    ret[mIdx] = 1.0;



    checkCuda(cudaMemcpy(res, ret, DSIZE * sizeof(float), cudaMemcpyHostToDevice));



    cudaDeviceSynchronize(); checkLastCudaError();

}





void local_LMO(float* cache, int* active, float* x, float* res, int len, int dim, int* minIdx)

{

    float inner[MAX_CACHE_SIZE] = { 0 };

    float* d_inner;

    cudaMalloc(&d_inner, MAX_CACHE_SIZE * sizeof(float));



    int l = 0;

    int idxs[DSIZE] = { 0 };



    for (int i = 0; i < dim; i++)

    {

        if (active[i] > 0)

        {

            inner[l] = x[i];

            idxs[l] = i;

            l = l + 1;

        }

    }



    int maxId = 0;

    int maxV = inner[0];



    for (int i = 0; i < l; i++)

    {

        if (inner[i] < maxV)

        {

            maxV = inner[i];

            maxId = i;

        }

    }



    float* h_res = (float*)malloc(dim * sizeof(float));



    int mIdx = idxs[maxId];



    checkCuda(cudaMemcpy(minIdx, &mIdx, sizeof(int), cudaMemcpyHostToDevice));



    float ret[DSIZE] = { 0 };



    ret[mIdx] = 1.0;



    checkCuda(cudaMemcpy(res, ret, DSIZE * sizeof(float), cudaMemcpyHostToDevice));



}



__global__  void update_x(float* alpha, int* active, float* x_t, float step, float* d, int len, int dim)

{

    // compute current thread id

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over vector: grid can be smaller than vector, it is therefore

    // required that each thread iterate over more than one vector element

    while (xIndex < DSIZE ) {

        if (active[xIndex] > 0)
        {

            x_t[xIndex] = alpha[xIndex];

        } 
        else
        {
            x_t[xIndex] = 0;
                
        }

        xIndex += gridDim.x * blockDim.x;

    }

}



__global__  void add_alpha(float* alpha, float* a, int* id, int n)

{

    float mul = 1 - *a;

    for (int j = 0; j < n; j++)

    {

        alpha[j] = alpha[j] * mul;

    }

    alpha[*id] = alpha[*id] + *a;

}





__global__  void apdate_active_idx(int* activeIdx, float* alpha, int n)

{
    for (int j = 0; j < n; j++)

    {
        if (alpha[j] > 0)
        {
            activeIdx[j] = 1;
        }
        else
        {
            alpha[j] = 0;
            activeIdx[j] = 0;
        }
    }
}


void bpcg_optimizer(int maxIter, float* cache, float* alpha, float* x_t, float** objects, float* probs, int numObjs, int numCoords, int vecDim, float *logs)

{



    // Define variables

    int L = 1;

    int i = 0;

    int it = 0;

    int k = 0;

    int j;

    int c_len = numObjs;

    size_t sizeVec = vecDim * sizeof(float);

    const int NM = numObjs * numCoords;

    float dimObjects[3000] = { 0 };


    //float** dimObjects;
    //malloc2D(dimObjects, numCoords, numObjs, float); 
    float h_alpha[DSIZE] = { 0 };

    for (i = 0; i < numCoords; i++)

    {

        for (j = 0; j < numObjs; j++)

        {

            dimObjects[numObjs * i + j] = objects[j][i];
            //dimObjects[i][j] = objects[j][i];

        }

    }






    int number_drop = 0;

    int vertex_added = 0;

    float* d_primal;

    float* deviceObjects;

    float* d_grad;

    float* dd_FW;

    float* dd_LFW;

    float* dd_AW;

    float* d_movingdirection;

    float h_probs[DSIZE] = { 0 };

    float step = 0;

    float zero = 0;



    size_t vecSize = vecDim * sizeof(float);



    float* primal = (float*)malloc(sizeof(float));



    *primal = 0;



    float grad[DSIZE] = { 0 };

    float d_FW[DSIZE] = { 0 };

    float d_LFW[DSIZE] = { 0 };

    float d_AW[DSIZE] = { 0 };

    float h_a[DSIZE] = { 0 };

    float movingdirection[DSIZE] = { 0 };

    int active[DSIZE] = { 0 };

    int* I_active;

    active[1] = 1;



    checkCuda(cudaMalloc(&d_primal, sizeof(float)));

    checkCuda(cudaMalloc(&deviceObjects, numObjs * numCoords * sizeof(float)));

    checkCuda(cudaMalloc(&d_grad, sizeVec));

    checkCuda(cudaMalloc(&dd_FW, sizeVec));

    checkCuda(cudaMalloc(&dd_LFW, sizeVec));

    checkCuda(cudaMalloc(&dd_AW, sizeVec));

    checkCuda(cudaMalloc(&d_movingdirection, sizeVec));

    checkCuda(cudaMalloc(&I_active, sizeVec));

    checkCuda(cudaMemcpy(I_active, active, DSIZE * sizeof(int), cudaMemcpyHostToDevice));

    cudaMemcpy(deviceObjects, dimObjects, numObjs * numCoords * sizeof(float), cudaMemcpyHostToDevice);


#if DEBUG

    float h_objects[DSIZE * 3] = { 0 };

    float h_x[DSIZE] = { 0 };

    cudaMemcpy(h_objects, deviceObjects, numObjs* numCoords * sizeof(float), cudaMemcpyDeviceToHost);

#endif

    int threadsPerBlock = THREAD_PER_BLOCK;

    int blocksPerGrid = (vecDim + threadsPerBlock - 1) / threadsPerBlock;



    // Define variable to store moving direction 

    float v[DSIZE] = { 0 };

    float zero_vec[DSIZE] = { 0 };

    float* dv;



    checkCuda(cudaMalloc(&dv, sizeVec));

    //checkCuda(cudaMemcpy(dv, v, sizeVec, cudaMemcpyHostToDevice));



    // Define variable to store away vertex 

    int* d_id_AW;

    checkCuda(cudaMalloc(&d_id_AW, sizeof(int)));

    int* id_AW = (int*)malloc(sizeof(int));

    *id_AW = 0;



    // Define variable to store FW vertex 

    int* d_id_FW;

    checkCuda(cudaMalloc(&d_id_FW, sizeof(int)));

    int* id_FW = (int*)malloc(sizeof(int));

    *id_FW = 0;



    // Define variable to store FW vertex 

    int* d_id_LFW;

    checkCuda(cudaMalloc(&d_id_LFW, sizeof(int)));

    int* id_LFW = (int*)malloc(sizeof(int));

    *id_LFW = 0;



    float* dual_gap = (float*)malloc(sizeof(float));

    *dual_gap = 0;

    float* away_gap = (float*)malloc(sizeof(float));

    *away_gap = 0;



    // Define variable to hold dual gap

    float* d_dual_gap;

    checkCuda(cudaMalloc(&d_dual_gap, sizeof(float)));

   // checkCuda(cudaMemcpy(d_dual_gap, dual_gap, sizeof(float), cudaMemcpyHostToDevice));



    // Define variable to hold away gap

    float* d_away_gap;

    checkCuda(cudaMalloc(&d_away_gap, sizeof(float)));

    checkCuda(cudaMemcpy(d_away_gap, away_gap, sizeof(float), cudaMemcpyHostToDevice));



    // Define variable to hold gap

    float* gap = (float*)malloc(sizeof(float));

    *gap = 0;

    float* d_gap;

    checkCuda(cudaMalloc(&d_gap, sizeof(float)));

    //checkCuda(cudaMemcpy(d_gap, gap, sizeof(float), cudaMemcpyHostToDevice));



    // Define variable to hold gap

    float* d = (float*)malloc(sizeof(float));

    *d = 0;

    float* d_d;

    checkCuda(cudaMalloc(&d_d, sizeof(float)));

    //checkCuda(cudaMemcpy(d_d, d, sizeof(float), cudaMemcpyHostToDevice));



    // Define variable to hold step size

    float* d_step;



    checkCuda(cudaMalloc((void**)&d_step, sizeof(float)));

    printf("define var done !");

#if DEBUG

    checkCuda(cudaMemcpy(h_alpha, alpha, DSIZE * sizeof(int), cudaMemcpyDeviceToHost));

#endif



    while (it < maxIter)

    {

        //checkCuda(cudaMemcpy(d_away_gap, &zero, sizeof(float), cudaMemcpyHostToDevice));

        //checkCuda(cudaMemcpy(d_dual_gap, &zero, sizeof(float), cudaMemcpyHostToDevice));

        /* code */

        *primal = 0;

        printf("start iteration \n");

        // Caculate v = x - mu for gradient

        substr_vectors << <blocksPerGrid, threadsPerBlock >> > (x_t, probs, dv, vecDim);

        cudaDeviceSynchronize(); checkLastCudaError();



        //Caculate gradient

        gradient_function(numCoords, numObjs, deviceObjects, dv, grad, vecDim);

        checkCuda(cudaMemcpy(d_grad, grad, sizeVec, cudaMemcpyHostToDevice));

        cudaDeviceSynchronize(); checkLastCudaError();



        // Local minimization oracle;

        local_LMO(cache, active, grad, dd_LFW, c_len, vecDim, d_id_LFW);

        cudaDeviceSynchronize(); checkLastCudaError();

        checkCuda(cudaMemcpy(id_LFW, d_id_LFW, sizeof(int), cudaMemcpyDeviceToHost));



        // Away step   

        local_away_vertex(cache, active, h_alpha, grad, dd_AW, c_len, vecDim, d_id_AW);

        cudaDeviceSynchronize(); checkLastCudaError();

        checkCuda(cudaMemcpy(id_AW, d_id_AW, sizeof(int), cudaMemcpyDeviceToHost));



        // Linear minimization oracle over simplex

        compute_extreme_point(d_grad, active, dd_FW, d_id_FW);

        checkCuda(cudaMemcpy(id_FW, d_id_FW, sizeof(int), cudaMemcpyDeviceToHost));

        checkCuda(cudaMemcpy(d_FW, dd_FW, DSIZE * sizeof(int), cudaMemcpyDeviceToHost));

        cudaDeviceSynchronize(); checkLastCudaError();



        // printf("Extreme id %d \n", *id_FW);

         // caculate FW direction

        checkCuda(cudaMemcpy(d_movingdirection, zero_vec, DSIZE * sizeof(int), cudaMemcpyHostToDevice));


        substr_vectors << <blocksPerGrid, threadsPerBlock >> > (x_t, dd_FW, d_movingdirection, vecDim);

        cudaDeviceSynchronize(); checkLastCudaError();

#if DEBUG

        cudaMemcpy(h_x, x_t, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

#endif


        checkCuda(cudaMemcpy(movingdirection, d_movingdirection, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));



        checkCuda(cudaMemcpy(grad, d_grad, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));



       // checkCuda(cudaMemcpy(d_dual_gap, &zero, sizeof(float), cudaMemcpyHostToDevice));



        // caculate dual gap

        VecInner(d_movingdirection, d_grad, dual_gap, vecDim);

        cudaDeviceSynchronize(); checkLastCudaError();



        //*dual_gap = 0;

       // checkCuda(cudaMemcpy(dual_gap, d_dual_gap, sizeof(float), cudaMemcpyDeviceToHost));

        checkCuda(cudaMemcpy(d_movingdirection, zero_vec, DSIZE * sizeof(int), cudaMemcpyHostToDevice));

        substr_vectors << <blocksPerGrid, threadsPerBlock >> > (dd_AW, dd_LFW, d_movingdirection, vecDim);

        cudaDeviceSynchronize(); checkLastCudaError();



      //  checkCuda(cudaMemcpy(d_away_gap, &zero, sizeof(float), cudaMemcpyHostToDevice));

        //checkCuda(cudaMemcpy(away_gap, d_away_gap, sizeof(float), cudaMemcpyDeviceToHost));



        VecInner(d_movingdirection, d_grad, away_gap, vecDim);

        cudaDeviceSynchronize(); checkLastCudaError();


        //*away_gap = 0;

       // checkCuda(cudaMemcpy(away_gap, d_away_gap, sizeof(float), cudaMemcpyDeviceToHost));

        printf("Compare gap %f >= %f \n", *away_gap, *dual_gap);



        if (*away_gap > *dual_gap)

        {

            printf("Case 1");

            printf("away gap %f \n", *away_gap);


            checkCuda(cudaMemcpy(d, &alpha[*id_AW], sizeof(float), cudaMemcpyDeviceToHost));



            substr_vectors << <blocksPerGrid, threadsPerBlock >> > (x_t, probs, dv, vecDim);



            cudaDeviceSynchronize(); checkLastCudaError();



            step = step_size(numCoords, numObjs, blocksPerGrid, threadsPerBlock, probs, d_grad, x_t, d_movingdirection, *away_gap, deviceObjects, d, dv, vecDim);



            checkCuda(cudaMemcpy(d_step, &step, sizeof(float), cudaMemcpyHostToDevice));



            if (step < *d)

            {



                printf("DESCENT step \n", *id_FW);



                checkCuda(cudaMemcpy(d, &alpha[*id_LFW], sizeof(float), cudaMemcpyDeviceToHost));



                *d = *d + step;



                checkCuda(cudaMemcpy(&alpha[*id_LFW], d, sizeof(float), cudaMemcpyHostToDevice));



                checkCuda(cudaMemcpy(d, &alpha[*id_AW], sizeof(float), cudaMemcpyDeviceToHost));



                *d = *d - step;



                checkCuda(cudaMemcpy(&alpha[*id_AW], d, sizeof(float), cudaMemcpyHostToDevice));



                checkCuda(cudaMemcpy(h_alpha, alpha, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));



            }

            else

            {

                printf("DROP STEP \n");



                float* tmp = (float*)malloc(sizeof(float));



                checkCuda(cudaMemcpy(tmp, &alpha[*id_LFW], sizeof(float), cudaMemcpyDeviceToHost));



                *d = step + *tmp;



                checkCuda(cudaMemcpy(&alpha[*id_LFW], d, sizeof(float), cudaMemcpyHostToDevice));



                *tmp = 0;



                checkCuda(cudaMemcpy(&alpha[*id_AW], tmp, sizeof(float), cudaMemcpyHostToDevice));



                number_drop = number_drop + 1;



                checkCuda(cudaMemcpy(h_alpha, alpha, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));

            }



        }

        else

        {



            printf("FW step \n");

#if DEBUG

            cudaMemcpy(h_x, x_t, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

#endif
            checkCuda(cudaMemcpy(d_movingdirection, zero_vec, DSIZE * sizeof(int), cudaMemcpyHostToDevice));


            checkCuda(cudaMemcpy(movingdirection, d_movingdirection, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));


            substr_vectors << <blocksPerGrid, threadsPerBlock >> > (x_t, dd_FW, d_movingdirection, vecDim);

            cudaDeviceSynchronize(); checkLastCudaError();

#if DEBUG
            checkCuda(cudaMemcpy(movingdirection, d_movingdirection, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));

#endif

  /*          cudaDeviceSynchronize(); checkLastCudaError();



            printf("moving direction \n");



            checkCuda(cudaMemcpy(d_gap, &zero, sizeof(float), cudaMemcpyHostToDevice));


            VecInner << <blocksPerGrid, threadsPerBlock >> > (d_movingdirection, d_grad, d_gap, vecDim);



            cudaDeviceSynchronize(); checkLastCudaError();



            checkCuda(cudaMemcpy(gap, d_gap, sizeof(float), cudaMemcpyDeviceToHost));*/



            float gama_max = 1;



            printf("dual_gap = %f \n", *dual_gap);



            substr_vectors << <blocksPerGrid, threadsPerBlock >> > (x_t, probs, dv, vecDim);



            cudaDeviceSynchronize(); checkLastCudaError();



            step = step_size(numCoords, numObjs, blocksPerGrid, threadsPerBlock, probs, d_grad, x_t, d_movingdirection, *dual_gap, deviceObjects, &gama_max, dv, vecDim);



            printf("Step size %f \n", step);



            checkCuda(cudaMemcpy(d_d, &step, sizeof(float), cudaMemcpyHostToDevice));



            printf("add cahe \n");



            add_alpha << <1, 1 >> > (alpha, d_d, d_id_FW, vecDim);

#ifdef DEBUG

            cudaDeviceSynchronize(); checkLastCudaError();

            checkCuda(cudaMemcpy(h_alpha, alpha, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));

#endif // DEBUG



            cudaDeviceSynchronize(); checkLastCudaError();



        }



        printf("update x %d len \n", c_len);



        apdate_active_idx << <1, 1 >> > (I_active, alpha, vecDim);



        checkCuda(cudaMemcpy(active, I_active, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));



        update_x << <blocksPerGrid, threadsPerBlock >> > (alpha, I_active, x_t, step, d_movingdirection, c_len, vecDim);

        cudaDeviceSynchronize(); checkLastCudaError();


#if DEBUG

        cudaMemcpy(h_x, x_t, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

#endif



        cudaDeviceSynchronize(); checkLastCudaError();



        substr_vectors << <blocksPerGrid, threadsPerBlock >> > (x_t, probs, dv, vecDim);

#if DEBUG

        cudaMemcpy(v, dv, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

#endif



        primal_function(numCoords, numObjs, deviceObjects, dv, d_grad, primal, vecDim);



        cudaDeviceSynchronize(); checkLastCudaError();



        it = it + 1;

        logs[it] = *primal;



        printf("primal value at %f iteration %d \n", *primal, it);



        checkCuda(cudaMemcpy(active, I_active, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG

        checkCuda(cudaMemcpy(h_alpha, alpha, DSIZE * sizeof(float), cudaMemcpyDeviceToHost));

#endif // DEBUG


    }



    // Free host memory

   // delete grad;

//    free(primal);

//    free(gap);

    //free(d_FW);

    //free(d_LFW);

    //free(d_AW);

    //free(movingdirection);

    //free(dimObjects);

    //free(v);



    // Free device memory

    //checkCuda(cudaFree(d_gap));

    //checkCuda(cudaFree(d_primal));

    //checkCuda(cudaFree(d_grad));

    //checkCuda(cudaFree(d_step));

    //checkCuda(cudaFree(dd_FW));

    //checkCuda(cudaFree(dd_LFW));

    //checkCuda(cudaFree(dd_AW));

    //checkCuda(cudaFree(d_movingdirection));

    //checkCuda(cudaFree(deviceObjects));

    //checkCuda(cudaFree(dv));

}



