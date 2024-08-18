#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define tbp 121

#define nblocks 1 

__global__ void kernel_min(float* a, float* d, int* index)
{
    __shared__ int sdata[tbp]; //"static" shared memory

    unsigned int tid = threadIdx.x;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = a[i];

    printf("tid = %d \n", tid);

    printf("id = %d \n", i);

    __syncthreads();

    for (unsigned int s = tbp / 2; s >= 1; s = s / 2)
    {
        if (tid < s)
        {
            if (sdata[tid] > sdata[tid + s])
            {
                sdata[tid] = sdata[tid + s];

                printf("ndex = %d \n", tid + s);

            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d[blockIdx.x] = sdata[0];

        *index = i;
    }
}

// https://stackoverflow.com/questions/27925979/thrustmax-element-slow-in-comparison-cublasisamax-more-efficient-implementat/27928463#27928463%5B/url%5D



__global__ void argmax(float* a, int* idx) {

    extern __shared__ int s[1];

    s[0] = a[0];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (a[i] < *idx)
    {
        *idx = i;

        printf("min = %f \n", a[i]);

        printf("id = %f \n", i);
    }

}

// __global__ void argmax(int *in_indexes, float *in_values, int *out_indexes, float *out_values, int rows, int cols) {

// 	extern __shared__ int s[];
// 	int *maxindexes = s;
// 	float *maxvalues = (float*)&maxindexes[blockDim.x*blockDim.y];

// 	unsigned int tidx = threadIdx.x;
// 	unsigned int tidy = threadIdx.y;
// 	unsigned int i = tidx + tidy*blockDim.x;

// 	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
// 	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
// 	maxindexes[i] = (x < rows && y < cols) ? in_indexes[x + rows*y] : 0;
// 	maxvalues[i] = (x < rows && y < cols) ? in_values[x + rows*y] : 0;

// 	__syncthreads();

// 	// do reduction in shared mem
// 	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
// 	{
// 		if (tidx < s  && x + s < rows && y < cols)
// 		{
// 			if (maxvalues[i + s] > maxvalues[i]) {
// 				maxvalues[i] = maxvalues[i + s];
// 				maxindexes[i] = maxindexes[i + s];
// 			}
// 		}

// 		__syncthreads();
// 	}

// 	if (tidx == 0 && y < cols) {
// 		out_indexes[gridDim.x*(tidy+ (blockIdx.y*blockDim.y)) + blockIdx.x] = maxindexes[blockDim.x * tidy];
// 		out_values[gridDim.x*(tidy + (blockIdx.y*blockDim.y)) + blockIdx.x] = maxvalues[blockDim.x * tidy];
// 	}
// }

int main()
{
    int i;
    const int N = tbp * nblocks;

    // const int N = nextPowerOfTwo(M);

    srand(time(NULL));

    float* a;
    int* in;
    a = (float*)malloc(N * sizeof(float));
    in = (int*)malloc(N * sizeof(int));
    float* d;
    d = (float*)malloc(nblocks * sizeof(float));

    float* dev_a, * dev_d;
    int* indexes;
    int* indexes_out;

    cudaMalloc(&dev_a, N * sizeof(float));
    cudaMalloc(&dev_d, nblocks * sizeof(float));
    cudaMalloc(&indexes, N * sizeof(int));
    cudaMalloc(&indexes_out, N * sizeof(int));

    int mmm = 100;
    int minid = 0;

    for (i = 0; i < N; i++)
    {
        a[i] = rand() % 100 + 5;
        in[i] = i;
        //printf("%d ",a[i]);
        if (mmm > a[i])
        {
            mmm = a[i];
            minid = i;
        }
    }

    in[0] = 100;
    printf("");
    printf("");
    printf("");

    printf("");

    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(indexes, in, N * sizeof(int), cudaMemcpyHostToDevice);


    kernel_min << < nblocks, tbp >> > (dev_a, dev_d, indexes);
    //  argmax<<< nblocks,tbp >>>(dev_a, indexes);

    cudaMemcpy(d, dev_d, nblocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(in, indexes, sizeof(float), cudaMemcpyDeviceToHost);


    printf("cpu min %d, gpu_min_id = %d  min_cpu_id = %d", mmm, *in, minid);

    cudaFree(dev_a);
    cudaFree(dev_d);

    printf("");

    return 0;
}