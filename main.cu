/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_main.c   (an sequential version)                      */
/*   Description:  This program shows an example on how to call a subroutine */
/*                 that implements a simple k-means clustering algorithm     */
/*                 based on Euclid distance.                                 */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 binary file: first 4-byte integer is the number of data   */
/*                 objects and 2nd integer is the no. of features (or        */
/*                 coordinates) of each object                               */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng   Liao
// Copyright (c) 2011 Serban Giuroiu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>


int      _debug;
#include "BPCG.h"
#include "util.cuh"


// Thread block size
#define BLOCK_SIZE 16


/*---< usage() >------------------------------------------------------------*/
static void usage(char* argv0, float threshold) {
    char* help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n"
        "       -d             : enable debug mode\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

__global__
void use_ptr2ptr(float** cache)
{
    int k;
    for (k = 0; k < 1; k++)
    {
        printf("ABC");
        printf("%d \n", cache[0][0]);
    }
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char** argv) {
    int     opt;
    extern char* optarg;
    extern int     optind;
    int     isBinaryFile, is_output_timing;

    int     numClusters, numCoords, numObjs;
    int* membership;    /* [numObjs] */
    char* filename;
    float** objects;       /* [numObjs][numCoords] data objects */
    float** objs;      /* [numClusters][numCoords] cluster center */
    float   threshold;
    double  timing, io_timing, clustering_timing;
    int     loop_iterations;

    /* some default values */
    _debug = 0;
    threshold = 0.001;
    numClusters = 0;
    isBinaryFile = 0;
    is_output_timing = 0;
    filename = "3D100.txt";

    /* read data points from file ------------------------------------------*/
    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);

    //objects = (float**)malloc(sizeof(float) * numCoords * numObjs);



    if (objects == NULL) exit(1);

    normalize(objects, objects, numCoords, numObjs);

    /* start the timer for the core computation -----------------------------*/

    float primals;

    float *logs =  (float*)malloc(numObjs * sizeof(float));


    float* x_star = (float*)malloc(numObjs * sizeof(float));

    float* probs = (float*)malloc(numObjs * sizeof(float));

    float* x0;

    int maxIter = 30;

    int j;

    int vecDim = numObjs;

    size_t size = vecDim * sizeof(float);
    float alpha[DSIZE] = {0 };

    // memset(x_star, 0, size);
    // memset(probs, float(1)/numObjs, sizeof(x_star[0]));
    for (j = 0; j < vecDim; j++)
    {
        x_star[j] = 0.0;
        probs[j] = 1.0 / float(vecDim);
    }

    x_star[1] = 1;
    alpha[1] = 1;

    size_t sizePtr = sizeof(float*);
    size_t sizeInt = vecDim * sizeof(int);

    float* cachePtr = new float[vecDim];
    int* id = new int[vecDim];
    

    float* d_x_start;
    float* d_probs;
    float* d_alpha;

    checkCuda(cudaMalloc(&d_x_start, size));

    checkCuda(cudaMalloc(&d_probs, size));

    checkCuda(cudaMalloc(&d_alpha, size));

    checkCuda(cudaMemcpy(d_x_start, x_star, size, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_probs, probs, size, cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_alpha, alpha, size, cudaMemcpyHostToDevice));

    //checkCuda(cudaMemset(d_alpha, 0, numObjs * sizeof(float)));


    float* h_0;
    float* d_0;
    float* S;


    h_0 = (float*)malloc(size);

    for (int i = 0; i < vecDim; ++i) {
        h_0[i] = 0;
    }

    h_0[0] = 1;

    checkCuda(cudaMalloc(&S, MAX_CACHE_SIZE * size));

    checkCuda(cudaMemcpy(S, h_0, size, cudaMemcpyHostToDevice));

    printf("Call BPCG \n");

    bpcg_optimizer(maxIter, S, d_alpha, d_x_start, objects, d_probs, numObjs, numCoords, vecDim, logs);

    /* Free the resources.*/
   // if (S) checkCuda(cudaFree(S));
   // if (d_0) checkCuda(cudaFree(d_0));
   // if (h_0) free(h_0);

   // free(objects[0]);
   // free(objects);

   // cudaFree(&cache);
   // /* output: the coordinates of the cluster centres ----------------------*/
    file_log(filename, logs);

   // free(membership);
   // free(clusters[0]);
    //free(clusters);

    //puts("Press <enter> to quit:");
    //getchar();
    //system("pause");
    return(0);
}

