#include <cuda_runtime.h>
#include <random>
#include "BPCG.h"



//
//void mean(float** objects, float* mean, int numCoords, int numObjs)
//{
//
//}
//
//void std_data(float** objects, float* std, int numCoords, int numObjs)
//{
//
//}

void normalize(float** objects, float** normalizedObjs, int numCoords, int numObjs)
{
    float mean[3] = { 0 };// = (float* ) malloc(numCoords * sizeof(float));
    float std[3] = { 0 };// = (float*)malloc(numCoords * sizeof(float));

    int i, j;

    for (i = 0; i < numObjs; i++)
    {
        for (j = 0; j < numCoords; j++)
        {
            mean[j] = mean[j] + objects[i][j];
        }
    }

    for (j = 0; j < numCoords; j++)
    {
        mean[j] = mean[j] / numObjs;
    }

    for (i = 0; i < numObjs; i++)
    {
        for (j = 0; j < numCoords; j++)
        {
            std[j] = std[j] + (objects[i][j] - mean[j]) * (objects[i][j] - mean[j]);
        }
    }

    for (j = 0; j < numCoords; j++)
    {
        std[j] =  sqrt(std[j] / numObjs);
    }


    for (i = 0; i < numObjs; i++)
    {
        for (j = 0; j < numCoords; j++)
        {
            normalizedObjs[i][j] = (objects[i][j] - mean[j]) / std[j];
        }
    }

}