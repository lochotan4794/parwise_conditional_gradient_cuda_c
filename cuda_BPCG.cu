
#include <stdio.h>
#include <stdlib.h>
#include "BPCG.h"

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void compute_extreme_point(int numCoords, float *grad, float *extreme_point)// [numCoords][numObjs]
                                                     
{
    int i;
    int imin = 0;
    float min_val = grad[0];
    for (i = 1; i < numCoords; i++)
    {
        if (grad[i] < min_val) 
        {
            min_val = grad[i];
            imin = i;
        }
    }
    extreme_point[imin] = 1;
}


__host__ __device__ void away_step(int numCoords, float *grad, float *extreme_point)// [numCoords][numObjs]                 
{
    int i;
    int imin = 0;
    float min_val = grad[0];
    for (i = 1; i < numCoords; i++)
    {
        if (grad[i] < min_val) 
        {
            min_val = grad[i];
            imin = i;
        }
    }
    extreme_point[imin] = 1;
}

__host__ __device__ void step_size(int numCoords, float *grad, float *extreme_point)// [numCoords][numObjs]
                                                     
{
    int i;
    int imin = 0;
    float min_val = grad[0];
    for (i = 1; i < numCoords; i++)
    {
        if (grad[i] < min_val) 
        {
            min_val = grad[i];
            imin = i;
        }
    }
    extreme_point[imin] = 1;
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
    float d = euclid_dist_2(numCoords, numObjs, objectId1, objectId2, objects);
    ans = (1 + sqrt(3) * d) * exp(-sqrt(3) * d);
    return (ans);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static float euclid_dist_2(int numCoords,
                                                      int numObjs,
                                                      int objectId1,
                                                      int objectId2,
                                                      float *objects) // [numCoords][numObjs]
{
    int i;
    float ans = 0.0;

    for (i = 0; i < numCoords; i++)
    {
        ans += (objects[numObjs * i + objectId1] - objects[numObjs * i + objectId2]) *
               (objects[numObjs * i + objectId1] - objects[numObjs * i + objectId2]);
    }

    return sqrt(ans);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static float primal_function(
    float *x,
    float *mean, // [numCoords][numObjs]
    int numCoords)

{
    int i;
    float ans = 0.0;
    for (i = 0; i < numCoords; i++)
    {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return (ans);
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
    for (i = 0; i < numCoords; i++)
    {
        v[i] = x[i] - mu[i];
    }

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
__host__ __device__ void local_linear_minimization_oracle(int numCoords, float *grad, float *extreme_point)// [numCoords][numObjs]
                                                     
{
    int i;
    int imin = 0;
    float min_val = grad[0];
    for (i = 1; i < numCoords; i++)
    {
        if (grad[i] < min_val) 
        {
            min_val = grad[i];
            imin = i;
        }
    }
    extreme_point[imin] = 1;
}

__host__ __device__ inline static void bpcg_optimizer(float *x0, int maxIter, cuda_cache cuda)
{
    int L = 1;
    int i;
    int number_drop;

    while (i < maxIter)
    {
        /* code */
        float *grad;
        gradient_function(numCoords, numObjs, numClusters, &objects, &mu, &x, &gradient );
        int id_FW  = local_linear_minimization_oracle(numCoords, *grad, *extreme_point);
        int id_AW  = away_step( numCoords, *grad, *extreme_point);
        int id_AW  = compute_extreme_point(numCoords, *grad, *extreme_point);

        float *dual_gap;
        gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k);

        float *away_gap;
        gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k);

        if (away_gap > dual_gap)
        {
             d_t = a_W - s_FW

            d = alpha_t[id_A]

            gap = np.dot(d_t.T,  grad)[0,0]

            step, _ = take_step_size(f=f, d_t=d_t, x_t=x_t, g_t=gap, L=L, gama_max=d)
           
            if step < d:

                alpha_t[id_FW] = alpha_t[id_FW] + step

                alpha_t[id_A] = alpha_t[id_A] - step

            else:

                alpha_t[id_FW] = alpha_t[id_FW] + d

                alpha_t[id_A] = 0.0

                I_active.remove(id_A);

                number_drop = number_drop + 1

        } 
        else 
        {
            d_t = x_t - w_FW

            gap = np.dot(d_t.T,  grad)[0,0]

            step, _ = take_step_size(f=f, d_t=d_t, x_t=x_t, g_t=gap, L=L, gama_max=1)

            alpha_t = (1-step) * alpha_t 

            alpha_t[id_W] = alpha_t[id_W] + step

            vertex_added = True

            if step > 1-eps:
                I_active = [id_W]
                alpha_t = alpha_t * 0
                alpha_t[id_W] = 1

        }

        I_active =  find(np.argwhere(alpha_t > 0))
        
        if vertex_added:
            I_active_lst.append(I_active)
            alpha_lst.append(alpha_t)

        x_t = update_x(alpha_t, I_active)

        primal= f(x_t)

        fvalues[it-1] = primal

        if len(I_active) > crr:
            crr = crr + 1
            xs.append(x_t)

        i++;
    }
    
}
