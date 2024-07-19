

typedef struct 
{
    /* data */
    float **coord;
    int id;
} vertex;

typedef struct 
{
    /* data */
    vertex *S;
    int *alpha;
    int len;
} cuda_cache;

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void remove_from_cache(int id)
{

}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void update_alpha(cuda_cache *cache, float *coord, float alpha, int id)
{

}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ int in_cache(cuda_cache cache, int id)
{
    int maxId = cache.len;
    int i;
    for (i=0; i<maxId;i++)
    {
        if (cache.S[i].id == id)
        {
            return id;
        }
    }
    return -1;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void add_to_cache(cuda_cache *cache, float *coord, float alpha, int id)
{
    int maxId = cache.len + 1;
    vertex v = (vertex*)malloc(vertex);
    v.coord= coord;
    v.id=id;
    cache.S[maxId] = v;
}