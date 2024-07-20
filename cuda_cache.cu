

typedef struct 
{
    /* data */
    float **S;
    int *alpha;
    int *cacheId;
    int len;

} cuda_cache;



/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void remove_from_cache(int id, cuda_cache *cache)
{
    cache->alpha[id] = 0;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void set_alpha(int id, cuda_cache *cache, float alpha)
{
    cache->alpha[id] = alpha;
}


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ void add_cache(int id, cuda_cache *cache, float alpha, float *v)
{
    float scale = 1 - alpha;
    cache->alpha[id] = alpha;
    int i;

    for(i=0;i< cache.len ;i++)
    {
        cache.alpha[i] = cache.alpha[i]  * scale;
    }

    cache->len = cache->len + 1;
    int len = cache.len;

    float* ptr = (float*)malloc(sizeof(float*)*len);
    ptr[len] = v;
    memcpy(ptr, cache.S, len - 1);
    cache.S = ptr;

    int* alphaPtr = (int*)malloc(sizeof(int*)*len);
    alphaPtr[len] = alpha;
    memcpy(alphaPtr, cache.alpha, len - 1);
    cache.alpha = alphaPtr;

    int* idCache = (int*)malloc(sizeof(int*)*len);
    idCache[len] = id;
    memcpy(idCache, cache.cacheId, len - 1);
    cache.cacheId = idCache;

}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ int in_cache(cuda_cache cache, int id)
{
    int maxId = cache.len;
    int i;
    for (i=0; i<maxId;i++)
    {
        if (cache.cacheId[i] == id)
        {
            return id;
        }
    }
    return -1;
}

