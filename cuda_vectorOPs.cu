

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


// Kernel definition
__global__ void VecMul(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] * B[i];
}



// Kernel definition
__global__ void VecSub(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] - B[i];
}


// Kernel definition
__global__ void VecInner(float* A, float* B, float C)
{
    int i = threadIdx.x;
    C = C + A[i] * B[i];
}

__global__ void VecNorm(float* A, float* B)
{
    
    VecInner<<1, N>>(A, B);
    B = sqrt(B);
}


// Kernel definition
__global__ void VectorSubWithScale(float* A, float* B, float* C, float scale)
{
    int i = threadIdx.x;
    C[i] = A[i] - B[i] * scale;
}

// int main()
// {
// // Kernel invocation with N threads
// VecAdd<<<1, N>>>(A, B, C);
// }