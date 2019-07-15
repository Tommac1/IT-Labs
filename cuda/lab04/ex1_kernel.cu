#include "ex1.h"

extern "C" __global__ void IncVect(float *x0, size_t *n, float *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < n[0]) {
        float tmp = a[x] * pow(x0[0], (float) x);

        __syncthreads();
        atomicAdd(&a[0], tmp);
    }
}
