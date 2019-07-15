#include "ex2.h"

extern "C" __global__ void mul(double *a)
{
    // plus one because threads are zero indexed
    int x = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (x < N)
        a[x] = (double)(4.0 * x * x)/(4.0 * x * x - 1.0);
}
