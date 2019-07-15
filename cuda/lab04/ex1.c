#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ex1.h"

#define UNREF(X) (void)(X)
#define ALIGN_UP(off, align) ((off) = ((off) + (align) - 1) & ~((align) - 1))

float hVect[N + 1];
int blocks;
float hX[1] = { 1.5 };
size_t hN[1] = { N + 1 };

void init()
{
    blocks = N / BLOCK_SIZE;
    if (N % BLOCK_SIZE)
        blocks++;

    for (size_t i = 0; i < hN[0]; ++i)
        hVect[i] = i;
}

void check()
{
    float fx = 0.0;
    float dFx = hVect[0];

    for (size_t i = 0; i < hN[0]; ++i)
        hVect[i] = i;

    for (size_t i = 0; i < hN[0]; ++i)
        fx += (hVect[i] * pow(hX[0], (float) i));

    printf("host: %.3f, device: %.3f\n", fx, dFx);
}

void run()
{
    CUdevice hDevice;
    CUcontext hContext;
    CUmodule hModule;
    CUfunction hFunction;

    CUDA_CALL( cuInit(0) );
    CUDA_CALL( cuDeviceGet(&hDevice, 0) );
    CUDA_CALL( cuCtxCreate(&hContext, 0, hDevice) );
    CUDA_CALL( cuModuleLoad(&hModule, "ex1_kernel.cubin") );
    CUDA_CALL( cuModuleGetFunction(&hFunction, hModule, "IncVect") );

    CUdeviceptr dVectA;
    CUdeviceptr dX0;
    CUdeviceptr dn;

    CUDA_CALL( cuMemAlloc(&dX0, sizeof(hX)) );
    CUDA_CALL( cuMemAlloc(&dn, sizeof(hN)) );
    CUDA_CALL( cuMemAlloc(&dVectA, sizeof(hVect)) );
    CUDA_CALL( cuMemcpyHtoD(dX0, hX, sizeof(hX)) );
    CUDA_CALL( cuMemcpyHtoD(dn, hN, sizeof(hN)) );
    CUDA_CALL( cuMemcpyHtoD(dVectA, hVect, sizeof(hVect)) );

    CUDA_CALL( cuFuncSetBlockShape(hFunction, BLOCK_SIZE, 1, 1) );

    int offset = 0;
    void *ptr = (void *)(size_t) dX0;
    ALIGN_UP(offset, __alignof(ptr));
    CUDA_CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
    offset += sizeof(ptr);

    ptr = (void *)(size_t) dn;
    ALIGN_UP(offset, __alignof(ptr));
    CUDA_CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
    offset += sizeof(ptr);

    ptr = (void *)(size_t) dVectA;
    ALIGN_UP(offset, __alignof(ptr));
    CUDA_CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
    offset += sizeof(ptr);

    CUDA_CALL( cuParamSetSize(hFunction, offset) );

    CUDA_CALL( cuLaunchGrid(hFunction, blocks, 1) );
    CUDA_CALL( cuCtxSynchronize() );

    CUDA_CALL( cuMemcpyDtoH((void *) hVect, dVectA, sizeof(hVect)) );
    CUDA_CALL( cuMemFree(dVectA) );
}

int main(int argc, char *argv[])
{
    UNREF(argc);
    UNREF(argv);

    init();
    run();
    check();

    printf("done\n");
}
