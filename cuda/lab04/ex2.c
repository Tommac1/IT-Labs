#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ex2.h"

#define UNREF(X) (void)(X)
#define ALIGN_UP(off, align) ((off) = ((off) + (align) - 1) & ~((align) - 1))

const double eps = 0.00000000001;
double hVect[N + 1];
double dPi = 0.0;
int blocks = 0;

void init()
{
    blocks = N / BLOCK_SIZE;
    if (N % BLOCK_SIZE)
        blocks++;

    for (size_t i = 0; i < N; ++i)
        hVect[i] = i + 1;
}

void check()
{
    double hPi = 1.0;
    double last_pi = 10.0;
    int n = 1;
    
    init(); // just hVect for check

    // CPU
    while (fabs(last_pi - hPi) >= eps) {
        double x = (double)(4.0 * n * n)/(4.0 * n * n - 1.0);
        last_pi = hPi;
        hPi *= x;
        ++n;
    }

    printf("host:\t%.15f\ndevice:\t%.15f\n", hPi * 2.0, dPi * 2.0);
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
    CUDA_CALL( cuModuleLoad(&hModule, "ex2_kernel.cubin") );
    CUDA_CALL( cuModuleGetFunction(&hFunction, hModule, "mul") );

    CUdeviceptr dArr;

    CUDA_CALL( cuMemAlloc(&dArr, sizeof(hVect)) );

    CUDA_CALL( cuFuncSetBlockShape(hFunction, BLOCK_SIZE, 1, 1) );

    int offset = 0;
    void *ptr = (void *)(size_t) dArr;
    ALIGN_UP(offset, __alignof(ptr));
    CUDA_CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
    offset += sizeof(ptr);

    CUDA_CALL( cuParamSetSize(hFunction, offset) );

    CUDA_CALL( cuLaunchGrid(hFunction, blocks, 1) );
    CUDA_CALL( cuCtxSynchronize() );

    CUDA_CALL( cuMemcpyDtoH((void *) hVect, dArr, sizeof(hVect)) );
    CUDA_CALL( cuMemFree(dArr) );

    // calculate the product of multiplication
    int n = 1;
    hVect[0] = 1.0;
    while (fabs(hVect[0] - dPi) >= eps) {
        dPi = hVect[0];
        hVect[0] *= hVect[n];
        ++n;
    }

    dPi = hVect[0];
}

int main(int argc, char *argv[])
{
    UNREF(argc);
    UNREF(argv);

    init();
    run();
    check();

    printf("done\n");

    return 0;
}
