#include <stdio.h>
#include <time.h>
#include <assert.h>


#define DIM 2048
#define N ((DIM) * (DIM))
#define BLOCK_SIZE 1024


typedef void (* kernel)(double *);


double h_arr[N];
double *d_arr;
int blocks;


// **********
// HELPERS
// **********
void prologue()
{
    blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void **) &d_arr, sizeof(h_arr));
    cudaMemcpy(d_arr, h_arr, sizeof(h_arr), cudaMemcpyHostToDevice);
}

void epilogue()
{
    cudaMemcpy(h_arr, d_arr, sizeof(h_arr), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

void init()
{
    for (int i = 0; i < N; ++i)
        h_arr[i] = i + 1;
}

void check()
{
    for (int i = 1; i < N; ++i) {
        if (abs(i * i - h_arr[i - 1]) >= 0.0001) {
            printf("%.4f != %.4f\n", (double) i * i, h_arr[i - 1]);
            break;
        }
    }
}


// **********
// KERNELS
// **********
__global__ void unroll1(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 1
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll2(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 2
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll4(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 4
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll8(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 8
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll16(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 16
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll32(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 32
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll64(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 64
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}

__global__ void unroll128(double *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
#pragma unroll 128
        for (int i = 0; i < 128; ++i) {
            a[x] += (a[x] * (i % 120 / 60));
        }
    }
}


// **********
// MAIN
// **********
int main(int argc, char *argv[])
{
    clock_t begin;
    clock_t end;
    const int UNROLLS[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
    const kernel KERNELS[8] = { unroll1, unroll2, unroll4, unroll8,
            unroll16, unroll32, unroll64, unroll128 };

    init();

    prologue();

    for (int i = 0; i < 8; ++i) {
        begin = clock();
        KERNELS[i]<<<blocks, BLOCK_SIZE>>>(d_arr);
        cudaThreadSynchronize();
        end = clock();

        printf("unrolls: %d, time taken: %2.3fms\n", UNROLLS[i],
                (double)(end - begin) / CLOCKS_PER_SEC * 1000);
    }

    epilogue();
    
//    check();
}
