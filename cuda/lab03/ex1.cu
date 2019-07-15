#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>


#define DIM 4096
#define N ((DIM) * (DIM))
#define BLOCK_SIZE 1024


float h_arr[N];
float h_arr2[N];
float *d_arr;
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
    for (int i = 0; i < N; ++i) {
        h_arr[i] = h_arr2[i] = i;
    }
}

void check()
{
    for (int i = 0; i < N; ++i) {
        //printf("%d %.5f %.5f\n", i, h_arr[i], h_arr2[i]);

        if (abs(h_arr2[i] - h_arr[i]) >= 0.0001) {
            printf("%d %.10f\n", i, abs(h_arr[i] - h_arr2[i]));
            //printf("%.4f %.4f, %d\n", h_arr2[i], h_arr[i], i);
        }
    }
}


// **********
// KERNELS
// **********
__global__ void kernel(float *a)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < N) {
        //      2x^3 + x^2 + 4x + 3
        // y = ---------------------
        //        x^2 - 4x - 3/2
        //a[x] = (2.0 * (a[x] * a[x] * a[x]) + (a[x] * a[x]) + (4.0 * a[x]) + 3.0) /
        //        ((a[x] * a[x]) - (4.0 * a[x]) - (3.0 / 2.0));
        a[x] = cosf(a[x]) * (pow(a[x], x % 10) / log2((double) x));
    }
}

void cpu()
{
    for (int i = 0; i < N; ++i) {
        //      2x^3 + x^2 + 4x + 3
        // y = ---------------------
        //        x^2 - 4x - 3/2
        //h_arr2[i] = (2.0 * (h_arr2[i] * h_arr2[i] * h_arr2[i]) + (h_arr2[i] * h_arr2[i]) + (4.0 * h_arr2[i]) + 3.0) /
        //        ((h_arr2[i] * h_arr2[i]) - (4.0 * h_arr2[i]) - (3.0 / 2.0));

        h_arr2[i] = cosf(h_arr2[i]) * (pow(h_arr2[i], i % 10) / log2((double) i));
    }
}


// **********
// MAIN
// **********
int main(int argc, char *argv[])
{
    clock_t begin;
    clock_t end;

    init();
    prologue();

    // GPU
    begin = clock();
    kernel<<<blocks, BLOCK_SIZE>>>(d_arr);
    cudaThreadSynchronize();
    end = clock();
    printf("gpu: %3.3fms\n", ((float) (end - begin) / CLOCKS_PER_SEC * 1000));

    epilogue();

    // CPU
    begin = clock();
    cpu();
    end = clock();
    printf("cpu: %3.3fms\n", ((float) (end - begin) / CLOCKS_PER_SEC * 1000));
    
    check();
}

