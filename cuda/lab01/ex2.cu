#include <stdio.h>
#include <time.h>

#include "ex2.h"

double  h_arr[N];
double *d_arr;
int blocks;

void prologue(void)
{
    for (int i = 0; i < N; ++i)
        h_arr[i] = i + 1;

    cudaMalloc((void **)&d_arr, sizeof(h_arr));
    cudaMemcpy(d_arr, h_arr, sizeof(h_arr), cudaMemcpyHostToDevice);
}

void epilogue(void)
{
    cudaMemcpy(h_arr, d_arr, sizeof(h_arr), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

__global__ void mul(double *a)
{
    // plus one because threads are zero indexed
    int x = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (x < N)
        a[x] = (double)(4.0 * x * x)/(4.0 * x * x - 1.0);
}

void cuda(double eps)
{
    blocks = N / BLOCK_SIZE;
    if (N % BLOCK_SIZE != 0)
        blocks++;

    // CUDA
    mul<<<blocks, BLOCK_SIZE>>>(d_arr);
    cudaThreadSynchronize();
}

double cpu(double eps)
{
    double pi = 1.0;
    double last_pi = 10.0;
    int n = 1;

    // CPU
    while (abs(last_pi - pi) >= eps) {
        double x = (double)(4.0 * n * n)/(4.0 * n * n - 1.0);
        last_pi = pi;
        pi *= x;
        ++n;
    }

    return pi;
}

int main(int argc, char *argv[])
{
    clock_t start;
    clock_t end;
    int dev_cnt;
    int n = 1;
    double pi = 0.0;
    double eps = 1.0;

    if (argc != 2) {
        printf("usage: %s 1*10^-eps\n", argv[0]);
        return -2;
    }

    int x = atoi(argv[1]);
    for (int i = 0; i < x; ++i)
        eps *= 0.1;

    printf("epsilon: %.15f\n", eps);

    cudaGetDeviceCount(&dev_cnt);
    if (dev_cnt == 0) {
        perror("no cuda device available");
        return -1;
    }

    prologue();
    start = clock();
    cuda(eps);
    end = clock();
    epilogue();

    // calculate the product of multiplication
    while (abs(h_arr[0] - pi) >= eps) {
        pi = h_arr[0];
        h_arr[0] *= h_arr[n];
        ++n;
    }

    pi = h_arr[0];

    printf("cuda:\t%.5fms,\tpi = %.10f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000, pi * 2.0);

    start = clock();
    pi = cpu(eps);
    end = clock();

    printf("cpu:\t%.5fms,\tpi = %.10f\n", (double)(end - start) / CLOCKS_PER_SEC * 1000, pi * 2.0);

    return 0;
}
