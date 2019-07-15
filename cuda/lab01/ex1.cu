#include <stdio.h>
#include <time.h>

#ifndef N
#define N (2048 * 2048)
#endif
#define BLOCK_SIZE 1024

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

void check(void)
{
    srand(time(NULL));
    for (int i = 0; i < 100000; ++i) {
        unsigned long j = (rand() % N) + 1;
        if (h_arr[j - 1] != j * j) {
            printf("error: %.3f != %lu\n", h_arr[j - 1], j * j);
            break;
        }
    }
}

__global__ void sqr(double *a)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x < N)
        a[x] = a[x] * a[x] * a[x];
}

int main(int argc, char *argv[])
{
    clock_t start;
    clock_t end;
    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);
    if (dev_cnt == 0) {
        perror("no cuda device available");
        goto error_ret;
    }

    printf("N = %d\n", N);

    prologue();
    blocks = N / BLOCK_SIZE;
    if (N % BLOCK_SIZE != 0)
        blocks++;

    // KERNEL
    start = clock();
    sqr<<<blocks, BLOCK_SIZE>>>(d_arr);
    cudaThreadSynchronize();
    end = clock();
    epilogue();

    printf("cuda: %.5fms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // CPU
    start = clock();
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i + 1;
        h_arr[i] *= h_arr[i];
    }
    end = clock();

    printf("cpu: %.5fms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    check();
    
    return 0;

error_ret:
    return -1;
}
