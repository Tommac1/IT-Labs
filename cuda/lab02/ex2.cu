#include <stdio.h>
#include <time.h>
#include <assert.h>

extern "C" {
#include "libbmp.h"
}

#define BLOCK_SIZE 512
#define CUDA_CALL(x)                                        \
    do {                                                    \
        cudaError_t cuda_error__ = (x);                     \
        if (cuda_error__) {                                 \
            printf("CUDA error: " #x " returned \"%s\"\n",  \
                    cudaGetErrorString(cuda_error__));      \
        }                                                   \
    } while (0)                                             \


bmp_pixel *h_arr;
bmp_pixel *d_arr;
int h = 0;
int w = 0;


/* 
 * flattens 2D(h, w) array into 1D(h * w). 
 * dest should be allocated before call
 */
void flatten(bmp_pixel *dest, bmp_pixel **src, int h, int w)
{
    for (int i = 0; i < h; ++i)
        memcpy((dest + i * w), src[i], w * sizeof(*dest));
}

/* 
 * converts 1D(h * w) array into 2D(h, w)
 * dest should be allocated before call
 */
void make2D(bmp_pixel **dest, bmp_pixel *src, int h, int w)
{
    for (int i = 0; i < h; ++i)
        memcpy(dest[i], (src + i * w), w * sizeof(*src));
}

void prologue(bmp_img *img)
{
    CUDA_CALL( cudaMalloc((void **) &d_arr, h * w * sizeof(bmp_pixel)) );
    h_arr = (bmp_pixel *) malloc(w * h * sizeof(bmp_pixel));
    flatten(h_arr, img->img_pixels, h, w);

    CUDA_CALL( cudaMemcpy(d_arr, h_arr, w * h * sizeof(bmp_pixel),
                cudaMemcpyHostToDevice) );
}

void epilogue(bmp_img *img)
{
    CUDA_CALL( cudaMemcpy(h_arr, d_arr, w * h * sizeof(bmp_pixel),
                cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaFree(d_arr) );

    make2D(img->img_pixels, h_arr, h, w);

    free(h_arr);
}

__device__ void swap(unsigned char *a, unsigned char *b)
{
    unsigned char t = *a;
    *a = *b;
    *b = t;
}

__device__ void sort(bmp_pixel *arr, int n) 
{ 
   int i;
   int j; 
   int swapped; 

   // RED
   for (i = 0; i < n - 1; i++) { 
     swapped = 0; 
     for (j = 0; j < n - i - 1; j++) { 
        if (arr[j].red > arr[j + 1].red) { 
           swap(&arr[j].red, &arr[j + 1].red); 
           swapped = 1; 
        } 
     } 

     if (swapped == 0) 
        break; 
   }

   // GREEN
   for (i = 0; i < n - 1; i++) { 
     swapped = 0; 
     for (j = 0; j < n - i - 1; j++) { 
        if (arr[j].green > arr[j + 1].green) { 
           swap(&arr[j].green, &arr[j + 1].green); 
           swapped = 1; 
        } 
     } 

     if (swapped == 0) 
        break; 
   }

   // BLUE
   for (i = 0; i < n - 1; i++) { 
     swapped = 0; 
     for (j = 0; j < n - i - 1; j++) { 
        if (arr[j].blue > arr[j + 1].blue) { 
           swap(&arr[j].blue, &arr[j + 1].blue); 
           swapped = 1; 
        } 
     } 

     if (swapped == 0) 
        break; 
   }
} 

__device__ void cuda_gather_pixels(bmp_pixel *buff, bmp_pixel *img, 
        int x, int w, int h)
{
    int sz = w * h;

    // gather neighbour pixels
    buff[0] = (x - w - 1) < 0 ? img[x] : img[x - w - 1];
    buff[1] = (x - w) < 0 ? img[x] : img[x - w];
    buff[2] = (x - w + 1) < 0 ? img[x] : img[x - w + 1];
    buff[3] = (x - 1) < 0 ? img[x] : img[x - 1];
    buff[4] = img[x];
    buff[5] = (x + 1) > sz ? img[x] : img[x + 1];
    buff[6] = (x + w - 1) > sz ? img[x] : img[x + w - 1];
    buff[7] = (x + w) > sz ? img[x] : img[x + w];
    buff[8] = (x + w + 1) > sz ? img[x] : img[x + w + 1];
}

// KERNEL
__global__ void cuda_median(bmp_pixel *a, int w, int h)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    bmp_pixel pix_arr[9] = { 0 };
    int sz = w * h;

    if (x < sz) {
        cuda_gather_pixels(pix_arr, a, x, w, h);

        sort(pix_arr, 9);

        __syncthreads();

        a[x].red = pix_arr[4].red;
        a[x].green = pix_arr[4].green;
        a[x].blue = pix_arr[4].blue;
    }
}

void run_gpu(bmp_img *img)
{
    int blocks = (h * w) / BLOCK_SIZE;
    if ((h * w) % BLOCK_SIZE != 0)
        blocks++;

    prologue(img);

    cuda_median<<<blocks, BLOCK_SIZE>>>(d_arr, w, h);

    CUDA_CALL( cudaThreadSynchronize() );
    CUDA_CALL( cudaPeekAtLastError() );
    CUDA_CALL( cudaDeviceSynchronize() );

    epilogue(img);
}

int cmp(const void *val1, const void *val2)
{
    int a = *((int *) val1);
    int b = *((int *) val2);

    if (a > b) 
        return 1;
    else if (b > a) 
        return -1;
    else
        return 0;
}

int median(int *v, int n)
{
    if (n & 1) {
        return v[n / 2];
    } else {
        return ((v[n / 2] + v[n / 2 + 1] / 2.0));
    }
}

// color: 1 - red, 2 - green, 3 - blue
void gather_pixels(bmp_img *img, int h, int w, int *pix_vect, int color)
{
    int vect_idx = 0;
    for (int i = h - 1; i <= h + 1; ++i) {
        for (int j = w - 1; j <= w + 1; ++j) {
            if (color == 1)
                pix_vect[vect_idx] = img->img_pixels[i][j].red;
            else if (color == 2)
                pix_vect[vect_idx] = img->img_pixels[i][j].green;
            else
                pix_vect[vect_idx] = img->img_pixels[i][j].blue;
        
            vect_idx++;
        }
    }
}

void run_cpu(bmp_img *img)
{
    int *pix_vect = (int *) malloc(9 * sizeof(int));
    assert(pix_vect);

    for (int i = 1; i < h - 1; ++i) {
        for (int j = 1; j < w - 1; ++j) {
            // red
            gather_pixels(img, i, j, pix_vect, 1);
            qsort(pix_vect, 9, sizeof(int), cmp);
            img->img_pixels[i][j].red = median(pix_vect, 9);
            // green
            gather_pixels(img, i, j, pix_vect, 2);
            qsort(pix_vect, 9, sizeof(int), cmp);
            img->img_pixels[i][j].green = median(pix_vect, 9);
            // blue
            gather_pixels(img, i, j, pix_vect, 3);
            qsort(pix_vect, 9, sizeof(int), cmp);
            img->img_pixels[i][j].blue = median(pix_vect, 9);
        }
    }

    free(pix_vect);
}

int run(const char *img_name)
{
    bmp_img cpu_img;
    bmp_img gpu_img;
    double time;
    clock_t begin;
    clock_t end;

    enum bmp_error err_cpu = bmp_img_read(&cpu_img, img_name);
    enum bmp_error err_gpu = bmp_img_read(&gpu_img, img_name);

    if (err_cpu != BMP_OK || err_gpu != BMP_OK) {
        printf("error opening file %s (cpu %d gpu %d)\n",
                img_name, err_cpu, err_gpu);
        return -3;
    }

    h = cpu_img.img_header.biHeight;
    w = cpu_img.img_header.biWidth;

    // CPU
    begin = clock();
    run_cpu(&cpu_img);
    end = clock();
    
    time = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
    printf("cpu time = %.3fms\n", time);

    // CUDA
    begin = clock();
    run_gpu(&gpu_img);
    end = clock();

    time = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
    printf("gpu time = %.3fms\n", time);


    bmp_img_write(&cpu_img, "illidan_cpu.bmp");
    bmp_img_write(&gpu_img, "illidan_gpu.bmp");

    bmp_img_free(&cpu_img);
    bmp_img_free(&gpu_img);

    return 0;
}

int main(int argc, char *argv[])
{
    int dev_cnt;
    cudaGetDeviceCount(&dev_cnt);

    if (dev_cnt == 0) {
        printf("no cuda devices available\n");
        return -1;
    }

    return run("illidan.bmp");
}
