#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "libbmp.h"

#define DIFF_HIST_SIZE 511

typedef struct {
    int val;
    int x;
    int y;
} max_delta;

void print_header(const bmp_img *img);

static double calculate_psnr(const bmp_img *img1, const bmp_img *img2, int *equal, 
        max_delta *delta)
{
    int width = img1->img_header.biWidth;
    int height = img1->img_header.biHeight;
    long long sum = 0;
    long double nomin = 255.0 * 255.0 * width * height;
    int max_val = 0;
    int max_x = 0;
    int max_y = 0;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int diff = img1->img_pixels[i][j] - img2->img_pixels[i][j];
            if (abs(diff) > max_val) {
                max_val = abs(diff);
                max_x = i;
                max_y = j;
            }
            diff *= diff;
            sum += diff;
        }
    }

    delta->val = max_val;
    delta->x = max_x;
    delta->y = max_y;

    if (sum == 0) {
        *equal = 1;
        return 0.0;
    }

    double psnr = (nomin / (double)sum);

    psnr = 10.0 * log10(psnr);

    return psnr;
}

void count_diff_image(const bmp_img *img1, const bmp_img *img2, short **diffs)
{
    int width = img1->img_header.biWidth;
    int height = img1->img_header.biHeight;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            diffs[i][j] = img1->img_pixels[i][j] - img2->img_pixels[i][j];
        }
    }
}

void count_diff_histogram(const bmp_img *img, short **diffs, int *hist)
{
    int width = img->img_header.biWidth;
    int height = img->img_header.biHeight;
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            hist[diffs[i][j]] += 1;
}

void count_histogram(const bmp_img *img, int *hist)
{
    int width = img->img_header.biWidth;
    int height = img->img_header.biHeight;
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            hist[img->img_pixels[i][j]] += 1;
}

int calc_error(int error)
{
    if (error >= 0)
        return 2 * error;
    else
        return -2 * error - 1;
}

void write_diff(const bmp_img *diff_img, const int *diff_hist, int hist_size,
        short **diffs, const char *diff_name)
{
    char diff_fname[256] = { 0 };

    // WRITE HISTOGRAM
    strncpy(diff_fname, diff_name, 256);
    strcat(diff_fname, "_hist");

    FILE *file = fopen(diff_fname, "w");
    for (int i = 0; i < hist_size; ++i)
        fprintf(file, "%d\n", diff_hist[i]);
    fclose(file);

    // WRITE ERROR
    strncpy(diff_fname, diff_name, 256);
    strcat(diff_fname, "_error");

    file = fopen(diff_fname, "wb");
    int h = diff_img->img_header.biHeight;
    int w = diff_img->img_header.biWidth;

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            unsigned short error = calc_error(diffs[i][j]);
            fwrite(&error, sizeof(unsigned short), 1, file);
        }
    }
}

void handle_psnr(const bmp_img *img1, const bmp_img *img2)
{
    int equal = 0;
    max_delta delta = { 0 };
    double psnr = calculate_psnr(img1, img2, &equal, &delta);

    if (equal) {
        printf("images are equal\n");
    }
    else {
        printf("psnr = %.10f\n", psnr);
        printf("max_delta = %d @ [%d, %d]\n", delta.val, delta.x, delta.y);
    }
}

void handle_diff_img(bmp_img *img1, bmp_img *img2, char *diff_name)
{
    bmp_img diff_img;
    bmp_img_copy(&diff_img, img1);

    // alloc diff image
    short **diffs = malloc(sizeof(short *) * 
            diff_img.img_header.biHeight);
    assert(diffs);
    for (int i = 0; i < diff_img.img_header.biHeight; ++i) {
        diffs[i] = malloc(sizeof(short) * diff_img.img_header.biWidth);
        assert(diffs[i]);
    }

    count_diff_image(img1, img2, diffs);
    int *diff_hist = calloc(DIFF_HIST_SIZE, sizeof(int));

    count_diff_histogram(&diff_img, diffs, diff_hist + 255);    

    // write diff histogram and error image
    write_diff(&diff_img, diff_hist, DIFF_HIST_SIZE, diffs, diff_name);

    free(diff_hist);
    bmp_img_free(&diff_img);
}

void run(bmp_img *img1, bmp_img *img2, char *diff_name)
{
    handle_psnr(img1, img2);
    handle_diff_img(img1, img2, diff_name);
}

int main (int argc, char *argv[])
{
    if (argc != 4) {
        printf("usage: %s img1 img2 diff_name\n", argv[0]);
        return -3;
    }
    
    bmp_img before_img;
    bmp_img after_img;
    enum bmp_error before_ret = bmp_img_read(&before_img, argv[1]);
    enum bmp_error after_ret = bmp_img_read(&after_img, argv[2]);

    if (before_ret != BMP_OK || after_ret != BMP_OK) {
        fprintf(stderr, "error opening file (%d = succ): bef %d aft %d\n", 
                BMP_OK, before_ret, after_ret);
        return -1;
    }

    if ((before_img.img_header.biWidth != after_img.img_header.biWidth)
            || (before_img.img_header.biHeight != after_img.img_header.biHeight)) {
        fprintf(stderr, "image sizes do not match\n");
        return -2;
    }

    run(&before_img, &after_img, argv[3]);

    bmp_img_free(&before_img);
    bmp_img_free(&after_img);

    return 0;
}
