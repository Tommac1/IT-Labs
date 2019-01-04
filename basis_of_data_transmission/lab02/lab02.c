#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "kissfft/kiss_fftr.h"

#define PLOTS_NUM 7

struct sine_wave {
    double *values[2];
    int samples;
};

typedef double (*exc_fun)(double x);

// =========================================================
// Excersises functions
// =========================================================

static double ex1_fun(double x)
{
    return (0.45/(x + 20.0) * cos(2.0 * M_PI * 20.0 * (x / 600.0) - 2.0 * M_PI));
}

static double ex2_yn(double x)
{
    return (cos(20.0 * x) / 4.0);
}

static double ex2a_fun(double x)
{
    double fx = ex1_fun(x);
    double yx = ex2_yn(x);

    return (yx * (sin(0.6 * M_PI * x / 600.0) * fabs(fx)));
}

static double ex2b_fun(double x)
{
    double fx = ex1_fun(x);
    double yx = ex2_yn(x);

    return (sqrt(fabs(fx)) * cos(yx/2.0));
}

static double ex3_fun(double x)
{
    double result = 0.0;

    if (x >= 0 && x < 0.2)
        result = (4.0 + (-x * sin(18000.0 * M_PI * (x - 0.2)/8000.0) + 
                    cos(45000 * M_PI * (x - 0.2)/8000.0)));
    else if (x >= 0.2 && x < 0.7)
        result = 1.0/x;
    else if (x >= 0.7 && x < 1.0)
        result = ((0.5 * cos(12.0 * M_PI * 3000.0 * (x - 0.7)/8000.0)) + 0.92);
            
    return result;
}

static double ex4_sum(double t, int n)
{
    return ((pow(-1.0, (double)n)/pow((double)n, 2.0)) * cos((2.0 * M_PI * (double)n * t) / 8000.0));
}

static double ex4a_fun(double t)
{
    double result = 0.0;
    int n = 1;
    const int H = 5;

    for (n = 1; n <= H; ++n)
        result += ex4_sum(t, n);

    return ((pow(M_PI, 2.0) / 3.0) + (4.0 * result));
}

static double ex4b_fun(double t)
{
    double result = 0.0;
    int n = 1;
    const int H = 50;

    for (n = 1; n <= H; ++n)
        result += ex4_sum(t, n);

    return ((pow(M_PI, 2.0) / 3.0) + (4.0 * result));
}

static double ex4c_fun(double t)
{
    double result = 0.0;
    int n = 1;
    const int H = 100;

    for (n = 1; n <= H; ++n)
        result += ex4_sum(t, n);

    return ((pow(M_PI, 2.0) / 3.0) + (4.0 * result));
}

// =========================================================
// Helper functions
// =========================================================

static void create_sine(struct sine_wave *sw, int samples)
{
    sw->values[1] = malloc(sizeof(double) * samples);
    sw->values[0] = malloc(sizeof(double) * samples);

    sw->samples = samples;

    assert(sw->values[0] && sw->values[1]);
}

static void generate_sine(double period, double freq_s, struct sine_wave *out, exc_fun fun)
{
    int i = 0;
    int samples = freq_s * period;
    double step = 1.0 / freq_s;
    double x_val = 0.0;

    for (i = 0; i < samples; ++i) {
        x_val += step;
        out->values[0][i] = x_val;
        out->values[1][i] = fun(x_val);
    }
}

static void free_sine_wave(struct sine_wave *sw)
{
    free(sw->values[0]);
    free(sw->values[1]);
    sw->values[0] = NULL;
    sw->values[1] = NULL;
}

static void print_sine(struct sine_wave *sw, int file, int ftw)
{
    int i = 0;
    FILE *fp;
    char filename[20];
    if (ftw)
        strncpy(filename, "./plotx_ftw.csv", 20);
    else
        strncpy(filename, "./plotx.csv", 20);

    filename[6] = file + 1 + '0';

    fp = fopen(filename, "w+");
    assert(fp);

    for (i = 0; i < sw->samples; ++i)
        fprintf(fp, "%.15f, %.15f\n", sw->values[0][i], sw->values[1][i]);

    fclose(fp);
}

int round_pow2(int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

static void dft_sum(double *out, int k, int N, double *x)
{
    double fi = 0.0;
    int n;

    for (n = 0; n < N; ++n) {
        fi = (2.0 * M_PI * (double)k * (double)n / (double)N);
        out[0] += (x[n] * cos(fi)); // Re
        out[1] += (-x[n] * sin(fi)); // Im
    }
}

static void dft(exc_fun xn, struct sine_wave *fw, double fs)
{
    int i;
    double z[2] = { 0 };

    for (i = 0; i < fw->samples / 2 - 1; ++i) {
        (i & 1023) ? : printf("%d\n", i);
        dft_sum(z, i, fw->samples, fw->values[1]);
        fw->values[0][i] = ((double)i * (fs / (double)fw->samples));
        fw->values[1][i] = sqrt(pow(z[0], 2.0) + pow(z[1], 2.0));
        fw->values[1][i] = 10.0 * log10(fw->values[1][i]);
    }
    fw->samples /= 2;
    fw->samples--;
}

// =========================================================
// Main function
// =========================================================

int main(int argc, char *argv[])
{
    const exc_fun funcs[PLOTS_NUM] = { ex1_fun, ex2a_fun, ex2b_fun, ex3_fun, 
        ex4a_fun, ex4b_fun, ex4c_fun };
    const double freqs_s[PLOTS_NUM] = { 600.0, 600.0, 600.0, 800.0, 800.0, 800.0, 800.0 };
    const double periods[PLOTS_NUM] = { 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0 };

    double time_used;
    clock_t start;
    clock_t end;
    int i = 0;
    int j = 0;
    struct sine_wave sw;
    struct sine_wave fw;

    for (i = 0; i < PLOTS_NUM; ++i) {
        create_sine(&sw, freqs_s[i] * periods[i]);
        generate_sine(periods[i], freqs_s[i], &sw, funcs[i]);
        print_sine(&sw, i, 0);

        
        create_sine(&fw, freqs_s[i] * periods[i]);
        start = clock();
        dft(funcs[i], &fw, freqs_s[i]);
        end = clock();
        print_sine(&fw, i, 1);

        time_used = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Plot %d DFT time taken: %.5fs\n", i, time_used);


        sw.samples /= periods[i];
        kiss_fftr_cfg kf_cfg = kiss_fftr_alloc(sw.samples, 0, NULL, NULL);
        kiss_fft_scalar *in = malloc(sw.samples * sizeof(kiss_fft_scalar));

        for (j = 0; j < sw.samples; ++j)
            in[j] = sw.values[1][j];

        kiss_fft_cpx *out = malloc((sw.samples/2+1) * sizeof(kiss_fft_cpx));

        start = clock();
        kiss_fftr(kf_cfg, in, out);
        end = clock();

        time_used = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Plot %d FFT time taken: %.5fs\n", i, time_used);

        free(in);
        free(out);
        free(kf_cfg);
        free_sine_wave(&sw);
        free_sine_wave(&fw);
    }

    return 0;
}
