#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

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
    return (0.7 * sin(2.0 * M_PI * 4.0 * (x / 100.0) + 7.0 * M_PI / 9.0) * x);
}

static double ex2_yn(double x)
{
    return (sin(8.0 * M_PI * (x / 100.0)) * cos(10.0 * x));
}

static double ex2a_fun(double x)
{
    double fx = ex1_fun(x);
    double yx = ex2_yn(x);

    return (yx - 3000.0 * abs(fx));
}

static double ex2b_fun(double x)
{
    double fx = ex1_fun(x);
    double yx = ex2_yn(x);

    return (fx * (abs(yx) + 2.6) * (fx - yx));
}

static double ex3_fun(double x)
{
    double result = 0.0;

    if (x >= 0 && x < 0.4)
        result = (pow(0.8, x) - 1.0);
    else if (x >= 0.4 && x < 0.6)
        result = -1.8 * x * cos(16.0 * M_PI * (x/4000.0) + M_PI);
    else if (x >= 0.6 && x < 0.8)
        result = 0.72 * x;
    else if (x >= 0.8 && x < 1.0)
        result = 0.29 * pow(x, 8) * sin(31.0 * M_PI * (x/4000.0) + 0.55);
            
    return result;
}

static double ex4_sum(double t, int n)
{
    return (sin(0.5 * M_PI * t * (double)n) / (2.0 * (double)n + 1.0));
}

static double ex4a_fun(double t)
{
    double result = 0.0;
    int n = 0;
    const int H = 2;

    for (n = 0; n <= H - 1; ++n)
        result += ex4_sum(t, n);

    return ((4.0 / M_PI) * result);
}

static double ex4b_fun(double t)
{
    double result = 0.0;
    int n = 0;
    const int H = 25;

    for (n = 0; n <= H - 1; ++n)
        result += ex4_sum(t, n);

    return ((4.0 / M_PI) * result);
}

static double ex4c_fun(double t)
{
    double result = 0.0;
    int n = 0;
    const int H = 100;

    for (n = 0; n <= H - 1; ++n)
        result += ex4_sum(t, n);

    return ((4.0 / M_PI) * result);
}

// =========================================================
// Helper functions
// =========================================================

static void generate_sine(double period, double freq_s, struct sine_wave *out, exc_fun fun)
{
    int i = 0;
    int samples = freq_s * period;
    double step = period / freq_s;
    double x_val = 0.0;

    out->values[1] = malloc(sizeof(double) * samples);
    out->values[0] = malloc(sizeof(double) * samples);

    assert(out->values[0] && out->values[1]);

    for (i = 0; i < samples; ++i) {
        x_val += step;
        out->values[0][i] = x_val;
        out->values[1][i] = fun(x_val);
    }

    out->samples = samples;
}

static void free_sine_wave(struct sine_wave *sw)
{
    free(sw->values[0]);
    free(sw->values[1]);
    sw->values[0] = NULL;
    sw->values[1] = NULL;
}

static void print_sine(struct sine_wave *sw, int file)
{
    int i = 0;
    FILE *fp;
    char filename[] = "./plotx.csv";
    filename[6] = file + 1 + '0';

    fp = fopen(filename, "w+");
    assert(fp);

    for (i = 0; i < sw->samples; ++i)
        fprintf(fp, "%.5f, %.5f\n", sw->values[0][i], sw->values[1][i]);

    fclose(fp);
}

// =========================================================
// Main function
// =========================================================

int main(int argc, char *argv[])
{
    exc_fun funcs[PLOTS_NUM] = { ex1_fun, ex2a_fun, ex2b_fun, ex3_fun, 
        ex4a_fun, ex4b_fun, ex4c_fun };
    double freqs_s[PLOTS_NUM] = { 100.0, 100.0, 100.0, 4000.0, 10000.0, 10000.0, 10000.0 };
    double periods[PLOTS_NUM] = { 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0 };

    int i = 0;
    struct sine_wave sw;


    for (i = 0; i < PLOTS_NUM; ++i) {
        generate_sine(periods[i], freqs_s[i], &sw, funcs[i]);
        print_sine(&sw, i);
        free_sine_wave(&sw);
    }

    return 0;
}
