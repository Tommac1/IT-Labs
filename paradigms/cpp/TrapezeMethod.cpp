#include "TrapezeMethod.h"

double TrapezeMethod::CalculateIntegral(double a, double b, Function fun, int n)
{
    double integral = 0.0;
    double step = calculate_step(a, b, n);
    double *x = make_vector(a, b, n);
    double sum = 0.0;
    size_t len = n + 1; // make_vector()'s return size is always n + 1
    size_t i;

    for (i = 1; i < (len - 1); ++i) {
        sum += fun(x[i]);
    }

    sum += (fun(a)/2.0 + fun(b)/2.0);
    integral = sum * step;

    free(x);
    return integral;
}
