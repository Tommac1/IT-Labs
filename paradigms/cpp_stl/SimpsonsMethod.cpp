#include "SimpsonsMethod.h"

double SimpsonsMethod::CalculateIntegral(double a, double b, Function fun, int n)
{
    double step = calculate_step(a, b, n);
    std::vector<double> x = make_vector(a, b, n);
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum = 0.0;
    double integral = 0.0;
    size_t len = n + 1; // make_vector()'s return size is always n + 1
    size_t i;

    for (i = 1; i < (len - 1); ++i) {
        if ((i & 2) == 0) {
            sum1 += fun(x[i]);
        }
        else {
            sum2 += fun(x[i]);
        }
    }

    sum1 *= 4;
    sum2 *= 2;

    sum = sum1 + sum2 + fun(a) + fun(b);
    integral = (step/3) * sum;

    return integral;
}
