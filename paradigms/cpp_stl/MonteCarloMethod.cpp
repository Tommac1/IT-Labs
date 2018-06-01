#include "MonteCarloMethod.h"

double MonteCarloMethod::CalculateIntegral(double a, double b, Function fun, int n)
{
    double integral = 0.0;
    std::vector<double> x = make_vector(a, b, n);
    double new_a = a;
    int i = 0;

    if (sign(fun(x[i])) == 0) i++;

    for (; i < n; ++i) {
        // If the function crosses x-axis, we have to divide area
        if (sign(fun(x[i])) != sign(fun(x[i + 1]))) {
            integral += _calculateIntegral(new_a, x[i], fun, n);

            i++;
            new_a = x[i];
        }
    }

    integral += _calculateIntegral(new_a, x[i], fun, n);

    //delete x;

    return integral;
}

double MonteCarloMethod::_calculateIntegral(double a, double b, Function fun, int n)
{
    double integral = 0.0;
    std::vector<double> x = make_vector(a, b, n);
    double fmax;
    double fmin;
    double randX = 0.0;
    double randY = 0.0;
    int hits = 0;
    int i = 0;
    int fun_sign = sign(fun(a));

    while ((fun_sign == 0) && (i < n)) {
        fun_sign = sign(fun(x[i]));
        i++;
    }

    if (i == n)
        return 0.0; // f(x) = 0

    if (fun_sign == 1)
        fmax = max(x, fun, n + 1);
    else
        fmin = min(x, fun, n + 1);


    for (i = 0; i < MONTE_CARLO_SHOTS; ++i) {
        // rand()/(RAND_MAX/a) => random number [0..a]
        randX = (double)rand()/(double)(RAND_MAX/(b - a));
        randX += a;

        if (fun_sign == 1) {
            randY = (double)rand()/(double)(RAND_MAX/fmax);

            if (fun(randX) >= randY)
                hits++;
        }
        else {
            // To generate [-a..0] -> generate [0..a] and substract a
            // Note: fmin is negative, so opposite sign
            randY = (double)rand()/(double)(RAND_MAX/(-fmin));
            randY += fmin;

            if (fun(randX) <= randY)
                hits++;
        }
    }

    integral = ((double)hits/(double)MONTE_CARLO_SHOTS);

    if (fun_sign == 1)
        integral *= ((b - a) * fmax);
    else
        integral *= ((b - a) * fmin);

    //delete x;

    return integral;
}
