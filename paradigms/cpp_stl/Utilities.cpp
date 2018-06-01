#include "Utilities.h"

int is_sign(std::string s)
{
    int ret = 0;

    if ((s.compare("+") == 0) || s.compare("-") == 0)
        ret = 1;

    return ret;
}

int sign(double a)
{
    int ret = 0;

    if (a > 0)
        ret = 1;
    else if (a < 0)
        ret = -1;

    return ret;
}

void my_swap(double *a, double *b)
{
	double tmp = *a;
	*a = *b;
	*b = tmp;
}

double max(std::vector<double> v, Function fun, int size)
{
    double res = 0.0;
    int i;

    res = fun(v[0]);
    for (i = 1; i < size; ++i)
        if (res < fun(v[i]))
            res = fun(v[i]);

    return res;
}

double min(std::vector<double> v, Function fun, int size)
{
    double res = 0.0;
    int i;

    res = fun(v[0]);
    for (i = 1; i < size; ++i)
        if (res > fun(v[i]))
            res = fun(v[i]);

    return res;
}
