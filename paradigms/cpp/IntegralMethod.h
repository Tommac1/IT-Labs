#ifndef INTEGRAL_H
#define INTEGRAL_H


#include "Utilities.h"

class IntegralMethod {
protected:
	double calculate_step(double a, double b, int n)
	{
	    return (b - a) / n;
	}

	double *make_vector(double a, double b, int n)
	{
	    int i;
	    double step = calculate_step(a, b, n);
	    double *ret = new double[n + 1];

	    for (i = 0; i <= n; ++i) {
	        ret[i] = a + (i * step);
	    }
	    return ret;
	}

public:
	virtual double CalculateIntegral(double a, double b, Function fun, int n) = 0;

	IntegralMethod() {};
	virtual ~IntegralMethod() {};
};

#endif
