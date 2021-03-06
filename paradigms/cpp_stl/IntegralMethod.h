#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <iterator>
#include "Utilities.h"
#include "main.h"

class IntegralMethod {
protected:
	double calculate_step(double a, double b, int n)
	{
	    return (b - a) / n;
	}

	std::vector<double> make_vector(double a, double b, int n)
	{
	    int i;
	    double step = calculate_step(a, b, n);
	    std::vector<double> ret (n + 1);
	    int length = std::distance(ret.begin(), ret.end());

	    for (i = 0; i < length; ++i) {
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
