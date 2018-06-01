#ifndef MONTECARLOMETHOD_H
#define MONTECARLOMETHOD_H

#include "IntegralMethod.h"

class MonteCarloMethod : public IntegralMethod {
private:
	double _calculateIntegral(double  a, double  b, Function fun, int n);

public:
	double CalculateIntegral(double  a, double  b, Function fun, int n);

	MonteCarloMethod() {};
	~MonteCarloMethod() {};
};

#endif
