#ifndef SIMPSONSMETHOD_H
#define SIMPSONSMETHOD_H

#include "IntegralMethod.h"

class SimpsonsMethod : public IntegralMethod {
private:

public:
	double CalculateIntegral(double  a, double  b, Function fun, int n);

	SimpsonsMethod() {};
	~SimpsonsMethod() {};
};

#endif
