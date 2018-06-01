#ifndef TRAPEZEMETHOD_H
#define TRAPEZEMETHOD_H

#include "IntegralMethod.h"

class TrapezeMethod : public IntegralMethod {
private:

public:
	double CalculateIntegral(double a, double b, Function fun, int n);

	TrapezeMethod() {};
	~TrapezeMethod() {};
};

#endif
