#ifndef RECTANGLESMETHOD_H
#define RECTANGLESMETHOD_H

#include "IntegralMethod.h"

class RectanglesMethod : public IntegralMethod {
private:

public:
	double CalculateIntegral(double a, double b, Function fun, int n);

	RectanglesMethod() {};
	~RectanglesMethod() {};
};

#endif
