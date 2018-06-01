#include "RectanglesMethod.h"

double RectanglesMethod::CalculateIntegral(double a, double b, Function fun, int n)
{
	double step = calculate_step(a, b, n);
	double integral = 0;
	int i;

	for (i = 0; i < n; ++i) {
		integral += (fun(a + (i * step)) * step);
	}

	return integral;
}

