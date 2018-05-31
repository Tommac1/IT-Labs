#ifndef UTILITIES_H
#define UTILITIES_H

#include "Results.h"

typedef double (*Function)(double);
typedef double (*Integral)(double, double, Function, int);

double min(double *v, Function fun, int size);
double max(double *v, Function fun, int size);
void my_swap(double *a, double *b);
int is_sign(std::string s);
int sign(double a);

#endif
