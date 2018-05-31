#ifndef RESULTS_H
#define RESULTS_H

#include <cmath>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <string>

#include "Utilities.h"
#include "main.h"

class Results {
private:
	double calculate_mean(int size);
	double calculate_deviation(double mean, double value);

	double upper_bound = 0.0;
	double lower_bound = 0.0;
	double MEAN = 0.0;

	std::string METHOD_NAMES[4] = {
	    "Rectangle's Method",
	    "Trapeze's Method",
	    "Simspon's Method",
	    "Monte Carlo's Method"
	};

public:
	void print_outputs(int argc, char *argv[]);
	void calculate_deviations();

	double results[4];
	double deviations[4];

	Results(double lb, double ub) {
		this->lower_bound = lb;
		this->upper_bound = ub;

		for (int i = 0; i < 4; ++i) {
			results[i] = 0.0;
			deviations[i] = 0.0;
		}
	};
	~Results() {};
};

#endif
