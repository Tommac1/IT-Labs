#include "Results.h"

double Results::calculate_deviation(double mean, double value)
{
    double nomin = fabs(value - mean);
    double denomin = mean;

    return (nomin / denomin) * 100;
}

double Results::calculate_mean(int size)
{
    double res = 0.0;
    int i;

    for (i = 0; i < size; ++i) {
        res += this->results[i];
    }

    return (res / size);
}

void Results::calculate_deviations()
{
    double mean = 0.0;
    int i;

    mean = calculate_mean(NUM_METHODS);

    // Calculate deviations
    for (i = 0; i < NUM_METHODS; ++i)
        this->deviations[i] = calculate_deviation(mean, this->results[i]);
}

void Results::print_outputs(int argc, char *argv[])
{
    int i;

    // Print outputs
    std::cout << "Integral between " << lower_bound <<" and " << upper_bound << " of";

    for (i = 4; i < argc; ++i) {
    	std::string s(argv[i]);
		std::cout << " " << s;
    }
    std::cout << "\n";

    for (i = 0; i < NUM_METHODS; ++i)
    	std::cout << std::setprecision(6) << METHOD_NAMES[i] << " "
    			  << this->results[i] << " (" << this->deviations[i] << "%)\n";

}
