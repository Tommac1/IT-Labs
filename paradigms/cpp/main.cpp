#include <iostream>
#include <string>
#include <cmath>
#include <regex>
#include <locale>

#include "main.h"
#include "IntegralMethod.h"
#include "RectanglesMethod.h"
#include "TrapezeMethod.h"
#include "SimpsonsMethod.h"
#include "MonteCarloMethod.h"
#include "Results.h"
#include "Utilities.h"


// GLOBAL DATA =================================================================

double LOWER_BOUND = 0.0;
double UPPER_BOUND = 0.0;
double MY_POW_EXP = 0.0;
double MY_EXPY_BASE = 0.0;
double MY_CONST = 0.0;
double MULTIPLIER = 1.0;
int N_INTERVALS = 100000;

// Gather all funtion patterns and functions in linked arrays
std::string FUNCTION_PATTERNS[NUM_PATTERNS] = { SIN_PATTERN, COS_PATTERN,
    TAN_PATTERN, SQRT_PATTERN, LN_PATTERN, LOG10_PATTERN, LOG2_PATTERN,
    EXP_PATTERN, POW_PATTERN, EXPY_PATTERN, CONST_PATTERN };

Function FUNCTIONS[NUM_PATTERNS] = { sin, cos, tan, sqrt, log, log10, log2,
    exp, my_pow, my_expy, my_const };


void process_stubs(int n, std::string function);
Function regex_match_function(std::string function);
void run(int argc, char *argv[], Results *res, IntegralMethod **methods);
void calculate_function(std::string function, int negative, Results *res, IntegralMethod **methods);
void apply_multiplier(std::string function, Function fun);


int main(int argc, char *argv[])                 
{                                                  
    srand(time(NULL));

    IntegralMethod *methods[4];

    IntegralMethod *rect = new RectanglesMethod();
    IntegralMethod *trap = new TrapezeMethod();
    IntegralMethod *simp = new SimpsonsMethod();
    IntegralMethod *carl = new MonteCarloMethod();

    init_env(argc, argv);

    methods[0] = rect;
    methods[1] = trap;
    methods[2] = simp;
    methods[3] = carl;

    Results *results = new Results(LOWER_BOUND, UPPER_BOUND);

    run(argc, argv, results, methods);

    results->calculate_deviations();

    results->print_outputs(argc, argv);

    delete results;
    delete rect;
    delete trap;
    delete simp;
    delete carl;

    return 0;
}

void init_env(int argc, char *argv[])
{
    if (argc < 5) {
        printf("Usage: %s LOW_BOUND UP_BOUND N f1(x) +- f2(x) +- ... +- fn(x)\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    LOWER_BOUND = atof(argv[1]);
    UPPER_BOUND = atof(argv[2]);

    if (LOWER_BOUND > UPPER_BOUND) {
        // Swap bounds
        my_swap(&LOWER_BOUND, &UPPER_BOUND);
    }

    N_INTERVALS = atoi(argv[3]);
}

void calculate_function(std::string function, int negative, Results *res, IntegralMethod **methods)
{
    Function fun = nullptr;
    std::locale loc;
    double integral = 0.0;
    int i;

    fun = regex_match_function(function);
    if (isdigit(function[0], loc) && (fun != my_expy)) {
        apply_multiplier(function, fun);
    }
    else {
        MULTIPLIER = 1.0;
    }

    if (fun != NULL) {
        // Calculate integrals
        for (i = 0; i < NUM_METHODS; ++i) {
            integral = methods[i]->CalculateIntegral(LOWER_BOUND, UPPER_BOUND,
                    fun, N_INTERVALS);

            if (negative) {
                res->results[i] -= (integral * MULTIPLIER);
            }
            else {
            	res->results[i] += (integral * MULTIPLIER);
            }

        }
    }
    else {
    	std::cout << "Unknown function: " << function << "\n";

    }
}

void apply_multiplier(std::string function, Function fun)
{
    if (fun != my_const) {
    	MULTIPLIER = atof(function.c_str());
    }
}

void run(int argc, char *argv[], Results *res, IntegralMethod **methods)
{
    int i;
    int negative_func = 0;

    for (i = 4; i < argc; ++i) {
    	std::string str(argv[i]);
        if (is_sign(str)) {
            // Sign argument
            negative_func = (str.compare("+") == 0) ? 0 : 1;
        }
        else {
            calculate_function(str, negative_func, res, methods);
        }
    }
}

Function regex_match_function(std::string function)
{
    int i;
    Function ret = NULL;

    for (i = 0; i < NUM_PATTERNS; ++i) {
    	std::regex self_regex(FUNCTION_PATTERNS[i], std::regex_constants::icase);

        if (std::regex_search(function, self_regex)) {
            ret = FUNCTIONS[i];

            if (i >= 8 && i <= 10) {
                process_stubs(i, function);
            }
        }
    }

    return ret;
}

void process_stubs(int n, std::string function)
{
    int i = 0;
    size_t len = function.length();
    char *tmp = new char[len + 1];
    assert(tmp);
    function.copy(tmp, 0);

    switch (n) {
    case 8:
        // x^y function
        while (function[i] != '^') i++;
        i++;
        MY_POW_EXP = atof(function.c_str() + i);
        std::cout << "MY_POW_EXP " << MY_POW_EXP << "\n";
        break;
    case 9:
        // y^x function
        MY_EXPY_BASE = atof(function.c_str());
        std::cout << "MY_EXPY_BASE " << MY_EXPY_BASE << "\n";
        break;
    case 10:
        // const function
        MY_CONST = atof(function.c_str());
        std::cout << "MY_CONST " << MY_CONST << "\n";
        break;
    }

    delete tmp;
}

double my_pow(double x)
{
    return pow(x, MY_POW_EXP);
}

double my_expy(double x)
{
    return pow(MY_EXPY_BASE, x);
}

double my_const(double x)
{
    MULTIPLIER = 1.0; // Const cannot be multiplied
    return MY_CONST;
}
