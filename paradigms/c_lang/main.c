#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <regex.h>
#include <time.h>
#include <assert.h>
#include <ctype.h>

#define UNREF(x)            (void)(x)
#define NUM_METHODS         (4)
#define MONTE_CARLO_SHOTS   (100000)

#define SIN_PATTERN         "^[[:digit:]|.]*sin\\(x\\)$"
#define COS_PATTERN         "^[[:digit:]|.]*cos\\(x\\)$" 
#define TAN_PATTERN         "^[[:digit:]|.]*tan\\(x\\)$"
#define SQRT_PATTERN        "^[[:digit:]|.]*sqrt\\(x\\)$"
#define LN_PATTERN          "^[[:digit:]|.]*ln\\(x\\)$"
#define LOG10_PATTERN       "^[[:digit:]|.]*log10\\(x\\)$"
#define LOG2_PATTERN        "^[[:digit:]|.]*log2\\(x\\)$"
#define EXP_PATTERN         "^[[:digit:]|.]*e\\^x$"
#define POW_PATTERN         "^[[:digit:]|.]*x\\^[[:digit:]]+$"
#define EXPY_PATTERN        "^[[:digit:]|.]+\\^x$" 
#define CONST_PATTERN       "^[[:digit:]|.]+$"

#define NUM_PATTERNS        11

#ifndef M_PI
 #define M_PI 3.14159265358979323846 
#endif

// TYPES =======================================================================

typedef double (*Fun)(double);
typedef double (*Integral)(double, double, Fun, int);
typedef struct Result {
    double result;
    double deviation;
} Result;

// FUNCTION DECLATIONS =========================================================

double rectangles_method(double a, double b, Fun fun, int n);
double trapeze_method(double a, double b, Fun fun, int n);
double simpsons_method(double a, double b, Fun fun, int n);
double monte_carlo_method(double a, double b, Fun fun, int n);
double _monte_carlo_method(double a, double b, Fun fun, int n);


void calculate_function(double a, double b, char *function, int negative);
Fun regex_match_function(char *function);
void apply_multiplier(char *funtion, Fun fun);
void run(int argc, char *argv[]);
void init_env(int argc, char *argv[]);
double calculate_step(double a, double b, int n);
double *make_vector(double a, double b, int n);
double calculate_deviation(double mean, double value);
void calculate_deviations();
double calculate_mean(Result *v, int size);

double max(double *v, Fun fun, int size);
double min(double *v, Fun fun, int size);
int is_sign(char *s);
int sign(double a);
void print_outputs(int argc, char *argv[]);

void process_stubs(int n, char *function);
double my_pow(double x);
double my_expy(double x);
double my_const(double x);


// GLOBAL DATA =================================================================

double LOWER_BOUND = 0.0;
double UPPER_BOUND = 0.0;
double MY_POW_EXP = 0.0;
double MY_EXPY_BASE = 0.0;
double MY_CONST = 0.0;
double MEAN = 0.0;
double MULTIPLIER = 0.0;
int N_INTERVALS = 100000;

Integral methods[NUM_METHODS] = {
    rectangles_method,
    trapeze_method,
    simpsons_method,
    monte_carlo_method
};

const char *METHOD_NAMES[NUM_METHODS] = {
    "Rectangle's Method",
    "Trapeze's Method",
    "Simspon's Method",
    "Monte Carlo's Method"
};

// Initialize every field of every structure in array with zeros
Result results[NUM_METHODS] = {  { 0.0 } };

// Gather all funtion patterns and functions in linked arrays
const char *FUNCTION_PATTERNS[NUM_PATTERNS] = { SIN_PATTERN, COS_PATTERN, 
    TAN_PATTERN, SQRT_PATTERN, LN_PATTERN, LOG10_PATTERN, LOG2_PATTERN, 
    EXP_PATTERN, POW_PATTERN, EXPY_PATTERN, CONST_PATTERN }; 

Fun FUNCTIONS[NUM_PATTERNS] = { sin, cos, tan, sqrt, log, log10, log2,
    exp, my_pow, my_expy, my_const };

// FUNCTION DEFINITIONS ========================================================
                                               
int main(int argc, char *argv[])                 
{                                                  
    srand(time(NULL));                         
                                               
    init_env(argc, argv);

    run(argc, argv);

    calculate_deviations();

    print_outputs(argc, argv);

    return 0;
}

void init_env(int argc, char *argv[])
{
    double tmp = 0.0;
    if (argc < 5) {
        printf("Usage: %s LOW_BOUND UP_BOUND N f1(x) +- f2(x) +- ... +- fn(x)\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    LOWER_BOUND = atof(argv[1]);
    UPPER_BOUND = atof(argv[2]);

    if (LOWER_BOUND > UPPER_BOUND) {
        // Swap bounds
        tmp = LOWER_BOUND;
        LOWER_BOUND = UPPER_BOUND;
        UPPER_BOUND = tmp;
    }

    N_INTERVALS = atoi(argv[3]);
}

void calculate_deviations()
{
    double mean = 0.0;
    int i;

    mean = calculate_mean(results, NUM_METHODS);
    
    // Calculate deviations
    for (i = 0; i < NUM_METHODS; ++i) 
        results[i].deviation = calculate_deviation(mean, results[i].result);

}

void run(int argc, char *argv[])
{
    int i;
    int negative_func = 0;

    for (i = 4; i < argc; ++i) {
        if (is_sign(argv[i])) {
            // Sign argument
            negative_func = (*argv[i] == '+') ? 0 : 1;
        }
        else {
            calculate_function(LOWER_BOUND, UPPER_BOUND, argv[i], negative_func);
        }
    }
}


void calculate_function(double a, double b, char *function, int negative)
{
    Fun fun = NULL;
    double integral = 0.0;
    int i;

    fun = regex_match_function(function);
    if (isdigit(*function)) {
        apply_multiplier(function, fun);
    }
    else {
        MULTIPLIER = 1.0;
    }

    if (fun != NULL) {
        // Calculate integrals
        for (i = 0; i < NUM_METHODS; ++i) {
            integral = methods[i](LOWER_BOUND, UPPER_BOUND, 
                    fun, N_INTERVALS); 

            if (negative) {
                results[i].result -= (integral * MULTIPLIER); 
            }
            else {
                results[i].result += (integral * MULTIPLIER);
            }
                
        }
    }
    else {
        printf("Unknown function: %s\n", function);
    }
}

void apply_multiplier(char *function, Fun fun)
{
//    double old_multiplier = 0.0;
    size_t len;
    int i;
    char *tmp; 

    if (fun != my_const) {
        len = strlen(function);            
        tmp = malloc(sizeof(char) * (len + 1));
        assert(tmp);
        tmp = strcpy(tmp, function);

        i = 0;
        while (isdigit(tmp[i]) || (tmp[i] == '.')) i++;
        tmp[i] = '\0';

//        old_multiplier = MULTIPLIER;
        MULTIPLIER = atof(tmp);
        
        free(tmp);
    }
}

Fun regex_match_function(char *function)
{
    regex_t regex;
    char msgbuf[100]; // In case of errors.
    int temp = 0;
    int i;
    Fun ret = NULL;

    for (i = 0; i < NUM_PATTERNS; ++i) {
        temp = regcomp(&regex, FUNCTION_PATTERNS[i], REG_ICASE | REG_EXTENDED);
        assert(!temp);

        temp = regexec(&regex, function, 0, NULL, 0);
        if (!temp) {
            ret = FUNCTIONS[i];

            if (i >= 8 && i <= 10) {
                process_stubs(i, function);
            }
        }
        else if (temp != REG_NOMATCH) {
            regerror(temp, &regex, msgbuf, sizeof(msgbuf));
            fprintf(stderr, "Regex for match %s function with pattern %s failed: %s\n",
                    function, FUNCTION_PATTERNS[i], msgbuf);
            
        }

        regfree(&regex);
    }

    return ret;
}

void process_stubs(int n, char *function)
{
    int i = 0;
    size_t len = strlen(function);
    char *tmp = malloc(sizeof(char) * (len + 1));
    assert(tmp);
    tmp = strncpy(tmp, function, len + 1);

    switch (n) {
    case 8:
        // x^y function
        while (tmp[i] != '^') i++;
        i++;
        MY_POW_EXP = atof(tmp + i);
        break;
    case 9:
        // y^x function
        while (tmp[i] != '^') i++;
        tmp[i] = '\0';
        MY_EXPY_BASE = atof(tmp);
        break;
    case 10:
        // const function
        MY_CONST = atof(function);
        break;
    }

    free(tmp);
}

void print_outputs(int argc, char *argv[])
{
    int i;

    // Print outputs
    printf("Integral between %.2f and %.2f of", 
            LOWER_BOUND, UPPER_BOUND);

    for (i = 4; i < argc; ++i) {
        if (is_sign(argv[i])) {
            printf(" %c", *argv[i]);
        }
        else {
            printf(" %s", argv[i]);
        }
    }
    printf("\n");

    for (i = 0; i < NUM_METHODS; ++i) 
        printf("%s: %.4f (%.2f%%)\n", METHOD_NAMES[i], results[i].result, results[i].deviation);

}


double rectangles_method(double a, double b, Fun fun, int n)
{
    double step = calculate_step(a, b, n);
    double integral = 0;
    int i;

    for (i = 0; i < n; ++i) {
        integral += (fun(a + (i * step)) * step); 
    } 

    return integral;
}

double trapeze_method(double a, double b, Fun fun, int n)
{
    double integral = 0.0;
    double step = calculate_step(a, b, n);
    double *x = make_vector(a, b, n);
    double sum = 0.0;
    size_t len = n + 1; // make_vector()'s return size is always n + 1
    int i;

    for (i = 1; i < (len - 1); ++i) {
        sum += fun(x[i]);
    }

    sum += (fun(a)/2.0 + fun(b)/2.0);
    integral = sum * step;

    free(x);
    return integral;
}

double simpsons_method(double a, double b, Fun fun, int n)
{
    double step = calculate_step(a, b, n);
    double *x = make_vector(a, b, n);
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum = 0.0;
    double integral = 0.0;
    size_t len = n + 1; // make_vector()'s return size is always n + 1
    int i;

    for (i = 1; i < (len - 1); ++i) {
        if ((i & 2) == 0) {
            sum1 += fun(x[i]);
        }
        else {
            sum2 += fun(x[i]);
        }
    }

    sum1 *= 4;
    sum2 *= 2;

    sum = sum1 + sum2 + fun(a) + fun(b);
    integral = (step/3) * sum;

    free(x);
    return integral;
}

double monte_carlo_method(double a, double b, Fun fun, int n)
{
    double integral = 0.0;
    double *x = make_vector(a, b, n);
    double new_a = a;
    int i = 0;

    if (sign(fun(x[i])) == 0) i++;

    for (; i < n; ++i) {
        // If the function crosses x-axis, we have to divide area
        if (sign(fun(x[i])) != sign(fun(x[i + 1]))) {
            integral += _monte_carlo_method(new_a, x[i], fun, n);

            i++;
            new_a = x[i];
        }
    }

    integral += _monte_carlo_method(new_a, x[i], fun, n);

    free(x);

    return integral;
}

double _monte_carlo_method(double a, double b, Fun fun, int n)
{
    double integral = 0.0;
    double *x = make_vector(a, b, n);
    double fmax; 
    double fmin;
    double randX = 0.0;
    double randY = 0.0;
    int hits = 0;
    int i = 0;
    int fun_sign = sign(fun(a));

    while ((fun_sign == 0) && (i < n)) { 
        fun_sign = sign(fun(x[i]));
        i++;
    }

    if (i == n)
        return 0.0; // f(x) = 0

    if (fun_sign == 1)
        fmax = max(x, fun, n + 1);
    else
        fmin = min(x, fun, n + 1);


    for (i = 0; i < MONTE_CARLO_SHOTS; ++i) {
        // rand()/(RAND_MAX/a) => random number [0..a]
        randX = (double)rand()/(double)(RAND_MAX/(b - a));
        randX += a;

        if (fun_sign == 1) {
            randY = (double)rand()/(double)(RAND_MAX/fmax);

            if (fun(randX) >= randY)
                hits++;
        }
        else {
            // To generate [-a..0] -> generate [0..a] and substract a
            // Note: fmin is negative, so opposite sign
            randY = (double)rand()/(double)(RAND_MAX/(-fmin));
            randY += fmin;

            //printf("%.4f randY %.4f fun(randX) %.4f randX\n", randY, fun(randX), randX);

            if (fun(randX) <= randY)
                hits++;
        }
    }

    integral = ((double)hits/(double)MONTE_CARLO_SHOTS);

    if (fun_sign == 1) 
        integral *= ((b - a) * fmax);
    else
        integral *= ((b - a) * fmin);

    free(x);

    return integral;
}



double calculate_step(double a, double b, int n)
{
    return (b - a) / n;
}

double *make_vector(double a, double b, int n)
{
    int i;
    double step = calculate_step(a, b, n);
    double *ret = malloc(sizeof(double) * (n + 1));
    
    for (i = 0; i <= n; ++i) {
        ret[i] = a + (i * step);
    }
    return ret;
}

double calculate_deviation(double mean, double value)
{
    double nomin = fabs(value - mean);
    double denomin = mean;

    return (nomin / denomin) * 100;
}

double max(double *v, Fun fun, int size)
{
    double res = 0.0;
    int i;

    res = fun(v[0]);
    for (i = 1; i < size; ++i)
        if (res < fun(v[i]))
            res = fun(v[i]);

    return res;
}

double min(double *v, Fun fun, int size)
{
    double res = 0.0;
    int i;

    res = fun(v[0]);
    for (i = 1; i < size; ++i)
        if (res > fun(v[i]))
            res = fun(v[i]);

    return res;
}

double calculate_mean(Result *v, int size)
{
    double res = 0.0;
    int i;

    for (i = 0; i < size; ++i) {
        res += v[i].result;
    }
    
    return (res / size);
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

int is_sign(char *s)
{
    int ret = 0;

    if (*s == '+' || *s == '-')
        ret = 1;

    return ret;
}

int sign(double a)
{
    int ret = 0;
    
    if (a > 0)
        ret = 1;
    else if (a < 0)
        ret = -1;

    return ret;
}
