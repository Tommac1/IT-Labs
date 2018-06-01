#ifndef MAIN_H
#define MAIN_H

#include "Utilities.h"

// DEFINES =====================================================================

typedef double (*Function)(double);
typedef double (*Integral)(double, double, Function, int);

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

// FUNCTION DEFINITIONS ========================================================

Function regex_match_function(char *function);
void calculate_function(double a, double b, char *function, int negative);
void apply_multiplier(std::string function, Function fun);
void init_env(int argc, char *argv[]);
void run(int argc, char *argv[]);

extern double my_pow(double);
extern double my_const(double);
extern double my_expy(double);

#endif /* MAIN_H */
