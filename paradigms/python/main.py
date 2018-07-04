#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import math
import my_funcs
import methods
import random


# GLOBAL DATA
# ======================================
NUM_METHODS = 4;

SIN_PATTERN = '^[\d|.]*sin\(x\)$';
COS_PATTERN = '^[\d|.]*cos\(x\)$';
TAN_PATTERN = '^[\d|.]*tan\(x\)$';
SQRT_PATTERN = '^[\d|.]*sqrt\(x\)$';
LN_PATTERN = '^[\d|.]*ln\(x\)$';
LOG10_PATTERN =	'^[\d|.]*log10\(x\)$';
LOG2_PATTERN = '^[\d|.]*log2\(x\)$';
EXP_PATTERN = '^[\d|.]*e\^x$';
POW_PATTERN = '^[\d|.]*x\^\d$';
EXPY_PATTERN = '^[\d|.]+\^x$';
CONST_PATTERN = '^[\d|.]+$';
NUM_PATTERNS = 11;

LOWER_BOUND = 0.0;
UPPER_BOUND = 0.0;
MY_POW_EXP = 0.0;
MY_EXPY_BASE = 0.0;
MY_CONST = 0.0;
MEAN = 0.0;
MULTIPLIER = 0.0;
N_INTERVALS = 100000;

METHOD_NAMES = [ 'Rectangles Method', 'Trapezes Method',
    'Simspons Method', 'Monte Carlos Method' ];

METHODS = {};

RESULTS = [
    { 'result': 0, 'deviation': 0 },
    { 'result': 0, 'deviation': 0 },
    { 'result': 0, 'deviation': 0 },
    { 'result': 0, 'deviation': 0 } ];

FUNCTION_PATTERNS = [ SIN_PATTERN, COS_PATTERN, TAN_PATTERN,
    SQRT_PATTERN, LN_PATTERN, LOG10_PATTERN, LOG2_PATTERN,
    EXP_PATTERN, POW_PATTERN, EXPY_PATTERN, CONST_PATTERN ];

FUNCTIONS = { 0: my_funcs.my_sin, 1: my_funcs.my_cos, 2: my_funcs.my_tan, 3: my_funcs.my_sqrt,
    4: my_funcs.my_log, 5: my_funcs.my_log10, 6: my_funcs.my_log2, 7: my_funcs.my_expe,
    8: my_funcs.my_pow, 9: my_funcs.my_expy, 10: my_funcs.my_const };


# FUNCTIONS
# =====================================
def main():
    ret = 0;

    ret = init_env(sys.argv);

    run(sys.argv);

    calculate_deviations();

    if (ret == 0):
        output = print_outputs(sys.argv);
    else:
        output = "Usage: lower_bound upper_bound interval f1(x) ± f2(x) ± ... ± fn(x)";

    print output
    return



def run(cmd_args):
    negative_func = False;

    for i in range(4, len(cmd_args)):
        if (is_sign(cmd_args[i])):
            negative_func = False if (cmd_args[i] == "+") else True;
        else:
            calculate_function(cmd_args[i], negative_func);


def regex_match_function(func):
    ret = None;


    for i in range(0, NUM_PATTERNS):
        regex = re.match(FUNCTION_PATTERNS[i], func, 0);

        if (regex):
            ret = FUNCTIONS[i];

            if (i >= 8 and i <= 10):
                process_stubs(i, func);

            break


    return ret;


def process_stubs(n, fun):
    tmp = re.match(r"^(-?\d+\.?\d*)", fun)

    if (tmp != None):
        if (n == 8):
            # x^y func
            my_funcs.MY_POW_EXP = float(str(tmp.group(1)));
        elif (n == 9):
            # y^x func
            my_funcs.MY_EXPY_BASE = float(str(tmp.group(1)));
        elif (n == 10):
            # const
            my_funcs.MY_CONST = float(str(tmp.group(1)));


def calculate_function(arg, negative):
    integral = 0.0;
    tmp = 0.0;

    fun = regex_match_function(arg);
    # tmp = float(arg);
    tmp = re.match(r"^(-?\d+\.?\d*)", arg)

    if (tmp != None):
        my_funcs.MULTIPLIER = float(str(tmp.group(1)));
    else:
        my_funcs.MULTIPLIER = 1.0;

    if (None != fun):
        # Calculate integrals
        for i in range(0, NUM_METHODS):
            integral = METHODS[i].Calculate(fun);

            if (negative):
                RESULTS[i]['result'] -= (integral * my_funcs.MULTIPLIER);
            else:
                RESULTS[i]['result'] += (integral * my_funcs.MULTIPLIER);

    else:
        print 'Unknown function: ' + arg;


def calculate_deviations():
    mean = 0.0;

    mean = calculate_mean(RESULTS, NUM_METHODS);

    for i in range(0, NUM_METHODS):
        RESULTS[i]['deviation'] = calculate_deviation(mean, RESULTS[i]['result']);



def init_env(cmd_args):
    global LOWER_BOUND
    global UPPER_BOUND
    ret = 0;

    if (len(cmd_args) < 4):
        ret = 1;
    else:
        LOWER_BOUND = float(cmd_args[1]);
        UPPER_BOUND = float(cmd_args[2]);

        if (UPPER_BOUND < LOWER_BOUND):
            # Swap bounds
            tmp = UPPER_BOUND;
            UPPER_BOUND = LOWER_BOUND;
            LOWER_BOUND = tmp;

        N_INTERVALS = int(cmd_args[3], 10);

        # fill the methods dictionary
        METHODS[0] = methods.RectanglesMethod(LOWER_BOUND, UPPER_BOUND, N_INTERVALS)
        METHODS[1] = methods.TrapezeMethod(LOWER_BOUND, UPPER_BOUND, N_INTERVALS)
        METHODS[2] = methods.SimpsonsMethod(LOWER_BOUND, UPPER_BOUND, N_INTERVALS)
        METHODS[3] = methods.MonteCarloMethod(LOWER_BOUND, UPPER_BOUND, N_INTERVALS)


    # clear the results structure
    for i in range(0, NUM_METHODS):
        RESULTS[i]['deviation'] = 0;
        RESULTS[i]['result'] = 0;


    return ret;


def print_outputs(cmd_args):
    ret = 'Integral between ' + str(LOWER_BOUND) + ' and ' + str(UPPER_BOUND) + ' of';

    for i in range(4, len(cmd_args)):
        ret += (" " + cmd_args[i]);

    ret += ':\n';

    for i in range(0, NUM_METHODS):
        ret += METHOD_NAMES[i] + ': ' + str(RESULTS[i]['result']) + ' (' + str(RESULTS[i]['deviation']) + '%)\n';


    return ret;



def calculate_deviation(mean, value):
    nomin = math.fabs(value - mean);
    denomin = mean;

    if (denomin == 0):
        return nomin * 100;
    else:
        return (nomin / denomin) * 100;




def calculate_mean(v, size):
    res = 0.0;

    for i in range(0, size):
        res += v[i]['result'];


    return (res / size);


def is_sign(s):
    if (('+' in s) or ('-' in s)):
        return 1;
    else:
        return 0;



if __name__ == '__main__':
    main()

