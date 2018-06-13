#!/bin/usr/python
import math

MY_POW_EXP = 0.0
MY_EXPY_BASE = 0.0
MY_CONST = 0.0
MULTIPLIER = 0.0

def my_pow(x):
    return math.pow(x, MY_POW_EXP);


def my_expy(x):
    return math.pow(MY_EXPY_BASE, x);


def my_const(x):
    global MULTIPLIER
    MULTIPLIER = 1.0;
    return MY_CONST;


def my_sin(x):
    return math.sin(x);


def my_cos(x):
    return math.cos(x);


def my_tan(x):
    return math.tan(x);


def my_sqrt(x):
    return math.sqrt(x);


def my_log(x):
    return math.log(x);


def my_log10(x):
    return math.log10(x);


def my_log2(x):
    return math.log2(x);


def my_expe(x):
    return math.exp(x);

