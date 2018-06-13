#!/bin/usr/python
import random


MONTE_CARLO_SHOTS = 100000;



def calculate_step(a, b, n):
    return (b - a) / n;



def make_vector(a, b, n):
    step = calculate_step(a, b, n);
    ret = [];

    for i in range(0, n + 1):
        ret.append(a + (i * step));

    return ret;



def sign(a):
    if (a < 0):
        return -1
    elif (a > 0):
        return 1;
    else:
        return 0;



def my_max(v, fun, size):
    res = 0.0;

    res = fun(v[0]);
    for i in range(1, size - 1):
        if (res < fun(v[i])):
            res = fun(v[i]);

    return res;



def my_min(v, fun, size):
    res = 0.0;

    res = fun(v[0]);
    for i in range(1, size - 1):
        if (res > fun(v[i])):
            res = fun(v[i]);

    return res;



def rectangles_method(a, b, fun, n):
    step = calculate_step(a, b, n);
    integral = 0.0;

    for i in range(0, n):
        integral += (fun(a + (i * step)) * step);

    return integral;



def trapeze_method(a, b, fun, n):
    integral = 0.0;
    step = calculate_step(a, b, n);
    v = make_vector(a, b, n);
    sum = 0.0;
    len = n + 1;

    for i in range(1, len - 1):
        sum += fun(v[i]);

    sum += (fun(a)/2.0 + fun(b)/2.0);
    integral = sum * step;

    return integral;



def simpsons_method(a, b, fun, n):
    step = calculate_step(a, b, n);
    v = make_vector(a, b, n);
    sum1 = 0.0;
    sum2 = 0.0;
    sum = 0.0;
    integral = 0.0;
    len = n + 1;

    for i in range(1, len - 1):
        if ((i & 1) == 0):
            sum1 += fun(v[i]);
        else:
            sum2 += fun(v[i]);

    sum1 *= 4;
    sum2 *= 2;

    sum = sum1 + sum2 + fun(a) + fun(b);
    integral = (step/3) * sum;

    return integral;



def monte_carlo_method(a, b, fun, n):
    random.seed()
    integral = 0.0;
    v = make_vector(a, b, n);
    new_a = a;
    i = 0;

    if (sign(fun(v[i])) == 0):
        i += 1;

    while (i < n):
        # If the function crosses x-axis, we have to divide area
        if (sign(fun(v[i])) != sign(fun(v[i + 1]))):
            integral += _monte_carlo_method(new_a, v[i], fun, n);

            i += 1;
            new_a = v[i];
        i += 1;


    integral += _monte_carlo_method(new_a, v[i], fun, n);

    return integral;



def _monte_carlo_method(a, b, fun, n):
    integral = 0.0;
    v = make_vector(a, b, n);
    randX = 0.0;
    randY = 0.0;
    hits = 0;
    i = 0;
    fun_sign = sign(fun(a));

    while ((fun_sign == 0) and (i < n)):
        fun_sign = sign(fun(v[i]));
        i += 1;


    if (i == n):
        return 0.0; # f(x) = 0

    if (fun_sign == 1):
        fmax = my_max(v, fun, n + 1);
    else:
        fmin = my_min(v, fun, n + 1);


    for i in range(0, MONTE_CARLO_SHOTS):
        randX = random.uniform(a, b)

        if (fun_sign == 1):
            randY = random.uniform(0, fmax)

            if (fun(randX) >= randY):
                hits += 1;

        else:
            # To generate [-a..0] -> generate [0..a] and substract a
            # Note: fmin is negative, so opposite sign
            randY = random.uniform(fmin, 0)

            if (fun(randX) <= randY):
                hits += 1;


    integral = (hits/float(MONTE_CARLO_SHOTS));

    if (fun_sign == 1):
        integral *= ((b - a) * fmax);
    else:
        integral *= ((b - a) * fmin);

    return integral;
