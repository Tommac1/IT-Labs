#!/bin/usr/python
import random


MONTE_CARLO_SHOTS = 100000;


class Method(object):
    def __init__(self):
        pass

    def make_vector(self, a, b, n):
        step = (b - a) / n;
        ret = [];

        for i in range(0, n + 1):
            ret.append(a + (i * step));

        return ret;

    def sign(self, a):
        if (a < 0):
            return -1
        elif (a > 0):
            return 1;
        else:
            return 0;

    def calculate_step(self, a, b, n):
        return (b - a) / n;

    def my_max(self, v, fun, size):
        res = 0.0;

        res = fun(v[0]);
        for i in range(1, size - 1):
            if (res < fun(v[i])):
                res = fun(v[i]);

        return res;

    def my_min(self, v, fun, size):
        res = 0.0;

        res = fun(v[0]);
        for i in range(1, size - 1):
            if (res > fun(v[i])):
                res = fun(v[i]);

        return res;

    def Calculate(self, a, b, fun, n):
        raise NotImplementedError("Subclass must override this method!")
        pass


class RectanglesMethod(Method):
    def Calculate(self, a, b, fun, n):
        step = super(RectanglesMethod, self).calculate_step(a, b, n);
        integral = 0.0;

        for i in range(0, n):
            integral += (fun(a + (i * step)) * step);

        return integral;


class TrapezeMethod(Method):
    def Calculate(self, a, b, fun, n):
        integral = 0.0;
        step = super(TrapezeMethod, self).calculate_step(a, b, n);
        v = super(TrapezeMethod, self).make_vector(a, b, n);
        sum = 0.0;
        len = n + 1;

        for i in range(1, len - 1):
            sum += fun(v[i]);

        sum += (fun(a)/2.0 + fun(b)/2.0);
        integral = sum * step;

        return integral;


class SimpsonsMethod(Method):
    def Calculate(self, a, b, fun, n):
        step = super(SimpsonsMethod, self).calculate_step(a, b, n);
        v = super(SimpsonsMethod, self).make_vector(a, b, n);
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


class MonteCarloMethod(Method):
    def __monte_carlo_method(self, a, b, fun, n):
        integral = 0.0;
        v = super(MonteCarloMethod, self).make_vector(a, b, n);
        randX = 0.0;
        randY = 0.0;
        hits = 0;
        i = 0;
        fun_sign = super(MonteCarloMethod, self).sign(fun(a));

        while ((fun_sign == 0) and (i < n)):
            fun_sign = super(MonteCarloMethod, self).sign(fun(v[i]));
            i += 1;


        if (i == n):
            return 0.0; # f(x) = 0

        if (fun_sign == 1):
            fmax = super(MonteCarloMethod, self).my_max(v, fun, n + 1);
        else:
            fmin = super(MonteCarloMethod, self).my_min(v, fun, n + 1);


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


    def Calculate(self, a, b, fun, n):
        random.seed()
        integral = 0.0;
        v = super(MonteCarloMethod, self).make_vector(a, b, n);
        new_a = a;
        i = 0;

        if (super(MonteCarloMethod, self).sign(fun(v[i])) == 0):
            i += 1;

        while (i < n):
            # If the function crosses x-axis, we have to divide area
            if (super(MonteCarloMethod, self).sign(fun(v[i])) !=
                    super(MonteCarloMethod, self).sign(fun(v[i + 1]))):
                integral += self.__monte_carlo_method(new_a, v[i], fun, n);

                i += 1;
                new_a = v[i];
            i += 1;


        integral += self.__monte_carlo_method(new_a, v[i], fun, n);

        return integral;

