#!/bin/usr/python
import random


MONTE_CARLO_SHOTS = 100000;


class Method(object):
    def __init__(self, a, b, n):
        self._a = a
        self._b = b
        self._n = n
        self._step = self.calculate_step()
        pass

    def make_vector(self):
        self._step = (self._b - self._a) / self._n;
        ret = [];

        for i in range(0, self._n + 1):
            ret.append(self._a + (i * self._step));

        return ret;

    def sign(self, a):
        if (a < 0):
            return -1
        elif (a > 0):
            return 1;
        else:
            return 0;

    def calculate_step(self):
        return (self._b - self._a) / self._n;

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

    def Calculate(self, fun):
        raise NotImplementedError("Subclass must override this method!")
        pass


class RectanglesMethod(Method):
    def __init__(self, a, b, n):
        super(RectanglesMethod, self).__init__(a, b, n)

    def Calculate(self, fun):
        integral = 0.0;

        for i in range(0, self._n):
            integral += (fun(self._a + (i * self._step)) * self._step);

        return integral;


class TrapezeMethod(Method):
    def __init__(self, a, b, n):
        super(TrapezeMethod, self).__init__(a, b, n)

    def Calculate(self, fun):
        integral = 0.0;
        v = super(TrapezeMethod, self).make_vector();
        sum = 0.0;
        len = self._n + 1;

        for i in range(1, len - 1):
            sum += fun(v[i]);

        sum += (fun(self._a)/2.0 + fun(self._b)/2.0);
        integral = sum * self._step;

        return integral;


class SimpsonsMethod(Method):
    def __init__(self, a, b, n):
        super(SimpsonsMethod, self).__init__(a, b, n)

    def Calculate(self, fun):
        v = super(SimpsonsMethod, self).make_vector();
        sum1 = 0.0;
        sum2 = 0.0;
        sum = 0.0;
        integral = 0.0;

        for i in range(1, self._n):
            if ((i & 1) == 0):
                sum1 += fun(v[i]);
            else:
                sum2 += fun(v[i]);

        sum1 *= 4;
        sum2 *= 2;

        sum = sum1 + sum2 + fun(self._a) + fun(self._b);
        integral = (self._step/3) * sum;

        return integral;


class MonteCarloMethod(Method):
    def __init__(self, a, b, n):
        super(MonteCarloMethod, self).__init__(a, b, n)

    def __monte_carlo_method(self, fun):
        integral = 0.0;
        v = super(MonteCarloMethod, self).make_vector();
        randX = 0.0;
        randY = 0.0;
        hits = 0;
        i = 0;
        fun_sign = super(MonteCarloMethod, self).sign(fun(self._a));

        while ((fun_sign == 0) and (i < self._n)):
            fun_sign = super(MonteCarloMethod, self).sign(fun(v[i]));
            i += 1;


        if (i == self._n):
            return 0.0; # f(x) = 0

        if (fun_sign == 1):
            fmax = super(MonteCarloMethod, self).my_max(v, fun, self._n + 1);
        else:
            fmin = super(MonteCarloMethod, self).my_min(v, fun, self._n + 1);


        for i in range(0, MONTE_CARLO_SHOTS):
            randX = random.uniform(self._a, self._b)

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
            integral *= ((self._b - self._a) * fmax);
        else:
            integral *= ((self._b - self._a) * fmin);

        return integral;


    def Calculate(self, fun):
        random.seed()
        integral = 0.0;
        v = super(MonteCarloMethod, self).make_vector();
        new_a = self._a;
        i = 0;

        if (super(MonteCarloMethod, self).sign(fun(v[i])) == 0):
            i += 1;

        while (i < self._n):
            # If the function crosses x-axis, we have to divide area
            if (super(MonteCarloMethod, self).sign(fun(v[i])) !=
                    super(MonteCarloMethod, self).sign(fun(v[i + 1]))):
                self._b = v[i]
                integral += self.__monte_carlo_method(fun);

                i += 1;
                new_a = v[i]
                self._a = v[i];
            i += 1;

        self._b = v[i]
        self._a = new_a

        integral += self.__monte_carlo_method(fun);

        return integral;

