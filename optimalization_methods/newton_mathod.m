clear all

fun = 'x.^3 + x.^2 - (20 * x)';
f = inline(fun);
eps = 0.001;
h = 0.05;
a = 0;
b = 6;
n = 0;
xn = a;
xnp1 = b;
x = 0;
while (abs(xnp1 - xn) > eps)
   xn = xnp1;
   fp = ((f(xn + h) - f(xn))/h);
   fpp = ((f(xn + 2 * h) - 2 * f(xn + h) + f(xn))/(h.^2));
   xnp1 = xn - (fp / fpp);
   n = n + 1;
end

disp(xn)
disp(n)