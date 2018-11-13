clear all

fun = 'x.^3 + x.^2 - (20 * x)';
f = inline(fun);
eps = 0.01;
a = 0;
b = 6;
n = 0;

while ((b - a) > eps)
    L = b - a;

    x1 = a + 0.382 * L;
    x2 = a + 0.618 * L;
    xm = (a + b) / 2;
    
    if (f(x1) < f(x2))
        b = x2;
    else
        a = x1;
    end
    n = n + 1;
end

disp(xm)
disp(n)