clear all

fun = 'x.^3 + x.^2 - (20 * x)';
f = inline(fun);
eps = 0.01;
a = 0;
b = 6;
n = 0;

while ((b - a) > eps)
    L = b - a;

    x1 = a + 0.25 * L;
    x2 = b - 0.25 * L;
    xm = (a + b) / 2;
    
    if (f(x1) < f(x2))
        b = xm;
    elseif (f(xm) < f(x2))
        a = x1;
        b = x2;
    else
        a = xm;
    end
    n = n + 1;
end

disp(xm)
disp(n)