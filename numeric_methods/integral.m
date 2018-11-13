clear all

a = 0;
b = 10;
n = 10;
h = ((b-a)/n);
x = a : h : b;
l = length(x);
F = 'x.^cos(x)';
f = inline(F);

% Rectangle's method
i1 = h * (sum(f(x)) - f(b));
disp('Rectangles method');
disp(i1);

% Trapezow method
sum = 0;
for i = 2 : (l - 1)
    sum = sum + f(x(i));
end
sum = sum + (f(a)/2) + (f(b)/2);
i2 = sum * h;

disp('Trapezow method');
disp(i2);

% Simpson's method
sum1 = 0;
sum2 = 0;
for i = 2 : (l - 1)
    if (mod(i, 2) == 0)
       sum1 = sum1 + f(x(i));
    else
       sum2 = sum2 + f(x(i));
    end
end
sum1 = sum1 * 4;
sum2 = sum2 * 2;

sum = sum1 + sum2 + f(a) + f(b);
i3 = (h/3) * sum;

disp('Simpsons method');
disp(i3);

% quad() function
i4 = quad(f, a, b);
disp('quad() function');
disp(i4);

% Monte Carlo method
plot(x, f(x));
hold on
shots = 500;
hit = 0;
fmax = max(f(x));
for i = 1 : shots
    randX = rand() * (b - a);
    randY = rand() * fmax;
    if (f(randX) >= randY)
        % If the point is under the function
        hit = hit + 1;
        plot(randX, randY, 'og')
        hold on
    else
        plot(randX, randY, 'or')
        hold on
    end
end
% Number of shots that fit the area under the function
% Multiplied by the area of rectangle 
% x=(a, b); y=(fmin, fmax)
i5 = (hit/shots) * ((b - a) * fmax) ;
disp('Monte Carlo method');
disp(i5);

hold off
