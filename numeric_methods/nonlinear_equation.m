clear all

% Note : for e^x use exp(x) function
F = 'exp(x) + x.^2 - 2'; %F = input('Input F(x) : ', 's');
% Convert string to inline function
f = inline(F)

% Preapre env for plot drawing
x = -2 : 0.01 : 2;
n = length(x);
y = zeros(n);

for i = 1 : n
    y(i) = f(x(i));
end

% Draw a plot
plot(x, y)

% Input a and b regardng the plot drawn to have proper results
in_a = input('Input a : ');
in_b = input('Input b : ');
a = in_a;
b = in_b;

% Error margin
E = 0.01;

% Half-cutting method
while (b - a >= E)
    Xm = (a + b) / 2;
    if ((f(a) * f(Xm)) < 0)
        b = Xm;
    else
        a = Xm;
    end
end
Xm = (a + b) / 2;
X = Xm;

% Print out the result  
disp('Result of Half-cutting method: ')
disp(X);

% FALSI METHOD
a = in_a;
b = in_b;

x = zeros(n);
i = 1;
x(i) = ((a*f(b)) - (b*f(a))) / (f(b) - f(a));

while (abs(x(i + 1) - x(i)) >= E)
    if (f(a)*f(x(i)) < 0)
        x(i + 1) = ((x(i)*f(a) - a*f(x(i))) / (f(a) - f(x(i))));
    end
    if (f(b)*f(x(i)) < 0)
        x(i + 1) = ((x(i)*f(b) - b*f(x(i))) / (f(b) - f(x(i))));
    end
    i = i+1;
end
X = x(i - 1);

% Print out the result  
disp('Result of Falsi method: ')
disp(X)
    
    
    
    