clear all

h = 0.1;
x = 0 : 0.1 : 10;
F = 'x.^cos(x)'; % input('Input F(x) : ');
f = inline(F);
n = length(x);

% Draw a plot #1
plot(x, f(x))
hold on

% Draw a plot #2
f1 = (((cos(x) ./ x) - (sin(x) .* log(x))) .* (x.^cos(x)));
plot(x, f1);
hold on

% Draw a plot #3
f1 = ((f(x + h) - f(x)) / h);
plot(x, f1);
hold on

% Draw a plot #4
f1 = ((f(x + h) - f(x - h)) / (2*h));
plot(x, f1);
hold on

% Draw a plot #5
f1 = ((1/(12*h))*(f(x-(2*h)) - (8*f(x-h)) + (8*f(x+h)) - f(x+(2*h))));
plot(x, f1);
hold on

legend('Function', 'Analitic', '2pkt', '3pkt', '5pkt');
