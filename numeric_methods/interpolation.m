clear all
X = [1 3 2 7]; % input('Input X: ');
Y = [6 4 8 2]; % input('Input Y: ');

n = length(X);
V = zeros(n);

% Calculate Vardermond's matrix
for i = 1 : n
    for j = 1 : n
        V(i, j) = X(i)^(j-1);
    end
end

% Print Vardermond's matrix
V;

% Calculate attributes matrix
A = inv(V) * Y';

% Print A matrix
A;

% A1 = polyfit(X, Y, n - 1)

% Draw points f(X) = Y
plot(X, Y, 'or')
hold on

x = min(X) : 0.1 : max(X);

W = 0;
for k = 1 : n
    W = W + A(k)*x.^(k - 1);
end

% Draw plot
plot(x, W)

% Set proper name to the plot
tit = ('W(x) = ');
for i = 1 : n - 1
    A1 = num2str(A(i));
    tit = [tit A1];
    tit = [tit 'x\^'];
    tit = [tit num2str(i - 1)];
    tit = [tit ' + '];
end

A1 = num2str(A(n));
tit = [tit A1];
tit = [tit 'x\^' num2str(n - 1)];
title(tit)
