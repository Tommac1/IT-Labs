clear all
X = [1 3 2 7 4 10 12 0]; % input('Input X: ');
Y = [6 4 8 2 2 11 10 3]; % input('Input Y: ');
P = 1;         % input('Input P: ');

n = length(X);

% Calculate psi matrix
psi = zeros(P + 1);
for k = 1 : n
    for i = 1 : (P + 1)
        for j = 1 : (P + 1)
            psi(i, j) =  psi(i, j) + X(k)^(i + j - 2);
        end
    end
end

% Calculate F matrix
F = zeros(1, P + 1);
for k = 1 : n
    for i = 1 : (P + 1)
        F(i) = F(i) + (Y(k) * X(k)^(i - 1));
    end
end

% Calculate attributes matrix
A = inv(psi) * F';

plot(X, Y, 'or')
hold on

x = min(X) : 0.1 : max(X);

W = 0;
for k = 1 : (P + 1)
    W = W + A(k)*x.^(k - 1);
end

% Draw plot
plot(x, W)

% Set proper name to the plot
tit = ('W(x) = ');
for i = 1 : (P)
    A1 = num2str(A(i));
    tit = [tit A1];
    tit = [tit 'x\^'];
    tit = [tit num2str(i - 1)];
    tit = [tit ' + '];
end

A1 = num2str(A(P + 1));
tit = [tit A1];
tit = [tit 'x\^' num2str(P)];
title(tit)