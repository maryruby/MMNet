function [S] = sum_of_row(n, x)
S = 0;
for i=0:(n-1)
    S = S + (-1)^n * sqrt((n+1)*x)/(n+1);
end
disp(S)

