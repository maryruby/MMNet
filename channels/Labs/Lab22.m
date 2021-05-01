close all
clc
clear all

h0 = 0.7 ;
h1 = 0.7;
h2 = -0.5;
h3 = 0.4;
h = [h0 h1 h2 h3]; 

N = 2048;
std = 0.1;
s = randsrc(1, N, [-1 1]);

x = filter(h, 1, s);

Ps = mean(abs(x).^2); 
xn = awgn(x, 10*log10((1/std)^2), 10*log10(Ps));
x=firwiener(3,x,xn)
delay = N-255;
k = delay:N;
ke = 1:100;
for l = [4 8]
l_fil = l;
mu_max = 2/(mean(abs(s).^2)*l_fil);
disp(mu_max);
e_fil = zeros(1, length(ke));
for n = 1:ke(end)
ha = dsp.LMSFilter(l, 'StepSize', mu_max*n/ke(end));
y = zeros(N, 1);
e = zeros(N, 1);
w = zeros(1, l_fil);
[y, e, w] = step(ha, s.', xn.');
e_fil(n) = mean((abs(e(k, :).')).^2);
if mu_max*n/ke(end) == mu_max/2
w.'
figure
plot(e)'
title('Error with filter length L =', num2str(l));
ylabel('error signal'); xlabel('number of symbol');
end
end

figure
semilogy(mu_max*ke/ke(end), e_fil)
title('Error with filter length L =', num2str(l));
ylabel('filtration error');
xlabel('μ');
grid on

end



E_fil = zeros(1, 21);
for z = 0:20
S = [zeros(1, z) s];
xn_s = [xn zeros(1, z)];
l_fil = 16;
mu_max = 2/(mean(abs(xn).^2)*l_fil);
e_fil = zeros(1, length(ke));
if z == 9
for n = 1:ke(end)
ha = dsp.LMSFilter(l_fil, 'StepSize', mu_max*n/ke(end));
y = zeros(N+z, 1);
e = zeros(N+z, 1);
w = zeros(1, l_fil);
[y, e, w] = step(ha, xn_s.', S.');
e_fil(n) = mean((abs(e(k, :).')).^2);
if mu_max*n/ke(end) == mu_max/2
w.'
figure
impz(w)
figure
plot(y, '.')
ylabel('error signal, delay = 9');xlabel('number');
grid on
figure
plot(e)
ylabel('error signal, delay = 9');xlabel('number');
end
end
figure
semilogy(mu_max*ke/ke(end), e_fil)
ylabel('error signal, delay = 9'); xlabel('μ');
grid on
end
mu = mu_max/2;
ha = dsp.LMSFilter(l_fil, 'StepSize', mu);
y = zeros(N+z, 1);
e = zeros(N+z, 1);
w = zeros(1, l_fil);
[y, e, w] = step(ha, xn_s.', S.');
E_fil(z+1) = mean((abs(e(k, :).')).^2);
end

figure
semilogy([0:20], E_fil)
ylabel('error signal'); xlabel('delay');
grid on
