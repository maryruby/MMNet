%signal
N = 2048;
L = 4;
h = [0.7 0.7 -0.5 0.4];
std = 0.1;
signal = randsrc(1, N, [-1 1]);
x = filter(h, 1, d);
power = mean(abs(x).^2);
x_with_noise = awgn(x, 10*log10((1/std)^2), 10*log10(power));
plot(x_with_noise);

%channel
delay = N-255;
k = delay:N;
ke = 1:100;
for filter_length = [4 8]
mu_max = 2/(mean(abs(signal).^2)*filter_length);
e_fil = zeros(1, length(ke));
    for n = 1:ke(end)
    ha = dsp.LMSFilter(filter_length, 'StepSize', mu_max*n/ke(end));
    y = zeros(N, 1);
    e = zeros(N, 1);
    w = zeros(1, filter_length);
    [y, e, w] = step(ha, signal.', x_with_noise.');
    e_fil(n) = mean((abs(e(k, :).')).^2);
    if mu_max*n/ke(end) == mu_max/2
    w.'
    figure
    semilogy(mu_max*ke/ke(end), e_fil);
    title('Error with filter length L =', num2str(filter_length));
    ylabel('filtration error'); 
    xlabel('μ');
    grid;
    end
    end
end

% %[y, e, w_new] = ha(x, d); 
% y = zeros (N, 1);
% e = zeros (N, 1);
% w = zeros (L, N);
% for n = 1:N-delay
%     %[y(n), e(n), w(:, n)] = H.step(mu(delay+n), e(n));
%     ha = dsp.LMSFilter(4, 'StepSize', 0.3);
%     [y(n), e(n), w(:, n)] = H.step(ha, e(n));
%     %[y, e, w_new] = ha(x, d); 
%     subplot(2,1,2)
%     stem(w(1,n), 'filled', 'LineWidth',2)
%     title(['Time =' num2str(n)])
%     ylabel('(w(k))')
%     ylim([-0.4 0.4])
%     subplot(2,2,1)
%     plot(y, '.')
%     ylabel('y(k)')
%     subplot(2,2,2)
%     plot(e)
%     ylabel('e(k)')
%     drawnow
% end

