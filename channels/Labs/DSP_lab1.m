N = 4096; % number of frequencies
x = randn(N, 10); % discrete white noise

p1 = 0.996*exp(0.25*1i*pi);
p2 = 0.999*exp(0.80*1i*pi);
p3 = conj(p1);
p4 = conj(p2);
p = [p1 p2 p3 p4]; % matrix of poles

[b,a] = zp2tf([],p,1); % matrix b and a for polinome
y = filter(b, a, x); % output signal
h = impz(b, a, N); % impulse characteristic
corr_h = xcorr(h); %cross-correlation
[H, f] = freqz(b, a, N, 2); % complex frequency characteristic
nf = normalize(f, 'range'); %normalize f

figure();
plot(nf, db(H)); % graph of theoretical magnitude response
title('Magnitude Response (dB)');
ylabel('Magnitude (dB)');
xlabel('Normalized frequency');
grid;

figure(); % diy periodogram
periodogram_whole = ((abs(fft(y))).^2)/N;
periodogramy = mean(periodogram_whole(1:(N/2), :), 2);
f_p = (0:N/2-1)/N*2;
plot(f_p, pow2db(periodogramy));
title('Periodogram from (abs(fft(y))^2)/N, (dB)');
ylabel('Averaged periodogram, dB/(rad/sample))');
xlabel('Normalized frequency');
grid;

%figure();
%pxx,w] = periodogram(y);
%plot(normalize(w, 'range'), db(mean(pxx, 2)));
%title('Periodogram from periodogram(y), (dB)');
%ylabel('Averaged periodogram, dB/(rad/sample))');
%xlabel('Normalized frequency');
%grid;

figure(); % Relative standart deviation of diy periodogram
msrange_p = (std(periodogram_whole(1:(N/2), :)'))'./periodogramy;
plot(f_p, msrange_p);
title('Relative standart deviation of periodogram');
ylabel('Relative standart deviation');
xlabel('Normalized frequency');
grid;

figure(); % daniell
dan_whole = filter(ones(1,5)/5, 1, periodogram_whole);
dan = mean(dan_whole(1:(N/2), :), 2);
plot(f_p, pow2db(dan));
title('Daniell periodogram, (dB)');
ylabel('Periodogram, dB/(rad/sample))');
xlabel('Normalized frequency');
grid;

figure(); % Relative standart deviation of daniell periodogram
msrange_pd = (std(dan_whole(1:(N/2), :)'))'./dan;
plot(f_p, msrange_pd);
title('Relative standart deviation of daniell periodogram');
ylabel('Relative standart deviation');
xlabel('Normalized frequency');
grid;

for i = [2048, 1024, 512, 256, 128]

    figure(); % welch 
    [welch_whole, f] = pwelch(y, i, [], [], 2);
    welch = mean(welch_whole, 2);
    plot(normalize(f, 'range'), pow2db(welch));
    titlegraph = strcat('Welch periodogram, ', ' ', num2str(i), '/', num2str(i/2));
    title(titlegraph);
    ylabel('Periodogram, dB/(rad/sample))');
    xlabel('Normalized frequency');
    grid;

    figure(); % Relative standart deviation of welch periodogram 
    msrange_p = (std(welch_whole')')./welch;
    plot(normalize(f, 'range'), msrange_p);
    titlegraph = strcat('Relative standart deviation of welch periodogram, ', num2str(i), '/', num2str(i/2));
    title(titlegraph);
    ylabel('Relative standart deviation');
    xlabel('Normalized frequency');
    grid;
    
end

figure(); % Autoregr of a single implementation
y_first = y(:, 1)';
for p = 1:8
    [autoregr, f] = pcov(y_first, p, [], 2);
    plot(f, pow2db(autoregr), 'Color', [0 (1-p/8) p/8]);
    hold on 
end
title('Autoregressive analysis of a single implementation');
ylabel('AA, dB/(rad/sample)');
xlabel('Normalized frequency');
legend('1', '2', '3', '4', '5', '6', '7', '8');
grid;

figure();  % Autoregr of 10 implementations
autor_matrix = zeros(129, 10);
for p = 1:10
    [autoregr, f] = pcov(y(:,p)', 4, [], 2);
    autor_matrix(:,p) = autoregr;
    plot(f, pow2db(autoregr), 'Color', [(1-p/10) 0 p/10]);
    hold on 
end
title('Autoregressive analysis of 10 implementations');
ylabel('AA, dB/(rad/sample)');
xlabel('Normalized frequency');
legend('1', '2', '3', '4', '5', '6', '7', '8', '9', '10');
grid;  
hold off

aut = mean(autor_matrix, 2);
figure();
plot(f, pow2db(aut));
title('Autoregressive analysis, mean, (dB)');
ylabel('Autoregressive analysis, dB/(rad/sample))');
xlabel('Normalized frequency');
grid;

figure(); % Relative standart deviation of autoregr
    msrange_aut = (std(autor_matrix')')./aut;
    plot(normalize(f, 'range'), msrange_aut);
    title('Relative standart deviation of regressive analysis');
    ylabel('Relative standart deviation');
    xlabel('Normalized frequency');
    grid;


