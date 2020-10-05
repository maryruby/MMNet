clear all

NT = 4; % number of transmintting antennas
NR = 4; % numner of receiving antennas
M = 4; % constellation size
m = log2(M); % number of bits per symbol (for one antenna)
% channel matrix
H = [     0.4250 - 0.2708i  -1.1116 + 0.4020i  -0.3511 + 0.8706i  -0.2075 - 0.6166i
    -1.8547 + 0.4397i  -0.3172 + 0.5392i   1.3820 + 0.3308i   0.3403 - 0.6986i
    0.8586 - 0.5061i   0.8379 - 0.7336i  -0.8607 - 1.3479i  -0.4981 - 1.4236i
    0.3064 - 0.1843i   2.4265 - 0.2684i   0.2069 + 1.5479i  -1.4214 - 0.1096i
    ];
H = H/sqrt(sum(abs(H(:)).^2)/NT); % channel gain normalization

H_r = real(H);
H_i = imag(H);
hdf5write('/Users/mary/mimo/channel_sequences.hdf5', 'H_r', H_r, 'H_i', H_i)
% 
% K = 1e5; % number of symbols 
% qdb = 0:20; % Eb/N0 in dB
% A = modnorm(qammod(0:M-1, M), 'avpow', 1)/sqrt(NT); % power normalization factor
% 
% % set of reference signals for ML algorithm
% [st1, st2, st3, st4] = ndgrid(qammod(0:M-1, M, 0, 'gray')*A);
% st1 = st1(:).';
% st2 = st2(:).';
% st3 = st3(:).';
% st4 = st4(:).';
% st = [st1;st2;st3;st4];
% sr = H*st; % set of possible received signals (columns)
% 
% yt_est = zeros(NT, K);
% ber = zeros(size(qdb));
% berb = zeros(m*NT, length(qdb));
% for kq = 1:length(qdb)
%     x = randi([0 M-1], NT, K); % transmitted symbols (integer values 0...M-1)
%     b = reshape(de2bi(x(:))', m*NT, K);
%     s = qammod(x, M, 0, 'gray')*A; % modulation
%     sh = H*s; % channel
%     y = awgn(sh, qdb(kq)+10*log10(m*NT)); % AWGN
%     % ML receiver
%     for kk = 1:K % loop over all possible signals
%         d = bsxfun(@minus, sr, y(:, kk));
%         e = sum(abs(d).^2); % squared Euclidean distances
%         [~, ind] = min(e); % look for the most close combination
%         yt_est(:,kk) = st(:, ind); % estimate of transmitted combination
%     end
%     z = qamdemod(yt_est/A, M, 0, 'gray'); % demodulation
%     [~, ber(kq)] = biterr(x, z, m);
%     %xb = reshape(de2bi(reshape(x',[],1),m), m*NT, [])';
%     zb = reshape(de2bi(z(:))', m*NT, K);
%     [~, berb(1:m*NT,kq)] = symerr(b, zb, 'row-wise');
%     fprintf('SNR = %g dB, BER = %g\n', qdb(kq), ber(kq))
%     if ber(kq)==0
%         break
%     end
% end
% 
% semilogy(qdb, ber, 'LineWidth', 2)
% hold on
% semilogy(qdb, berb')
% hold off
% grid on
% xlabel('SNR, dB')
% ylabel('BER')



