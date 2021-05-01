Nt = 4; %number of receivers
Nr = 6; %number of transmitter
N = 10;  %number of sample

H = complex(randn(Nr*Nt,N), randn(Nr*Nt,N))/sqrt(2); %Random complex H

DeltaK = 0:1:(N-1);
fdT = 3e-4;
z = 2*pi*fdT*DeltaK;
r = besselj(0, z);
T_coordinate = toeplitz(r);

DeltaK = 0:1:(Nt-1);
alpha = 0.01; %???
Rtx = alpha.^((DeltaK/(Nt-1)).^2);
RTX = toeplitz(Rtx);

DeltaK = 0:1:(Nr-1);
beta = 0.01;  %???
Rrx = beta.^((DeltaK/(Nr-1)).^2);
RRX = toeplitz(Rtx);
R = kron(RTX, RRX);

[V,D] = eig(R);
A = V*sqrt(D);

