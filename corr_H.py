Nt = 16; #number of receivers
Nr = 64; #number of transmitter
N = 100; #number of samples

def make_random_complex_matrix(Nt, Nr, Nsamples):
    _size = Nt*Nr*Nsamples
    x = np.random.multivariate_normal([0,0], [[0.5,0],[0,0.5]], _size).reshape(2, _size)
    return np.vectorize(complex)(x[0,:], x[1, :]).reshape(Nt*Nr,Nsamples)
X = make_random_complex_matrix(Nt,Nr,N)

def make_temporal_correlations(Nsamples):
    fdT = 3e-4
    z = 2 * np.pi * fdT * np.arange(Nsamples)
    r = scipy.special.jv(0, z) # Bessel function of the first kind of real order and complex argument
    return scipy.linalg.toeplitz(r)

def make_cross_antenna_correlations(Nt, Nr):
    alpha = 0.01
    Rtx = np.power(alpha, (np.arange(Nt)/(Nt-1.))**2)
    RTX = scipy.linalg.toeplitz(Rtx)
    beta = 0.01
    Rrx = np.power(beta, (np.arange(Nr)/(Nr-1.))**2)
    RRX = scipy.linalg.toeplitz(Rrx)
    return np.kron(RTX, RRX)

def apply_correlation(X, R, rowwise=True):
    V, D = np.linalg.eig(R)
    A = V * np.sqrt(D)
    if rowwise:
        return np.matmul(X, A)
    else:
        return np.matmul(A, X)

X_ = apply_correlation(
                       apply_correlation(X, make_temporal_correlations(N)),
                       make_cross_antenna_correlations(Nt, Nr),
                       rowwise=False)