import tensorflow.compat.v1 as tf
import numpy as np
import scipy


def make_random_complex_matrix(Nt, Nr, Nsamples):
    """result shape is (Nsamples, Nr * Nt)"""
    Hr = tf.random_normal(shape=[Nr * Nt, Nsamples], dtype=tf.float32)
    Hi = tf.random_normal(shape=[Nr * Nt, Nsamples], dtype=tf.float32)
    return Hr, Hi


def make_temporal_correlations(Nsamples):
    fdT = 4e-3
    z = 2 * np.pi * fdT * np.arange(Nsamples)
    r = scipy.special.j0(z)  # Bessel function of the first kind of zero order
    R_tau = scipy.linalg.toeplitz(r)
    return R_tau


def make_cross_antenna_correlations(Nt, Nr):
    alpha = 0.3
    Rtx = np.power(alpha, (np.arange(Nt) / (Nt - 1.)) ** 2)
    RTX = scipy.linalg.toeplitz(Rtx)
    beta = 0.9
    Rrx = np.power(beta, (np.arange(Nr) / (Nr - 1.)) ** 2)
    RRX = scipy.linalg.toeplitz(Rrx)
    return np.kron(RTX, RRX)


def prepare_correlation_transform(R, name):
    D, V, W = scipy.linalg.eig(R, left=True)
    D = D.real*np.eye(D.shape[0])
    D_poz = np.maximum.reduce([D, np.zeros(D.shape)])
    A = np.matmul(V, np.sqrt(D_poz))
    return tf.Variable(A, trainable=False, name=name, dtype=tf.float32)


def generate_correlated_matrix(Nt, Nr, Nsamples):
    """result shape is (Nsamples, Nr, Nt)"""

    # make iid matrix
    Xr, Xi = make_random_complex_matrix(Nt, Nr, Nsamples)  # shape = [Nr * Nt, Nsamples]
    # make transforms for correlations
    temporal_corr = prepare_correlation_transform(make_temporal_correlations(Nsamples), "temporal_correlations")
    antenna_corr = prepare_correlation_transform(make_cross_antenna_correlations(Nt, Nr), "cross_antenna_correlations")

    # apply correlation bw antennas
    Xr_, Xi_ = tf.matmul(antenna_corr, Xr), tf.matmul(antenna_corr, Xi)
    # apply temporal correlation
    Xr__, Xi__ = tf.matmul(temporal_corr, tf.transpose(Xr_)), tf.matmul(temporal_corr, tf.transpose(Xi_))
    return Xr__, Xi__
