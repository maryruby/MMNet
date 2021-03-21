import tensorflow.compat.v1 as tf
import numpy as np
import scipy


def make_random_complex_matrix(Nt, Nr, Nsamples):
    """result shape is (Nsamples, Nr * Nt)"""
    Hr = tf.random_normal(shape=[Nsamples, Nr * Nt], stddev=1. / np.sqrt(2. * Nr), dtype=tf.float32)
    Hi = tf.random_normal(shape=[Nsamples, Nr * Nt], stddev=1. / np.sqrt(2. * Nr), dtype=tf.float32)
    return Hr, Hi


def make_temporal_correlations(Nsamples):
    fdT = 3e-4
    z = 2 * np.pi * fdT * np.arange(Nsamples)
    r = scipy.special.j0(z)  # Bessel function of the first kind of zero order
    t = scipy.linalg.toeplitz(r)
    return t


def make_cross_antenna_correlations(Nt, Nr):
    alpha = 0.01
    Rtx = np.power(alpha, (np.arange(Nt) / (Nt - 1.)) ** 2)
    RTX = scipy.linalg.toeplitz(Rtx)
    beta = 0.01
    Rrx = np.power(beta, (np.arange(Nr) / (Nr - 1.)) ** 2)
    RRX = scipy.linalg.toeplitz(Rrx)
    return np.kron(RRX, RTX)


def prepare_correlation_transform(R, name):
    V, D = np.linalg.eig(R)
    A = np.abs(V * np.sqrt(D))  # is that correct?
    return tf.Variable(A, trainable=False, name=name, dtype=tf.float32)


def generate_correlated_matrix(Nt, Nr, Nsamples):
    """result shape is (Nsamples, Nr, Nt)"""
    Xr, Xi = make_random_complex_matrix(Nt, Nr, Nsamples)

    temporal_corr = prepare_correlation_transform(make_temporal_correlations(Nsamples), "temporal_correlations")
    antenna_corr = prepare_correlation_transform(make_cross_antenna_correlations(Nt, Nr), "cross_antenna_correlations")

    Xr_, Xi_ = tf.matmul(temporal_corr, Xr), tf.matmul(temporal_corr, Xi)

    Xr__, Xi__ = tf.reshape(tf.matmul(Xr_, antenna_corr), (Nsamples, Nr, Nt)), \
                 tf.reshape(tf.matmul(Xi_, antenna_corr), (Nsamples, Nr, Nt))
    return Xr__, Xi__
