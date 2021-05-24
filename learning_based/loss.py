import tensorflow.compat.v1 as tf
from utils import batch_matvec_mul, demodulate, demodulate_raw


def loss_fun(xhat, x, constellation, params):
    loss_type = params.get("loss_type", "sum_layers")
    if loss_type == "sum_layers":
        return loss_sum_layers(xhat, x)
    elif loss_type == "mse":
        return loss_mse(xhat, x)
    elif loss_type == "softmax":
        return loss_softmax(xhat, x, constellation, params)
    else:
        raise Exception("Unknown type of loss")


def loss_softmax(xhat, x, constellation, params):
    with tf.name_scope("loss_softmax"):
        x_indices = demodulate(x, constellation)
        x_onehot = tf.one_hot(x_indices, params['K'])
        xhat_onehot = demodulate_raw(xhat[-1], constellation)
        lk = tf.losses.softmax_cross_entropy(onehot_labels=x_onehot, logits=xhat_onehot)
        tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
        tf.summary.scalar("total_loss", lk)
        return lk


def loss_mse(xhat, x):
    with tf.name_scope("loss_mse"):
        lk = tf.losses.mean_squared_error(labels=x, predictions=xhat[-1])
        tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
        tf.summary.scalar("total_loss", lk)
        return lk


def loss_sum_layers(xhat, x):
    with tf.name_scope("loss_sum_layers"):
        loss = 0.
        for i, xhatk in enumerate(xhat):
            lk = tf.losses.mean_squared_error(labels=x, predictions=xhatk)
            loss += lk
            tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
            tf.summary.scalar("loss_layer_%s" % i, lk)
        tf.summary.scalar("total_loss", lk)
        return loss


def loss_yhx(y, xhat, H):
    loss = 0.
    for xhatk in xhat:
        lk = tf.losses.mean_squared_error(labels=y, predictions=batch_matvec_mul(H, xhatk))
        loss += lk
        tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
    return loss
