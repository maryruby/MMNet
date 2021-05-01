import os
from tf_session import *
import argparse
import tensorflow.compat.v1 as tf
from utils import model_eval, plot_result_graph, dump_result_to_file
from exp import get_data
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
tf.disable_v2_behavior()


def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi, Hr], axis=2)
    inp = np.concatenate([h1, h2], axis=1)
    return inp


def parse_args():
    parser = argparse.ArgumentParser(description='MIMO signal detection simulator')

    parser.add_argument('--x-size', '-xs',
                        type=int,
                        required=True,
                        help='Number of senders')

    parser.add_argument('--y-size', '-ys',
                        type=int,
                        required=True,
                        help='Number of receivers')

    parser.add_argument('--layers',
                        type=int,
                        required=True,
                        help='Number of neural net blocks')

    parser.add_argument('--snr-min',
                        type=float,
                        required=True,
                        help='Minimum SNR in dB')

    parser.add_argument('--snr-max',
                        type=float,
                        required=True,
                        help='Maximum SNR in dB')

    parser.add_argument('--learn-rate', '-lr',
                        type=float,
                        required=True,
                        help='Learning rate')

    parser.add_argument('--batch-size',
                        type=int,
                        required=True,
                        help='Batch size')

    parser.add_argument('--test-every',
                        type=int,
                        required=True,
                        help='number of training iterations before each test')

    parser.add_argument('--train-iterations',
                        type=int,
                        required=True,
                        help='Number of training iterations')

    parser.add_argument('--modulation', '-mod',
                        type=str,
                        required=True,
                        help='Modulation type which can be BPSK, 4PAM, or MIXED')

    parser.add_argument('--gpu',
                        type=str,
                        required=False,
                        default="0",
                        help='Specify the gpu core')

    parser.add_argument('--test-batch-size',
                        type=int,
                        required=True,
                        help='Size of the test batch')

    parser.add_argument('--data',
                        action='store_true',
                        help='Use dataset to train/test')

    parser.add_argument('--linear',
                        type=str,
                        required=True,
                        help='linear transformation step method')

    parser.add_argument('--denoiser',
                        type=str,
                        required=True,
                        help='denoiser function model')

    parser.add_argument('--loss-type',
                        type=str,
                        required=False,
                        help='Loss type',
                        choices=["sum_layers", "mse"],
                        default="sum_layers")

    parser.add_argument('--exp',
                        type=str,
                        required=False,
                        help='experiment name')

    parser.add_argument('--corr-analysis',
                        action='store_true',
                        help='fetch covariance matrices')

    parser.add_argument('--start-from',
                        type=str,
                        required=False,
                        default='',
                        help='Saved model name to start from')

    parser.add_argument('--log',
                        action='store_true',
                        help='Log data mode')

    parser.add_argument('--log-file',
                        help='File for result logging (append)')

    parser.add_argument('--correlated-h',
                        action='store_true',
                        help='Generate correlated H for both time and antennas')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Ignore if you do not have multiple GPUs
    return args


def offline_training(args):
    # Simulation parameters
    params = {
        'N': args.y_size,  # Number of receive antennas
        'K': args.x_size,  # Number of transmit antennas
        'L': args.layers,  # Number of layers
        'SNR_dB_min': args.snr_min,  # Minimum SNR value in dB for training and evaluation
        'SNR_dB_max': args.snr_max,  # Maximum SNR value in dB for training and evaluation
        'seed': 1,  # Seed for random number generation
        'batch_size': args.batch_size,
        'modulation': args.modulation,
        'correlation': False,
        'start_from': args.start_from,
        'data': args.data,
        'linear_name': args.linear,
        'denoiser_name': args.denoiser,
        'loss_type': args.loss_type,
        'use_correlated_H': args.correlated_h
    }

    if args.data:
        train_data_ref, test_data_ref, Hdataset_powerdB = get_data(args.exp)
        print(train_data_ref.shape)
        params['Hdataset_powerdB'] = Hdataset_powerdB

    # Build the computational graph
    mmnet = MMNet_graph(params)
    nodes = mmnet.build()

    # Get access to the nodes on the graph
    sess = nodes['sess']
    x = nodes['x']
    H = nodes['H']
    x_id = nodes['x_id']
    constellation = nodes['constellation']
    train = nodes['train']
    snr_db_min = nodes['snr_db_min']
    snr_db_max = nodes['snr_db_max']
    lr = nodes['lr']
    batch_size = nodes['batch_size']
    accuracy = nodes['accuracy']
    mmse_accuracy = nodes['mmse_accuracy']
    loss = nodes['loss']
    logs = nodes['logs']
    measured_snr = nodes['measured_snr']
    merged = nodes['merged']

    # Training loop
    record = {'before': [], 'after': []}
    record_flag = False

    if args.data:
        train_data = train_data_ref
        test_data = test_data_ref
    else:
        test_data = []
        train_data = []
    for it in range(args.train_iterations):
        feed_dict = {
            batch_size: args.batch_size,
            lr: args.learn_rate,
            snr_db_max: params['SNR_dB_max'],
            snr_db_min: params['SNR_dB_min'],
        }
        if args.data:
            sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
            feed_dict[H] = train_data[sample_ids]
        if record_flag:
            feed_dict_test = {
                batch_size: args.test_batch_size,
                lr: args.learn_rate,
                snr_db_max: params['SNR_dB_max'],
                snr_db_min: params['SNR_dB_min'],
            }
            if args.data:
                sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
                feed_dict[H] = test_data[sample_ids]
            before_acc = 1. - sess.run(accuracy, feed_dict_test)
            record['before'].append(before_acc)

        # summary, _  = sess.run([merged, train], feed_dict)
        H_generated = sess.run(H, feed_dict)
        print("H gen shape:", H_generated.shape)
        with open("H.csv", "w") as H_out:
            for sample_id in range(H_generated.shape[0]):
                for r in range(H_generated.shape[1]):
                    H_out.write(",".join(map(str, H_generated[sample_id, r, :])) + "\n")
        # np.savetxt("H.csv", H_generated.transpose(2, 0, 1).reshape(H_generated.shape[-1], -1).T, delimiter=",")
        with open("H.txt", 'w') as H_out:
            H_out.write(str(H_generated))
        sys.exit(1)
        mmnet.write_tensorboard_summary(summary, it, test=False)

        # Test
        if it % args.test_every == 0:
            feed_dict = {
                batch_size: args.test_batch_size,
                snr_db_max: params['SNR_dB_max'],
                snr_db_min: params['SNR_dB_max'],
            }
            if args.data:
                sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
                feed_dict[H] = test_data[sample_ids]
            if args.log:
                summary, test_accuracy_, test_loss_, logs_, x_, H_ = sess.run([merged, accuracy, loss, logs, x, H], feed_dict)
                mmnet.write_tensorboard_summary(summary, it, test=True)
                np.save('log.npy', logs_)
                break
            else:
                summary, test_accuracy_, test_loss_, measured_snr_ = sess.run([merged, accuracy, loss, measured_snr], feed_dict)
                print((it, 'SER: {:.2E}'.format(1. - test_accuracy_), test_loss_, measured_snr_))
                mmnet.write_tensorboard_summary(summary, it, test=True)
            if args.corr_analysis:
                log_ = sess.run(logs, feed_dict)
                for l in range(1, int(args.layers) + 1):
                    c = log_['layer' + str(l)]['linear']['I_WH']
                    print((np.linalg.norm(c, axis=(1, 2))[0]))

    result = model_eval(test_data,
                        params['SNR_dB_min'], params['SNR_dB_max'],
                        mmse_accuracy, accuracy, batch_size,
                        snr_db_min, snr_db_max,
                        H, sess,
                        n_samples=args.batch_size)
    plot_result_graph(result, args.x_size, args.y_size, args.modulation, args.linear, args.denoiser)
    dump_result_to_file(result, params, args.log_file)


if __name__ == "__main__":
    arguments = parse_args()
    offline_training(arguments)
