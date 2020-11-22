import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from offlineTraining import offline_training


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
                        required=False,
                        default=100,
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
                        default=5000,
                        help='Size of the test batch')

    parser.add_argument('--data',
                        action='store_true',
                        help='Use dataset to train/test')

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

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Ignore if you do not have multiple GPUs
    return args


LINEAR_LAYERS = ["MMNet", "MMNet_iid", "Ht", "lin_DetNet", "identity", "OAMPNet"]
DENOISER_LAYERS = ["gaussian_test", "DetNet", "MMNet", "OAMPNet", "identity", "naive_nn", "featurous_nn"]


def main(args):
    args.linear = "MMNet_iid"
    for denoiser in DENOISER_LAYERS:
        args.denoiser = denoiser
        print("EXPERIMENT: linear=%s denoiser=%s" % (args.linear, args.denoiser))
        try:
            offline_training(args)
        except Exception as e:
            print("FAILED EXPERIMENT linear=%s denoiser=%s" % (args.linear, args.denoiser))
            print(e)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
