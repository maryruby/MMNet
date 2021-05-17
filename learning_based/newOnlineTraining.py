import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
from tqdm import tqdm

from tf_session import *
import pickle
from parser import parse
from dataset import read_channels_dataset


if __name__ == '__main__':
    params, args = parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.data:
        train_data_ref, test_data_ref, power_db = read_channels_dataset(
            args.channels_dir,
            args.num_channel_samples,
            args.x_size, args.y_size,
            from_csv=args.load_data_from_csv,
            sequences_size=args.repeated_H_sequences)
        params['Hdataset_powerdB'] = power_db
    else:
        raise Exception("data is required here")
        # train_data_ref = []
        # test_data_ref = []

    # Build the computational graph
    mmnet = MMNet_graph(params)
    nodes = mmnet.build()

    # Get access to the nodes on the graph
    sess = nodes['sess']
    x = nodes['x']
    y = nodes['y']
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
    init = nodes['init']

    snrs = np.linspace(params['SNR_dB_min'], params['SNR_dB_max'],
                       round(params['SNR_dB_max'] - params['SNR_dB_min']) + 1)

    train_data_ref = train_data_ref.reshape(train_data_ref.shape[0] * train_data_ref.shape[1], train_data_ref.shape[2], train_data_ref.shape[3])

    test_results = []

    path = args.output_dir + '/OnlineTraining_%s_NT%sNR%s_%s_%s/' % (args.modulation, args.x_size, args.y_size, args.linear, args.denoiser)
    if not os.path.exists(path):
        os.makedirs(path)
    savePath = path + 'results1.json'
    with open(savePath, 'w') as output_fd:

        for snr in snrs:
            print("SNR:", snr)
            # Train
            print("TRAIN")
            with tqdm(total=train_data_ref.shape[0] * args.train_iterations) as progress_bar:
                for r in range(train_data_ref.shape[0]):
                    train_data = train_data_ref[r]
                    feed_dict = {
                        batch_size: args.batch_size,
                        lr: args.learn_rate,
                        snr_db_max: params['SNR_dB_max'],
                        snr_db_min: params['SNR_dB_min'],
                    }
                    train_H = np.zeros((args.batch_size, train_data.shape[0], train_data.shape[1]), dtype=train_data.dtype)
                    for i in range(args.batch_size):
                        train_H[i] = train_data.copy()
                    feed_dict[H] = train_H
                    for it in range(args.train_iterations):
                        sess.run(train, feed_dict)
                        progress_bar.update(1)
                # Test
                print("TEST")
                with tqdm(total=test_data_ref.shape[0] * test_data_ref.shape[1] * args.test_iterations * len(snrs)) as progress_bar:
                    for sequence in range(test_data_ref.shape[0]):
                        for r in range(test_data_ref.shape[1]):
                            test_data = test_data_ref[sequence][r]
                            feed_dict = {
                                batch_size: args.test_batch_size,
                                lr: args.learn_rate,
                                snr_db_max: snr,
                                snr_db_min: snr,
                            }
                            test_H = np.zeros((args.test_batch_size, test_data.shape[0], test_data.shape[1]), dtype=test_data.dtype)
                            for i in range(args.test_batch_size):
                                test_H[i] = test_data.copy()
                            feed_dict[H] = test_H

                            current_results = []
                            mmse_test_accuracy_, test_accuracy_, test_loss_, measured_snr_ = sess.run([mmse_accuracy, accuracy, loss, measured_snr], feed_dict)
                            test_results.append({"sequence": sequence, "h_in_sequence": r, "iteration": 0, "ser": float(1. - test_accuracy_), 'mmse': float(1. - mmse_test_accuracy_), "snr": snr, "measured_snr": float(measured_snr_), "loss": float(test_loss_)})
                            output_fd.write(json.dumps({"sequence": sequence, "h_in_sequence": r, "iteration": 0, "ser": float(1. - test_accuracy_), 'mmse': float(1. - mmse_test_accuracy_), "snr": snr, "measured_snr": float(measured_snr_), "loss": float(test_loss_)}) + "\n")
                            for it in range(args.test_iterations):
                                sess.run(train, feed_dict)
                                test_accuracy_, test_loss_, measured_snr_ = sess.run([accuracy, loss, measured_snr], feed_dict)
                                test_results.append({"sequence": sequence, "h_in_sequence": r, "iteration": it+1, "ser": float(1. - test_accuracy_), 'mmse': float(1. - mmse_test_accuracy_), "snr": snr, "measured_snr": float(measured_snr_), "loss": float(test_loss_)})
                                output_fd.write(json.dumps({"sequence": sequence, "h_in_sequence": r, "iteration": it+1, "ser": float(1. - test_accuracy_), 'mmse': float(1. - mmse_test_accuracy_), "snr": snr, "measured_snr": float(measured_snr_), "loss": float(test_loss_)}) + "\n")
                                progress_bar.update(1)
    # Save results
    #     json.dump(test_results, output_fd)
    print('Results are saved at', savePath)
