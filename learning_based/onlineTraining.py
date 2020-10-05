import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
from tf_session import *
import pickle
from parser import parse
from dataset import read_channels_dataset

if __name__ == "__main__":
    params, args = parse()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    num_channel_samples = 100

    if args.data:
        train_data_ref, test_data_ref, power_db = read_channels_dataset(args.channels_dir, num_channel_samples)
        params['Hdataset_powerdB'] = power_db
    else:
        train_data_ref = []
        test_data_ref = []

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
    init = nodes['init']

    # Training loop
    for r in range(num_channel_samples):
        sess.run(init)
        train_data = np.expand_dims(train_data_ref[r], axis=0)
        test_data = np.expand_dims(test_data_ref[r], axis=0)
        results = {}
        for it in range(args.train_iterations + 1):
            feed_dict = {
                batch_size: args.batch_size,
                lr: args.learn_rate,
                snr_db_max: params['SNR_dB_max'],
                snr_db_min: params['SNR_dB_min'],
            }
            if args.data:
                sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
                feed_dict[H] = train_data[sample_ids]

            sess.run(train, feed_dict)

            # Test
            if it == args.train_iterations:
                for snr_ in range(int(params['SNR_dB_min']), int(params['SNR_dB_max']) + 1):
                    feed_dict = {
                        batch_size: args.test_batch_size,
                        snr_db_max: snr_,
                        snr_db_min: snr_,
                    }
                    if args.data:
                        sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
                        feed_dict[H] = test_data[sample_ids]

                    test_accuracy_, test_loss_, measured_snr_, log_ = sess.run([accuracy, loss, measured_snr, logs],
                                                                               feed_dict)
                    print('Test SER of %f on channel realization %d after %d iterations at SNR %f dB' % (
                    1. - test_accuracy_, r, it, measured_snr_))
                    results[str(snr_)] = {}
                    for k in log_:
                        results[str(snr_)][k] = log_[k]['stat']
                    results[str(snr_)]['accuracy'] = test_accuracy_
                results['cond'] = np.linalg.cond(test_data[sample_ids][0])
                path = args.output_dir + '/OnlineTraining_%s_NT%sNR%s_%s/' % (
                args.modulation, args.x_size, args.y_size, args.linear)
                if not os.path.exists(path):
                    os.makedirs(path)
                savePath = path + 'results%d.pkl' % r
                with open(savePath, 'wb') as f:
                    pickle.dump(results, f)
                print('Results saved at %s' % savePath)
