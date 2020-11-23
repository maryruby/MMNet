import tensorflow.compat.v1 as tf
from utils import *
from detector import detector
from loss import loss_fun
from sample_generator import generator


class MMNet_graph():
    def __init__(self, params):
        self.params = params
        self.train_writer = None
        self.test_writer = None

    def build(self):

        with tf.device('/gpu:0'):
            tf.reset_default_graph()
            tf.set_random_seed(self.params['seed'])

            # Placeholders for feed dict
            batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
            lr = tf.placeholder(tf.float32, shape=(), name='lr')
            snr_db_max = tf.placeholder(tf.float32, shape=(), name='snr_db_max')
            snr_db_min = tf.placeholder(tf.float32, shape=(), name='snr_db_min')
            train_flag = tf.placeholder(tf.bool, shape=(), name='train_flag')

            # MIMO sample generator model
            mimo = generator(self.params, batch_size)

            # Generate transmitt signals
            constellation = mimo.constellation
            indices = mimo.random_indices()
            x = mimo.modulate(indices)

            # Send x through the channel
            if self.params['data']:
                H = tf.placeholder(tf.float32, shape=(None, 2 * self.params['N'], 2 * self.params['K']), name='H')
                y, noise_sigma, actual_snrdB = mimo.channel(x, snr_db_min, snr_db_max, H, self.params['data'],
                                                            self.params['correlation'])
            else:
                y, H, noise_sigma, actual_snrdB = mimo.channel(x, snr_db_min, snr_db_max, [], self.params['data'],
                                                               self.params['correlation'])

            # Zero-forcing detection
            # x_mmse = mmse(y, H)
            x_mmse = mmse(y, H, noise_sigma)
            x_mmse_idx = demodulate(x_mmse, constellation)
            x_mmse = tf.gather(constellation, x_mmse_idx)
            with tf.name_scope("accuracy_mmse"):
                acc_mmse = accuracy(indices, demodulate(x_mmse, constellation))
                tf.summary.scalar("total_accuracy_mmse", acc_mmse)

            # MMNet detection
            x_NN, helper = detector(self.params, constellation, x, y, H, noise_sigma, indices, batch_size).create_graph()
            loss = loss_fun(x_NN, x, self.params)

            with tf.name_scope("accuracy_nn"):
                print("REPORTING MAX ACCURACY")
                temp = []
                for i in range(self.params['L']):
                    layer_acc = accuracy(indices, demodulate(x_NN[i], constellation))
                    temp.append(layer_acc)
                    tf.summary.scalar("accuracy_nn_layer_%s" % i, layer_acc)
                    # acc_NN = accuracy(indices, mimo.demodulate(x_NN[train_layer_no-1], modtypes))
                acc_NN = tf.reduce_max(temp)
                tf.summary.scalar("total_accuracy_nn", acc_NN)

            # Training operation
            print("tf_session: Optimizing for the total loss")
            train_step = tf.train.AdamOptimizer(lr).minimize(tf.reduce_mean(loss))
            # train_step = tf.train.AdamOptimizer(lr).minimize(loss[train_layer_no-1])
            # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss[train_layer_no-1])

            # Init operation
            init = tf.global_variables_initializer()

            # #### Training
            # define saver
            saver = tf.train.Saver()

            # Create summary writer
            # merged = [0]
            merged = tf.summary.merge_all()
            # print "merged"
            # Create session and initialize all variables
            sess = tf.Session()

            self.train_writer = tf.summary.FileWriter('./reports/' + 'model1' + '/log/train', sess.graph)
            self.test_writer = tf.summary.FileWriter('./reports/' + 'model1' + '/log/test', sess.graph)

            if len(self.params['start_from']) > 1:
                saver.restore(sess, self.params['start_from'])
            else:
                sess.run(init)
            self.train_writer.flush()
            self.test_writer.flush()

            nodes = {'measured_snr': actual_snrdB, 'batch_size': batch_size, 'lr': lr, 'snr_db_min': snr_db_min,
                     'snr_db_max': snr_db_max, 'x': x, 'x_id': indices, 'H': H, 'y': y, 'sess': sess,
                     'train': train_step, 'accuracy': acc_NN, 'loss': loss, 'mmse_accuracy': acc_mmse,
                     'constellation': constellation, 'logs': helper, 'init': init,
                     'merged': merged}
        return nodes

    def write_tensorboard_summary(self, summary, global_step=None, test=False):
        if test:
            if self.test_writer:
                self.test_writer.add_summary(summary, global_step)
                self.test_writer.flush()
        else:
            if self.train_writer:
                self.train_writer.add_summary(summary, global_step)
                self.train_writer.flush()
