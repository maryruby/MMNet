import h5py
import numpy as np


def complex_to_real_dataset(Hr, Hi):
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi, Hr], axis=2)
    out = np.concatenate([h1, h2], axis=1)
    return out


# TODO: train and test are equal
def read_channels_dataset(channels_dataset_file, num_channel_samples):
    with h5py.File(channels_dataset_file, "r") as input_f:
        H_dataset = complex_to_real_dataset(input_f['H_r'].value, input_f['H_i'].value)
        power_db = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.)
        print('Channels dataset power (dB): %f' % power_db)

        train_data_ref = H_dataset
        test_data_ref = H_dataset
        print('Channels dataset shape:', H_dataset.shape)
        rnd_index = np.random.randint(0, train_data_ref.shape[0], num_channel_samples)
        train_data_ref = train_data_ref[rnd_index]
        test_data_ref = test_data_ref[rnd_index]
        print('Sampled channel indices:', rnd_index)
        return train_data_ref, test_data_ref, power_db
