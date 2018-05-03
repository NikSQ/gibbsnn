from scipy.io import loadmat, savemat
import numpy as np


def get_nlp_names():
    names = ['20News_comp',
            '20News_elec',
            'domain0',
            'domain1',
            'reuters_I81_I83',
            'reuters_I654_I65']
    return names


def get_dataset(name, train_ratio=0.5):
    dataset = loadmat(get_filename(name, True))
    n_samples = dataset['T']
    train_size = int(float(n_samples) * train_ratio)
    X_tr = dataset['data'][:train_size, :]
    X_tst = dataset['data'][train_size:, :]
    Y_tr = dataset['labels'][:train_size, :]
    Y_tst = dataset['labels'][train_size:, :]
    return X_tr, Y_tr, X_tst, Y_tst


# Transforms features such that they can be represented by integers
def transform_nlp_data():
    dataset_names = get_nlp_names()

    for name in dataset_names:
        file_name = get_filename(name)
        dataset = loadmat(file_name)
        input_data = dataset['data']

        transformed_data = np.copy(input_data)
        for column_idx in range(input_data.shape[1]):
            unique = np.unique(transformed_data[:, column_idx])
            for idx, val in enumerate(unique):
                transformed_data[transformed_data[:, column_idx] == val, column_idx] = idx
        dataset['data'] = transformed_data

        target_data = dataset['labels']
        one_hot = np.zeros((target_data.shape[0], 2), dtype=np.int32)
        target_data[target_data == -1] = 0
        target_data = np.squeeze(target_data)
        one_hot[target_data == 0, 0] = 1
        one_hot[target_data == 1, 1] = 1
        dataset['labels'] = one_hot

        savemat(get_filename(name, True), dataset)


def get_filename(name, processed=False):
    if processed:
        return '../data/Classification/' + name + '/processed.mat'
    else:
        return '../data/Classification/' + name + '/processed_data.mat'

