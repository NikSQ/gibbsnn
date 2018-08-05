from scipy.io import loadmat
import numpy as np

filenames = {'mnist': '../datasets/mnist/mnist.mat',
             'mnist_basic': '../datasets/mnist_basic/mnist_basic.mat',
             'mnist_basic_pca_50_zeromean': '../datasets/mnist_basic/mnist_basic_pca50_zeromean.mat',
             'mnist_background': '../datasets/mnist_background/mnist_background.mat',
             'mnist_background_random': '../datasets/mnist_background_random/mnist_background_random.mat',
             'mnist_rotated': '../datasets/mnist_rotated/mnist_rotated.mat',
             'mnist_rotated_background': '../datasets/mnist_rotated_background/mnist_rotated_background',
             'ebp_reuters_i6': '../datasets/ebp_reuters_i6.mat',
             'uci_iris': '../datasets/uci_iris/iris.mat'}

def load_dataset(dataset_name, binarize=False):
    if 'mnist' in dataset_name:
        return load_mnist_dataset(dataset_name, binarize)
    elif 'uci' in dataset_name:
        return load_uci_dataset(dataset_name)

def load_uci_dataset(dataset_name):
    filename = filenames[dataset_name]
    data = loadmat(filename)
    data['t'] = np.squeeze(data['t']) - 1
    n_samples = data['x'].shape[0]
    tr_batch_size = int(n_samples * 0.7)
    va_batch_size = n_samples - tr_batch_size
    t_tr = np.zeros((tr_batch_size, np.max(data['t'] + 1)))
    t_va = np.zeros((va_batch_size, np.max(data['t'] + 1)))
    x_tr = np.zeros((tr_batch_size, data['x'].shape[1]))
    x_va = np.zeros((va_batch_size, data['x'].shape[1]))
    tr_ind = np.random.choice(np.arange(n_samples), replace=False, size=(tr_batch_size))
    va_ind = list(set(np.arange(n_samples)) - set(tr_ind))
    x_tr[:, :] = data['x'][tr_ind, :]
    x_va[:, :] = data['x'][va_ind, :]
    t_tr[np.arange(tr_batch_size), data['t'][tr_ind]] = 1
    t_va[np.arange(va_batch_size), data['t'][va_ind]] = 1

    return x_tr, t_tr, x_va, t_va, None, None


def load_mnist_dataset(dataset_name, binarize):
    filename = filenames[dataset_name] 
    data = loadmat(filename)
    data['t_tr'] = np.squeeze(data['t_tr']) - 1
    data['t_va'] = np.squeeze(data['t_va']) - 1
    data['t_te'] = np.squeeze(data['t_te']) - 1
    t_tr = np.zeros((data['t_tr'].shape[0], 10))
    t_tr[np.arange(t_tr.shape[0]), data['t_tr']] = 1
    t_va = np.zeros((data['t_va'].shape[0], 10))
    t_va[np.arange(t_va.shape[0]), data['t_va']] = 1
    t_te = np.zeros((data['t_te'].shape[0], 10))
    t_te[np.arange(t_te.shape[0]), data['t_te']] = 1
    data['x_tr'] = data['x_tr'] * 2 - 1
    data['x_va'] = data['x_va'] * 2 - 1
    data['x_te'] = data['x_te'] * 2 - 1
    
    print('Input binarization: {}'.format(binarize))

    if binarize:
        return (binarize_data(data['x_tr']), t_tr, binarize_data(data['x_va']), t_va, 
                binarize_data(data['x_te']), t_te)
    else:
        return (data['x_tr'], t_tr, data['x_va'], t_va, data['x_te'], t_te)


def binarize_data(data):
    data[data < 0] = -1
    data[data >= 0] = 1
    return data







    



