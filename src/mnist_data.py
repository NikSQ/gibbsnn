from scipy.io import loadmat
import numpy as np

filenames = {'mnist': '../datasets/mnist/mnist.mat',
              'mnist_basic': '../datasets/mnist_basic/mnist_basic.mat',
              'mnist_basic_pca_50_zeromean': '../datasets/mnist_basic/mnist_basic_pca50_zeromean.mat',
              'mnist_background': '../datasets/mnist_background/mnist_background.mat',
              'mnist_background_random': '../datasets/mnist_background_random/mnist_background_random.mat',
              'mnist_rotated': '../datasets/mnist_rotated/mnist_rotated.mat',
              'mnist_rotated_background': '../datasets/mnist_rotated_background/mnist_rotated_background',
              'ebp_reuters_i6': '../datasets/ebp_reuters_i6.mat'}

def load_dataset(dataset_name):
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
    return (binarize_data(data['x_tr']), t_tr, binarize_data(data['x_va']), t_va, 
            binarize_data(data['x_te']), t_te)


def binarize_data(data):
    data[data < 0.5] = -1
    data[data >= 0.5] = 1
    return data




    



