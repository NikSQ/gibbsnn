import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('../')

from src.nn import NN
from src.activation import get_activation_function
from src.storage_tools import Saver
from src.mnist_data import load_dataset

# TODO: job name should include the actual index of the job
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
dir_name = 'ternary_act_do'
job_name = 'job_' + str(task_id)
n_epochs = 10
block_size = 4
store_activations = True
store_acts_every = 1

act_func = get_activation_function('extended_ternary_sign')
act_func.set_params([1 + 2 * task_id])
act_funcs = [act_func, act_func]

x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset('mnist_basic')

config = {'layout': [x_tr.shape[1], 40, 40, y_tr.shape[1]],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [.8, .8],
          'flat_factor': [1.0, 1.0, 1.0],
          'sampling_sequence': 'stochastic'}

saver = Saver(dir_name, job_name)

nn = NN(config)
nn.create_gibbs_graph(x_tr.shape[0], x_va.shape[0], block_size)

tr_mis = []
va_mis = []
tr_acts_hists = []
tr_acts_epochs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(nn.load_train_set_op, feed_dict={nn.X_placeholder: x_tr, nn.Y_placeholder: y_tr})
    sess.run(nn.load_val_set_op, feed_dict={nn.X_placeholder: x_va, nn.Y_placeholder: y_va})

    for i in range(n_epochs):
        if store_activations and i % store_acts_every == 0:
            tr_acts_hists.append(nn.get_activation_histogram(sess))
            tr_acts_epochs.append(i)

        tr_mis.append(nn.get_misclassification(sess, False))
        va_mis.append(nn.get_misclassification(sess, True))
        nn.perform_gibbs_iteration(sess)

saver.store_sequence(name='tr_mis', sequence=tr_mis, epochs=np.arange(n_epochs))
saver.store_sequence(name='va_mis', sequence=va_mis, epochs=np.arange(n_epochs))
saver.store_act_hists(name='tr_acts', hists=tr_acts_hists, epochs=tr_acts_epochs)
print('Train misclassification')
print(tr_mis)
print('Validation misclassification')
print(va_mis)


