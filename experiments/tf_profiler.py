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
dir_name = '../results/profiler/'
n_epochs = 2
block_size = 4
store_activations = True
store_acts_every = 1

act_func = get_activation_function('binary_sign')
act_func.set_params([])
act_funcs = [act_func, act_func]

x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset('mnist_basic')
x_tr = x_tr[:, :10]
x_va = x_va[:, :10]

config = {'layout': [x_tr.shape[1], 10, 10, y_tr.shape[1]],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1.0, 1.0],
          'flat_factor': [1.0, 1.0, 1.0],
          'sampling_sequence': 'stochastic'}

nn = NN(config)
nn.create_gibbs_graph(x_tr.shape[0], x_va.shape[0], block_size)

tr_mis = []
va_mis = []
tr_acts_hists = []
tr_acts_epochs = []

with tf.Session() as sess:
    writer = tf.summary.FileWriter("../results/demo_graph/x")
    writer.add_graph(sess.graph)
    writer.close()
    exit(0)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(tf.global_variables_initializer())
    sess.run(nn.load_train_set_op, feed_dict={nn.X_placeholder: x_tr, nn.Y_placeholder: y_tr})
    sess.run(nn.load_val_set_op, feed_dict={nn.X_placeholder: x_va, nn.Y_placeholder: y_va})

    for epoch in range(n_epochs):
        nn.profile_gibbs_iteration(sess, options, run_metadata, epoch, dir_name)



