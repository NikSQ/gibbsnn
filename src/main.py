from src.nn import NN
import numpy as np
import tensorflow as tf
from src.activation import get_activation_function
from src.mnist_data import load_dataset
import matplotlib.pyplot as plt



# Here just a dummy task is implemented, to test whether training works
# Dropout is currently only partially implemented, namely when creating the entries in the lookup table
# If we sample from layer i, only the following layers will thus be affected by dropout.
# That can be easily changed though.
file_names = {''}
dir_name = 'test'

# Create activation functions
act_funcs = []
act_func = get_activation_function('stair')
act_func.set_params([1, 3])

act_funcs.append(act_func)
act_funcs.append(act_func)
# plotter.plot_activation_function(act_func, np.arange(-15, 15))
# activation_function = get_activation_function('binary_sign')


# data_handler.transform_nlp_data()
x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset('mnist_basic')

config = {'layout': [x_tr.shape[1], 30, 30, y_tr.shape[1]],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1.0, 1.0],
          'flat_factor': [1.0, 1.0, 1.0],
          'sampling_sequence': 'stochastic'}


nn = NN(config)
nn.create_gibbs_graph(x_tr.shape[0], y_tr.shape[0], 2)

train_mis = []
val_mis = []

with tf.Session() as sess:
    writer = tf.summary.FileWriter("../results/demo_graph")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(nn.load_train_set_op, feed_dict={nn.X_placeholder: x_tr, nn.Y_placeholder: y_tr})
    sess.run(nn.load_val_set_op, feed_dict={nn.X_placeholder: x_va, nn.Y_placeholder: y_va})

    for i in range(10):
        #if i % 3 == 0:
            #plotter.save_activation_plots(nn.get_activation_histogram(sess, X_tr), str(i))
        #quit()
        #print(np.sum(sess.run(nn.full_network.likelihoods, feed_dict={nn.X: X_tr, nn.Y: Y_tr})))
        train_mis.append(nn.get_misclassification(sess, False))
        val_mis.append(nn.get_misclassification(sess, True))
        nn.perform_gibbs_iteration(sess)



