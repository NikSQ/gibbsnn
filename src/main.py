from src.nn import NN
import numpy as np
import tensorflow as tf
from src.activation import get_activation_function
import src.nlp_data as data_handler
from src.plot_tools import Plotter
from src.storage_tools import Loader

exp_name = 'ternary_act'
job_name = 'job_1'
loader = Loader(exp_name, job_name)
loader.load_act_hists('tr_acts')
print(loader.load_sequence('tr_mis'))
loader.generate_plots()
quit()

# Here just a dummy task is implemented, to test whether training works
# Dropout is currently only partially implemented, namely when creating the entries in the lookup table
# If we sample from layer i, only the following layers will thus be affected by dropout.
# That can be easily changed though.
file_names = {''}
dir_name = 'test'
plotter = Plotter(dir_name)

# Create activation functions
act_funcs = []
act_func = get_activation_function('stair')
act_func.set_params([1, 3])

act_funcs.append(act_func)
act_funcs.append(act_func)
# plotter.plot_activation_function(act_func, np.arange(-15, 15))
# activation_function = get_activation_function('binary_sign')


# data_handler.transform_nlp_data()
nlp_dataset_names = data_handler.get_nlp_names()
X_tr, Y_tr, X_val, Y_val = data_handler.get_dataset(nlp_dataset_names[0])
X_tr = X_tr[:, :20]
X_val = X_val[:, :20]
n_samples = X_tr.shape[0]
n_val_samples = X_val.shape[0]

config = {'layout': [X_tr.shape[1], 30, 30, Y_tr.shape[1]],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1.0, 1.0],
          'flat_factor': [1.0, 1.0, 1.0],
          'sampling_sequence': 'stochastic'}


nn = NN(config)
nn.create_gibbs_graph(n_samples, n_val_samples, 2)

train_mis = []
val_mis = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(nn.load_train_set_op, feed_dict={nn.X_placeholder: X_tr, nn.Y_placeholder: Y_tr})
    sess.run(nn.load_val_set_op, feed_dict={nn.X_placeholder: X_val, nn.Y_placeholder: Y_val})

    for i in range(10):
        if i % 3 == 0:
            plotter.save_activation_plots(nn.get_activation_histogram(sess, X_tr), str(i))
        #quit()
        #print(np.sum(sess.run(nn.full_network.likelihoods, feed_dict={nn.X: X_tr, nn.Y: Y_tr})))
        train_mis.append(nn.get_misclassification(sess, False))
        val_mis.append(nn.get_misclassification(sess, True))
        nn.perform_gibbs_iteration(sess)

plotter.plot_misclassification(np.arange(10), train_mis, val_mis)

