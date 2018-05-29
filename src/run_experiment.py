import tensorflow as tf
import sys
import numpy as np

sys.path.append('../')

from src.nn import NN
from src.mnist_data import load_dataset
from src.tools import print_nn_config, print_run_config, get_init_values
from src.ensemble import Ensemble


def run_experiment(exp_config, init_config, nn_config, dataset):
    print_nn_config(nn_config)
    print_run_config(exp_config)

    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset(dataset)
    nn_config['layout'].insert(0, x_tr.shape[1])
    nn_config['layout'].append(y_tr.shape[1])

    w_init_vals, b_init_vals = get_init_values(nn_config, init_config, x_tr, y_tr)

    nn = NN(nn_config, w_init_vals, b_init_vals)
    nn.create_gibbs_graph(x_tr.shape[0], x_va.shape[0], exp_config['block_size'])
    ensemble_tr = Ensemble(nn.Y_tr, y_tr.shape, 'tr', nn.full_network.activation)
    ensemble_va = Ensemble(nn.Y_val, y_va.shape, 'va', nn.full_network.activation)

    with tf.Session() as sess:
        writer_tr = tf.summary.FileWriter(exp_config['path'] + 'tr/')
        writer_va = tf.summary.FileWriter(exp_config['path'] + 'va/')

        sess.run(tf.global_variables_initializer())
        sess.run(nn.load_train_set_op, feed_dict={nn.X_placeholder: x_tr, nn.Y_placeholder: y_tr})
        sess.run(nn.load_val_set_op, feed_dict={nn.X_placeholder: x_va, nn.Y_placeholder: y_va})

        for epoch in range(exp_config['n_epochs']):
            nn.perform_gibbs_iteration(sess)

            if exp_config['store_vars'] and epoch % exp_config['store_vars_every'] == 0:
                if exp_config['store_method'] == 'both' or exp_config['store_method'] == 'tensorboard':
                    s_var = sess.run(nn.var_summary_op)
                    writer_va.add_summary(s_var, epoch)
                if exp_config['store_method'] == 'both' or exp_config['store_method'] == 'log':
                    tr_acc, tr_ce = sess.run([nn.full_network.accuracy, nn.full_network.cross_entropy],
                                              feed_dict={nn.validate: False})
                    va_acc, va_ce = sess.run([nn.full_network.accuracy, nn.full_network.cross_entropy],
                                              feed_dict={nn.validate: True})
                    print('Epoch: {}, AccTr: {}, AccVa: {}, CeTr: {}, CeVa: {}'.format(epoch+1,tr_acc, va_acc, tr_ce, va_ce))

            if exp_config['store_acts'] and epoch % exp_config['store_acts_every'] == 0:
                s_tr = sess.run(nn.full_network.summary_op, feed_dict={nn.validate: False})
                s_va = sess.run(nn.full_network.summary_op, feed_dict={nn.validate: True})
                writer_tr.add_summary(s_tr, epoch)
                writer_va.add_summary(s_va, epoch)

            if exp_config['burn_in'] <= epoch + 1 and (epoch + 1 - exp_config['burn_in']) % exp_config['thinning'] == 0:
                tr_acc, tr_ce = sess.run([ensemble_tr.accuracy, ensemble_tr.cross_entropy], feed_dict={nn.validate: False})
                va_acc, va_ce = sess.run([ensemble_va.accuracy, ensemble_va.cross_entropy], feed_dict={nn.validate: True})
                print('ENSEMBLE | Tr_Acc: {}, Tr_CE: {}, Va_Acc: {}, Va_CE: {}'.format(tr_acc, tr_ce, va_acc, va_ce))



# TEST1 Bias mean sampling with old fashioned lookuptable
# TEST2 Just weight sampling with old fashioned lookuptable (where the fuck is the error)
# TEST3 Bias mean sampling but with argmax