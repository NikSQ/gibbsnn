import tensorflow as tf
import sys

sys.path.append('../')

from src.nn import NN
from src.mnist_data import load_dataset
from src.tools import print_nn_config, print_run_config


def run_experiment(exp_config, nn_config, dataset):
    print_nn_config(nn_config)
    print_run_config(exp_config)

    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset(dataset)
    nn_config['layout'].insert(0, x_tr.shape[1])
    nn_config['layout'].append(y_tr.shape[1])

    nn = NN(nn_config)
    nn.create_gibbs_graph(x_tr.shape[0], x_va.shape[0], exp_config['block_size'])

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
                    acc_tr, llh_tr = sess.run([nn.full_network.accuracy, nn.full_network.total_likelihood],
                                              feed_dict={nn.validate: False})
                    acc_va, llh_va = sess.run([nn.full_network.accuracy, nn.full_network.total_likelihood],
                                              feed_dict={nn.validate: True})
                    print('Epoch: {}, AccTr: {}, AccVa: {}, LlhTr: {}, LlhVa: {}'.format(epoch+1,acc_tr, acc_va, llh_tr, llh_va))
            if exp_config['store_acts'] and epoch % exp_config['store_acts_every'] == 0:
                s_tr = sess.run(nn.full_network.summary_op, feed_dict={nn.validate: False})
                s_va = sess.run(nn.full_network.summary_op, feed_dict={nn.validate: True})
                writer_tr.add_summary(s_tr, epoch)
                writer_va.add_summary(s_va, epoch)

            if exp_config['burn_in'] <= epoch + 1 and (epoch + 1 - exp_config['burn_in']) % exp_config['thinning'] == 0:
                tr_acts = sess.run(nn.full_network.activation, feed_dict={nn.validate: False})
                va_acts = sess.run(nn.full_network.activation, feed_dict={nn.validate: True})


