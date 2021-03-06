import tensorflow as tf
import sys
import numpy as np
import copy

sys.path.append('../')

from src.nn import NN
from src.mnist_data import load_dataset
from src.tools import print_nn_config, print_run_config, get_init_values
from src.ensemble import Ensemble
from src.activation import get_activation_function
from src.genetic_algo import GeneticSolver
from src.simulated_annealing import SASolver


def run_experiment(exp_config, init_config, nn_config_primitive, dataset, run=0):
    tf.reset_default_graph()
    nn_config = copy.deepcopy(nn_config_primitive)

    act_funcs = []
    for layer_idx, act_func_name in enumerate(nn_config['act_func_names']):
        act_func = get_activation_function(act_func_name)
        act_func.set_params(nn_config['act_func_params'][layer_idx])
        act_funcs.append(act_func)
    nn_config['act_funcs'] = act_funcs

    x_tr, y_tr, x_va, y_va, x_te, y_te = load_dataset(dataset)
    nn_config['layout'].insert(0, x_tr.shape[1])
    nn_config['layout'].append(y_tr.shape[1])

    w_init_vals, b_init_vals = get_init_values(nn_config, init_config, x_tr, y_tr)

    nn = NN(nn_config, w_init_vals, b_init_vals)
    nn.create_gibbs_graph(x_tr.shape[0], x_va.shape[0], exp_config['block_size'])
    ensemble_tr = Ensemble(nn.Y_tr, y_tr.shape, 'tr', nn.full_network.activation)
    ensemble_va = Ensemble(nn.Y_val, y_va.shape, 'va', nn.full_network.activation)
    final_ensemble_acc = None
    final_ensemble_ce = None
    final_acc = None
    final_ce = None

    print_nn_config(nn_config)
    print_run_config(exp_config)

    init_pop = []
    tr_accs = []
    va_accs = []

    with tf.Session() as sess:
        writer_tr = tf.summary.FileWriter(exp_config['path'] + 'tr/')
        writer_va = tf.summary.FileWriter(exp_config['path'] + 'va/')

        sess.run(tf.global_variables_initializer())
        sess.run(nn.load_train_set_op, feed_dict={nn.X_placeholder: x_tr, nn.Y_placeholder: y_tr})
        sess.run(nn.load_val_set_op, feed_dict={nn.X_placeholder: x_va, nn.Y_placeholder: y_va})

        tr_acc, tr_ce = sess.run([nn.full_network.accuracy, nn.full_network.cross_entropy],
                                              feed_dict={nn.validate: False})
        va_acc, va_ce = sess.run([nn.full_network.accuracy, nn.full_network.cross_entropy],
                                              feed_dict={nn.validate: True})
        print('Epoch: {}, AccTr: {}, AccVa: {}, CeTr: {}, CeVa: {}'.format(0,tr_acc, va_acc, tr_ce, va_ce))

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
                    final_acc = va_acc
                    final_ce = va_ce
                    tr_accs.append(tr_acc)
                    va_accs.append(va_acc)
                    print('Epoch: {}, AccTr: {}, AccVa: {}, CeTr: {}, CeVa: {}'.format(epoch+1,tr_acc, va_acc, tr_ce, va_ce))

            if exp_config['store_acts'] and epoch % exp_config['store_acts_every'] == 0:
                s_tr = sess.run(nn.full_network.summary_op, feed_dict={nn.validate: False})
                s_va = sess.run(nn.full_network.summary_op, feed_dict={nn.validate: True})
                writer_tr.add_summary(s_tr, epoch)
                writer_va.add_summary(s_va, epoch)

            if exp_config['burn_in'] <= epoch + 1 and (epoch + 1 - exp_config['burn_in']) % exp_config['thinning'] == 0:
                tr_acc, tr_ce = sess.run([ensemble_tr.accuracy, ensemble_tr.cross_entropy], feed_dict={nn.validate: False})
                va_acc, va_ce = sess.run([ensemble_va.accuracy, ensemble_va.cross_entropy], feed_dict={nn.validate: True})
                final_ensemble_acc = va_acc
                final_ensemble_ce = va_ce
                print('ENSEMBLE | Tr_Acc: {}, Tr_CE: {}, Va_Acc: {}, Va_CE: {}'.format(tr_acc, tr_ce, va_acc, va_ce))

            if (exp_config['mode'] == 'ga' or exp_config['mode'] == 'sa') and exp_config['init_burn_in'] <= epoch+1 and (epoch + 1 - exp_config['init_burn_in']) % exp_config['init_thinning'] == 0:
                init_pop.append(nn.get_weights(sess))

    if exp_config['mode'] == 'ga':
        print('Starting genetic algorithm')
        return run_ga_solver(exp_config, nn_config, x_tr, y_tr, x_va, y_va, init_pop)

    if exp_config['mode'] == 'sa':
        print('Starting simulated annealing')
        return run_sa_solver(exp_config, nn_config, x_tr, y_tr, x_va, y_va, init_pop)

    return final_ensemble_acc, final_ensemble_ce, final_acc, final_ce
    if len(tr_accs) != 0:
        np.save('../training/' + exp_config['expname'] + '_tr_accs' + str(run), tr_accs)
        np.save('../training/' + exp_config['expname'] + '_va_accs' + str(run), va_accs)

def run_ga_solver(ga_config, nn_config, x_tr, y_tr, x_va, y_va, ga_init_pop):
    tf.reset_default_graph()
    solver = GeneticSolver(nn_config, ga_config, x_tr.shape[0], x_va.shape[0], ga_init_pop)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(solver.load_train_set_op, feed_dict={solver.x_placeholder: x_tr, solver.y_placeholder: y_tr})
        sess.run(solver.load_val_set_op, feed_dict={solver.x_placeholder: x_va, solver.y_placeholder: y_va})
        return solver.perform_ga(sess)


def run_sa_solver(sa_config, nn_config, x_tr, y_tr, x_va, y_va, sa_init_pop):
    tf.reset_default_graph()
    solver = SASolver(nn_config, sa_config, x_tr.shape[0], x_va.shape[0], sa_init_pop)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(solver.load_train_set_op, feed_dict={solver.x_placeholder: x_tr, solver.y_placeholder: y_tr})
        sess.run(solver.load_val_set_op, feed_dict={solver.x_placeholder: x_va, solver.y_placeholder: y_va})
        return solver.perform_sa(sess)



# TEST1 Bias mean sampling with old fashioned lookuptable
# TEST2 Just weight sampling with old fashioned lookuptable (where the fuck is the error)
# TEST3 Bias mean sampling but with argmax
