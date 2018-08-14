import tensorflow as tf
import numpy as np


def get_init_values(nn_config, train_config, x, y):
    w_list = []
    b_list = []
    q_w_vals = []
    q_b_vals = []

    with tf.variable_scope('backpropnn'):
        x_init = tf.constant_initializer(value=x, dtype=tf.float32)
        y_init = tf.constant_initializer(value=y, dtype=tf.float32)
        x = tf.get_variable(name='x', shape=(x.shape[0], nn_config['layout'][0]), dtype=tf.float32,
                                 trainable=False, initializer=x_init)
        y = tf.get_variable(name='y', shape=(y.shape[0], nn_config['layout'][-1]), dtype=tf.float32,
                                 trainable=False, initializer=y_init)
        layer_input = x
        l2_loss = 0

        for layer_idx in range(0, len(nn_config['layout']) - 1):
            with tf.variable_scope('layer' + str(layer_idx)):
                w_init = tf.random_normal_initializer(mean=0.0,
                                                      stddev=np.divide(1, np.sqrt(nn_config['layout'][layer_idx])))
                b_init = tf.zeros_initializer(dtype=tf.float32)
                w = tf.get_variable(name='W', shape=(nn_config['layout'][layer_idx], nn_config['layout'][layer_idx+1]),
                                    initializer=w_init)
                b = tf.get_variable(name='b', shape=(1, nn_config['layout'][layer_idx + 1 ]), initializer=b_init)
                w_list.append(w)
                b_list.append(b)
                l2_loss = l2_loss + tf.nn.l2_loss(w)

                activation = tf.matmul(layer_input, w)
                if layer_idx == len(nn_config['layout']) - 2:
                    activation = activation + b
                    continue

                neg_act_means = -tf.reduce_mean(activation, axis=0)
                with tf.control_dependencies([tf.assign(b, tf.expand_dims(neg_act_means, axis=0))]):
                    layer_input = tf.nn.tanh(activation + neg_act_means)

    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))
    prediction = tf.argmax(tf.nn.softmax(activation), axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y, axis=1)), dtype=tf.float32))
    train_op = tf.train.AdamOptimizer(learning_rate=train_config['learning_rate'])\
        .minimize(ce + l2_loss * train_config['reg'])
    train_config = train_config

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(train_config['n_epochs']):
            sess.run(train_op)
        print('Pretrained network accuracy: {}'.format(sess.run(accuracy)))
        w_vals = sess.run(w_list)
        b_vals = sess.run(b_list)

        for layer_idx in range(0, len(nn_config['layout']) - 1):
            if nn_config['weight_type'] == 'ternary':
                ratio = np.var(w_vals[layer_idx])
                if ratio >= 0.9:
                    ratio = 0.9
                threshold = np.sort(np.abs(np.reshape(w_vals[layer_idx], (-1,))))[int(np.size(w_vals[layer_idx]) * (1 - ratio))]
                q_w_val = np.zeros_like(w_vals[layer_idx])
                q_w_val[w_vals[layer_idx] >= threshold] = 1
                q_w_val[w_vals[layer_idx] <= -threshold] = -1
                q_w_vals.append(q_w_val)
            else:
                q_w_val = np.sign(w_vals[layer_idx])
                q_w_val[q_w_val == 0] = 1
                q_w_vals.append(q_w_val)
            q_b_vals.append(b_vals[layer_idx].astype(np.int32))
    return q_w_vals, q_b_vals


def print_nn_config(config):
    print('NN CONFIG')

    print('Weight-type:\t\t{}'.format(config['weight_type']))
    #head = 'name\tl0\tl1\t\l2\tl3'
    #fmt = "{name:s}"

    layout_str = "Layout\t\t\t\t\t\t\t\t\t"
    for idx, n_neurons in enumerate(config['layout']):
        layout_str += 'Layer #' + str(idx+1) + ': ' + str(n_neurons) + '\t'
    print(layout_str)

    act_str = "Activation Functions\t\t\t\t\t"
    for idx, act_func in enumerate(config['act_funcs']):
        act_str += 'Layer #' + str(idx+1) + ': ' + act_func.name + '\t'
    print(act_str)

    probs_str = "Keep probabilities\t\t"
    for idx, keep_prob in enumerate(config['keep_probs']):
        probs_str += 'Layer #' + str(idx) + ': ' + str(keep_prob) + '\t'
    print(probs_str)

    flat_str = "Flat factors\t\t\t\t\t\t\t"
    for idx, flat_factor in enumerate(config['flat_factor']):
        flat_str += 'Layer #' + str(idx +1) + ': ' + str(flat_factor) + '\t'
    print(flat_str)

    noise_str = "Activation Noise\t\t\t\t\t\t"
    for idx, noise in enumerate(config['act_noise']):
        noise_str += 'Layer #' + str(idx + 1) + ': ' + str(noise) + '\t'
    print(noise_str)

    print('Prior: {}'.format(config['prior_value']))
    print('\n===================================\n')


def print_run_config(config):
    print("RUN CONFIG")
    print('Gibbs  Epochs: {}, Blocksize: {}'.format(config['n_epochs'], config['block_size']))
    print('\n===================================\n')
    print('Gibbs Ensemble | Burn In: {}, Thinning: {}'.format(config['burn_in'], config['thinning']))

    if config['mode'] == 'ga':
        print('GA Init Pop | Burn In: {}, Thinning: {}'.format(config['init_burn_in'], config['init_thinning']))
        if config['recombination'] == 'default':
            print('Recombination Method: {}, Crossover_P: {}, Mutation_P: {}'.format(config['recombination'], config['crossover_p'], config['mutation_p']))
        else:
            print('Recombination Method: {}, N_Neurons: {}, Layer-Wise: {}'.format(config['recombination'], config['n_neurons'], config['layer_wise']))
            if config['layer_wise'] == True:
                print('Gen Per Layer: {}, Layer Mutation: {}'.format(config['gen_per_layer'], config['p_layer_mutation']))
        print('Max GA Generations: {}, Population Size: {}, Fit_Individuals: {}, N_Recombinations: {}'.format(config['max_generations'], config['pop_size'], config['n_fit_individuals'], config['n_recombinations']))
        print('GA Ensemble | Burn In: {}, Thinning: {}'.format(config['ens_burn_in'], config['ens_thinning']))
    if config['mode'] == 'sa':
        print('SA Init Pop | Burn In: {}, Thinning: {}'.format(config['init_burn_in'], config['init_thinning']))
        print('Temperature | Start: {}, Epochs Per T: {}, Decrement: {}'.format(config['T_start'], config['epochs_per_T'], config['T_decremental']))
        print('Max Epochs: {}, Non Zero Temperature Epochs: {}'.format(config['max_epochs'], np.minimum(config['max_epochs'], int(np.ceil(config['T_start'] / config['T_decremental']) * config['epochs_per_T']))))
        print('SA Ensemble | Population Size: {}, Calculate Every {}th Epoch'.format(config['pop_size'], config['ens_calc']))
        print('Mutation P: {}'.format(config['mutation_p']))





def print_stats(text, values):
    values_arr = np.asarray(values)
    mean = np.mean(values_arr)
    std = np.std(values_arr)
    print(text + ' || Mean: {}\t Std: {}'.format(mean, std))
