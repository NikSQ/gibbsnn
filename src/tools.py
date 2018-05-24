def print_nn_config(config):
    print('NN CONFIG')

    print('Weight-type:\t\t\t{}'.format(config['weight_type']))

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
    print('\n===================================\n')


def print_run_config(config):
    print("RUN CONFIG")
    print('Epochs: {}, Blocksize: {}'.format(config['n_epochs'], config['block_size']))
    print('\n===================================\n')
