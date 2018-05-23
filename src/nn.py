import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from src.fc_layer import FCLayer
from src.sub_nn import SubNN

class NN:
    def __init__(self, config):
        self.config = config
        self.weight_type = config['weight_type']
        self.layers = []
        self.dropout_masks = []
        self.n_layers = len(config['layout']) - 1
        self.act_func = config['act_funcs'][0]
        self.lookup_table = None
        self.log_prior_w = None
        self.likelihoods = None
        self.likelihood_op = None
        self.load_train_set_op = None
        self.load_val_set_op = None
        self.set_dropout_masks_op = None
        self.batch_size = None
        # Number of weights that are sampled at once
        self.block_size = None
        self.full_network = None
        self.var_summary_op = None

        for layer_idx in range(self.n_layers):
            shape = (config['layout'][layer_idx], config['layout'][layer_idx+1])
            self.layers.append(FCLayer(shape, 'layer' + str(layer_idx + 1), config, layer_idx))

        self.X_tr = None
        self.Y_tr = None
        self.X_val = None
        self.Y_val = None
        self.X_placeholder = None
        self.Y_placeholder = None
        self.validate = None

        self.compute_log_priors()

    def compute_log_priors(self):
        if self.weight_type == 'binary':
            self.log_prior_w = np.log(np.asarray([0.5, 0.5], dtype=np.float32))
        elif self.weight_type == 'ternary':
            self.log_prior_w = np.log(np.asarray([0.1, 0.8, 0.1], dtype=np.float32))

    # This function creates the execution graph for Gibbs Sampling (lookup table updates, weight update, bias update)
    # TODO: Load validation set into gpu
    def create_gibbs_graph(self, batch_size, n_val_samples, block_size):
        with tf.variable_scope('global'):
            self.X_placeholder = tf.placeholder(tf.float32, [None, self.config['layout'][0]])
            self.Y_placeholder = tf.placeholder(tf.float32, [None, self.config['layout'][self.n_layers]])
            self.X_tr = tf.get_variable('X_tr', shape=[batch_size, self.config['layout'][0]], dtype=tf.float32)
            self.Y_tr = tf.get_variable('Y_tr', shape=[batch_size, self.config['layout'][self.n_layers]], dtype=tf.float32)
            self.X_val = tf.get_variable('X_val', shape=[n_val_samples, self.config['layout'][0]], dtype=tf.float32)
            self.Y_val = tf.get_variable('Y_val', shape=[n_val_samples, self.config['layout'][self.n_layers]], dtype=tf.float32)
            self.validate = tf.placeholder(tf.bool)

            op1 = tf.assign(self.X_tr, self.X_placeholder)
            op2 = tf.assign(self.Y_tr, self.Y_placeholder)
            self.load_train_set_op = tf.group(*[op1, op2])

            op1 = tf.assign(self.X_val, self.X_placeholder)
            op2 = tf.assign(self.Y_val, self.Y_placeholder)
            self.load_val_set_op = tf.group(*[op1, op2])

        self.batch_size = batch_size
        self.block_size = block_size

        # Create connection matrix and corresponding prior probabilities
        # connection_matrix contains all possible combinations of weights that can be obtained
        # by one sampling process
        # log_pw contains prior probabilities
        if self.weight_type == 'binary':
            connection_matrix_shape = (block_size, 2**block_size)
            connection_matrix = np.zeros(connection_matrix_shape, dtype=np.float32)
            for i in range(2**block_size):
                for j in range(block_size):
                    connection_matrix[j, i] = (i & (1 << j)) == 0
            log_pw = np.sum(self.log_prior_w[connection_matrix.astype(np.int32)], axis=0)
            connection_matrix[connection_matrix == 0] = -1
            n_input_var = 2 ** block_size
        elif self.weight_type == 'ternary':
            connection_matrix_shape = (block_size, 3**block_size)
            connection_matrix = np.zeros(connection_matrix_shape, dtype=np.float32)
            for i in range(3 ** block_size):
                for j in range(block_size):
                    connection_matrix[j, i] = np.asarray(int(i / (3 ** j)) % 3)
            log_pw = np.sum(self.log_prior_w[connection_matrix.astype(np.int32)], axis= 0)
            connection_matrix -= 1
            n_input_var = 3 ** block_size

        # Create lookup table
        with tf.variable_scope('global'):
            lookup_shape = (batch_size, self.act_func.n_values)
            lookup_init = tf.constant_initializer(np.zeros(lookup_shape, dtype=np.float32))
            tiled_lookup_shape = (batch_size, self.act_func.n_values, n_input_var)
            self.lookup_table = tf.get_variable('lookup', shape=lookup_shape, initializer=lookup_init)
            tiled_lookup_table = tf.get_variable('tiled_lookup', shape=tiled_lookup_shape)

        # Create indices required to update lookup_table
        indices = np.tile(np.reshape(np.arange(batch_size).astype(dtype=np.int32), newshape=(1, -1, 1)),
                          reps=[self.act_func.n_values, 1, 1])
        act_indices = np.zeros((1, batch_size, 1), dtype=np.int32)
        for lookup_index in range(1, self.act_func.n_values):
            act_indices = np.concatenate((act_indices, np.ones((1, batch_size, 1), dtype=np.int32) * lookup_index),
                                         axis=0)
        lookup_indices = np.concatenate((indices, act_indices), axis=2)

        # This variable is used to create the indices to access the lookup_table
        batch_range = np.tile(np.reshape(np.arange(batch_size), (-1, 1, 1)).astype(np.int32),
                              reps=(1, connection_matrix_shape[1], 1))

        # This creates variables in the layers that hold input, activation and output and also the necessary
        # operations to update them.
        set_dropout_ops = []
        layer_input = self.X_tr
        for layer_idx in range(self.n_layers):
            if layer_idx != self.n_layers - 1:
                dropout_mask = tf.placeholder(tf.float32, shape=(1, self.config['layout'][layer_idx +1]))
                self.layers[layer_idx].create_variables(layer_input, batch_size, dropout_mask)
                self.dropout_masks.append(dropout_mask)
                set_dropout_ops.append(self.layers[layer_idx].set_dropout_mask_op)
            else:
                self.layers[layer_idx].create_variables(layer_input, batch_size)
            layer_input = self.layers[layer_idx].output
        self.set_dropout_masks_op = tf.group(*set_dropout_ops)

        # Iterate through all layers and create the graph for updating the lookup table aswell as the graph that
        # performs the sampling
        # also collect summary ops
        var_summary_ops = []
        for layer_idx in range(self.n_layers):
            if layer_idx != self.n_layers - 1:
                self.layers[layer_idx].create_lookup_graph(self.lookup_table, tiled_lookup_table, lookup_indices,
                                                           self.layers[layer_idx+1:], self.Y_tr)
            self.layers[layer_idx].create_sampling_graph(block_size, connection_matrix, batch_range, log_pw,
                                                         self.Y_tr)
            var_summary_ops.append(self.layers[layer_idx].var_summary_op)
        self.var_summary_op = tf.summary.merge(var_summary_ops)

        # This network allows for a fast forward pass using all current weights. This network is not affected
        # and does not affect the layer variables that store input, activation and ouput
        x = tf.cond(self.validate, lambda: self.X_val, lambda: self.X_tr)
        y = tf.cond(self.validate, lambda: self.Y_val, lambda: self.Y_tr)
        self.full_network = SubNN(self.layers, x, y, True)

    def get_misclassification(self, sess, validation):
        return 1 - sess.run(self.full_network.accuracy, feed_dict={self.validate: validation})


    # Performs a forward pass using the dataset X and returns a list containing the histograms (one per layer) of the
    # activations
    def get_activation_histogram(self, sess):
        layer_activations = []
        for layer_idx in range(self.n_layers):
            curr_layer = self.layers[layer_idx]
            sess.run(curr_layer.forward_op)

            activation = sess.run(curr_layer.activation)
            unique, counts = np.unique(activation, return_counts=True)
            layer_activations.append(np.concatenate([np.expand_dims(unique, axis=1),
                                                     np.expand_dims(counts, axis=1)], axis=1))
        return layer_activations

    # This function performs one complete iteration of gibbs sampling
    def perform_gibbs_iteration(self, sess):
        # Compute dropout masks for each layer
        dropout_masks = []
        for layer_idx in range(self.n_layers-1):
            tries = 20
            while tries > 0:
                mask = np.random.binomial(n=1, p=self.config['keep_probs'][layer_idx],
                                          size=(1, self.config['layout'][layer_idx + 1])).astype(np.float32)
                if np.sum(mask) >= self.block_size:
                    dropout_masks.append(np.divide(mask, self.config['keep_probs'][layer_idx]))
                    break

                tries -= 1
                if tries == 0:
                    raise Exception('Could not generate dropout mask with at least block_size neurons active')
        sess.run(self.set_dropout_masks_op, feed_dict={i: d for i, d in zip(self.dropout_masks, dropout_masks)})

        for layer_idx in range(self.n_layers):
            curr_layer = self.layers[layer_idx]

            if layer_idx > 0:
                input_perm = np.flatnonzero(dropout_masks[layer_idx-1])
                n_inputs = len(input_perm)
            else:
                n_inputs = self.config['layout'][layer_idx]
                input_perm = np.arange(n_inputs)
            input_perm = np.expand_dims(input_perm, axis=1)

            if layer_idx < self.n_layers - 1:
                neuron_perm = np.flatnonzero(dropout_masks[layer_idx])
                n_neurons = len(neuron_perm)
            else:
                n_neurons = self.config['layout'][layer_idx+1]
                neuron_perm = np.arange(n_neurons)

            # forward_op updates the variables containing input, activation and output of the current layer, using
            # either the output of the previous layer or the input data
            sess.run(curr_layer.forward_op)

            if self.config['sampling_sequence'] == 'stochastic':
                np.random.shuffle(neuron_perm)

            for neuron_idx in range(n_neurons):
                # Perform the lookup operation for non output layers and do the bias sampling first
                if layer_idx != self.n_layers - 1:
                    sess.run(curr_layer.lookup_op,
                             feed_dict={curr_layer.neuron_idx: np.expand_dims(neuron_perm[neuron_idx], axis=1)})

                #sess.run(curr_layer.b_sample_op, feed_dict={curr_layer.neuron_idx: neuron_perm[neuron_idx]})

                # For each block of input connections sample the weights
                if self.config['sampling_sequence'] == 'stochastic':
                    np.random.shuffle(input_perm)

                for block_idx in range(0, n_inputs, self.block_size):
                    if block_idx + self.block_size > n_inputs:
                        block_start_idx = n_inputs - self.block_size
                    else:
                        block_start_idx = block_idx
                    block_end_idx = block_start_idx + self.block_size
                    block_range = range(block_start_idx, block_end_idx)

                    sess.run(curr_layer.w_sample_op,
                             feed_dict={curr_layer.neuron_idx: neuron_perm[neuron_idx],
                                        curr_layer.input_indices: input_perm[block_range, :]})

    def profile_gibbs_iteration(self, sess, options, run_metadata, epoch, filepath):
        trace_path = filepath + 'e_' + str(epoch) + '_'
        # Compute dropout masks for each layer
        dropout_masks = []
        for layer_idx in range(self.n_layers-1):
            tries = 20
            while tries > 0:
                mask = np.random.binomial(n=1, p=self.config['keep_probs'][layer_idx],
                                          size=(1, self.config['layout'][layer_idx + 1])).astype(np.float32)
                if np.sum(mask) >= self.block_size:
                    dropout_masks.append(mask)
                    break

                tries -= 1
                if tries == 0:
                    raise Exception('Could not generate dropout mask with at least block_size neurons active')
        sess.run(self.set_dropout_masks_op, feed_dict={i: d for i, d in zip(self.dropout_masks, dropout_masks)},
                 options=options, run_metadata=run_metadata)

        do_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
        pass_traces = []
        lookup_traces = []
        sample_block_traces  = []
        sample_bias_traces = []

        for layer_idx in range(self.n_layers):
            print('layer {}'.format(layer_idx))
            curr_layer = self.layers[layer_idx]

            if layer_idx > 0:
                n_inputs = np.sum(dropout_masks[layer_idx-1]).astype(np.int32)
                input_perm = np.flatnonzero(dropout_masks[layer_idx-1])
            else:
                n_inputs = self.config['layout'][layer_idx]
                input_perm = np.arange(n_inputs)
            input_perm = np.expand_dims(input_perm, axis=1)

            if layer_idx < self.n_layers - 1:
                n_neurons = np.sum(dropout_masks[layer_idx]).astype(np.int32)
                neuron_perm = np.flatnonzero(dropout_masks[layer_idx])
            else:
                n_neurons = self.config['layout'][layer_idx+1]
                neuron_perm = np.arange(n_neurons)


            # forward_op updates the variables containing input, activation and output of the current layer, using
            # either the output of the previous layer or the input data
            sess.run(curr_layer.forward_op, options=options, run_metadata=run_metadata)
            pass_traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

            if self.config['sampling_sequence'] == 'stochastic':
                np.random.shuffle(neuron_perm)

            profile_neuron_idx = np.random.randint(0, n_neurons)
            profile_block_idx = np.random.randint(0, n_inputs - self.block_size)

            for neuron_idx in range(n_neurons):
                # Perform the lookup operation for non output layers and do the bias sampling first
                if layer_idx != self.n_layers - 1:
                    sess.run(curr_layer.lookup_op,
                             feed_dict={curr_layer.neuron_idx: np.expand_dims(neuron_perm[neuron_idx], axis=1)},
                             options=options, run_metadata=run_metadata)
                    if neuron_idx == profile_neuron_idx:
                        lookup_traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

                sess.run(curr_layer.b_sample_op, feed_dict={curr_layer.neuron_idx: neuron_perm[neuron_idx]},
                         options=options, run_metadata=run_metadata)
                if neuron_idx == profile_neuron_idx:
                    sample_bias_traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

                # For each block of input connections sample the weights
                if self.config['sampling_sequence'] == 'stochastic':
                    np.random.shuffle(input_perm)

                for block_idx in range(0, n_inputs, self.block_size):
                    if block_idx + self.block_size > n_inputs:
                        block_start_idx = n_inputs - self.block_size
                    else:
                        block_start_idx = block_idx
                    block_end_idx = block_start_idx + self.block_size
                    block_range = range(block_start_idx, block_end_idx)

                    sess.run(curr_layer.w_sample_op,
                             feed_dict={curr_layer.neuron_idx: neuron_perm[neuron_idx],
                                        curr_layer.input_indices: input_perm[block_range, :]},
                             options=options, run_metadata=run_metadata)
                    if neuron_idx == profile_neuron_idx and block_end_idx > profile_block_idx:
                        sample_block_traces.append(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())

        with open(trace_path + 'dropout.json', 'w') as f:
            f.write(do_trace)

        for layer_idx in range(len(lookup_traces)):
            path = trace_path + 'l' + str(layer_idx) + '_'
            with open(path + 'lookup.json', 'w') as f:
                f.write(lookup_traces[layer_idx])
            with open(path + 'pass.json', 'w') as f:
                f.write(pass_traces[layer_idx])
            with open(path + 'bsample.json', 'w') as f:
                f.write(sample_bias_traces[layer_idx])
            with open(path + 'blocksample.json', 'w') as f:
                f.write(sample_block_traces[layer_idx])


