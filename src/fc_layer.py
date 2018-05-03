import tensorflow as tf
import numpy as np
# from src.tf_tools import softmax_activation #get_parent_ops, get_operation_list
from src.sub_nn import SubNN


class FCLayer:
    # Here we create and initialize all variables that are independent of batch_size and block_size
    def __init__(self, shape, var_scope, config, layer_idx):
        self.config = config
        self.layer_idx = layer_idx
        self.n_neurons = shape[1]
        self.weight_shape = shape
        self.bias_shape = (1, self.n_neurons)
        self.var_scope = var_scope
        self.is_output = (layer_idx == len(config['layout']) - 2)
        if self.is_output == False:
            self.act_func = config['act_funcs'][layer_idx]
        self.input = None
        self.activation = None
        self.output = None
        self.new_output = None
        self.dropout_mask = None
        self.batch_size = None
        self.w_sample_op = None
        self.b_sample_op = None
        self.forward_op = None
        self.set_dropout_mask_op = None
        self.lookup_op = None
        self.lookup_table = None
        self.bias_vals = config['bias_vals'][layer_idx]

        if self.bias_vals is None:
            self.bias_vals = np.arange(-shape[0], shape[0]).astype(np.int32)

        self.n_bias_vals = len(self.bias_vals)

        with tf.variable_scope(var_scope):
            rng = np.random.RandomState()
            if self.config['weight_type'] == 'binary':
                w_vals = rng.binomial(n=1, p=0.5, size=self.weight_shape).astype(np.int32)
                w_vals[w_vals == 0] = -1
            elif self.config['weight_type'] == 'ternary':
                w_vals = rng.multinomial(n=1, pvals=np.asarray([1., 1., 1.])/3., size=self.weight_shape)\
                    .astype(np.int32)
                w_vals = np.cumsum(w_vals, axis=2)
                w_vals = np.sum(w_vals, axis=2) - 2
            b_vals = np.zeros(self.bias_shape, dtype=np.int32)

            w_init = tf.constant_initializer(w_vals)
            b_init = tf.constant_initializer(b_vals)

            self.W = tf.get_variable(name='W', shape=self.weight_shape, initializer=w_init, dtype=tf.int32)
            self.b = tf.get_variable(name='b', shape=self.bias_shape, initializer=b_init, dtype=tf.int32)
            self.neuron_idx = tf.placeholder(name='neuron_idx', dtype=np.int32)
            self.input_indices = None

    # Creates the execution graph for setting all the entries in the lookup table for a given neuron (this neuron is
    # given via a placeholder, which means that it can be set at runtime)
    def create_lookup_graph(self, lookup_table, lookup_indices, following_layers, targets):
        self.lookup_table = lookup_table

        # neuron_indices are required to access (and update) the output of a specific neuron in the variable containing
        # the output
        indices = np.reshape(np.arange(self.batch_size).astype(dtype=np.int32), newshape=(-1, 1))
        neuron_indices = tf.concat([indices, tf.tile(tf.expand_dims(self.neuron_idx, axis=1),
                                                     multiples=[self.batch_size, 1])], axis=1)

        # First we create a copy of the output variable, and just operate on this copy
        lookup_ops = [tf.assign(self.new_output, self.output)]

        # We iterate over all possible output values of the activation function
        # For each possible output value, we set the output of the current neuron to that value, and create a graph
        # for completing the forward pass to obtain the likelihoods. This graph is created by SubNN.
        # Then we update the lookup table
        # The control dependencies ensure that those operations are not run in parallel
        for lookup_idx, output_val in enumerate(self.act_func.values):
            with tf.control_dependencies(lookup_ops):
                out_update_op = tf.scatter_nd_update(self.new_output, neuron_indices,
                                                     tf.constant(output_val, shape=(self.batch_size,)))
                with tf.get_default_graph().control_dependencies([out_update_op]):
                    subnn = SubNN(following_layers, self.new_output, targets)
                    lookup_ops.append(tf.scatter_nd_update(lookup_table, lookup_indices[lookup_idx], subnn.likelihoods))

        # We simply set our lookup op to be the last operation in our list. This operation will automatically run
        # the others because of the control dependencies
        self.lookup_op = lookup_ops[-1]

    # This creates the execution graph that performs sampling
    def create_sampling_graph(self, block_size, connection_matrix, w_batch_range, log_pw, Y=None):
        # bias vals (which contains all considered values of the bias) needs to be tiled to fit the batch_siz
        # as the tensorflow operation operating on it does not support broadcasting
        self.bias_vals = np.tile(np.reshape(self.bias_vals, (1, -1)), reps=[self.batch_size, 1])

        # Like w_batch_range, this is used to construct indices to access the lookup table
        b_batch_range = np.tile(np.reshape(np.arange(self.batch_size), (-1, 1, 1)).astype(np.int32),
                                reps=(1, self.bias_vals.shape[1], 1))

        # input indices contain the indices of those input neurons, whose weights we want to sample
        with tf.variable_scope(self.var_scope):
            self.input_indices = tf.placeholder(name='input_indices', shape=(block_size, 1), dtype=np.int32)
            self.bias_vals = tf.constant(self.bias_vals, dtype=tf.int32)

        # Rest of this function is more or less a tensorflow version of the Theano code.
        # Variables that have 'indices' in their name, are used to index Tensorflow tensors either in the methods:
        # gather_nd (creates a new tensor by only taking those elements of the reference tensor, that are specified
        # in the index variable)
        # scatter_nd_update (updates elements - specified by indices - of a TF variable with the given update values)
        input_block = tf.transpose(tf.gather_nd(tf.transpose(self.input), self.input_indices))
        weight_indices = tf.concat([self.input_indices, tf.tile(tf.expand_dims(tf.expand_dims(self.neuron_idx, axis=0), axis=1),
                                                                (block_size, 1))], axis=1)
        weight_block = tf.expand_dims(tf.gather_nd(self.W, weight_indices), axis=1)
        neuron_activation = tf.slice(self.activation, begin=[0, self.neuron_idx], size=[self.batch_size, 1])

        w_removed_activation = neuron_activation - tf.matmul(input_block, weight_block)
        w_added_activation = tf.matmul(input_block, connection_matrix) + w_removed_activation

        b_removed_activation = neuron_activation - tf.gather_nd(self.b, [0, self.neuron_idx])
        b_added_activation = self.bias_vals + b_removed_activation

        update_var_indices = tf.concat([np.reshape(np.arange(self.batch_size), newshape=(-1, 1)),
                                       tf.tile(tf.expand_dims(tf.expand_dims(self.neuron_idx, axis=0), axis=1),
                                               (self.batch_size, 1))], axis=1)

        if self.is_output:
            activation = tf.cast(self.activation, dtype=tf.float32)
            neuron_activation = tf.cast(neuron_activation, dtype=tf.float32)
            w_added_activation_float = tf.cast(w_added_activation, dtype=tf.float32)
            b_added_activation_float = tf.cast(b_added_activation, dtype=tf.float32)

            aux_max1 = tf.reduce_max(activation, axis=1, keep_dims=True)
            smax_old = tf.log(tf.reduce_sum(tf.exp(activation - aux_max1), axis=1, keep_dims=True) -
                              tf.exp(neuron_activation - aux_max1)) + aux_max1
            aux_max2 = tf.maximum(smax_old, w_added_activation_float)
            smax_updated = -tf.log(tf.exp(smax_old - aux_max2) + tf.exp(w_added_activation_float - aux_max2)) - aux_max2
            t = tf.cast(tf.expand_dims(tf.argmax(Y, axis=1), axis=1), dtype=tf.int32)
            gather_indices = tf.concat((np.expand_dims(np.arange(self.batch_size), axis=1), t), axis=1)

            output_true_class = tf.expand_dims(tf.gather_nd(activation, gather_indices), axis=1)
            w_output_true_class = tf.tile(output_true_class, [1, connection_matrix.shape[1]])
            b_output_true_class = tf.tile(output_true_class, [1, self.n_bias_vals])
            aux_idx = tf.equal(t, self.neuron_idx)
            w_aux_idx = tf.tile(aux_idx, [1, connection_matrix.shape[1]])
            b_aux_idx = tf.tile(aux_idx, [1, self.n_bias_vals])

            log_probs = smax_updated + tf.where(w_aux_idx, w_added_activation_float, w_output_true_class)
            sample_idx = self.calc_sample_idx(log_probs, log_pw)
            new_activation_vals = tf.slice(w_added_activation, begin=[0, sample_idx], size=[self.batch_size, 1])
            new_weights = tf.slice(connection_matrix, begin=[0, sample_idx], size=[block_size, 1])
            sample_op = tf.scatter_nd_update(self.W, weight_indices, tf.squeeze(new_weights))
            self.w_sample_op = self.create_update_var_graph(sample_op, update_var_indices, new_activation_vals)

            aux_max3 = tf.maximum(smax_old, b_added_activation_float)
            smax_updated = -tf.log(tf.exp(smax_old - aux_max3) + tf.exp(b_added_activation_float - aux_max3)) - aux_max3
            log_probs = smax_updated + tf.where(b_aux_idx, b_added_activation_float, b_output_true_class)
            sample_idx = self.calc_sample_idx(log_probs)
            new_activation_vals = tf.slice(b_added_activation, begin=[0, sample_idx], size=[self.batch_size, 1])
            new_weight = tf.gather_nd(self.bias_vals, [0, sample_idx])
            sample_op = tf.scatter_nd_update(self.b, [[0, self.neuron_idx]], [new_weight])
            self.b_sample_op = self.create_update_var_graph(sample_op, update_var_indices, new_activation_vals)

        else:
            output_values = self.act_func.get_output(w_added_activation)
            lookup_indices = self.act_func.get_lookup_indices(w_added_activation)
            g_lookup_indices = tf.concat((w_batch_range, tf.expand_dims(lookup_indices, axis=2)), axis=2)
            sample_idx = self.calc_sample_idx(tf.gather_nd(self.lookup_table, g_lookup_indices), log_pw)
            new_weights = tf.slice(connection_matrix, begin=[0, sample_idx], size=[block_size, 1])
            new_output_vals = tf.slice(output_values, begin=[0, sample_idx], size=[self.batch_size, 1])
            new_activation_vals = tf.slice(w_added_activation, begin=[0, sample_idx], size=[self.batch_size, 1])
            sample_op = tf.scatter_nd_update(self.W, weight_indices, tf.squeeze(new_weights))
            self.w_sample_op = self.create_update_var_graph(sample_op, update_var_indices, new_activation_vals,
                                                            new_output_vals)

            output_values = self.act_func.get_output(b_added_activation)
            lookup_indices = self.act_func.get_lookup_indices(b_added_activation)
            g_lookup_indices = tf.concat((b_batch_range, tf.expand_dims(lookup_indices, axis=2)), axis=2)
            sample_idx = self.calc_sample_idx(tf.gather_nd(self.lookup_table, g_lookup_indices))
            new_weight = tf.gather_nd(self.bias_vals, [0, sample_idx])
            new_output_vals = tf.slice(output_values, begin=[0, sample_idx], size=[self.batch_size, 1])
            new_activation_vals = tf.slice(b_added_activation, begin=[0, sample_idx], size=[self.batch_size, 1])
            sample_op = tf.scatter_nd_update(self.b, [[0, self.neuron_idx]], [new_weight])
            self.b_sample_op = self.create_update_var_graph(sample_op, update_var_indices, new_activation_vals,
                                                            new_output_vals)

    # This updates activation and output variables after the sampling operation
    # As a control dependency is made on control op, the returned operation both performs sampling and the update
    # of variables
    def create_update_var_graph(self, control_op, update_indices, new_activation_vals, new_output_vals=None):
        new_ops = []
        with tf.control_dependencies([control_op]):
            new_ops.append(tf.scatter_nd_update(self.activation, update_indices, tf.squeeze(new_activation_vals)))
            if new_output_vals is not None:
                new_ops.append(tf.scatter_nd_update(self.output, update_indices, tf.squeeze(new_output_vals)))
            op = tf.group(*new_ops)
        return op

    # Calculate the sample index for given logprobs (before it is reduced over batch)
    # Currently prior is ignored and mode is taken instead of sampling for testint
    # TODO: After testing remove taking the mode and add those priors
    def calc_sample_idx(self, log_probs, log_pw=None):
        if log_pw is None:
            log_probs = tf.reduce_sum(log_probs, axis=0) / self.config['flat_factor'][self.layer_idx]
        else:
            log_probs = (tf.reduce_sum(log_probs, axis=0)) / self.config['flat_factor'][self.layer_idx]

        probs = tf.cumsum(tf.exp(log_probs - tf.reduce_max(log_probs)))
        sample_idx = tf.reduce_sum(tf.cast(tf.less(probs, tf.random_uniform((1,))*tf.reduce_max(probs)), tf.int32))
        sample_idx = tf.argmax(log_probs)
        return sample_idx


    # Creates TF variables required during Gibbs sampling, for storing input, activation and output.
    # There are two output variables, one actually stores the output, the other is tampered when updating lookup table.
    # This function also creates the operation to fill the variables
    def create_variables(self, layer_input, batch_size, dropout_mask=None):
        self.batch_size = batch_size

        with tf.variable_scope(self.var_scope):
            var_shape = (self.batch_size, self.n_neurons)
            input_shape = layer_input.shape

            if self.is_output:
                d_type = tf.float32
            else:
                d_type = tf.int32

            self.input = tf.get_variable(name='input', shape=input_shape, dtype=tf.int32)
            self.activation = tf.get_variable(name='activation', shape=var_shape, dtype=tf.int32)
            self.new_output = tf.get_variable(name='new_output', shape=var_shape, dtype=tf.int32)
            self.output = tf.get_variable(name='output', shape=var_shape, dtype=d_type)


            if self.is_output == False:
                dropout_init = tf.constant_initializer(np.ones((1, self.n_neurons)).astype(np.int32))
                self.dropout_mask = tf.get_variable(name='dropout', shape=(1, self.n_neurons), initializer=dropout_init,
                                                dtype=tf.int32)
                self.set_dropout_mask_op = tf.assign(self.dropout_mask, dropout_mask)

        input_op = tf.assign(self.input, layer_input).op
        activation = tf.matmul(layer_input, self.W) + self.b
        act_op = tf.assign(self.activation, activation).op

        activation = tf.cast(activation, d_type)
        if self.is_output == False:
            output = self.act_func.get_output(activation)
            output = tf.multiply(output, self.dropout_mask)
            output_op = tf.assign(self.output, output).op

            # Calling this operation will set the variables
            # This operation (aswell as the variables) are not used when we want to do a quick forward pass, only when
            # we want to access and keep a backup of the input, activation and output of each layer (so usually just during
            # Gibbs sampling)
            self.forward_op = tf.group(*[input_op, act_op, output_op])
        else:
            self.forward_op = tf.group(*[input_op, act_op])


    # This creates the forward pass graph of a classic feed forward network
    # No TF variables are used to store any interim values here. It can be used also when the current layer holds
    # different values for input, activation or output in it's variables. But it reuses the weights.
    # Number of return values depends on whether it's an output layer or not
    def forward_pass(self, layer_input, targets=None):
        activation = tf.matmul(layer_input, self.W) + self.b
        if self.is_output:
            activation = tf.cast(activation, dtype=tf.float32)
            return tf.nn.softmax(activation), \
                   tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=targets)
        else:
            layer_output = self.act_func.get_output(activation)
            return tf.multiply(layer_output, self.dropout_mask)






