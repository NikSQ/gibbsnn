import tensorflow as tf
import numpy as np
# from src.tf_tools import softmax_activation #get_parent_ops, get_operation_list
from src.sub_nn import SubNN


class FCLayer:
    # Here we create and initialize all variables that are independent of batch_size and block_size
    def __init__(self, shape, var_scope, config, layer_idx, w_init_vals, b_init_vals):
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
        self.sample_op = None
        self.forward_op = None
        self.set_dropout_mask_op = None
        self.lookup_op = None
        self.lookup_table = None
        self.bias_vals = config['bias_vals'][layer_idx]
        self.var_summary_op = None

        if self.bias_vals is None:
            #bits = int(np.log2(shape[0]))
            #self.bias_vals = np.power(np.arange(-bits, bits), 2).astype(np.int32)
            b_range = 10 * int(np.sqrt(shape[0]))
            self.bias_vals = np.arange(-b_range, b_range)

        self.n_bias_vals = len(self.bias_vals)

        with tf.variable_scope(var_scope):
            rng = np.random.RandomState()

            if w_init_vals is None:
                if self.config['weight_type'] == 'binary':
                    w_init_vals = rng.binomial(n=1, p=0.5, size=self.weight_shape).astype(np.float32)
                    w_init_vals[w_init_vals == 0] = -1
                elif self.config['weight_type'] == 'ternary':
                    w_init_vals = rng.multinomial(n=1, pvals=np.asarray([1., 1., 1.])/3., size=self.weight_shape)\
                        .astype(np.float32)
                    w_init_vals = np.cumsum(w_init_vals, axis=2)
                    w_init_vals = np.sum(w_init_vals, axis=2) - 2
            if b_init_vals is None:
                b_init_vals = np.zeros(self.bias_shape, dtype=np.int32)

            w_init = tf.constant_initializer(w_init_vals)
            b_init = tf.constant_initializer(b_init_vals)

            self.W = tf.get_variable(name='W', shape=self.weight_shape, initializer=w_init, dtype=tf.float32)
            self.b = tf.get_variable(name='b', shape=self.bias_shape, initializer=b_init, dtype=tf.float32)
            self.saver = tf.train.Saver({'W' + str(layer_idx): self.W, 'b' + str(layer_idx): self.b})
            self.neuron_idx = tf.placeholder(name='neuron_idx', dtype=np.int32)
            self.input_indices = None

            summary_ops = []
            summary_ops.append(tf.summary.histogram('weights', self.W))
            summary_ops.append(tf.summary.histogram('biases', self.b))
            self.var_summary_op = tf.summary.merge(summary_ops)

    def create_var_assign_op(self, w_vals):
        w_assign_op = tf.assign(self.W, tf.cast(w_vals, dtype=tf.float32))
        return w_assign_op
        #b_assign_op = tf.assign(self.b, b_vals)
        #return tf.group(*[w_assign_op, b_assign_op])

    # Creates the execution graph for setting all the entries in the lookup table for a given neuron (this neuron is
    # given via a placeholder, which means that it can be set at runtime)
    def create_lookup_graph(self, lookup_table, tiled_lookup_table, lookup_indices, following_layers, targets):
        with tf.name_scope(self.var_scope + '/lookup'):
            indices = np.reshape(np.arange(self.batch_size).astype(dtype=np.int32), newshape=(-1, 1))
            neuron_indices = tf.concat([indices, tf.tile(tf.expand_dims(self.neuron_idx, axis=1),
                                                         multiples=[self.batch_size, 1])], axis=1)

            lookup_ops = [tf.assign(self.new_output, self.output)]
            for lookup_idx, output_val in enumerate(self.act_func.values):
                with tf.control_dependencies(lookup_ops):
                    out_update_op = tf.scatter_nd_update(self.new_output, neuron_indices,
                                                         tf.constant(output_val, shape=(self.batch_size,)))
                    with tf.get_default_graph().control_dependencies([out_update_op]):
                        subnn = SubNN(following_layers, self.new_output, targets)
                        lookup_ops.append(tf.scatter_nd_update(lookup_table, lookup_indices[lookup_idx], subnn.likelihoods))

            with tf.control_dependencies([lookup_ops[-1]]):
                op = tf.assign(tiled_lookup_table, tf.tile(tf.expand_dims(lookup_table, axis=2), multiples=[1, 1, tf.shape(tiled_lookup_table)[2]]))
                self.lookup_table = tiled_lookup_table

        self.lookup_op = op

    # This creates the execution graph that performs sampling
    def create_sampling_graph(self, block_size, connection_matrix, w_batch_range, log_pw, Y=None):
        with tf.name_scope(self.var_scope + '/sampling'):
            # bias vals (which contains all considered values of the bias) needs to be tiled to fit the batch_siz
            # as the tensorflow operation operating on it does not support broadcasting
            self.bias_vals = np.tile(np.reshape(self.bias_vals, (1, -1)), reps=[self.batch_size, 1])

            # Like w_batch_range, this is used to construct indices to access the lookup table
            b_batch_range = np.tile(np.reshape(np.arange(self.batch_size), (-1, 1, 1)).astype(np.int32),
                                    reps=(1, self.bias_vals.shape[1], 1))

            # input indices contain the indices of those input neurons, whose weights we want to sample
            with tf.variable_scope(self.var_scope):
                self.input_indices = tf.placeholder(name='input_indices', shape=(block_size, 1), dtype=np.int32)
                self.bias_vals = tf.constant(self.bias_vals, dtype=tf.float32)
                w_batch_range = tf.constant(w_batch_range, dtype=tf.int32)
                b_batch_range = tf.constant(b_batch_range, dtype=tf.int32)
                #l_range = tf.constant(np.reshape(np.arange(self.batch_size).astype(np.int32), (-1,)) * self.act_func.n_values, dtype=np.int32)

            input_block = tf.transpose(tf.gather_nd(tf.transpose(self.input, name='inp_block_trans1'), self.input_indices, name='inp_block_gather'), name='inp_block_trans2')
            weight_indices = tf.concat([self.input_indices, tf.tile(tf.expand_dims(tf.expand_dims(self.neuron_idx, axis=0), axis=1),
                                                                    (block_size, 1), name='w_ind_tile')], axis=1, name='w_ind_concat')
            weight_block = tf.expand_dims(tf.gather_nd(self.W, weight_indices, name='w_block_gather'), axis=1)
            neuron_activation = tf.slice(self.activation, begin=[0, self.neuron_idx], size=[self.batch_size, 1], name='extract_neuron_activation')

            w_removed_activation = neuron_activation - tf.matmul(input_block, weight_block, name='curr_inp_block_influence')
            w_added_activation = tf.matmul(input_block, connection_matrix, name='calc_inp_block_influence') + w_removed_activation

            b_removed_activation = neuron_activation - tf.gather_nd(self.b, [0, self.neuron_idx], name='extract_bias_activation')
            b_added_activation = self.bias_vals + b_removed_activation

            update_var_indices = tf.concat([np.reshape(np.arange(self.batch_size), newshape=(-1, 1)),
                                           tf.tile(tf.expand_dims(tf.expand_dims(self.neuron_idx, axis=0), axis=1),
                                                   (self.batch_size, 1), name='var_indices_tile')], axis=1, name='var_indices_concat')

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
                bias_value = tf.gather_nd(self.b, (0, self.neuron_idx))
                act_means = tf.reduce_mean(w_added_activation, axis=0)
                correction = act_means
                w_added_activation = w_added_activation - correction + \
                                     tf.random_normal(tf.shape(w_added_activation), mean=0.0,
                                                      stddev=self.config['act_noise'][self.layer_idx])
                output_values = self.act_func.get_output(w_added_activation)
                lookup_indices = self.act_func.get_lookup_indices(w_added_activation)

                log_probs = tf.where(tf.equal(lookup_indices, 0), self.lookup_table[:, 0, :],
                                     self.lookup_table[:, self.act_func.n_values - 1, :])
                for idx, value in enumerate(self.act_func.values):
                    if idx + 1 != self.act_func.n_values and idx > 0:
                        log_probs = tf.where(tf.equal(lookup_indices, idx), self.lookup_table[:, idx, :], log_probs)
                sample_idx = self.calc_sample_idx(log_probs, log_pw)

                #g_lookup_indices = tf.concat((w_batch_range, tf.expand_dims(lookup_indices, axis=2, name='lookup_exp')), axis=2, name='look_ind_concat')
                #sample_idx = self.calc_sample_idx(tf.gather_nd(self.lookup_table, g_lookup_indices, name='lookup_w'), log_pw)
                new_weights = tf.slice(connection_matrix, begin=[0, sample_idx], size=[block_size, 1], name='get_new_weight')
                new_output_vals = tf.slice(output_values, begin=[0, sample_idx], size=[self.batch_size, 1])
                new_activation_vals = tf.slice(w_added_activation, begin=[0, sample_idx], size=[self.batch_size, 1])
                sample_op = tf.scatter_nd_update(self.W, weight_indices, tf.squeeze(new_weights), name='sample_w')
                w_sample_op = self.create_update_var_graph(sample_op, update_var_indices, new_activation_vals,
                                                                new_output_vals)
                b_sample_op = tf.scatter_nd_update(self.b, [[0, self.neuron_idx]], [bias_value-correction[sample_idx]], name='sample_b')
                self.sample_op = tf.group(*[w_sample_op, b_sample_op])

                output_values = self.act_func.get_output(b_added_activation)
                lookup_indices = self.act_func.get_lookup_indices(b_added_activation)
                g_lookup_indices = tf.concat((b_batch_range, tf.expand_dims(lookup_indices, axis=2)), axis=2, name='look_ind_concat')
                sample_idx = self.calc_sample_idx(tf.gather_nd(self.lookup_table, g_lookup_indices, name='lookup_b'))
                new_weight = tf.gather_nd(self.bias_vals, [0, sample_idx], name='get_new_weight')
                new_output_vals = tf.slice(output_values, begin=[0, sample_idx], size=[self.batch_size, 1])
                new_activation_vals = tf.slice(b_added_activation, begin=[0, sample_idx], size=[self.batch_size, 1])
                sample_op = tf.scatter_nd_update(self.b, [[0, self.neuron_idx]], [new_weight], name='sample_b')
                #self.b_sample_op = self.create_update_var_graph(sample_op, update_var_indices, new_activation_vals,
                                                                #new_output_vals)

    # This updates activation and output variables after the sampling operation
    # As a control dependency is made on control op, the returned operation both performs sampling and the update
    # of variables
    def create_update_var_graph(self, control_op, update_indices, new_activation_vals, new_output_vals=None):
        new_ops = []
        with tf.control_dependencies([control_op]):
            new_ops.append(tf.scatter_nd_update(self.activation, update_indices, tf.squeeze(new_activation_vals), name='activation_update'))
            if new_output_vals is not None:
                new_ops.append(tf.scatter_nd_update(self.output, update_indices, tf.squeeze(new_output_vals), name='output_update'))
            op = tf.group(*new_ops)
        return op

    # Calculate the sample index for given logprobs (before it is reduced over batch)
    # Currently prior is ignored and mode is taken instead of sampling for testint
    # TODO: After testing remove taking the mode and add those priors
    def calc_sample_idx(self, log_probs, log_pw=None):
        if log_pw is None:
            log_probs = tf.reduce_sum(log_probs, axis=0, name='calc_mean_log_prob') / self.config['flat_factor'][self.layer_idx]
        else:
            log_probs = (tf.reduce_sum(log_probs, axis=0, name='calc_mean_log_prob') + self.log_pw) / self.config['flat_factor'][self.layer_idx]

        probs = tf.cumsum(tf.exp(log_probs - tf.reduce_max(log_probs)))
        sample_idx = tf.reduce_sum(tf.cast(tf.less(probs, tf.random_uniform((1,))*tf.reduce_max(probs)), tf.int32))
        #sample_idx = tf.argmax(log_probs)
        return sample_idx


    # Creates TF variables required during Gibbs sampling, for storing input, activation and output.
    # There are two output variables, one actually stores the output, the other is tampered when updating lookup table.
    # This function also creates the operation to fill the variables
    def create_variables(self, layer_input, batch_size, log_pw, dropout_mask=None):
        self.batch_size = batch_size
        self.log_pw = log_pw

        with tf.variable_scope(self.var_scope):
            var_shape = (self.batch_size, self.n_neurons)
            input_shape = layer_input.shape


            self.input = tf.get_variable(name='input', shape=input_shape, dtype=tf.float32)
            self.activation = tf.get_variable(name='activation', shape=var_shape, dtype=tf.float32)
            self.new_output = tf.get_variable(name='new_output', shape=var_shape, dtype=tf.float32)
            self.output = tf.get_variable(name='output', shape=var_shape, dtype=tf.float32)

            if self.is_output == False:
                dropout_init = tf.constant_initializer(np.ones((1, self.n_neurons)).astype(np.float32))
                self.dropout_mask = tf.get_variable(name='dropout', shape=(1, self.n_neurons), initializer=dropout_init,
                                                dtype=tf.float32)
                self.set_dropout_mask_op = tf.assign(self.dropout_mask, dropout_mask)

        with tf.name_scope(self.var_scope + '/slow_forward_pass'):
            input_op = tf.assign(self.input, layer_input).op
            activation = tf.matmul(layer_input, self.W) + self.b
            act_op = tf.assign(self.activation, activation).op

            activation = tf.cast(activation, tf.float32)

            if self.is_output == False:
                output = self.act_func.get_output(activation)
                output = tf.multiply(output, self.dropout_mask)
                output_op = tf.assign(self.output, output).op

                self.forward_op = tf.group(*[input_op, act_op, output_op])
            else:
                self.forward_op = tf.group(*[input_op, act_op])


    # This creates the forward pass graph of a classic feed forward network
    # No TF variables are used to store any interim values here. It can be used also when the current layer holds
    # different values for input, activation or output in it's variables. But it reuses the weights.
    # Number of return values depends on whether it's an output layer or not
    def forward_pass(self, layer_input, record_variables, targets=None):
        with tf.name_scope(self.var_scope + '/fast_forward_pass'):
            activation = tf.matmul(layer_input, self.W) + self.b
            act_summary = None
            if record_variables:
                act_summary = tf.summary.histogram('activations', activation)

            if self.is_output:
                return tf.nn.softmax(activation), \
                       tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=targets), act_summary, \
                       activation
            else:
                layer_output = self.act_func.get_output(activation)
                if record_variables:
                    return layer_output, act_summary
                return tf.multiply(layer_output, self.dropout_mask), act_summary

    def adapt_bias(self, layer_input):
        activation = tf.matmul(layer_input, self.W)
        act_means = tf.reduce_mean(activation, axis=0, keep_dims=True)
        assign_op = tf.assign(self.b, -act_means)
        if self.is_output:
            return activation, assign_op

        output = self.act_func.get_output(activation - act_means)
        return output, assign_op







