import tensorflow as tf
import numpy as np
from src.fc_layer import FCLayer
from src.sub_nn import SubNN

class StaticNN:
    def __init__(self, config, x_tr, y_tr, x_va, y_va):
        self.config = config
        self.layers = []
        self.n_layers = len(config['layout']) - 1
        self.w_init_vals = []

        for layer_idx in range(self.n_layers):
            shape = (config['layout'][layer_idx], config['layout'][layer_idx+1])
            self.layers.append(FCLayer(shape, 'layer' + str(layer_idx + 1), config, layer_idx, None,
                                       None))
            self.w_init_vals.append(tf.placeholder(shape=self.layers[-1].weight_shape, dtype=tf.int32))

        self.validate = tf.placeholder(dtype=bool)
        x = tf.cond(self.validate, lambda: x_va, lambda: x_tr)
        y = tf.cond(self.validate, lambda: y_va, lambda: y_tr)
        self.full_network = SubNN(self.layers, x, y, True, True)

        assign_ops =[]
        for layer_idx in range(self.n_layers):
            assign_ops.append(self.layers[layer_idx].create_var_assign_op(self.w_init_vals[layer_idx]))
        self.assign_op = tf.group(*assign_ops)

    def evaluate(self, sess, w_init_vals):
        w_dict = {i: d for i, d in zip(self.w_init_vals, w_init_vals)}
        sess.run(self.assign_op, feed_dict=w_dict)
        sess.run(self.full_network.adapt_bias_op, feed_dict={self.validate: False})

        tr_acc, tr_ce = sess.run([self.full_network.accuracy, self.full_network.cross_entropy],
                                 feed_dict={self.validate: False})
        va_acc, va_ce = sess.run([self.full_network.accuracy, self.full_network.cross_entropy],
                                 feed_dict={self.validate: True})
        return tr_acc, tr_ce, va_acc, va_ce









