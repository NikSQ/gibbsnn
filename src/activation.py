import tensorflow as tf
import numpy as np


# Container for activation function
class ActivationFunction:
    def __init__(self, name, value_function, index_function, param_func):
        self.name = name
        # This is the actual activation function - computes the output for given activations
        self.value_function = value_function
        # This is a function that returns the indices to access the lookup table for given activations
        # Indices range from [0, self.n_values] and each one corresponds to an output value. Index 0 refers to smallest
        # output value
        self.index_function = index_function
        # Contains the possible output values of the activation function in ascending order
        self.param_func = param_func
        self.values = None
        self.n_values = None
        self.params = None

    def get_output(self, X):
        return self.value_function(X, self.params)

    def get_lookup_indices(self, X):
        return self.index_function(X, self.params)

    def set_params(self, params=[]):
        self.params = params
        self.values = self.param_func(params)
        self.n_values = len(self.values)

    def set_function_values(self, function_values):
        self.values = function_values
        self.n_values = len(self.values)



# Binary Sign Function
def bs_value_function(X, params):
    sign = tf.sign(X)
    cond = tf.equal(sign, 0)
    replace_vals = tf.ones_like(X)
    return tf.where(cond, replace_vals, sign)


def bs_index_function(X, params):
    sign = tf.sign(X)
    return tf.cast(tf.sign(sign + 1), dtype=tf.int32)


def bs_meta_function(params):
    return [-1., 1.]




# Ternary Sign Function
def ts_value_function(X, params):
    return tf.sign(X)


def ts_index_function(X, params):
    return tf.cast(tf.sign(X), dtype=tf.int32) + 1


def ts_meta_function(params):
    return [-1., 0., 1.]




# Extended Ternary Sign Function (param[0] = one-sided length of 0 stair)
def ets_value_function(X, params):
    cond = tf.logical_and(tf.less(X, params[0]), tf.greater(X, -params[0]))
    replace_vals = tf.zeros_like(X)
    return tf.where(cond, replace_vals, tf.sign(X))


def ets_index_function(X, params):
    cond = tf.logical_and(tf.less(X, params[0]), tf.greater(X, -params[0]))
    replace_vals = tf.zeros_like(X)
    return tf.cast(tf.where(cond, replace_vals, tf.sign(X)), dtype=tf.int32) + 1


def ets_meta_function(params):
    return [-1., 0., 1.]



# Staircase Function (param[0] = scale, param[1] = bits)
def stair_value_function(X, params):
    scaled_X = tf.divide(X, params[0])
    range = 2**(params[1] - 1)
    upper_cond = tf.greater(scaled_X, range)
    upper_replace_vals = tf.ones_like(X) * range
    lower_cond = tf.less(scaled_X, 1 - range)
    lower_replace_vals = - tf.ones_like(X) * (range - 1)
    return tf.where(lower_cond, lower_replace_vals, tf.where(upper_cond, upper_replace_vals, scaled_X))


def stair_index_function(X, params):
    scaled_X = tf.divide(X, params[0])
    range = 2**(params[1] - 1)
    upper_cond = tf.greater(scaled_X, range)
    upper_replace_vals = tf.ones_like(X) * range
    lower_cond = tf.less(scaled_X, 1 - range)
    lower_replace_vals = - tf.ones_like(X) * (range - 1)
    return tf.cast(tf.where(lower_cond, lower_replace_vals, tf.where(upper_cond, upper_replace_vals, scaled_X)) + range - 1, dtype=tf.int32)


def stair_meta_function(params):
    return np.arange(1 - 2**(params[1] - 1), 2**(params[1] - 1) + 1).astype(np.float32)



def get_activation_function(name):
    if name == 'stair':
        return ActivationFunction('stair', stair_value_function, stair_index_function, stair_meta_function)
    elif name == 'bs':
        return ActivationFunction('bs', bs_value_function, bs_index_function, bs_meta_function)
    elif name == 'ts':
        return ActivationFunction('ts', ts_value_function, ts_index_function, ts_meta_function)
    elif name == 'extended_ternary_sign':
        return ActivationFunction('extended_ternary_sign', ets_value_function, ets_index_function,
                                  ets_meta_function)
   
        





