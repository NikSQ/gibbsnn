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


# Here we just create some activation functions.
function_list = []

# Binary Sign Function
def bs_value_function(X, params):
    sign = tf.sign(X)
    cond = tf.equal(sign, 0)
    replace_vals = tf.ones_like(X)
    return tf.where(cond, replace_vals, sign)


def bs_index_function(X, params):
    sign = tf.sign(X)
    return tf.sign(sign + 1)


def bs_meta_function(params):
    return [-1, 1]


function_list.append(ActivationFunction('binary_sign', bs_value_function, bs_index_function, bs_meta_function))


# Ternary Sign Function
def ts_value_function(X, params):
    return tf.sign(X)


def ts_index_function(X, params):
    return tf.sign(X) + 1


def ts_meta_function(params):
    return [-1, 0, 1]


function_list.append(ActivationFunction('ternary_sign', ts_value_function, ts_index_function, ts_meta_function))


# Extended Ternary Sign Function (param[0] = one-sided length of 0 stair)
def ets_value_function(X, params):
    cond = tf.logical_and(tf.less(X, params[0]), tf.greater(X, -params[0]))
    replace_vals = tf.zeros_like(X)
    return tf.where(cond, replace_vals, tf.sign(X))


def ets_index_function(X, params):
    cond = tf.logical_and(tf.less(X, params[0]), tf.greater(X, -params[0]))
    replace_vals = tf.zeros_like(X)
    return tf.where(cond, replace_vals, tf.sign(X)) + 1


def ets_meta_function(params):
    return [-1, 0, 1]


function_list.append(ActivationFunction('extended_ternary_sign', ets_value_function, ets_index_function,
                                        ets_meta_function))


# Staircase Function (param[0] = scale, param[1] = bits)
def stair_value_function(X, params):
    scaled_X = tf.cast(tf.divide(X, params[0]), dtype=tf.int32)
    range = 2**(params[1] - 1)
    upper_cond = tf.greater(scaled_X, range)
    upper_replace_vals = tf.ones_like(X) * range
    lower_cond = tf.less(scaled_X, 1 - range)
    lower_replace_vals = - tf.ones_like(X) * (range - 1)
    return tf.where(lower_cond, lower_replace_vals, tf.where(upper_cond, upper_replace_vals, scaled_X))


def stair_index_function(X, params):
    scaled_X = tf.cast(tf.divide(X, params[0]), dtype=tf.int32)
    range = 2**(params[1] - 1)
    upper_cond = tf.greater(scaled_X, range)
    upper_replace_vals = tf.ones_like(X) * range
    lower_cond = tf.less(scaled_X, 1 - range)
    lower_replace_vals = - tf.ones_like(X) * (range - 1)
    return tf.where(lower_cond, lower_replace_vals, tf.where(upper_cond, upper_replace_vals, scaled_X)) + range - 1


def stair_meta_function(params):
    return np.arange(1 - 2**(params[1] - 1), 2**(params[1] - 1) + 1).astype(np.int32)


function_list.append(ActivationFunction('stair', stair_value_function, stair_index_function, stair_meta_function))


def get_activation_function(name):
    for func in function_list:
        if func.name == name:
            return func


