import tensorflow as tf
import numpy as np
import copy


# This class creates a complete forward pass using the given layers
# The given layers are reused (shallow copy)
class SubNN:
    def __init__(self, layers, input, targets, full_network=False):
        self.n_layers = len(layers)
        self.input = input
        self.targets = targets
        self.layers = layers

        # Iterates over all layers and build a forward pass graph
        layer_input = self.input
        for layer_idx in range(self.n_layers - 1):
            layer_input = self.layers[layer_idx].forward_pass(layer_input)

        output, cross_entropy = self.layers[self.n_layers - 1].forward_pass(layer_input, targets=self.targets)
        self.likelihoods = -cross_entropy
        self.total_likelihood = tf.reduce_sum(self.likelihoods)

        if full_network:
            self.output = output
            self.prediction = tf.argmax(output, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.targets, axis=1)),
                                           dtype=tf.float32))






