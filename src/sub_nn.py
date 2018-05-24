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
        summary_ops = []
        for layer_idx in range(self.n_layers - 1):
            layer_input, act_summary_op = self.layers[layer_idx].forward_pass(layer_input, full_network)
            summary_ops.append(act_summary_op)

        output, cross_entropy, act_summary_op, activation = self.layers[self.n_layers - 1].forward_pass(layer_input, full_network, targets=self.targets)
        summary_ops.append(act_summary_op)
        self.likelihoods = -cross_entropy
        self.total_likelihood = tf.reduce_sum(self.likelihoods)
        self.activation = activation

        if full_network:
            self.output = output
            self.prediction = tf.argmax(output, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.targets, axis=1)),
                                           dtype=tf.float32))
            summary_ops.append(tf.summary.scalar('accuracy', self.accuracy))
            summary_ops.append(tf.summary.scalar('likelihood', self.total_likelihood))
            self.summary_op = tf.summary.merge(summary_ops)






