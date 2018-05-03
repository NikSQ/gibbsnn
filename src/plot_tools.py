import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class Plotter:
    def __init__(self, path_name):
        self.path_name = '../results/' + path_name + '/'

    def save_activation_plots(self, histograms, name):
        for idx, histogram in enumerate(histograms):
            plt.figure()
            plt.bar(histogram[:, 0], histogram[:, 1], width=0.9, align='center')
            plt.title('Activation histogram of layer ' + str(idx))
            plt.xlabel('activation')
            plt.ylabel('amount')
            plt.savefig(self.path_name + name + '_layer_' + str(idx), bbox_inches='tight')

    def plot_activation_function(self, act_func, input_vals):
        X = tf.constant(input_vals, dtype=tf.int32)
        act_op = act_func.get_output(X)
        lookup_indices_op = act_func.get_lookup_indices(X)
        with tf.Session() as sess:
            activations = sess.run(act_op)
            lookup_indices = sess.run(lookup_indices_op)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.step(input_vals, activations)
        plt.title('Activation Function')
        plt.xlabel('input')
        plt.ylabel('output')
        plt.subplot(2, 1, 2)
        plt.step(input_vals, lookup_indices)
        plt.title('Lookup Indices for Input')
        plt.xlabel('input')
        plt.ylabel('indices')
        plt.show()

    def plot_misclassification(self, it_sequence, train_error=None, val_error=None):
        plt.figure()
        legend = []
        if train_error is not None:
            plt.plot(it_sequence, train_error)
            legend.append('training set')
            print('plotting train')
        if val_error is not None:
            plt.plot(it_sequence, val_error)
            legend.append('validation set')
            print('plotting val')
        plt.legend(tuple(legend))
        plt.xlabel('epochs')
        plt.ylabel('misclassification rate')
        plt.title('Misclassification rate')
        plt.savefig(self.path_name + 'error', bbox_inches='tight')







