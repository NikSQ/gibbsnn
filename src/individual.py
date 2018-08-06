import copy
import matplotlib.pyplot as plt
import numpy as np

class Individual:
    def __init__(self, w_vals, main_parent=None):
        self.w_vals = w_vals
        self.tr_acc = None
        self.tr_ce = None
        self.va_acc = None
        self.va_ce = None

        if main_parent is None:
            self.n_ancestors = 0
            self.w_change_count = []
            for layer_idx in range(len(self.w_vals)):
                self.w_change_count.append(np.zeros_like(self.w_vals[layer_idx]))
        else:
            self.n_ancestors = main_parent.n_ancestors + 1
            self.w_change_count = copy.deepcopy(main_parent.w_change_count)
            for layer_idx in range(len(self.w_vals)):
                self.w_change_count[layer_idx] += np.not_equal(self.w_vals[layer_idx], main_parent.w_vals[layer_idx]).astype(np.int32)

    def evaluate(self, sess, static_nn):
        self.tr_acc, self.tr_ce, self.va_acc, self.va_ce = static_nn.evaluate(sess, self.w_vals)

    def print_counts(self):
        for layer_idx in range(len(self.w_vals)):
            plt.subplot(len(self.w_vals), 1, layer_idx + 1)
            plt.imshow(self.w_change_count[layer_idx], cmap='gray')
            plt.colorbar()
        plt.show()


