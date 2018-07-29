class Individual:
    def __init__(self, w_vals, main_parent=None):
        self.w_vals = w_vals
        self.tr_acc = None
        self.tr_ce = None
        self.va_acc = None
        self.va_ce = None

        if main_parent is None:
            self.n_ancestors = 0
        else:
            self.n_ancestors = main_parent.n_ancestors + 1

    def evaluate(self, sess, static_nn):
        self.tr_acc, self.tr_ce, self.va_acc, self.va_ce = static_nn.evaluate(sess, self.w_vals)

