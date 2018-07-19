class Individual:
    def __init__(self, w_vals):
        self.w_vals = w_vals
        self.tr_acc = None
        self.tr_ce = None
        self.va_acc = None
        self.va_ce = None

    def evaluate(self, sess, static_nn):
        self.tr_acc, self.tr_ce, self.va_acc, self.va_ce = static_nn.evaluate(sess, self.w_vals)

