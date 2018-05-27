import tensorflow as tf


class Ensemble:
    def __init__(self, y, y_shape, var_scope, activation):
        with tf.variable_scope(var_scope):
            zero_init = tf.zeros_initializer(dtype=tf.float32)
            self.activation_sum = tf.get_variable(name='sum', shape=y_shape, initializer=zero_init)
            self.counter = tf.get_variable(name='counter', shape=1, initializer=zero_init)
            self.activation_ph = tf.placeholder(dtype=tf.float32, shape=y_shape)

        add_op = tf.assign(self.activation_sum, self.activation_sum + activation).op
        inc_counter_op = tf.assign(self.counter, self.counter + 1)
        add_model_op = tf.group(*[add_op, inc_counter_op])

        with tf.control_dependencies([add_model_op]):
            mean_activation = tf.divide(self.activation_sum, self.counter)
            self.output = tf.nn.softmax(mean_activation)
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=mean_activation))
            self.prediction = tf.argmax(self.output, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(y, axis=1)), dtype=tf.float32))

