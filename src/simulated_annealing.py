import tensorflow as tf
import numpy as np
from src.individual import Individual
from src.static_nn import StaticNN
from src.ensemble import Ensemble
import copy

class SASolver:
    def __init__(self, nn_config, sa_config, tr_batch_size, va_batch_size, sa_init_pop=None):
        self.nn_config = nn_config
        self.sa_config = sa_config
        self.tr_batch_size = tr_batch_size
        self.va_batch_size = va_batch_size
        self.sa_init_pop = sa_init_pop

        self.x_placeholder = tf.placeholder(tf.float32, [None, self.nn_config['layout'][0]])
        self.y_placeholder = tf.placeholder(tf.float32, [None, self.nn_config['layout'][-1]])
        self.x_tr = tf.get_variable('x_tr', shape=[tr_batch_size, self.nn_config['layout'][0]], dtype=tf.float32)
        self.y_tr = tf.get_variable('y_tr', shape=[tr_batch_size, self.nn_config['layout'][-1]], dtype=tf.float32)
        self.x_va = tf.get_variable('x_val', shape=[va_batch_size, self.nn_config['layout'][0]], dtype=tf.float32)
        self.y_va = tf.get_variable('y_val', shape=[va_batch_size, self.nn_config['layout'][-1]], dtype=tf.float32)

        op1 = tf.assign(self.x_tr, self.x_placeholder)
        op2 = tf.assign(self.y_tr, self.y_placeholder)
        self.load_train_set_op = tf.group(*[op1, op2])

        op1 = tf.assign(self.x_va, self.x_placeholder)
        op2 = tf.assign(self.y_va, self.y_placeholder)
        self.load_val_set_op = tf.group(*[op1, op2])

        self.static_nn = StaticNN(nn_config, self.x_tr, self.y_tr, self.x_va, self.y_va)
        self.ensemble_tr = Ensemble(self.y_tr, (self.tr_batch_size, self.nn_config['layout'][-1]), 'tr', self.static_nn.full_network.activation)
        self.ensemble_va = Ensemble(self.y_va, (self.va_batch_size, self.nn_config['layout'][-1]), 'va', self.static_nn.full_network.activation)
        self.population = []

    def create_individuals(self, amount, sess):
        print('sa init pop len: {}'.format(len(self.sa_init_pop)))
        if self.sa_init_pop is None:
            for i in range(amount):
                self.population.append(Individual(self.create_init_vals()))
        else:
            for i in range(amount):
                if i < len(self.sa_init_pop):
                    self.population.append(Individual(self.sa_init_pop[i]))
                else:
                    self.population.append(Individual(self.create_init_vals()))
        for p in self.population:
            p.evaluate(sess, self.static_nn)

    def create_init_vals(self):
        w_vals_list = []
        rng = np.random.RandomState()

        for layer_idx in range(self.static_nn.n_layers):
            w_vals = rng.binomial(n=1, p=0.5, size=self.static_nn.layers[layer_idx].weight_shape)
            w_vals[w_vals == 0] = -1
            w_vals_list.append(w_vals)

        return w_vals_list

    def print_generations_stats(self, generation):
        results = np.zeros(shape=(len(self.population), 5), dtype=np.float32)
        for idx, individual in enumerate(self.population):
            results[idx, 0] = individual.tr_acc
            results[idx, 1] = individual.tr_ce
            results[idx, 2] = individual.va_acc
            results[idx, 3] = individual.va_ce
            results[idx, 4] = individual.n_ancestors
        result_m = np.mean(results, axis=0)
        result_s = np.std(results, axis=0)

        print('Generation: {}\nTrAcc: {} +- {}\tTrCE: {} +- {}\nVaAcc: {} +- {}\tVa_CE: {} +- {}\nAncestors: {} +- {}'
              .format(generation, result_m[0], result_s[0], result_m[1], result_s[1], result_m[2], result_s[2],
                      result_m[3], result_s[3], result_m[4], result_s[4]))
        print('pop len {}'.format(len(self.population)))


    def mutate_population(self, sess, T):
        for p_idx in range(len(self.population)):
            tr_ce = self.population[p_idx].tr_ce
            w_vals = copy.deepcopy(self.population[p_idx].w_vals)
            for l in range(len(self.population[p_idx].w_vals)):
                mutation_map = np.random.binomial(n=1, p=self.sa_config['mutation_p'],
                                                  size=self.population[p_idx].w_vals[l].shape) * (-2) + 1
                self.population[p_idx].w_vals[l] *= mutation_map
            self.population[p_idx].evaluate(sess, self.static_nn)
            if tr_ce < self.population[p_idx].tr_ce:
                if T == 0 or np.random.binomial(n=1, p= np.exp((tr_ce - self.population[p_idx].tr_ce) / T)) == 0:
                    for l in range(len(self.population[p_idx].w_vals)):
                        self.population[p_idx].w_vals[l] = w_vals[l]
                    self.population[p_idx].tr_ce = tr_ce

    def perform_sa(self, sess):
        self.create_individuals(self.sa_config['pop_size'], sess)
        final_ensemble_acc = None
        final_ensemble_ce = None
        best_acc = 0
        generation = None
        T = self.sa_config['T_start']


        for epoch in range(self.sa_config['max_epochs']):
            if epoch % self.sa_config['print_every_n_epochs'] == 0:
                self.print_generations_stats(epoch)
            self.population.sort(key=lambda x: x.tr_ce, reverse=False)
            final_acc = self.population[0].va_acc
            final_ce = self.population[0].va_ce
            if final_acc > best_acc:
                best_acc = final_acc
                generation = epoch

            if epoch % self.sa_config['ens_calc'] == 0:
                sess.run([self.ensemble_tr.clear_activation_op, self.ensemble_va.clear_activation_op])
                for p in self.population:
                    self.static_nn.evaluate(sess, p.w_vals)
                    tr_acc, tr_ce = sess.run([self.ensemble_tr.accuracy, self.ensemble_tr.cross_entropy], feed_dict={self.static_nn.validate: False})
                    va_acc, va_ce = sess.run([self.ensemble_va.accuracy, self.ensemble_va.cross_entropy], feed_dict={self.static_nn.validate: True})
                final_ensemble_acc = va_acc
                final_ensemble_ce = va_ce
                print('ENSEMBLE | Tr_Acc: {}, Tr_CE: {}, Va_Acc: {}, Va_CE: {}'.format(tr_acc, tr_ce, va_acc, va_ce))

            self.mutate_population(sess, T)

            if (epoch+1) % self.sa_config['epochs_per_T'] == 0:
                T = T - self.sa_config['T_decremental']
                if T <= 0:
                    T = 0

        return final_ensemble_acc, final_ensemble_ce, best_acc, generation














