import tensorflow as tf
import numpy as np
from src.individual import Individual
from src.static_nn import StaticNN

class GeneticSolver:
    def __init__(self, nn_config, ga_config, tr_batch_size, va_batch_size):
        self.nn_config = nn_config
        self.ga_config = ga_config

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
        self.population = []

    def create_individuals(self, amount):
        for i in range(amount):
            self.population.append(Individual(self.create_init_vals()))

    def create_init_vals(self):
        w_vals_list = []
        rng = np.random.RandomState()

        for layer_idx in range(self.static_nn.n_layers):
            w_vals = rng.binomial(n=1, p=0.5, size=self.static_nn.layers[layer_idx].weight_shape)
            w_vals[w_vals == 0] = -1
            w_vals_list.append(w_vals)

        return w_vals_list

    def print_generations_stats(self, generation):
        results = np.zeros(shape=(len(self.population), 4), dtype=np.float32)
        for idx, individual in enumerate(self.population):
            results[idx, 0] = individual.tr_acc
            results[idx, 1] = individual.tr_ce
            results[idx, 2] = individual.va_acc
            results[idx, 3] = individual.va_ce
        result_m = np.mean(results, axis=0)
        result_s = np.std(results, axis=0)

        print('Generation: {}\nTrAcc: {} +- {}\nTrCE: {} +- {}\nVaAcc: {} +- {}\nVa_CE: {} +- {}'
              .format(generation, result_m[0], result_s[0], result_m[1], result_s[1], result_m[2], result_s[2],
                      result_m[3], result_s[3]))

    def recombination(self, pair):
        parent1 = self.population[pair[0]].w_vals
        parent2 = self.population[pair[1]].w_vals
        offspring1 = []
        offspring2 = []
        for layer_idx in range(len(parent1)):
            heritage_map = np.random.binomial(n=1, p=self.ga_config['crossover_p'], size=parent1[layer_idx].shape)
            inv_heritage_map = heritage_map * (-1) + 1
            offspring1.append(np.multiply(parent1[layer_idx], heritage_map) + np.multiply(parent2[layer_idx], inv_heritage_map))
            offspring2.append(np.multiply(parent1[layer_idx], heritage_map) + np.multiply(parent2[layer_idx], inv_heritage_map))
        return [Individual(offspring1), Individual(offspring2)]

    def mutate_population(self, population):
        for p in range(len(population)):
            for l in range(len(population[p].w_vals)):
                mutation_map = np.random.binomial(n=1, p=self.ga_config['mutation_p'],
                                                  size=population[p].w_vals[l].shape) * (-2) + 1
                population[p].w_vals[l] *= mutation_map
        return population

    def evaluate_population(self, sess):
        for p in self.population:
            p.evaluate(sess, self.static_nn)

    def perform_ga(self, sess):
        self.create_individuals(self.ga_config['pop_size'])

        for generation in range(self.ga_config['max_generations']):
            self.evaluate_population(sess)
            self.print_generations_stats(generation)
            self.population.sort(key=lambda x: x.tr_ce, reverse=False)

            offspring = []
            for idx in range(self.ga_config['n_recombinations']):
                pair = np.random.choice(self.ga_config['n_fit_individuals'], size=(2,), replace=False)
                offspring = offspring + self.recombination(pair)

            n_survivors = self.ga_config['pop_size'] - len(offspring)
            self.population = self.mutate_population(self.population[:n_survivors] + offspring)
            print(self.population[0].tr_ce)










