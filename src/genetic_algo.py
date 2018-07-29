import tensorflow as tf
import numpy as np
from src.individual import Individual
from src.static_nn import StaticNN
import copy

class GeneticSolver:
    def __init__(self, nn_config, ga_config, tr_batch_size, va_batch_size, ga_init_pop=None):
        self.nn_config = nn_config
        self.ga_config = ga_config
        self.ga_init_pop = ga_init_pop

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
        print('ga init pop len: {}'.format(len(self.ga_init_pop)))
        if self.ga_init_pop is None:
            for i in range(amount):
                self.population.append(Individual(self.create_init_vals()))
        else:
            for i in range(amount):
                if i < len(self.ga_init_pop):
                    self.population.append(Individual(self.ga_init_pop[i]))
                else:
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

    def recombination(self, pair, current_layer):
        parent1 = self.population[pair[0]].w_vals
        parent2 = self.population[pair[1]].w_vals

        if self.ga_config['recombination'] == 'neuron' or self.ga_config['recombination'] == 'o_neuron' \
                or self.ga_config['i_neuron']:
            offspring1 = copy.deepcopy(parent1)
            offspring2 = copy.deepcopy(parent2)

            for n_exchanged_neurons in range(self.ga_config['n_neurons']):
                if self.ga_config['layer_wise']:
                    layer_idx = current_layer
                else:
                    layer_idx = np.random.randint(0, len(parent1) - 1)
                neuron_idx = np.random.randint(0, parent1[layer_idx].shape[1])
                if self.ga_config['recombination'] == 'neuron' or self.ga_config['recombination'] == 'i_neuron':
                    offspring1[layer_idx][:, neuron_idx] = parent2[layer_idx][:, neuron_idx]
                    offspring2[layer_idx][:, neuron_idx] = parent1[layer_idx][:, neuron_idx]
                if self.ga_config['recombination'] == 'neuron' or self.ga_config['recombination'] == 'o_neuron':
                    offspring2[layer_idx+1][neuron_idx, :] = parent1[layer_idx+1][neuron_idx, :]
                    offspring1[layer_idx+1][neuron_idx, :] = parent2[layer_idx+1][neuron_idx, :]
            return[Individual(offspring1, self.population[pair[0]]), Individual(offspring2, self.population[pair[1]])]

        else:
            offspring1 = []
            offspring2 = []
            for layer_idx in range(len(parent1)):
                heritage_map = np.random.binomial(n=1, p=self.ga_config['crossover_p'], size=parent1[layer_idx].shape)
                inv_heritage_map = heritage_map * (-1) + 1
                offspring1.append(np.multiply(parent1[layer_idx], heritage_map) + np.multiply(parent2[layer_idx], inv_heritage_map))
                offspring2.append(np.multiply(parent1[layer_idx], heritage_map) + np.multiply(parent2[layer_idx], inv_heritage_map))
            return [Individual(offspring1), Individual(offspring2)]

    def mutate_population(self, population, layer=None, prob=None):
        mutation_prob = prob
        if prob is None:
            mutation_prob = self.ga_config['mutation_p']

        for p in range(len(population)):
            if layer is not None:
                if layer != p:
                    continue
            for l in range(len(population[p].w_vals)):
                mutation_map = np.random.binomial(n=1, p=mutation_prob,
                                                  size=population[p].w_vals[l].shape) * (-2) + 1
                population[p].w_vals[l] *= mutation_map
        return population

    def evaluate_population(self, sess):
        for p in self.population:
            p.evaluate(sess, self.static_nn)

    def perform_ga(self, sess):
        self.create_individuals(self.ga_config['pop_size'])
        current_layer = 0

        for generation in range(self.ga_config['max_generations']):
            self.evaluate_population(sess)
            self.print_generations_stats(generation)
            self.population.sort(key=lambda x: x.tr_ce, reverse=False)

            offspring = []
            for idx in range(self.ga_config['n_recombinations']):
                pair = np.random.choice(self.ga_config['n_fit_individuals'], size=(2,), replace=False)
                offspring = offspring + self.recombination(pair, current_layer)

            n_survivors = self.ga_config['pop_size'] - len(offspring)
            print('survivors {}'.format(n_survivors))
            if n_survivors > 0:
                self.population = self.mutate_population(self.population[:n_survivors] + offspring)
            print(self.population[0].tr_ce)

            if self.ga_config['layer_wise'] == True and (generation + 1) % self.ga_config['gen_per_layer'] == 0:
                current_layer += 1
                if current_layer == len(self.nn_config['layout']) - 2:
                    current_layer = 0
                self.population = [self.population[0]] * len(self.population)
                self.population = self.mutate_population(self.population, current_layer, self.ga_config['p_layer_mutation'])












