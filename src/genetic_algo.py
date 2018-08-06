import tensorflow as tf
import numpy as np
from src.individual import Individual
from src.static_nn import StaticNN
from src.ensemble import Ensemble
import copy

class GeneticSolver:
    def __init__(self, nn_config, ga_config, tr_batch_size, va_batch_size, ga_init_pop=None):
        self.nn_config = nn_config
        self.ga_config = ga_config
        self.ga_init_pop = ga_init_pop
        self.tr_batch_size = tr_batch_size
        self.va_batch_size = va_batch_size

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

    def exchange_weights(self, offspring, parent, layer_idx, ex_input):
        neuron_idx = np.random.randint(0, parent[layer_idx].shape[ex_input == True])
        if ex_input:
            offspring[layer_idx][:, neuron_idx] = parent[layer_idx][:, neuron_idx]
        else:
            offspring[layer_idx][neuron_idx, :] = parent[layer_idx][neuron_idx, :]
        return offspring


    def recombination(self, pair, current_layer):
        parent1 = self.population[pair[0]].w_vals
        parent2 = self.population[pair[1]].w_vals

        if self.ga_config['recombination'] == 'neuron' or self.ga_config['recombination'] == 'o_neuron' \
                or self.ga_config['recombination'] == 'i_neuron' or self.ga_config['recombination'] == 'io_neuron':
            offspring1 = copy.deepcopy(parent1)
            offspring2 = copy.deepcopy(parent2)

            for n_exchanged_neurons in range(self.ga_config['n_neurons']):
                layer_idx = current_layer
                if self.ga_config['recombination'] == 'neuron':
                    if self.ga_config['layer_wise'] == False:
                        layer_idx = np.random.randint(0, len(parent1) - 1)
                    offspring1 = self.exchange_weights(offspring1, parent2, layer_idx, True)
                    offspring1 = self.exchange_weights(offspring1, parent2, layer_idx+1, False)
                    offspring2 = self.exchange_weights(offspring2, parent1, layer_idx, True)
                    offspring2 = self.exchange_weights(offspring2, parent1, layer_idx+1, False)
                else:
                    if self.ga_config['layer_wise'] == False:
                        layer_idx = np.random.randint(0, len(parent1))
                    ex_input = self.ga_config['recombination'] == 'i_neuron'
                    if self.ga_config['recombination'] == 'io_neuron':
                        ex_input = bool(np.random.randint(0, 2))
                    offspring1 = self.exchange_weights(offspring1, parent2, layer_idx, ex_input)
                    offspring2 = self.exchange_weights(offspring2, parent1, layer_idx, ex_input)

            return[Individual(offspring1, self.population[pair[0]]), Individual(offspring2, self.population[pair[1]])]

        else:
            offspring1 = []
            offspring2 = []
            for layer_idx in range(len(parent1)):
                heritage_map = np.random.binomial(n=1, p=self.ga_config['crossover_p'], size=parent1[layer_idx].shape)
                inv_heritage_map = heritage_map * (-1) + 1
                offspring1.append(np.multiply(parent1[layer_idx], heritage_map) + np.multiply(parent2[layer_idx], inv_heritage_map))
                offspring2.append(np.multiply(parent1[layer_idx], heritage_map) + np.multiply(parent2[layer_idx], inv_heritage_map))
            return [Individual(offspring1, self.population[pair[0]]), Individual(offspring2, self.population[pair[1]])]

    def mutate_population(self, population, layer=None, prob=None):
        mutation_prob = prob
        if prob is None:
            mutation_prob = self.ga_config['mutation_p']

        for p in range(len(population)):
            for l in range(len(population[p].w_vals)):
                if layer is None or (layer == l or (layer == l-1 and self.ga_config['recombination'] == 'neuron')):
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
        self.simplify_pop(current_layer)
        final_ensemble_acc = None
        final_ensemble_ce = None
        final_acc = None
        final_ce = None

        for generation in range(self.ga_config['max_generations']):
            self.evaluate_population(sess)
            if generation % 1 == 0:
                self.print_generations_stats(generation)
            self.population.sort(key=lambda x: x.tr_ce, reverse=False)
            final_acc = self.population[0].va_acc
            final_ce = self.population[0].va_ce

            if self.ga_config['ens_burn_in'] <= generation + 1 and (generation - 1 - self.ga_config['ens_burn_in']) % self.ga_config['ens_thinning'] == 0:
                self.static_nn.evaluate(sess, self.population[0].w_vals)
                tr_acc, tr_ce = sess.run([self.ensemble_tr.accuracy, self.ensemble_tr.cross_entropy], feed_dict={self.static_nn.validate: False})
                va_acc, va_ce = sess.run([self.ensemble_va.accuracy, self.ensemble_va.cross_entropy], feed_dict={self.static_nn.validate: True})
                final_ensemble_acc = va_acc
                final_ensemble_ce = va_ce
                print('ENSEMBLE | Tr_Acc: {}, Tr_CE: {}, Va_Acc: {}, Va_CE: {}'.format(tr_acc, tr_ce, va_acc, va_ce))

            offspring = []
            for idx in range(self.ga_config['n_recombinations']):
                pair = np.random.choice(self.ga_config['n_fit_individuals'], size=(2,), replace=False)
                offspring = offspring + self.recombination(pair, current_layer)

            n_survivors = self.ga_config['pop_size'] - len(offspring)
            if n_survivors > 0:
                self.population = self.mutate_population(self.population[:n_survivors] + offspring)

            if self.ga_config['layer_wise'] == True and (generation + 1) % self.ga_config['gen_per_layer'] == 0:
                current_layer += 1
                if current_layer == len(self.nn_config['layout']) - 2 and self.ga_config['recombination'] == 'neuron':
                    current_layer = 0
                elif current_layer == len(self.nn_config['layout']) - 1:
                    current_layer = 0
                print('current_layer {}'.format(current_layer))
                self.simplify_pop(current_layer)
        return final_ensemble_acc, final_ensemble_ce, final_acc, final_ce

    def simplify_pop(self, layer_idx):

        self.population[0].print_counts()
        if self.ga_config['recombination'] == 'default' or self.ga_config['layer_wise'] == False:
            return
        for i in range(1, len(self.population)):
            self.population[i].w_vals = copy.deepcopy(self.population[0].w_vals)
        self.population = self.mutate_population(self.population, layer_idx, self.ga_config['p_layer_mutation'])















