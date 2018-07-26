import sys
import os

sys.path.append('../')

from src.run_experiment import run_experiment


#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/dropout/job_' + str(task_id) + '/'
n_runs = 1

# 0.8 + id * 0.05
# 0.0002

#0.7 + 0.1
# 0.00002
init_config = {'n_epochs': 100,
               'learning_rate': 0.1,
               'reg': 0.0001}

exp_config = {'n_epochs': 50,
              'block_size': 4,
              'store_acts': False,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'log',
              'burn_in': 50,
              'thinning': 3,
              'path': path,

              'is_ga': True,
              'ga_burn_in': 10,
              'ga_thinning': 3,
              'recombination': 'neuron',
              'crossover_p': 0.7,
              'mutation_p': 0.00001,
              'max_generations': 500,
              'n_fit_individuals': 10,
              'n_recombinations': 40,
              'pop_size': 50,
              'n_neurons': 1 + task_id}

layer_1 = 280
layer_2 = 200


config = {'layout': [layer_1, layer_2],
          'weight_type': 'binary',
          'act_func_names': ['bs', 'bs'],
          'act_func_params': [[], []],
          'bias_vals': [None, None, None],
          'keep_probs': [.9, .7, .7],
          'flat_factor': [1., 1., 1.],
          'act_noise': [0., 0., 0.],
          'prior_value': 0.9,
          'sampling_sequence': 'stochastic'}

run_experiment(exp_config, init_config, config, 'mnist_basic')

