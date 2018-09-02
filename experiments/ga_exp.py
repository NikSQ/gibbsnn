import sys
import os

sys.path.append('../')

from src.run_experiment import run_experiment
from src.tools import print_stats


task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
path = '../results/ga/job_' + str(task_id) + '/'
n_runs = 5

init_config = {'n_epochs': 100,
               'learning_rate': 0.1,
               'reg': 0.0001}

recomb = 'neuron'
mutation_p = 0
n_neurons = [10, 30, 100]


exp_config = {'n_epochs': 10,
              'block_size': 8,
              'store_acts': False,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'log',
              'burn_in': 50,
              'thinning': 3,
              'path': path,

              'mode': 'ga',
              'init_burn_in': 0,
              'init_thinning': 1,
              'recombination': recomb,
              'crossover_p': 0.7,
              'mutation_p': mutation_p,
              'max_generations': 100,
              'n_fit_individuals': 10,
              'n_recombinations': 50,
              'layer_wise': True,
              'pop_size': 100,
              'n_neurons': n_neurons[task_id],
              'gen_per_layer': 5,
              'p_layer_mutation': 0.005,
              'ens_burn_in': 30,
              'ens_thinning': 5}


config = {'layout': [280, 200],
          'weight_type': 'binary',
          'act_func_names': ['bs', 'bs'],
          'act_func_params': [[], []],
          'bias_vals': [None, None, None],
          'keep_probs': [.9, 1., 1.],
          'flat_factor': [1., 1., 1.],
          'act_noise': [0., 0., 0.],
          'prior_value': 0.9,
          'sampling_sequence': 'stochastic'}

va_acc_list = []
va_ce_list = []
ens_acc_list = []
ens_ce_list = []
for run in range(n_runs):
    ens_acc, ens_ce, va_acc, va_ce = run_experiment(exp_config, init_config, config, 'mnist_basic')
    va_acc_list.append(va_acc)
    va_ce_list.append(va_ce)
    ens_acc_list.append(ens_acc)
    ens_ce_list.append(ens_ce)

print_stats('Mod Accuracy', va_acc_list)
print_stats('Ens Accuracy', ens_acc_list)
print_stats('Mod CE', va_ce_list)
print_stats('Ens CE', ens_ce_list)



