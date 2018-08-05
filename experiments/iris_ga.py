import sys
import os

sys.path.append('../')

from src.run_experiment import run_experiment
from src.tools import print_stats


#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/ga/job_' + str(task_id) + '/'
n_runs = 1

# 0.8 + id * 0.05
# 0.0002

#0.7 + 0.1
# 0.00002
init_config = {'n_epochs': 100,
               'learning_rate': 0.1,
               'reg': 0.0001}

recombination = ['o_neuron', 'i_neuron', 'default']
recomb = recombination[task_id]


run_config = {'n_epochs': 0,
              'block_size': 8,
              'store_acts': False,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'log',
              'burn_in': 50,
              'thinning': 3,
              'path': path,

              'is_ga': True,
              'ga_burn_in': 0,
              'ga_thinning': 1,
              'recombination': recomb,
              'crossover_p': 0.7,
              'mutation_p': 0.00001,
              'max_generations': 500,
              'n_fit_individuals': 10,
              'n_recombinations': 20,
              'layer_wise': True,
              'pop_size': 50,
              'n_neurons': 1 + task_id,
              'gen_per_layer': 10,
              'p_layer_mutation': 0.5,
              'ens_burn_in': 10,
              'ens_thinning': 5}


config = {'layout': [],
          'weight_type': 'binary',
          'act_func_names': ['bs'],
          'act_func_params': [[]],
          'bias_vals': [None],
          'keep_probs': [.9],
          'flat_factor': [1., 1.],
          'act_noise': [0.],
          'prior_value': 0.9,
          'sampling_sequence': 'stochastic'}

va_acc_list = []
va_ce_list = []
ens_acc_list = []
ens_ce_list = []
for run in range(n_runs):
    ens_acc, ens_ce, va_acc, va_ce = run_experiment(run_config, init_config, config, 'uci_cancer')
    va_acc_list.append(va_acc)
    va_ce_list.append(va_ce)
    ens_acc_list.append(ens_acc)
    ens_ce_list.append(ens_ce)

print_stats('Mod Accuracy', va_acc_list)
print_stats('Ens Accuracy', ens_acc_list)
print_stats('Mod CE', va_ce_list)
print_stats('Ens CE', ens_ce_list)

