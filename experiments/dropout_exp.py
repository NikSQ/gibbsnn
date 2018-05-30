import sys
import os
import numpy as np

sys.path.append('../')

from src.run_experiment import run_experiment
from src.tools import print_stats

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#task_id = 0
path = '../results/dropout/job_' + str(task_id) + '/'
n_runs = 5

init_config = {'n_epochs': 100,
               'learning_rate': 0.1,
               'reg': 0.0001}

run_config = {'n_epochs': 30,
              'block_size': 8,
              'store_acts': True,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'both',
              'burn_in': 15,
              'thinning': 1,
              'path': path}

layer_1 = 80
layer_2 = 50
keep_probs1 = 0.3 + (0.2 * int(task_id / 4))
keep_probs2 = .5 + (0.15 * int(task_id % 4))


config = {'layout': [layer_1, layer_2],
          'weight_type': 'binary',
          'act_func_names': ['bs', 'bs'],
          'act_func_params': [[], []],
          'bias_vals': [None, None, None],
          'keep_probs': [keep_probs1, keep_probs2, keep_probs2],
          'flat_factor': [1., 1., 1.],
          'act_noise': [0., 0., 0.],
          'prior_value': 0.7,
          'sampling_sequence': 'stochastic'}

va_acc_list = []
va_ce_list = []
ens_acc_list = []
ens_ce_list = []
for run in range(n_runs):
    ens_acc, ens_ce, va_acc, va_ce = run_experiment(run_config, init_config, config, 'mnist_basic')
    va_acc_list.append(va_acc)
    va_ce_list.append(va_ce)
    ens_acc_list.append(ens_acc)
    ens_ce_list.append(ens_ce)

print_stats('Mod Accuracy', va_acc_list)
print_stats('Ens Accuracy', ens_acc_list)
print_stats('Mod CE', va_ce_list)
print_stats('Ens CE', ens_ce_list)



