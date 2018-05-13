import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
path = '../results/regu/job_' + str(task_id) + '_'
keep_probs = 0.85**(task_id % 2)
flat_factor = 1.5**(task_id / 2)

run_config = {'n_epochs': 9,
              'block_size': 8,
              'store_acts': True,
              'store_acts_every': 3,
              'store_vars': True,
              'store_vars_every': 3,
              'path': path}

act_func = get_activation_function('binary_sign')
act_func.set_params([])
act_funcs = [act_func, act_func]


config = {'layout': [350, 350],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [keep_probs, keep_probs],
          'flat_factor': [flat_factor, flat_factor, flat_factor],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')


