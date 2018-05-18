import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#task_id = 0
path = '../results/control/'

run_config = {'n_epochs': 15,
              'block_size': 5,
              'store_acts': True,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'path': path}

act_func1 = get_activation_function('stair')
act_func2 = get_activation_function('stair')

act_func1.set_params([5 + task_id / 2, 4])
act_func2.set_params([2 + task_id % 2, 4])
act_funcs = [act_func1, act_func2]


config = {'layout': [120, 90],
          'weight_type': 'ternary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [0.9, 0.9],
          'flat_factor': [1., 1., 1.],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')


