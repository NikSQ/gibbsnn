import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
path = '../results/control/job_' + str(task_id) + '/'

run_config = {'n_epochs': 10,
              'block_size': 5,
              'store_acts': True,
              'store_acts_every': 5,
              'store_vars': True,
              'store_vars_every': 5,
              'path': path}

act_func1 = get_activation_function('binary_sign')
act_func2 = get_activation_function('binary_sign')

act_func1.set_params([])
act_func2.set_params([])
act_funcs = [act_func1, act_func2]
layer_1 = 180 + 40 * int(task_id / 2)
layer_2 = 180 + 40 * int(task_id % 2)


config = {'layout': [layer_1, layer_2],
          'weight_type': 'ternary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1., 1.],
          'flat_factor': [1., 1., 1.],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')


