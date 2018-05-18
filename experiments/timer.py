import sys
import os
import time

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/timer/'

run_config = {'n_epochs': 2,
              'block_size': 8,
              'store_acts': False,
              'store_acts_every': 1,
              'store_vars': False,
              'store_vars_every': 1,
              'path': path}

act_func = get_activation_function('binary_sign')
act_func.set_params([])
act_funcs = [act_func, act_func]


config = {'layout': [200, 200],
          'weight_type': 'ternary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [0.9, 0.9],
          'flat_factor': [1., 1., 1.],
          'sampling_sequence': 'stochastic'}
start = time.time()
run_experiment(run_config, config, 'mnist_basic')
print('Elapsed time: {}'.format(str(time.time() - start)))


