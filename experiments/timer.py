import sys
import os
import time

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/timer/'

run_config = {'n_epochs': 5,
              'block_size': 2,
              'store_acts': False,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'both',
              'burn_in': 0,
              'thinning': 1,
              'path': path}

act_func1 = get_activation_function('ets')
act_func2 = get_activation_function('ets')
act_func1.set_params([3])
act_func2.set_params([3])
act_funcs = [act_func1, act_func2]


config = {'layout': [10, 20],
          'weight_type': 'ternary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1.0, 1., 1.],
          'flat_factor': [1., 1., 1.],
          'act_noise': [5., 5.],
          'sampling_sequence': 'stochastic'}

start = time.time()
run_experiment(run_config, config, 'mnist_basic')
print('Elapsed time: {}'.format(str(time.time() - start)))


