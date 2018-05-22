import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
path = '../results/test2/job_' + str(task_id) + '/'

run_config = {'n_epochs': 22,
              'block_size': 4,
              'store_acts': True,
              'store_acts_every': 7,
              'store_vars': True,
              'store_vars_every': 7,
              'path': path}

act_func1 = get_activation_function('stair')
act_func2 = get_activation_function('stair')

act_func1.set_params([4, 4])
act_func2.set_params([4, 4])
act_funcs = [act_func1, act_func2]
layer_1 = 250 
layer_2 = 250
keep_probs1 = 0.90 ** int(task_id / 3 + 1)
keep_probs2 = 0.90 ** int(task_id % 3 + 1)


config = {'layout': [layer_1, layer_2],
          'weight_type': 'ternary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [keep_probs1, keep_probs2],
          'flat_factor': [1., 1., 1.],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')


