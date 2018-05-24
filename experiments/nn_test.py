import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/nn_test/job_' + str(task_id) + '/'

run_config = {'n_epochs': 2,
              'block_size': 4,
              'store_acts': True,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'both',
              'path': path}

act_func1 = get_activation_function('bs')
act_func2 = get_activation_function('bs')

act_func1.set_params([])
act_func2.set_params([])
act_funcs = [act_func1, act_func2]
layer_1 = 200
layer_2 = 200
keep_probs1 = 1.0
keep_probs2 = 1.0


config = {'layout': [layer_1, layer_2],
          'weight_type': 'ternary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [keep_probs1, keep_probs2, keep_probs2],
          'flat_factor': [1., 1., 1.],
          'act_noise': [0., 0., 0.],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')


