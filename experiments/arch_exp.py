import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#task_id = 0
path = '../results/archs/job_' + str(task_id) + '/'

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
              'burn_in': 10,
              'thinning': 3,
              'path': path}

layer_1 = 200 + 30 * int(task_id / 5)
layer_2 = 200 + 30 * int(task_id % 5)
keep_probs1 = 1.#.5 + (0.2 * int(task_id / 2))
keep_probs2 = 1.#.7 + (0.2 * int(task_id % 2))


config = {'layout': [layer_1, layer_2],
          'weight_type': 'binary',
          'act_func_names': ['bs', 'bs'],
          'act_func_params': [[],[]],
          'bias_vals': [None, None, None],
          'keep_probs': [keep_probs1, keep_probs2, keep_probs2],
          'flat_factor': [1., 1., 1.],
          'act_noise': [0., 0., 0.],
          'prior_value': 0.7,
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, init_config, config, 'mnist_basic')


