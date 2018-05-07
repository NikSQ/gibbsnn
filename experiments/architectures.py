import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
path = '../results/archs/job_' + str(task_id) + '_'

run_config = {'n_epochs': 3,
              'block_size': 4,
              'store_acts': True,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'path': path}

size_layer1 = 50 + (int(task_id / 3) * 10)
size_layer2 = 30 + (int(task_id % 3) * 10)
print('Neurons in layer 1: {}, layer2: {}'.format(size_layer1, size_layer2))

act_func = get_activation_function('binary_sign')
act_func.set_params([])
act_funcs = [act_func, act_func]


config = {'layout': [size_layer1, size_layer2],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [.7, .7],
          'flat_factor': [.9, .9, .9],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')


