import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

# TODO: job name should include the actual index of the job
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

path = '../results/ternary_act/job_' + str(task_id)

run_config = {'n_epochs': 7,
              'block_size': 8,
              'store_acts': True,
              'store_acts_every': 3,
              'store_vars': True,
              'store_vars_every': 3,
              'path': path}

act_func1 = get_activation_function('extended_ternary_sign')
act_func2 = get_activation_function('extended_ternary_sign')
act_func1.set_params([5 + 8 * task_id])
act_func2.set_params([3 + 4 * task_id])
act_funcs = [act_func1, act_func2]


config = {'layout': [370, 350],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [0.8, 0.8],
          'flat_factor': [1.2, 1.2, 1.2],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')

