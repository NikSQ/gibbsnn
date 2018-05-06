import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

# TODO: job name should include the actual index of the job
#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

task_id = 0
path = '../results/ternary_act/job_' + str(task_id)

run_config = {'n_epochs': 4,
              'block_size': 2,
              'store_acts': True,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'path': path}

act_func = get_activation_function('extended_ternary_sign')
act_func.set_params([1 + 2 * task_id])
act_funcs = [act_func, act_func]


config = {'layout': [8, 8],
          'weight_type': 'binary',
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1.0, 1.0],
          'flat_factor': [1.0, 1.0, 1.0],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')

