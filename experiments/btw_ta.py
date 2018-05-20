import sys
import os

sys.path.append('../')

from src.activation import get_activation_function
from src.run_experiment import run_experiment

# TODO: job name should include the actual index of the job
#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0

path = '../results/btw_ta/job_' + str(task_id)

run_config = {'n_epochs': 4,
              'block_size': 8,
              'store_acts': True,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'path': path}


#act_func1 = get_activation_function('extended_ternary_sign')
#act_func2 = get_activation_function('extended_ternary_sign')
#act_func1.set_params([2 + 2 * task_id])
#act_func2.set_params([1 + 2 * task_id])
act_func = get_activation_function('binary_sign')
act_func.set_params()
act_funcs = [act_func, act_func]


config = {'layout': [40, 30],
          'weight_type': 'ternary', 
          'act_funcs': act_funcs,
          'bias_vals': [None, None, None],
          'keep_probs': [1., 1.],
          'flat_factor': [1., 1., 1.],
          'sampling_sequence': 'stochastic'}

run_experiment(run_config, config, 'mnist_basic')

