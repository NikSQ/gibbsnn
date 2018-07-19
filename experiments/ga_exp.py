import sys
import os

sys.path.append('../')

from src.run_experiment import run_experiment


#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/dropout/job_' + str(task_id) + '/'
n_runs = 1

ga_config = {'crossover_p': 0.80,
             'mutation_p': 0.005,
             'max_generations': 50,
             'n_fit_individuals': 10,
             'n_recombinations': 20,
             'pop_size': 50}

layer_1 = 280
layer_2 = 200


config = {'layout': [layer_1, layer_2],
          'weight_type': 'binary',
          'act_func_names': ['bs', 'bs'],
          'act_func_params': [[], []],
          'bias_vals': [None, None, None]}

run_experiment(ga_config, None, config, 'mnist_basic', True)

