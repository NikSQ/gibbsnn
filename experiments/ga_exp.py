import sys
import os

sys.path.append('../')

from src.run_experiment import run_experiment


task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
path = '../results/dropout/job_' + str(task_id) + '/'
n_runs = 1

# 0.8 + id * 0.05
# 0.0002

#0.7 + 0.1
# 0.00002
ga_config = {'crossover_p': 0.7 + float(task_id) * 0.1,
             'mutation_p': 0.00001,
             'max_generations': 5000,
             'n_fit_individuals': 20,
             'n_recombinations': 90,
             'pop_size': 200}

layer_1 = 280
layer_2 = 200


config = {'layout': [layer_1, layer_2],
          'weight_type': 'binary',
          'act_func_names': ['bs', 'bs'],
          'act_func_params': [[], []],
          'bias_vals': [None, None, None]}

run_experiment(ga_config, None, config, 'mnist_basic', True)

