import sys
import os

sys.path.append('../')

from src.run_experiment import run_experiment


#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
task_id = 0
path = '../results/ga/job_' + str(task_id) + '/'

# 0.8 + id * 0.05
# 0.0002

#0.7 + 0.1
# 0.00002
init_config = {'n_epochs': 100,
               'learning_rate': 0.1,
               'reg': 0.0001}

recombination = ['o_neuron', 'i_neuron', 'default']
recomb = recombination[task_id]


exp_config = {'n_epochs': 0,
              'block_size': 8,
              'store_acts': False,
              'store_acts_every': 1,
              'store_vars': True,
              'store_vars_every': 1,
              'store_method': 'log',
              'burn_in': 50,
              'thinning': 3,
              'path': path,

              'mode': 'sa',
              'init_burn_in': 0,
              'init_thinning': 1,
              'T_start': 0.5,
              'T_decremental': .005,
              'epochs_per_T': 10,
              'max_epochs': 2000,
              'ens_calc': 10,
              'pop_size': 5,
              'mutation_p': 0.001,
              'print_every_n_epochs': 10}



config = {'layout': [20],
          'weight_type': 'binary',
          'act_func_names': ['bs'],
          'act_func_params': [[]],
          'bias_vals': [None, None],
          'keep_probs': [.9, 1.],
          'flat_factor': [1., 1., 1.],
          'act_noise': [0., 0.],
          'prior_value': 0.9,
          'sampling_sequence': 'stochastic'}

run_experiment(exp_config, init_config, config, 'uci_cancer')

