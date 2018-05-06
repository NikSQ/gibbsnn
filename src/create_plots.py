from src.storage_tools import Loader


exp_names = ['ternary_act']
n_jobs = [7]

for jobs, exp_name in zip(n_jobs, exp_names):
    print('experiment {}'.format(exp_name))
    for job_id in range(jobs):
        job_name = 'job_' + str(job_id)
        loader = Loader(exp_name, job_name)
        loader.generate_plots()
