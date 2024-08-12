import torch
import os
import msprime
import numpy as np
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from aTMi.processing import read_file
from aTMi.config import population_time_interval
from aTMi.processing import obtain_mutation_densities
from aTMi.processing import calculate_transition_matrix
#import torch.multiprocessing as mp


data = '~/Datasets/aTMi/arabidopsis_L4_n10'
#data = '~/Datasets/aTMi/arabidopsis_L3_n10'

window_size = 20_000
mutation_rate = 7e-09
#empirical_scaling_factor = 3
empirical_scaling_factor = np.exp(8.2)

#dataset_path = "~/Datasets/aTMi/dataset_60k_window20k_L2_scaling3_diff.pth"
#dataset_path = "~/Datasets/aTMi/dataset_60k_window20k_L3_scaling3.pth"
#dataset_path = "~/Datasets/aTMi/dataset_60k_window10k_L2_scaling3.pth"
#dataset_path = "~/Datasets/aTMi/dataset_60k_window10k_L3_scaling3.pth"
dataset_path = "~/Datasets/aTMi/dataset_60k_window20k_L4_scaling3_diff.pth"


data_dir = Path(data).expanduser()
population_time =  population_time_interval(delta=0.001, K=42, tmax=2_100_000)
trees_files = sorted([data_dir/file for file in os.listdir(data_dir) if 'trees' in file])
population_size_files = sorted([data_dir/file for file in os.listdir(data_dir) if 'npy' in file])

def helper(idx):
    ts, population_size = read_file(idx, trees_files, population_size_files)
    ts = msprime.mutate(ts, rate=mutation_rate)
    mutation_densities = obtain_mutation_densities(ts, window_size=window_size)
    #tm = np.log(calculate_transition_matrix(np.log(mutation_densities.flatten()+1)*empirical_scaling_factor, np.log(population_time+1)) + 1)
    
    flattened_mutations = (mutation_densities.flatten() + 1) * empirical_scaling_factor # rescaling before the log
    #log_mutations = np.log(flattened_mutations)
    log_mutations = np.log(np.asarray(flattened_mutations))
    scaled_mutations = log_mutations #* empirical_scaling_factor
    log_population = np.log(population_time + 1)
    
    transition_matrix = calculate_transition_matrix(scaled_mutations, log_population)
    tm = np.log(transition_matrix + 1)
    
    population_size_log = np.log(population_size)
    #tm = torch.tensor(tm, dtype=torch.float32)
    #population_size_log = torch.tensor(population_size_log, dtype=torch.float32)
    return tm, population_size_log

num_files = 60_000
files = range(num_files)
pool = multiprocessing.Pool(multiprocessing.cpu_count())  
dataset = list(tqdm(pool.imap(helper, files), total=num_files))
pool.close()
pool.join()
#dataset = torch.tensor(dataset, dtype=torch.float32)
torch.save(dataset, Path(dataset_path).expanduser())
