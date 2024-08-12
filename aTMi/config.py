import numpy as np
from pathlib import Path

def population_time_interval(K, delta, tmax):
    i = np.arange(0, K)
    population_time = (np.exp((i/(K-1)) * np.log(1+(delta*tmax)))-1)/delta
    return population_time


def population_times(settings: str):
    if settings == 'arabidopsis_methyl':
        return population_time_interval(delta=0.001, K=42, tmax=2_100_000)
    else: raise ValueError(f'The "settings" parameter is not valid.') 

    
def population_size_params(settings: str):
    if settings == 'arabidopsis_methyl':
        return {
            'range': {'n_max': 1200000, 'n_min': 10000, 'num_time_windows': 42, 'constant': False},
            'window_size': 6
        }
    elif settings == 'arabidopsis_methyl_constant':
        return {
            'range': {'n_max': 200000, 'n_min': 10000, 'num_time_windows': 42, 'constant': True},
            'window_size': 6
        }
    else: raise ValueError(f'The "settings" parameter is not valid.') 

def tree_sequence_simulation_params(settings:str):
    
    
    if settings == 'arabidopsis_L2_n10':
        return {
                'simulation': {
                'discrete_genome': True,
                'num_sample': 10, 
                'ploidy': 1,
                'population_size': None,
                'population_time': None,
                'recombination_rate': 3.4e-8,
                'seed': None,
                'sequence_length': 2_000_000
            },
            'mutation_rate' : 7e-09 
        }
    elif settings == 'arabidopsis_L3_n10':
        return {
                'simulation': {
                'discrete_genome': True,
                'num_sample': 10, 
                'ploidy': 1,
                'population_size': None,
                'population_time': None,
                'recombination_rate': 3.4e-8,
                'seed': None,
                'sequence_length': 3_000_000
            },
            'mutation_rate' : 7e-09 
        }
    elif settings == 'arabidopsis_L4_n10':
        return {
                'simulation': {
                'discrete_genome': True,
                'num_sample': 10, 
                'ploidy': 1,
                'population_size': None,
                'population_time': None,
                'recombination_rate': 3.4e-8,
                'seed': None,
                'sequence_length': 4_000_000
            },
            'mutation_rate' : 7e-09 
        }
    else: raise ValueError(f'The "settings" parameter is not valid.')

def dataset_params(settings: str):
    if settings == 'arabidopsis_L2_n10':
        return {
            'chunk_path': Path("~/Datasets/aTMi/arabidopsis_L2_n10").expanduser(),
            'n_simulations':100,
            'ith_chunk':0,
        }
    elif settings == 'arabidopsis_L3_n10':
        return {
            'chunk_path': Path("~/Datasets/aTMi/arabidopsis_L3_n10").expanduser(),
            'n_simulations':100,
            'ith_chunk':0,
        }
    elif settings == 'arabidopsis_L4_n10':
        return {
            'chunk_path': Path("~/Datasets/aTMi/arabidopsis_L4_n10").expanduser(),
            'n_simulations':100,
            'ith_chunk':0,
        }
    else: raise ValueError(f'The "settings" parameter is not valid.')




def load_model_params(settings: str):
    if settings == 'few_params':
        return  {
            'seq_len':15,
            'enc_dim': 512,
            'enc_depth': 14,
            'enc_heads': 8,
            'in_dim': 500,
            'out_dim': 200,
            'sum_encoder_dim': False
        }
    else: raise ValueError(f'The "settings" parameter is not valid.')
     