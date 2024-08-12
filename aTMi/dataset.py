
import argparse
import tszip
import numpy as np

from aTMi.config import population_times
from aTMi.simulation import simulate_demographical_ts
from aTMi.simulation import sample_population_size
from aTMi.simulation import moving_average
from aTMi.config import population_size_params
from aTMi.config import tree_sequence_simulation_params
from aTMi.config import dataset_params

def simulate_ts(seed, parameters):

    np.random.seed(seed)
    population_time_settings = parameters['population_time_settings']
    population_size_settings = parameters['population_size_settings']
    simulations_settings = parameters['simulations_settings']
    population_time = population_times(population_time_settings)
    population_size = moving_average(
        sample_population_size(
            **population_size_params(population_size_settings)['range']
        ), window_size=population_size_params(population_size_settings)['window_size']
    )
    simulation_params = tree_sequence_simulation_params(simulations_settings)['simulation']
    simulation_params['population_time'] = population_time
    simulation_params['seed'] = seed

    simulation_params['population_size'] = population_size
    #population_size = [np.random.uniform(10_000, 200_000)] * 42
    #simulation_params['population_size'] = population_size

    ts = simulate_demographical_ts(**simulation_params)
    return ts, population_size


def simulate_chunk(parameters, ith_chunk):

    params = dataset_params(parameters['dataset_settings'])
    chunk_path = params['chunk_path']
    chunk_path.mkdir(parents=True, exist_ok=True)
    n_simulations = params['n_simulations']

    ith_simulation = 0
    while ith_simulation < n_simulations:
        np.random.seed(ith_simulation+(ith_chunk*n_simulations))
        seed = np.random.randint(0, 1e9)
        print(f'Chunky simulator: ith_chunk: {ith_chunk}, ith_simulation {ith_simulation}, seed {seed}')
        ts, population_size = simulate_ts(seed=seed, parameters=parameters)
        ts_file = chunk_path/('ts_' + str(ith_chunk) + '_'+ str(ith_simulation) + '.trees')
        population_size_file = chunk_path/('population_size_' + str(ith_chunk) + '_'+ str(ith_simulation) + '.npy')                  
        #ts.dump(ts_file, zlib_compression=True)

        tszip.compress(ts, ts_file)


        np.save(population_size_file, population_size)
        ith_simulation += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="aTMi")
    parser.add_argument('--population_time', type=str, required=True)
    parser.add_argument('--population_size', type=str, required=True)
    parser.add_argument('--simulation', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ith_chunk', type=int, required=True) # required for parallelization

    args = parser.parse_args()
    population_time_settings = args.population_time
    population_size_settings = args.population_size
    simulations_settings = args.simulation
    dataset_settings = args.dataset

    parameters = {
        'population_time_settings': population_time_settings,
        'population_size_settings': population_size_settings,
        'simulations_settings': simulations_settings,
        'dataset_settings': dataset_settings
    }
    
    simulate_chunk(parameters, args.ith_chunk)


