import msprime
import numpy as np
from typing import List

def sample_population_size(
    n_min:int = 10,
    n_max:int = 100_000,
    num_time_windows:int = 21,
    constant: bool = False,
) -> List[float]:
    if not constant:
        assert n_min < n_max, "Minimum population size (n_min) should be smaller than maximum population size (n_max)."
        n_min_log10 = np.log10(n_min)
        n_max_log10 = np.log10(n_max)
        population_size = [10 ** np.random.uniform(low=n_min_log10, high=n_max_log10)] 
        for j in range(num_time_windows - 1):
            population_size.append(10 ** n_min_log10 - 1)
            while population_size[-1] > 10 ** n_max_log10 or population_size[-1]  < 10 ** n_min_log10:
                population_size[-1] = population_size[-2] * 10 ** np.random.uniform(-1, 1)
    else:
        population_size = [np.random.uniform(n_min, n_max)] * num_time_windows
    return population_size

# population_time not used, but directly imported from config
def get_population_time(
    time_rate:float=0.06,
    tmax:int = 130_000,
    num_time_windows:int = 21
) -> List[float] :
    assert time_rate != 0, "Time rate cannot be 0."
    population_time = np.repeat([(np.exp(np.log(1 + time_rate * tmax) * i /
                              (num_time_windows - 1)) - 1) / time_rate for i in
                              range(num_time_windows)], 1, axis=0)
    return population_time.tolist()

def simulate_demographical_ts(
    population_size: List[float],
    population_time: List[float],
    recombination_rate: float,
    sequence_length: float,
    num_sample:int,
    ploidy:int,
    discrete_genome:bool, 
    seed: int = 69420,
    model = None,
):
    demography=msprime.Demography()
    demography.add_population(initial_size=(population_size[0]))
    for i, (time, size) in enumerate(zip(population_time, population_size)):
        demography.add_population_parameters_change(time=time, initial_size=size)
    ts = msprime.sim_ancestry(
        samples=num_sample, 
        recombination_rate=recombination_rate,
        sequence_length=sequence_length, 
        demography=demography,
        ploidy=ploidy, model=model,
        discrete_genome=discrete_genome,
        random_seed=seed)
    return ts

def simulate_constant_ts(
    population_size: float,
    population_time: List[float],
    recombination_rate: float,
    sequence_length: float,
    num_sample:int,
    ploidy:int,
    discrete_genome:bool, 
    seed: int = 69420,
    model = None,
):

    ts = msprime.sim_ancestry(
        samples=num_sample, 
        recombination_rate=recombination_rate,
        sequence_length=sequence_length, 
        population_size=population_size,
        ploidy=ploidy, model=model,
        discrete_genome=discrete_genome,
        random_seed=seed)
    return ts

def moving_average(data, window_size):
    """Calculates the moving average for a given list and window size."""
    if not 0 < window_size <= len(data):
        raise ValueError("Window size must be in range [1, len(data)]")
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='same')