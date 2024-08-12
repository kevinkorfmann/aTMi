from aTMi.misc import binomial_combinations_k2
import torch
import msprime
import numpy as np
from pathlib import Path
import tszip

def read_file(idx, trees_files, population_size_files):
    ts = tszip.decompress(trees_files[idx])
    population_size = np.load(population_size_files[idx])
    return ts, population_size

def optimized_block_sum(matrix, window_size=5):
    if matrix.ndim != 2: raise ValueError("Matrix must be 2D.")
    num_blocks = matrix.shape[1] // window_size
    block_sum = matrix[:, :num_blocks*window_size].reshape(matrix.shape[0], num_blocks, window_size).sum(axis=2)
    return block_sum

def obtain_densities(sequence, window_size=400):
    signal = optimized_block_sum(sequence, window_size=window_size)
    return torch.tensor(signal, dtype=torch.float32).sum(0)

def obtain_mutation_densities(ts, window_size=20_000):
    combinations = binomial_combinations_k2(ts.num_samples)
    genotype_matrix = ts.genotype_matrix().T
    positions = np.array([s.position for s in ts.sites()]).astype(np.int32)  # Ensure the right datatype from start
    sequence_length = int(ts.sequence_length)
    Xs = []
    for sample_0, sample_1 in combinations:
        binary_genotype_matrix = genotype_matrix[[sample_0, sample_1]]
        sum_matrix = binary_genotype_matrix.sum(0)
        valid_mask = (sum_matrix < 2) & (sum_matrix > 0)
        binary_genotype_matrix = binary_genotype_matrix[:, valid_mask]
        binary_positions = positions[valid_mask]
        full_sequence = np.zeros((2, sequence_length), dtype=np.int32)
        full_sequence[:, binary_positions] = binary_genotype_matrix
        X = obtain_densities(full_sequence, window_size=window_size)
        Xs.append(X)
    mutation_densities = torch.stack(Xs)
    return mutation_densities



def obtain_mutation_densities_2(ts, window_size=20_000):
    num_samples = ts.num_samples
    genotype_matrix = ts.genotype_matrix().T
    positions = np.array([s.position for s in ts.sites()]).astype(np.int32)
    sequence_length = int(ts.sequence_length)

    # Calculate the number of bins
    max_possible_bins = (sequence_length // window_size) + 1
    bins = np.arange(0, sequence_length + window_size, window_size)
    
    # Generate all combinations
    comb_indices = np.array(list(binomial_combinations_k2(num_samples)))

    # Initialize densities array
    num_combinations = len(comb_indices)
    densities_per_pair = np.zeros((num_combinations, max_possible_bins), dtype=np.float32)

    # Use broadcasting to compute the valid masks for all pairs at once
    sample_0_indices = comb_indices[:, 0]
    sample_1_indices = comb_indices[:, 1]

    # Get genotypes for the combinations
    sample_0_genotypes = genotype_matrix[sample_0_indices]
    sample_1_genotypes = genotype_matrix[sample_1_indices]

    # Calculate the valid masks
    sum_matrix = sample_0_genotypes + sample_1_genotypes
    valid_masks = (sum_matrix < 2) & (sum_matrix > 0)

    # Compute densities for all pairs in a vectorized manner
    for i in range(num_combinations):
        binary_positions = positions[valid_masks[i]]
        density, _ = np.histogram(binary_positions, bins=bins)
        densities_per_pair[i, :len(density)] = density

    # Convert densities to torch tensor
    mutation_densities = torch.tensor(densities_per_pair)
    return mutation_densities[:, :-1]



def calculate_transition_matrix(sequence, population_time):
    sequence = np.array(sequence)
    population_time = np.array(population_time)
    states = len(population_time)
    transition_matrix = np.zeros((states, states))
    discretized_population_times = np.searchsorted(population_time, sequence, side='right') - 1
    np.add.at(transition_matrix, (discretized_population_times[0:-2], discretized_population_times[1:-1]), 1)
    return transition_matrix




