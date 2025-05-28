
import torch
import msprime
import numpy as np
import torch.nn as nn 
from x_transformers import Encoder
from aTMi.processing import obtain_mutation_densities, calculate_transition_matrix
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from aTMi.misc import binomial_combinations_k2
from aTMi.processing import obtain_densities
import pandas as pd
from aTMi.config import population_times
population_time = population_times('arabidopsis_methyl')


def create_sawtooth_demogaphy_object(Ne = 10**4, magnitue=2):
    demography = msprime.Demography()
    demography.add_population(initial_size=(Ne))
    demography.add_population_parameters_change(time=20, population=None,growth_rate=6437.7516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=30, growth_rate=-378.691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=200, growth_rate=-643.77516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=300, growth_rate=37.8691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=2000, growth_rate=64.377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=3000, growth_rate=-3.78691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=20000, growth_rate=-6.4377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=30000, growth_rate=0.378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=200000, growth_rate=0.64377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=300000, growth_rate=-0.0378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=2000000, growth_rate=-0.064377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=3000000, growth_rate=0.00378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=20000000, growth_rate=0,initial_size=Ne)
    return demography


class CoalescenceTransformer(nn.Module):
    def __init__(self, seq_len, enc_dim, enc_depth,
                 enc_heads, in_dim, out_dim, sum_encoder_dim=True):
        super().__init__()
        self.sum_encoder_dim = sum_encoder_dim
        self.inp = nn.Linear(in_dim, enc_dim)
        self.encoder = Encoder(
            dim=enc_dim, depth=enc_depth,
            heads=enc_heads,ff_glu=True,
            residual_attn=True,
            rotary_pos_emb=True)
        self.out = nn.Linear(enc_dim, out_dim)
    def forward(self, x):
        x = self.inp(x)
        x = self.encoder(x)
        if self.sum_encoder_dim: x = x.sum(dim=1)
        return self.out(x)
    
def load_model_checkpoint(model, model_idx):
    """Load a model checkpoint and return the updated model index."""
    checkpoint_path = f"./models/model-dataset_60k_window20k_L4_scaling3_diff_checkpoint_{model_idx}.pth"
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    return model, model_idx + 1

def simulate_data(seed, demography_object, population_time, window_size, empirical_scaling_factor):
    """
    Simulate data using msprime, compute the mutation densities and transition matrix,
    and return both the numpy array version (for plotting) and the torch tensor versions.
    """
    # Define simulation parameters
    params = {
        'discrete_genome': True,
        'samples': 10, 
        'ploidy': 1,
        'demography': demography_object,
        'recombination_rate': 3.4e-8,
        'random_seed': seed,
        'sequence_length': 4_000_000
    }
    # Simulate ancestry and mutation process
    ts = msprime.sim_ancestry(**params)
    ts = msprime.mutate(ts, rate=7e-9, random_seed=seed)
    
    # Extract population size trajectory
    debugger = msprime.DemographyDebugger(demography=demography_object)
    population_size = debugger.population_size_trajectory(population_time).flatten()
    
    # Obtain mutation densities and compute the transition matrix (tm)
    mutation_densities = obtain_mutation_densities(ts, window_size=window_size)
    tm_np = np.log(
        calculate_transition_matrix(
            np.log((mutation_densities.flatten() + 1) * empirical_scaling_factor),
            np.log(population_time + 1)
        ) + 1
    )
    # Compute the log of the population size
    population_size_log = np.log(population_size)
    
    # Convert arrays to torch tensors for model inference
    tm_tensor = torch.tensor(tm_np, dtype=torch.float32)
    pop_size_tensor = torch.tensor(population_size_log, dtype=torch.float32)
    
    return tm_np, tm_tensor, pop_size_tensor

def run_inference(model, tm_tensor, pop_size_tensor, criterion, device):
    """Run model inference on the given data and return the predictions, targets, and loss."""
    dataset = [(tm_tensor, pop_size_tensor)]
    inf_dl = DataLoader(dataset, batch_size=1)
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in inf_dl:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            # Only one batch is expected so break after processing it
            break
            
    return predictions, targets, loss

def plot_simulation(ax, population_time, y_true, y_pred, color_true="black", color_pred="#0065bd"):
    """Plot the true and predicted population size trajectories on the given axis."""
    sns.lineplot(x=population_time, y=np.exp(y_true), drawstyle="steps-mid",
                 ax=ax, c=color_true, alpha=0.75)
    sns.lineplot(x=population_time, y=np.exp(y_pred), drawstyle="steps-mid",
                 ax=ax, c=color_pred, alpha=0.25)
    ax.set(yscale="log", xscale="log", ylim=(10000, 1_000_000))
    ax.set_ylabel("Population size [Nₑ]", fontsize=12)
    ax.set_xlabel("Population time [generations]", fontsize=12)
    ax.set_title("Sawtooth Demography", loc="left", fontsize=12, fontweight='bold')
    ax.grid(True)

def plot_average_predictions(ax, population_time, y_preds, color_pred="#0065bd"):
    """Plot the average of the predicted trajectories on the given axis."""
    avg_y = np.exp(np.array(y_preds).mean(axis=0))
    sns.lineplot(x=population_time, y=avg_y, drawstyle="steps-mid", ax=ax, c=color_pred)

def plot_transition_matrix(ax, tm_np, window_size):
    """Plot a heatmap of the transition matrix on the given axis."""
    sns.heatmap(tm_np, cmap="Blues", annot=False, fmt=".2f", linewidths=.5, ax=ax)
    ax.set_xlabel("log(Time intervals) x scaling factor")
    ax.set_ylabel("log(Time intervals) x scaling factor")
    ax.set_title(f"ρ/θ = 3.4x10⁸/ 7x10⁹ = 4.86 (ws = {window_size})",
                 fontsize=12, fontweight='bold', loc="left")




def encode_minor_alleles_as_derived(alleles):
    counts = Counter(alleles)
    minor = min(counts, key=counts.get)
    return [1 if allele == minor else 0 for allele in alleles]

def real_data2genotype_matrix_and_positions(data):
    data['alleles'] = data['alleles'].apply(lambda a: str(encode_minor_alleles_as_derived(a)))
    data = data[['chr', 'pos', 'alleles']]
    expanded_data = [
        [row['chr'], row['pos']] + row['alleles'][1:-1].split(', ')
        for _, row in data.iterrows()
    ]
    num_alleles = len(expanded_data[0]) - 2
    headers = ['chr', 'pos'] + [str(i) for i in range(num_alleles)]
    expanded_df = pd.DataFrame(expanded_data, columns=headers)
    positions = expanded_df['pos'].tolist()
    genotype_matrix = np.array(expanded_df.iloc[:, 2:].T, dtype=np.int32)
    assert len(expanded_df['chr'].unique()) == 1
    return genotype_matrix, np.array(positions)


def real_genotype_matrix2transition_matrix(genotype_matrix, sequence_start, sequence_end , positions, window_size, num_samples = 10):
    empirical_scaling_factor = np.exp(8.2)
    mask = np.logical_and(positions >= sequence_start, positions < sequence_end)

    genotype_matrix = genotype_matrix[:, mask]
    positions = positions[mask]

    combinations = binomial_combinations_k2(num_samples)
    sequence_length = int(sequence_end-sequence_start)
    Xs = []
    for sample_0, sample_1 in combinations:
        binary_genotype_matrix = genotype_matrix[[sample_0, sample_1]]

        sum_matrix = binary_genotype_matrix.sum(0)
        valid_mask = (sum_matrix < 2) & (sum_matrix > 0)
        binary_genotype_matrix = binary_genotype_matrix[:, valid_mask]
        binary_positions = positions[valid_mask]
        #print(binary_positions)

        full_sequence = np.zeros((2, int(sequence_length)), dtype=np.int32)
        full_sequence[:, binary_positions-int(sequence_start)] = binary_genotype_matrix
        #print(full_sequence.sum())
        X = obtain_densities(full_sequence, window_size=window_size)
        Xs.append(X)
    mutation_densities = torch.stack(Xs)
    mutation_densities = mutation_densities.flatten()
    #mutation_densities = mutation_densities[mutation_densities>0]
    tm = np.log(calculate_transition_matrix(np.log((mutation_densities.flatten()+1)*empirical_scaling_factor),
                                        np.log(population_time+1)) + 1)
    #tm = np.log(calculate_transition_matrix(np.log(mutation_densities+1)*empirical_scaling_factor, np.log(population_time+1)) + 1)
    tm = torch.tensor(tm, dtype=torch.float32)
    return tm

def real_genotype_matrix2mutation_sequence(genotype_matrix, sequence_start, sequence_end, positions, window_size, num_samples=10):
    mask = np.logical_and(positions >= sequence_start, positions < sequence_end)
    genotype_matrix = genotype_matrix[:, mask]
    positions = positions[mask]
    combinations = binomial_combinations_k2(num_samples)
    sequence_length = int(sequence_end - sequence_start)
    Xs = []
    for sample_0, sample_1 in combinations:
        binary_matrix = genotype_matrix[[sample_0, sample_1]]
        valid_mask = (binary_matrix.sum(0) < 2) & (binary_matrix.sum(0) > 0)
        binary_matrix = binary_matrix[:, valid_mask]
        binary_positions = positions[valid_mask]
        full_sequence = np.zeros((2, sequence_length), dtype=np.int32)
        full_sequence[:, binary_positions - int(sequence_start)] = binary_matrix
        Xs.append(obtain_densities(full_sequence, window_size=window_size))
    return torch.stack(Xs)


def infer_prediction(tm, model, device):
    """
    Convert a transition matrix sample to a tensor, run inference with the model,
    and return the prediction as a numpy array.
    """
    tm_tensor = torch.tensor(tm, dtype=torch.float32)
    # Dummy target; the model only needs the input for inference.
    population_size_log = torch.tensor(0, dtype=torch.float32)
    inf_dl = DataLoader([(tm_tensor, population_size_log)], batch_size=1)
    model.eval()
    with torch.no_grad():
        for inputs, _ in inf_dl:
            inputs = inputs.to(device)
            prediction = model(inputs)
            return prediction.detach().cpu().numpy()[0]


def process_model_predictions(model, model_idx_range, genotype_pair, sequence_starts, sequence_ends, window_size, device, log_list):
    """
    For each model checkpoint in the specified range, load the checkpoint,
    loop over the provided sequence windows to compute a transition matrix,
    record its sum in log_list, and run model inference.
    Returns the average prediction across models and sequences.
    
    genotype_pair: tuple containing (genotype_matrix, positions)
    """
    model_predictions = []  # List structure: [model][chromosome][sequence_sample_prediction]
    gm, positions = genotype_pair

    for model_idx in model_idx_range:
        checkpoint = f"./models/model-dataset_60k_window20k_L4_scaling3_diff_checkpoint_{model_idx}.pth"
        model.load_state_dict(torch.load(checkpoint)['model'])
        predictions_for_chromosome = []  # Here we assume one chromosome per genotype_pair
        
        predictions = []
        for seq_start, seq_end in zip(sequence_starts, sequence_ends):
            tm = real_genotype_matrix2transition_matrix(
                gm,
                sequence_start=seq_start,
                sequence_end=seq_end,
                positions=positions,
                window_size=window_size,
                num_samples=10
            )
            print(tm.sum())
            log_list.append(tm.sum().item())
            pred = infer_prediction(tm, model, device)
            predictions.append(pred)
        predictions_for_chromosome.append(predictions)
        model_predictions.append(predictions_for_chromosome)

    # Average over model checkpoints then over chromosomes (if more than one is provided)
    arr = np.array(model_predictions)
    # First average over model axis, then over chromosomes axis.
    avg_predictions = arr.mean(axis=0).mean(axis=1)
    return avg_predictions



# smp

def compute_transition_matrices(gm, positions, sequence_starts, sequence_ends, window_size, log_list, adjust_end=True):
    """
    Compute transition matrices for a given genotype matrix and positions.
    
    Parameters:
        gm: Genotype matrix.
        positions: Positions corresponding to the genotype data.
        sequence_starts: List of sequence start positions.
        sequence_ends: List of sequence end positions.
        window_size: Window size used in computing the transition matrix.
        log_list: List to which the sum of each transition matrix is appended.
        adjust_end: If True, subtract 1 from each sequence_end.
    
    Returns:
        List of computed transition matrices.
    """
    tms = []
    for start, end in zip(sequence_starts, sequence_ends):
        if adjust_end:
            end = end - 1
        tm = real_genotype_matrix2transition_matrix(
            gm,
            sequence_start=start,
            sequence_end=end,
            positions=positions,
            window_size=window_size,
            num_samples=10
        )
        print(tm.sum())
        log_list.append(tm.sum().item())
        tms.append(tm)
    return tms

def infer_from_tms(tms, model, device):
    """
    Run model inference for each transition matrix in the provided list.
    
    Parameters:
        tms: List of transition matrices.
        model: PyTorch model.
        device: Device to run inference on.
        
    Returns:
        List of predictions (one per transition matrix).
    """
    predictions = []
    for tm in tms:
        tm_tensor = torch.tensor(tm, dtype=torch.float32)
        # Dummy target, since we're only running inference.
        dummy_target = torch.tensor(0, dtype=torch.float32)
        inf_dl = DataLoader([(tm_tensor, dummy_target)], batch_size=1)
        
        model.eval()
        with torch.no_grad():
            for inputs, _ in inf_dl:
                inputs = inputs.to(device)
                pred = model(inputs)
                predictions.append(pred.detach().cpu().numpy()[0])
                break  # Only one batch is expected.
    return predictions

def process_genotype_predictions(model, genotype_pairs, sequence_starts, sequence_ends, window_size, device, log_list):
    """
    For each genotype pair (gm, positions), compute the transition matrices, run inference,
    and collect predictions.
    
    Parameters:
        model: PyTorch model.
        genotype_pairs: List of tuples (gm, positions).
        sequence_starts: List of sequence start positions.
        sequence_ends: List of sequence end positions.
        window_size: Window size to use in computing transition matrices.
        device: Device for model inference.
        log_list: List to store the sum of each transition matrix.
    
    Returns:
        List of predictions for each genotype pair.
    """
    y_preds_chromosomes = []
    for gm, positions in genotype_pairs:
        tms = compute_transition_matrices(gm, positions, sequence_starts, sequence_ends, window_size, log_list, adjust_end=True)
        preds = infer_from_tms(tms, model, device)
        y_preds_chromosomes.append(preds)
    return y_preds_chromosomes