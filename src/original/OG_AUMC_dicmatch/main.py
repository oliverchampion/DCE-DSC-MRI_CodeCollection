from src.original.OG_MO_AUMC_ICR_RMH_NL_UK.ExtendedTofts.DCE \
    import \
    Cosine8AIF_ExtKety
import numpy as np
from joblib import Parallel, delayed
from itertools import product
import faiss


def generate_dictionary(time_points, AIF, Kep_range, ve_range, vp_range, dt_range):
    """
    Generate a dictionary of signal curves for all combinations of Ktrans, ve, and vp.

    Parameters:
        time_points (array): Array of time points.
        AIF (array): Arterial input function values at corresponding time points.
        Ktrans_range (array): Array of Ktrans values to include in the dictionary.
        ve_range (array): Array of ve values to include in the dictionary.
        vp_range (array): Array of vp values to include in the dictionary.

    Returns:
        dictionary (dict): Dictionary with parameter tuples as keys and signal curves as values.
    """

    def compute_Ct(Kep, ve, vp, dt):
        Ct = Cosine8AIF_ExtKety(time_points, AIF, Kep, dt / 60, ve, vp)
        return (Kep, ve, vp, dt), Ct

    # Define parameter combinations
    param_combinations = list(product(Kep_range, ve_range, vp_range, dt_range))

    # Use joblib to parallelize computations
    results = Parallel(n_jobs=6,backend="loky",batch_size="auto")(delayed(compute_Ct)(Kep, ve, vp, dt) for Kep, ve, vp, dt in param_combinations)

    # Convert results into a dictionary
    dictionary = dict(results)
    return dictionary


def dictionary_matching(Ct_simulated, dictionary):
    """
    Match the simulated curve to the closest dictionary entry.

    Parameters:
        Ct_simulated (array): Simulated tissue concentration over time.
        dictionary (dict): Dictionary with parameter tuples as keys and signal curves as values.

    Returns:
        best_params (tuple): Parameters (Ktrans, ve, vp) that best match the simulated curve.
        best_fit_curve (array): The dictionary curve that best matches the simulated curve.
    """
    best_params = None
    min_mse = float('inf')
    for params, Ct_dict in dictionary.items():
        mse = np.mean((Ct_simulated - Ct_dict) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_params = params
            best_curve = Ct_dict
    return best_params, best_curve

def dictionary_fais(Ct_simulated, dictionary):
    param_keys = list(dictionary.keys())  # List of parameter tuples
    Ct_dict_matrix = np.stack(list(dictionary.values()))  # Shape: (num_dict_entries, num_time_points)
    Ct_dict_matrix[Ct_dict_matrix != Ct_dict_matrix] = 0
    #Ct_simulated[Ct_simulated is not Ct_simulated] = 0
    # Reduce dictionary and measured signals to lower dimensions
    N_dictionary, T = Ct_dict_matrix.shape

    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(T)  # L2 (Euclidean) norm matching
    index.add(Ct_dict_matrix.astype(np.float32))  # Add dictionary to index

    # Search for nearest dictionary entries for all measured signals
    D, I = index.search(Ct_simulated.astype(np.float32), 1)  # Find 1 nearest neighbor

    # I contains indices of best matches, D contains distances
    best_match_indices = I.flatten()
    param_keys=np.array(param_keys)
    param_matrix = param_keys[best_match_indices]  # Shape: (num_voxels, num_params)
    return param_matrix
