import numpy as np
from scipy.stats.qmc import LatinHypercube
import torch


def prepare_dataset_for_model(N_u, N_f, S_path, t_path, u_path, device):
    """
    Prepares dataset for model pre-training including data and collocation points

    :param N_u: Number of data points
    :param N_f: Number of collocation points
    :param S_path: Stock trajectory
    :param t_path: Time steps
    :param u_path: Options trajectory
    :param device: Device to store tensors on

    :return X_u_train_t: Processed data points
    :return X_f_train_t: Processed collocation points
    :return u_train_t: Ground truth for X_u_train_t
    """
    # --- Prepare data points ---
    # Switch between randomly chosen and evenly spaced points
    idx = np.random.choice(S_path.shape[0], N_u, replace=False)

    # Combine the S and t path values into one array for input to the model
    X_path_full = np.hstack((S_path.reshape(-1, 1), t_path.reshape(-1, 1)))
    u_path_full = u_path.reshape(-1, 1)

    # Index training points
    X_u_train = X_path_full[idx, :]
    u_train = u_path_full[idx, :]

    # --- Prepare collocation points ---
    S_min, S_max = S_path.min(), S_path.max()
    t_min, t_max = t_path.min(), t_path.max()

    # Define domain boundaries for collocation points
    lb = torch.tensor([S_min, t_min], device=device, dtype=torch.float32)
    ub = torch.tensor([S_max, t_max], device=device, dtype=torch.float32)

    # Sample random points uniformly distributed using hypercube
    X_f_train = lb.cpu().numpy() + (ub.cpu().numpy() - lb.cpu().numpy()) * LatinHypercube(d=2).random(n=N_f)

    X_f_train = np.vstack((X_f_train, X_u_train))  # Also enforce physics at data points

    # Convert to torch tensors
    X_u_train_t = torch.from_numpy(X_u_train).double().to(device)
    u_train_t = torch.from_numpy(u_train).double().to(device)
    X_f_train_t = torch.from_numpy(X_f_train).double().to(device)

    # For derivatives
    X_f_train_t.requires_grad = True

    return X_u_train_t, X_f_train_t, u_train_t


def build_library(S_vals, u_vals, u_S_vals, u_SS_vals):

    # S_vals should be of shape (-1, 1) for the model
    if len(S_vals.shape) == 1: S_vals = S_vals.unsqueeze(-1)

    """
    # Somewhat realistic library
    library = [u_vals, S_vals * u_S_vals, S_vals**2 * u_SS_vals, torch.ones_like(u_vals), u_S_vals, u_SS_vals, u_vals * u_S_vals]
    library_descriptions = ['u', 'S*u_S', 'S^2*u_SS', '1', 'u_S', 'u_SS', 'u*u_S']
    
    # Perfect Black Scholes library
    library = [u_vals, S_vals * u_S_vals, S_vals**2 * u_SS_vals]
    library_descriptions = ['u', 'S*u_S', 'S^2*u_SS']
    """

    # ??? A more realistic library perhaps
    library = [
        torch.ones_like(u_vals),  # Constant
        u_vals, S_vals, u_S_vals, u_SS_vals,
        u_vals * u_S_vals, S_vals * u_S_vals, u_vals ** 2 * u_S_vals, S_vals ** 2 * u_S_vals,
        u_vals * u_SS_vals, S_vals * u_SS_vals, u_vals ** 2 * u_SS_vals, S_vals ** 2 * u_SS_vals,
    ]
    library_descriptions = [
        '1',
        'u', 'S', 'u_S', 'u_SS',
        'u*u_S', 'S*u_S', 'u^2*u_S', 'S^2*u_S',
        'u*u_SS', 'S*u_SS', 'u^2*u_SS', 'S^2*u_SS'
    ]

    return torch.cat(library, dim=1), library_descriptions
