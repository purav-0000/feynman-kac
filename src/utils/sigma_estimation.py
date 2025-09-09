import itertools
from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.signal import fftconvolve
from scipy.stats import norm
from tqdm import tqdm


# -- Constant sigma estimation ---

def estimate_constant_sigma(S_path, dt):
    """
    Estimate sigma using quadratic variation and Riemann sum

    :param S_path: Stock trajectory
    :param dt: Time step

    :return sigma_estimated: Estimated sigma
    """
    qv = np.sum(np.diff(S_path) ** 2, axis=0)
    it = np.sum(S_path[:-1] ** 2 * dt, axis=0)
    sigma_estimated = np.sqrt(np.mean(qv) / np.mean(it))

    return sigma_estimated


# --- Data prep ---

def prepare_data(S_path, t_path, time_threshold=5.0):
    """
    Prepares time-series data for diffusion estimation by filtering out large time gaps.
    """
    logging.info("Preparing data and filtering time gaps...")
    dt = np.diff(t_path)
    dS = np.diff(S_path)

    valid_indices = np.where(dt <= time_threshold)[0]

    num_total = len(dt)
    num_valid = len(valid_indices)
    logging.info(f"Filtered {num_total - num_valid} of {num_total} increments due to time gaps.")

    S_path_filtered = S_path[valid_indices]
    dS_filtered = dS[valid_indices]
    dt_filtered = dt[valid_indices]
    dS_sq_per_dt = dS_filtered ** 2 / dt_filtered

    return S_path_filtered, dS_sq_per_dt


# --- SIGMA ESTIMATION ---

def estimate_diffusion_fft(S_path, dS_sq_per_dt, bandwidth=2.5, n_grid=10_000):
    """
    Estimates the diffusion coefficient sigma(x) using FFT-based kernel regression.
    """
    S_min, S_max = np.min(S_path), np.max(S_path)
    S_grid = np.linspace(S_min, S_max, n_grid)
    dS = S_grid[1] - S_grid[0]

    bin_idx = np.clip(((S_path - S_min) / dS).astype(int), 0, n_grid - 1)
    weighted_sum = np.bincount(bin_idx, weights=dS_sq_per_dt, minlength=n_grid)
    weight_sum = np.bincount(bin_idx, minlength=n_grid)

    kernel_support = np.arange(-n_grid // 2, n_grid // 2)
    kernel = np.exp(-0.5 * (kernel_support * dS / bandwidth) ** 2)
    kernel /= np.sum(kernel)

    smooth_weighted = fftconvolve(weighted_sum, kernel, mode='same')
    smooth_counts = fftconvolve(weight_sum, kernel, mode='same')

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_sq = np.where(smooth_counts > 1e-9, smooth_weighted / smooth_counts, 0.0)

    return S_grid, np.sqrt(sigma_sq)


def estimate_diffusion_unprocessed(S_path, t_path, time_threshold=5.0, bandwidth=2.5, n_grid=10_000):
    """
    Wrapper to estimate diffusion directly from unprocessed S_path and t_path arrays.
    """
    # Step 1: Internally prepare and clean the data
    S_path_clean, dS_sq_per_dt_clean = prepare_data(S_path, t_path, time_threshold)

    # Step 2: Call the core estimation function with the cleaned data
    S_grid, sigma_on_grid = estimate_diffusion_fft(S_path_clean, dS_sq_per_dt_clean, bandwidth, n_grid)

    logging.info("Interpolating sigma values back onto the original time path...")
    sigma_at_each_time = np.interp(S_path, S_grid, sigma_on_grid)

    return S_grid, sigma_on_grid, sigma_at_each_time


# --- SINDy MODEL TRAINING ---

def find_best_model_unprocessed(X, T, sigma_true_fn, all_functions, all_names, **kwargs):
    """
    High-level wrapper to find the best SINDy model directly from unprocessed data.
    """
    # Get specific params for the estimation step
    time_threshold = kwargs.get('time_threshold', 300.0)
    bandwidth = kwargs.get('bandwidth', 1.5)
    n_grid = kwargs.get('n_grid', 5000)

    # Step 1: Call the unprocessed estimation wrapper
    print("Step 1: Estimating diffusion from raw data...")
    x_grid, sigma_vals = estimate_diffusion_unprocessed(X, T, time_threshold, bandwidth, n_grid)

    # Step 2: Call the core cross-validation and model search function
    print("\nStep 2: Finding best symbolic model with SINDy...")
    top_models = best_model_cv(x_grid, sigma_vals, sigma_true_fn, all_functions, all_names, **kwargs)

    return top_models

def best_model_cv(x_grid, sigma_vals, sigma, all_functions, all_names, **kwargs):
    """
    Original function to find the best model using cross-validation.
    """
    # Get parameters from kwargs with defaults
    k = kwargs.get('k', 4)
    alpha = kwargs.get('alpha', 0)
    threshold = kwargs.get('threshold', 0)
    top_k = kwargs.get('top_k', 5)
    trim_length = kwargs.get('trim_length', 1500)
    plot_mode = kwargs.get('plot_mode', 'overlay')
    debugging = kwargs.get('debugging', False)
    normalize = kwargs.get('normalize', False)
    subset_sizes = kwargs.get('subset_sizes', [1, 2, 3])

    all_combinations = []
    for lib_size in subset_sizes:
        combos = list(itertools.combinations(range(len(all_functions)), lib_size))
        all_combinations.extend(combos)
    print(f"Total models to train: {len(all_combinations)}")

    x_grid_trimmed = x_grid[trim_length:-trim_length]
    sigma_vals_trimmed = sigma_vals[trim_length:-trim_length]

    results = []
    batch_size = 10
    n_workers = -1

    for i in tqdm(range(0, len(all_combinations), batch_size), desc="Model batches"):
        batch = all_combinations[i:i + batch_size]
        batch_results = Parallel(n_jobs=n_workers)(
            delayed(train_model_cv)(
                x_grid_trimmed, sigma_vals_trimmed, all_functions, all_names, k, idxs,
                normalize=normalize, alpha=alpha, threshold=threshold
            ) for idxs in batch
        )
        results.extend(r for r in batch_results if r is not None)

    top_results = sorted(results, key=lambda r: r["error"])[:top_k]


    if debugging:
        # Print all models and scores (debugging)
        for result in results:
            if result['error'] != np.inf:
                result['model'].print(lhs=['sigma(X)'])
                print(result['model'].coefficients()[0])
                print("CV error:", result['error'])
                print()

        # Plotting
    if plot_mode == 'overlay':
        plt.figure(figsize=(10, 6))
        plt.plot(x_grid_trimmed, sigma_vals_trimmed, '--', linewidth=2, label='Estimated σ(X)', color='darkorange')
        plt.plot(x_grid_trimmed, sigma(x_grid_trimmed), '--', linewidth=2, label='True σ(X)', color='red')

        for i, result in enumerate(top_results):
            features = result['lib'].fit_transform(x_grid_trimmed)
            prediction = np.sum(result['coeffs'] * features.reshape(-1, len(x_grid_trimmed)).T, axis=1)
            label = f"#{i + 1} σ(X) ≈ {result['equation']}"
            plt.plot(x_grid_trimmed, prediction, label=label)

        plt.title('Top Sparse Models for σ(X)', fontsize=14)
        plt.xlabel('X')
        plt.ylabel('σ(X)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output.png", dpi=300, bbox_inches='tight')
        plt.show()

    elif plot_mode == 'subplots':
        fig, axs = plt.subplots(top_k, 1, figsize=(10, 3.5 * top_k), sharex=True)

        for i, result in enumerate(top_results):
            features = result['lib'].fit_transform(x_grid_trimmed)
            prediction = np.sum(result['coeffs'] * features.reshape(-1, len(x_grid_trimmed)).T, axis=1)

            axs[i].plot(x_grid_trimmed, sigma_vals_trimmed, '--', label='Estimated σ(X)', color='darkorange')
            axs[i].plot(x_grid_trimmed, sigma(x_grid_trimmed), '--', label='True σ(X)', color='red')
            axs[i].plot(x_grid_trimmed, prediction, label='Predicted', color='royalblue')
            axs[i].set_title(f"#{i + 1} σ(X) ≈ {result['equation']}", fontsize=12)
            axs[i].legend()
            axs[i].grid(True)

        plt.xlabel('X')
        plt.tight_layout()
        plt.savefig("output.png", dpi=300, bbox_inches='tight')
        plt.show()

    else:
        raise ValueError("plot_mode must be either 'overlay' or 'subplots'")

    return top_results  # List of top-k result dictionaries

# Cross validation training function
def train_model_cv(x_grid, sigma_vals, all_functions, all_names, k, subset_indices, normalize=False, alpha=0, threshold=0):
    """
    Fit models with different libraries, use CV to determine goodness of fit
    """

    # Prepare trimmed dataset (values near the edges are often noisy)
    trim_length = 1000
    x_grid_trimmed = x_grid[trim_length: len(x_grid) - trim_length]
    sigma_vals_trimmed = sigma_vals[trim_length: len(sigma_vals) - trim_length]

    interval_size = len(x_grid_trimmed) // k

    funcs = [all_functions[i] for i in subset_indices]
    names = [all_names[i] for i in subset_indices]

    total_error = 0
    try:
        for i in range(k):
            # Split train/test
            test_start = i * interval_size
            test_end = (i + 1) * interval_size
            x_test = x_grid_trimmed[test_start:test_end]
            y_test = sigma_vals_trimmed[test_start:test_end]

            x_train = np.concatenate([x_grid_trimmed[:test_start], x_grid_trimmed[test_end:]])
            y_train = np.concatenate([sigma_vals_trimmed[:test_start], sigma_vals_trimmed[test_end:]])

            # Create model with custom library
            lib = ps.CustomLibrary(
                library_functions=funcs,
                function_names=names,
                include_bias=False
            )

            model = ps.SINDy(
                feature_library=lib,
                optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=normalize),
                feature_names=["X"]
            )

            model.fit(x=x_train, x_dot=y_train)

            # Calculate error on validation/test set
            features_test = lib.fit_transform(x_test)
            coeffs = model.coefficients()[0]
            y_pred = np.sum(coeffs * features_test.reshape(-1, len(x_test)).T, axis=1)

            total_error += np.mean((y_pred - y_test) ** 2)

        # Initialize again before saving to dictionary
        lib = ps.CustomLibrary(
            library_functions=funcs,
            function_names=names,
            include_bias=False
        )

        model = ps.SINDy(
            feature_library=lib,
            optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, normalize_columns=normalize),
            feature_names=["X"]
        )

        avg_error = total_error / k

        # Fit model on full data before passing to dictionary
        model.fit(x=x_grid_trimmed, x_dot=sigma_vals_trimmed)

        # Remove redundant libraries
        coeffs = model.coefficients()
        if coeffs is not None and np.sum(coeffs[0] <= 1e-3) >= 1:
            avg_error = np.inf

        return {
            "error": avg_error,
            "model": model,
            "indices": subset_indices,
            "coeffs": model.coefficients()[0],
            "lib": lib,
            "equation": model.equations()[0]
        }

    except Exception as e:
        print(f"Failed model for indices {subset_indices}: {e}")
        return None

