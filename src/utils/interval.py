import torch
import numpy as np
import logging

import pysindy as ps

from src.utils.dataclasses import DerivativeData, PathData, SINDyContext
from src.utils.sigma_estimation import estimate_diffusion_unprocessed
from src.utils.sindy import build_theta_matrix, estimate_constant_sigma, extract_brownian, run_sindy


def get_ensemble_sindy_intervals(
        paths: PathData,
        derivs: DerivativeData,
        ctx: SINDyContext,
        n_bootstraps=100,
        sample_ratio=0.9,
        alpha=0.1
):
    """
    Implements Ensemble-SINDy by first generating the global Library (Theta)
    and Target (dY), and then bootstrapping the rows.
    """

    # 1. Estimate global variables and Brownian motion
    # If t is uniform, then make certain assumptions about sigma and Brownian motion
    # Else make other assumptions
    if ctx.uniform_t:
        ctx.dt = paths.t[1] - paths.t[0]
        ctx.sigma_est = estimate_constant_sigma(paths.s, ctx.dt)
        ctx.recovered_dB = extract_brownian(ctx.assumed_R, paths.s, ctx.sigma_est, ctx.dt)
    else:
        ctx.dt = min(np.diff(paths.t))

        # Time threshold gets rid of points just before a time skip in the dataset
        # Necessary for better estimation of Brownian
        s_grid, sigma_on_grid = estimate_diffusion_unprocessed(paths.s, paths.t, time_threshold=ctx.dt)
        ctx.sigma_est = np.interp(paths.s, s_grid, sigma_on_grid)
        ctx.recovered_dB = extract_brownian(ctx.assumed_R, paths.s, ctx.sigma_est, ctx.dt)

        # We trim the last element because build_theta_matrix does that for every function
        ctx.sigma_est = ctx.sigma_est[:-1]

    # 2. Global Library Construction (Theta Matrix)
    # Unpack and convert to numpy flattened arrays for easy matrix math
    Theta_full, dY_full, feature_names = build_theta_matrix(
        paths, derivs, ctx, return_tensors=False
    )

    if not ctx.uniform_t:
        t_path = paths.t[:-1]
        valid_indices = np.where(np.diff(t_path) <= ctx.dt)[0]
        Theta_full = Theta_full[valid_indices, :]
        dY_full = dY_full[valid_indices]

    N_total = len(dY_full)
    n_samples_per_boot = int(N_total * sample_ratio)

    # The maximum starting index to ensure we don't run out of bounds
    max_start_idx = N_total - n_samples_per_boot

    optimizer_lin_reg = ps.STLSQ(threshold=0, alpha=0, normalize_columns=False, max_iter=1)
    optimizer_STLSQ_bs = ps.STLSQ(threshold=5e-2, normalize_columns=True)
    optimizer_STLSQ = ps.STLSQ(threshold=0.05, alpha=0.01, normalize_columns=True)
    optimizer_SR3 = ps.SR3(reg_weight_lam=0.5, regularizer='L2', normalize_columns=True)
    optimizer_linreg = ps.STLSQ(threshold=0, alpha=0, normalize_columns=False, max_iter=1)
    ensemble_coefs = []

    # 3. Time Block Bootstrap Loop
    for b in range(n_bootstraps):
        # A. Pick ONE random starting index
        start_idx = np.random.randint(0, max_start_idx + 1)

        # B. Create a contiguous sequence of indices
        boot_indices = np.arange(start_idx, start_idx + n_samples_per_boot)

        # C. Subset the pre-calculated Theta and dY (Preserves time-order)
        Theta_boot = Theta_full[boot_indices]
        dY_boot = dY_full[boot_indices]

        # D. Fit SINDy with Identity Library
        try:
            sindy_model = ps.SINDy(
                optimizer=optimizer_STLSQ,
                feature_library=ps.IdentityLibrary(),
            )
            sindy_model.fit(Theta_boot, x_dot=dY_boot.reshape(-1, 1), t=ctx.dt)

            ensemble_coefs.append(sindy_model.coefficients().flatten())

        except Exception as e:
            logging.warning(f"Bootstrap {b} failed: {e}")
            continue

    # 4. Aggregate Statistics
    ensemble_np = np.array(ensemble_coefs)  # Shape: (n_bootstraps, n_features)

    # Median (Updated from np.mean to match your intended logic)
    xi_median = np.median(ensemble_np, axis=0)

    # Confidence Intervals
    lower_bound = np.percentile(ensemble_np, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(ensemble_np, 100 * (1 - alpha / 2), axis=0)

    # Inclusion Probability (ratio of times coefficient was strictly non-zero)
    inclusion_prob = np.count_nonzero(ensemble_np, axis=0) / len(ensemble_np)

    # Log the results using feature names for clarity
    for i, name in enumerate(feature_names):
        logging.info(
            f"{name}: Med={xi_median[i]:.4f}, CI=[{lower_bound[i]:.4f}, {upper_bound[i]:.4f}], Prob={inclusion_prob[i]:.2f}"
        )

    return xi_median, lower_bound, upper_bound, inclusion_prob


def get_conformal_sindy_intervals(
        Theta_full, dY_full, xi_fit,
        alpha=0.1,
        n_steps=100,
        lr=1e-3
):
    """
    Implements Feature Conformal Prediction for SINDy Coefficients.
    Based on Teng et al. (2023) 'Feature Conformal Prediction'.
    """
    N_total = len(dY_full)
    # 1. Split into 'Training' (already used for xi_fit) and 'Calibration'
    # For simplicity, we use the full set if it wasn't used for xi_fit,
    # but a 50/50 split is standard CP practice[cite: 24, 102].
    cal_idx = np.random.choice(N_total, size=N_total // 2, replace=False)
    Theta_cal = Theta_full[cal_idx]
    dY_cal = dY_full[cal_idx]

    n_features = Theta_full.shape[1]
    scores = []

    logging.info(f"--- Calculating Non-Conformity Scores (N={len(cal_idx)}) ---")

    # 2. Calculate Non-Conformity Scores (Algorithm 2 in paper)
    # For each calibration point, find surrogate v such that Theta[i] @ v = dY[i]
    for i in range(len(cal_idx)):
        row = Theta_cal[i]
        target = dY_cal[i]

        # Start from the fitted coefficients [cite: 142, 155]
        v = xi_fit.copy().flatten()

        # Gradient descent to find surrogate feature v
        # Loss = (row @ v - target)^2
        for _ in range(n_steps):
            pred = np.dot(row, v)
            grad = 2 * (pred - target) * row
            v -= lr * grad

        # Score is the distance in 'feature' (coefficient) space [cite: 135, 160]
        score = np.linalg.norm(v - xi_fit.flatten())
        scores.append(score)

    # 3. Calculate the (1-alpha) quantile [cite: 105, 183]
    scores = np.array(scores)
    Q_val = np.percentile(scores, 100 * (1 - alpha))

    # 4. Construct Intervals (Band Estimation) [cite: 53, 152]
    # The interval for each coefficient is xi_fit +/- Q_val
    # Note: This provides a symmetric 'ball' in coefficient space.
    lower_bound = xi_fit.flatten() - Q_val
    upper_bound = xi_fit.flatten() + Q_val

    for i in range(len(xi_fit)):
        logging.info(
            f"Med={xi_fit[i]:.4f}, CI=[{lower_bound[i]:.4f}, {upper_bound[i]:.4f}]")

    return xi_fit.flatten(), lower_bound, upper_bound, Q_val

"""

def get_feature_cp_intervals(
        net_u,
        s_train, t_train, u_train, recovered_dB, dt,
        mask_indices,
        cfg,
        n_calibration=100,  # Lower count for speed, paper uses Jackknife (N)
        alpha=0.1
):
    Implements Feature Conformal Prediction (Feature-CP).

    1. Computes Global Hessian/Gradient on a large subset.
    2. Solves for xi_global (Best fit on Global).
    3. Iteratively solves KKT systems for removed points to find xi_surrogate.
    4. Computes score ||xi_global - xi_surrogate||.

    device = s_train.device
    N_total = len(recovered_dB)

    logging.info(f"--- Starting Feature-CP (Calibration N={n_calibration}) ---")

    # 1. Prepare "Global" Matrices
    # We use a large representative subset (e.g. 50k) to approximate the full dataset's Hessian.
    global_subset_size = min(50000, N_total)
    indices_global = torch.randperm(N_total, device=device)[:global_subset_size]

    # Get Raw Library (Subset of Rows, All Columns)
    Theta_raw, dY_subset = _get_theta_dy(
        net_u, s_train, t_train, u_train, recovered_dB, dt, torch.tensor([i for i in range(0, N_total)])
    )

    # Apply Feature Mask (All Rows, Subset of Columns)
    Theta_global = Theta_raw[:, mask_indices]

    # Pre-compute Global Hessian (H) and Gradient (g)
    H_global = torch.matmul(Theta_global.T, Theta_global)
    g_global = torch.matmul(Theta_global.T, dY_subset)

    # 2. Compute xi_global
    # We solve H * xi = g. This is the unconstrained SINDy solution on the global subset.
    # We assume this is the "Truth" we are comparing against.
    xi_global = torch.linalg.lstsq(H_global, g_global, driver='gels').solution

    logging.info(f"Computed xi_global on {global_subset_size} points.")

    # 3. Calibration Loop
    # We pick n_calibration distinct points to act as constraints
    cal_indices = torch.randperm(N_total, device=device)[:n_calibration]

    scores = []

    for i, idx in enumerate(cal_indices):
        # A. Get the calibration point(s)
        # Using Batch Size = 1 for KKT stability
        theta_i_raw, dy_i = _get_theta_dy(
            net_u, s_train, t_train, u_train, recovered_dB, dt,
            idx.reshape(1)
        )
        theta_i = theta_i_raw[:, mask_indices]  # Apply Column Mask

        # B. UPDATE H and g (The "Leave-One-Out" Step)
        # Remove the influence of point 'i' from the global landscape
        H_surr = H_global - torch.matmul(theta_i.T, theta_i)
        g_surr = g_global - torch.matmul(theta_i.T, dy_i)

        # C. Solve KKT System (Eq. 17 in paper)
        # Minimize ||Theta_{-i} xi - dY_{-i}||^2  s.t.  theta_i xi = dy_i

        n_feats = H_surr.shape[0]
        n_constraints = theta_i.shape[0]  # 1

        # Build KKT System
        KKT_left = torch.zeros((n_feats + n_constraints, n_feats + n_constraints), device=device, dtype=torch.double)

        # [ 2H   A^T ]
        # [ A     0  ]
        KKT_left[:n_feats, :n_feats] = 2 * H_surr
        KKT_left[:n_feats, n_feats:] = theta_i.T
        KKT_left[n_feats:, :n_feats] = theta_i

        # RHS: [ 2g ]
        #      [ b  ]
        KKT_right = torch.zeros((n_feats + n_constraints, 1), device=device, dtype=torch.double)
        KKT_right[:n_feats] = 2 * g_surr
        KKT_right[n_feats:] = dy_i

        # Solve
        solution = torch.linalg.lstsq(KKT_left, KKT_right, driver='gels').solution
        xi_tilde = solution[:n_feats]

        # D. Compute Score (Eq. 18: L1 norm difference)
        diff = xi_tilde - xi_global
        print("LOCO: ", xi_tilde)
        print("SOLUTION: ", xi_global)
        print()
        score = torch.norm(diff, p=1).item()
        scores.append(score)

    # 4. Compute Quantile
    scores = np.array(scores)
    q_val = np.quantile(scores, 1 - alpha)

    logging.info(f"Feature-CP: computed q_{1 - alpha} = {q_val:.6e} from {n_calibration} samples.")

    return q_val, xi_global


def get_ensemble_sindy_intervals(
        net_u,
        s_train, t_train, u_train, recovered_dB, dt,
        mask_indices,
        cfg,
        n_bootstraps=100,  # Number of ensemble models (q in the paper)
        sample_ratio=1.0,  # Fraction of data to sample (with replacement)
        alpha=0.1  # 90% Confidence
):

    Implements Ensemble-SINDy (E-SINDy) for Uncertainty Quantification.
    Based on 'Ensemble-SINDy' paper (Fasel et al., 2021).

    Uses Bagging (Bootstrap Aggregating) on the rows of Theta and dY.

    device = s_train.device
    N_total = len(recovered_dB)

    logging.info(f"--- Starting E-SINDy (Bootstraps q={n_bootstraps}) ---")

    # 1. Prepare Full Matrices ONCE
    # We need the full library to sample from.
    # If memory is tight, we can generate these on the fly inside the loop,
    # but pre-generating is faster if it fits.
    # Let's try to fit a reasonably large chunk (e.g. 50k) or full if possible.
    subset_size = min(50000, N_total)
    # subset_size = N_total # Uncomment if you have 100GB RAM

    # We use a fixed subset of the data as our "Population" to bootstrap from
    indices_population = torch.randperm(N_total, device=device)[:subset_size]

    # Get Raw Library
    Theta_raw, dY_pop = _get_theta_dy(
        net_u, s_train, t_train, u_train, recovered_dB, dt, indices_population
    )

    # Apply Mask (Active terms only)
    Theta_pop = Theta_raw[:, mask_indices]

    n_samples_per_boot = int(subset_size * sample_ratio)
    n_features = Theta_pop.shape[1]

    # Storage for coefficients
    # Shape: (n_bootstraps, n_features)
    ensemble_coefs = torch.zeros((n_bootstraps, n_features), device=device, dtype=torch.double)

    # 2. Bootstrap Loop
    for b in range(n_bootstraps):
        # A. Resample Indices with Replacement
        # We sample integers from [0, subset_size)
        boot_indices = torch.randint(0, subset_size, (n_samples_per_boot,), device=device)

        # B. Select Data
        Theta_boot = Theta_pop[boot_indices]
        dY_boot = dY_pop[boot_indices]

        # C. Solve Least Squares (Standard SINDy step)
        # No regularization needed here because we are restricted to the active mask
        try:
            xi_boot = torch.linalg.lstsq(Theta_boot, dY_boot, driver='gels').solution
        except RuntimeError:
            # Fallback for singular matrix
            xi_boot = torch.matmul(torch.linalg.pinv(Theta_boot), dY_boot)

        ensemble_coefs[b] = xi_boot.flatten()

    # 3. Aggregate Statistics (Bragging / Robust Bagging)
    # Move to CPU for quantile logic
    ensemble_np = ensemble_coefs.cpu().detach().numpy()

    # Median Coefficient (Robust Estimate)
    xi_median = np.median(ensemble_np, axis=0)

    # Confidence Intervals
    # 90% Interval = [5th percentile, 95th percentile]
    lower_bound = np.percentile(ensemble_np, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(ensemble_np, 100 * (1 - alpha / 2), axis=0)

    return xi_median, lower_bound, upper_bound


def _get_theta_dy(net_u, s_train, t_train, u_train, recovered_dB, dt, indices):
    Helper: Gets Theta and dY for indices using Ground Truth.
    # Prepare Inputs
    X_full = torch.cat([s_train.reshape(-1, 1), t_train.reshape(-1, 1)], dim=1)
    X_curr = X_full[indices].clone().requires_grad_(True)

    u_true_curr = u_train[indices].reshape(-1, 1)
    u_true_next = u_train[indices + 1].reshape(-1, 1)
    dB_batch = recovered_dB[indices].reshape(-1, 1)

    # Derivatives from Network
    u_pred, u_t, u_x, u_xx = net_u.get_derivatives(X_curr)

    # Target from Data
    dY_target = u_true_next - u_true_curr

    # Library Construction
    Theta = build_sindy_library_torch(
        S_path=X_curr[:, 0].reshape(-1, 1),
        u_path=u_true_curr,
        u_t_pred=u_t.reshape(-1, 1),
        u_S_pred=u_x.reshape(-1, 1),
        u_SS_pred=u_xx.reshape(-1, 1),
        recovered_dB=dB_batch,
        dt=dt
    )

    return Theta, dY_target
"""