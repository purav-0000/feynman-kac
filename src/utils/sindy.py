import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import sympy as sp
import torch

from src.utils.dataclasses import DerivativeData, PathData, SINDyContext
from src.utils.sigma_estimation import estimate_constant_sigma, estimate_diffusion_unprocessed


def get_sindy_mask(sindy_model, threshold=1e-5):
    """
    Extracts a boolean mask of active terms from a trained SINDy model.

    Returns:
        torch.Tensor: A boolean tensor of shape (n_features,) where True
                      indicates the term has a non-zero coefficient.
    """
    # coefficients() returns shape (n_targets, n_features).
    # For a single equation (dY), we take the first row [0].
    coeffs = sindy_model.coefficients()[0]

    # Create boolean mask
    # SINDy probably already applies a threshold and sets many terms to 0 but we still use a small number just to be
    # safe
    active_indices = np.abs(coeffs) > threshold

    # Return as a simple boolean list or tensor for easy indexing later
    return active_indices


def extract_brownian(assumed_r, S_path, sigma_estimate, dt):
    drift_term = assumed_r * S_path[:-1] * dt

    # If we have sigma values for all time points, then remove the last one
    # Just like for S_path
    if not isinstance(sigma_estimate, float):
        sigma_estimate = sigma_estimate[:-1]

    # If sigma is constant and not a function
    if isinstance(sigma_estimate, float):
        recovered_dB = (np.diff(S_path) - drift_term) / (sigma_estimate * S_path[:-1])
    else:
        recovered_dB = (np.diff(S_path) - drift_term) / (sigma_estimate)

    return recovered_dB


def save_updated_brownian(recovered_dB, dt, model_dir):
    plt.figure(figsize=(12, 4))
    plt.plot(
        recovered_dB,
        color="tab:blue",
        linewidth=1.2,
        alpha=0.9,
        label="Recovered dB"
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.title("Recovered Brownian Motion", fontsize=14, fontweight="bold")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Increment", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(frameon=False)
    plt.title(f"Mean: {np.mean(recovered_dB):.3e}, Std: {np.std(recovered_dB):.3e} (sqrt(dt): {np.sqrt(dt):.3e})")
    plt.tight_layout()
    plt.savefig(
        f"{model_dir}/recovered_brownian_motion.png",
        bbox_inches="tight",
        dpi=300
    )
    plt.close()


def generate_functions(term_strings, local_vars):
    """
    Takes a list of mathematical expressions as strings.
    Uses SymPy to algebraically simplify them (e.g. S * S -> S^2) for the names,
    and uses eval() to compute the actual numerical/tensor arrays.
    """
    evaluated_terms = []
    feature_names = []
    sym_locals = {key: sp.Symbol(key) for key in local_vars.keys()}

    for expr_str in term_strings:
        # 1. Smart Name Generation
        # Pass the symbolic locals so it knows exactly what the variables are
        sym_expr = sp.sympify(expr_str, locals=sym_locals)

        # Format the math into your preferred string style
        name = str(sym_expr).replace('**', '^').replace('*', '·')
        feature_names.append(name)

        # 2. Numerical Evaluation
        # Computes the actual Numpy/PyTorch arrays using your actual tensors/arrays
        evaluated_terms.append(eval(expr_str, {}, local_vars))

    return evaluated_terms, feature_names


def build_theta_matrix(paths: PathData, derivs: DerivativeData, ctx: SINDyContext, return_tensors: bool = False,
                       trim_percent: float = None):
    """
    Unified function to build the SINDy Theta matrix for both PySINDy (Numpy) and fine-tuning (PyTorch).
    Structure: dY = f*dt + Z*dB
    """

    # --- 1. Slicing and Trimming ---
    # Determine the slice length based on trimming
    total_len = len(paths.u)
    if trim_percent is not None:
        trim_size = int(total_len * trim_percent)
    else:
        # If no trim, we still drop the last element to align with dY/dB
        trim_size = total_len - 1

    # --- 2. Format Data (Numpy vs PyTorch) ---
    if return_tensors:
        # PyTorch format (keeps computational graph alive)
        S = paths.s[:trim_size].reshape(-1, 1)
        u = paths.u[:trim_size].reshape(-1, 1)
        u_t = derivs.u_t[:trim_size].reshape(-1, 1)
        u_s = derivs.u_s[:trim_size].reshape(-1, 1)
        u_ss = derivs.u_ss[:trim_size].reshape(-1, 1)
        dB = ctx.recovered_dB[:trim_size].reshape(-1, 1)

        # dY calculation for PyTorch
        dY = (paths.u[1:trim_size + 1] - paths.u[:trim_size]).reshape(-1, 1)

        # Helper for concatenation
        concat_fn = lambda x: torch.cat(x, dim=1)
    else:
        # Numpy format (detached for PySINDy)
        S = paths.s[:trim_size].flatten()
        u = paths.u[:trim_size].flatten()
        u_t = derivs.u_t.cpu().detach().numpy()[:trim_size].flatten()
        u_s = derivs.u_s.cpu().detach().numpy()[:trim_size].flatten()
        u_ss = derivs.u_ss.cpu().detach().numpy()[:trim_size].flatten()
        dB = ctx.recovered_dB[:trim_size].flatten()

        # dY calculation for Numpy
        dY = np.diff(paths.u[:trim_size + 1]).flatten()

        # Helper for concatenation
        concat_fn = lambda x: np.vstack(x).T

    # --- 3. Shared Mathematical Library ---
    math_env = {
        'S': S,
        'u': u,
        'u_t': u_t,
        'u_x': u_s,  # Naming it u_x here so it strings nicely
        'u_xx': u_ss,
        'r': ctx.assumed_R,
        'sigma': ctx.sigma_est,
        'epsilon': 1e-6
    }
    f_strings = [
        "u_t",                              # Actual Black-Scholes term
        "r * S * u_x",                      # Actual Black-Scholes term
        "0.5 * sigma**2 * S * S * u_xx",    # Actual Black-Scholes term
        # "sigma**2 * S",
        # "r * u",
        "sigma**2 * u**2 / (S + epsilon)",
        # "r * S**2 * u_xx"
    ]
    z_strings = [
        "sigma * S * u_x",                  # Actual Black-Scholes term
        # "sigma * S",
        # "sigma * u",
        # "sigma * S**2 * u_xx",
        # "sigma * u**2 / (S + epsilon)"
    ]

    f_terms, f_names = generate_functions(f_strings, math_env)
    z_terms, z_names = generate_functions(z_strings, math_env)

    # Build intermediate matrices
    f_matrix = concat_fn(f_terms)
    z_matrix = concat_fn(z_terms)

    # --- 4. Construct Final Theta ---
    if return_tensors:
        Theta = torch.cat([ctx.dt * f_matrix, dB * z_matrix], dim=1)
    else:
        Theta = np.hstack([ctx.dt * f_matrix, dB.reshape(-1, 1) * z_matrix])

    feature_names = [f"dt·{n}" for n in f_names] + [f"dB·{n}" for n in z_names]

    return Theta, dY, feature_names


def run_sindy(paths: PathData, derivs: DerivativeData, ctx: SINDyContext, sindy_model=None, trim_percent=None,
              save_dir=None):
    """
    Prepares data and evaluates a SINDy model.
    If no SINDy model is provided, it fits a new one.
    """

    # --- 1. Data Processing ---
    # If t is uniform, then make certain assumptions about sigma and Brownian motion
    # Else make other assumptions
    if ctx.uniform_t:
        ctx.dt = paths.t[1] - paths.t[0]
        ctx.sigma_est = estimate_constant_sigma(paths.s, ctx.dt)
        ctx.recovered_dB = extract_brownian(ctx.assumed_R, paths.s, ctx.sigma_est, ctx.dt)
        t_sindy = ctx.dt
    else:
        ctx.dt = min(np.diff(paths.t))

        # Time threshold gets rid of points just before a time skip in the dataset
        # Necessary for better estimation of Brownian
        s_grid, sigma_on_grid = estimate_diffusion_unprocessed(paths.s, paths.t, time_threshold=ctx.dt)
        ctx.sigma_est = np.interp(paths.s, s_grid, sigma_on_grid)
        ctx.recovered_dB = extract_brownian(ctx.assumed_R, paths.s, ctx.sigma_est, ctx.dt)

        # We trim the last element because build_theta_matrix does that for every function
        ctx.sigma_est = ctx.sigma_est[:-1]

        t_sindy = paths.t

    theta_matrix, dy, feature_names = build_theta_matrix(
        paths, derivs, ctx, return_tensors=False, trim_percent=trim_percent
    )

    # --- 2. Trimming & Masking ---
    # Manually trim t_path
    # This follows exactly what happens in build_theta_matrix
    if not ctx.uniform_t:   # If t_sindy is not dt
        if trim_percent is not None:
            trim_size = int(len(paths.u) * trim_percent)
            t_path = paths.t[:trim_size]
        else:
            t_path = paths.t[:-1]

        # Apply masks that get rid of big time jumps
        valid_indices = np.where(np.diff(t_path) <= ctx.dt)[0]
        theta_matrix = theta_matrix[valid_indices, :]
        dy = dy[valid_indices]
        t_sindy = t_path[valid_indices]
        ctx.recovered_dB = ctx.recovered_dB[valid_indices]

    if save_dir is not None:
        save_updated_brownian(ctx.recovered_dB, ctx.dt, save_dir)

    # --- 3. Model Fitting (If needed) ---
    # If they passed a sindy_model, it means they wanted to evaluate
    prefix = "train" if sindy_model is None else "test"
    if sindy_model is None:
        optimizer_lin_reg = ps.STLSQ(threshold=0, alpha=0, normalize_columns=False, max_iter=1)
        optimizer_STLSQ_bs = ps.STLSQ(threshold=5e-2, normalize_columns=True)
        optimizer_STLSQ = ps.STLSQ(threshold=0.05, alpha=0.01, normalize_columns=True)
        optimizer_SR3 = ps.SR3(reg_weight_lam=0.5, regularizer='L2', normalize_columns=True)

        sindy_model = ps.SINDy(
            optimizer=optimizer_STLSQ,
            feature_library=ps.IdentityLibrary(),
        )
        sindy_model.fit(theta_matrix, x_dot=dy.reshape(-1, 1), t=t_sindy, feature_names=feature_names)

    # --- 4. Evaluation & Debugging ---
    score = sindy_model.score(x=theta_matrix, x_dot=dy.reshape(-1, 1), t=t_sindy)
    logging.info(f"{prefix.capitalize()} SINDy score (R^2): {score}")

    """
    # Stats for debugging
    print(f"dy stats: mean={np.mean(dy):.3e}, std={np.std(dy):.3e}" )
    for i in range(theta_matrix.shape[1]):
        print(f"Theta col {i} stats: mean={np.mean(theta_matrix[:, i]):.3e}, std={np.std(theta_matrix[:, i]):.3e}")
    """

    # Equation for debugging
    # sindy_model.print(lhs=["dY"])

    # --- 5. Plotting ---
    if save_dir is not None:
        plt.figure(figsize=(16, 8))

        # Safe x-axis generation for Matplotlib
        if ctx.uniform_t:
            # t_sindy is a float (dt), so we must use the array for plotting
            t_plot = paths.t[:len(dy)]
        else:
            # t_sindy is already an array correctly subsetted by valid_indices
            t_plot = t_sindy

        plt.plot(t_plot, dy, label="True dY")
        plt.plot(t_plot, sindy_model.predict(theta_matrix), label="Predicted dY", alpha=0.7)
        plt.title(f"{prefix.capitalize()} dY Prediction (R^2: {score:.4f})")
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.savefig(Path(save_dir) / f"{prefix}_dy.png", bbox_inches="tight", dpi=300)
        plt.close()

    return sindy_model
