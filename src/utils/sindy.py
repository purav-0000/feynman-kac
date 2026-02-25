import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from sklearn.linear_model import LinearRegression
import torch

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


def prepare_theta_matrix(S_path, u_path, u_t_pred, u_S_pred, u_SS_pred, recovered_dB, dt, assumed_R, sigma_est,
                         trim_percent=None):

    # To numpy and flatten
    u_path_sindy = u_path.flatten()
    u_t_sindy = u_t_pred.cpu().detach().numpy().flatten()
    u_s_sindy = u_S_pred.cpu().detach().numpy().flatten()
    u_ss_sindy = u_SS_pred.cpu().detach().numpy().flatten()

    recovered_dB_sindy = recovered_dB
    # Trimming because derivatives get fuzzy near the end
    if trim_percent is not None:
        trim_size = int(len(u_path) * trim_percent)
        u_path_sindy = u_path_sindy[:trim_size]
        u_t_sindy = u_t_sindy[:trim_size]
        u_s_sindy = u_s_sindy[:trim_size]
        u_ss_sindy = u_ss_sindy[:trim_size]
        S_path_sindy = S_path[:trim_size]
        recovered_dB_sindy = recovered_dB[:trim_size]
    else:  # Account for the fact recovered_dB will be smaller by 1 element
        u_path_sindy = u_path_sindy[:-1]
        u_t_sindy = u_t_sindy[:-1]
        u_s_sindy = u_s_sindy[:-1]
        u_ss_sindy = u_ss_sindy[:-1]
        S_path_sindy = S_path[:-1]

    # dY for training target
    if trim_percent is not None:
        trim_size = int(len(u_path) * trim_percent)
        dY = np.diff(u_path[:trim_size + 1])
    else:
        dY = np.diff(u_path)

    """
    # --- Black-Scholes specific library ---
    rate_term = S_path_sindy * u_s_sindy
    f_candidate_terms_matrix = np.vstack([
        u_t_sindy,
        assumed_R * rate_term,
        (sigma_est ** 2) * (S_path_sindy ** 2) * u_ss_sindy,
    ]).T
    Z_candidate_terms_matrix = np.vstack([
        sigma_est * S_path_sindy * u_s_sindy,
    ]).T
    f_candidate_feature_names = ["u_t", "r * X * u_x", "sigma^2 * X^2 * u_xx"]
    Z_candidate_feature_names = ["sigma * X * u_x"]
    """

    # --- (Not) Black-Scholes specific library ---
    # --- 1. Drift Candidates (dt terms) ---
    # Intrinsic Units required: [Currency] / [Time]
    # When multiplied by dt [Time], the result is [Currency].

    f_candidate_terms_matrix = np.vstack([
        # --- A. Standard Black-Scholes Terms ---
        u_t_sindy,  # Time Decay (Theta)
        assumed_R * S_path_sindy * u_s_sindy,  # Drift (Rate · Delta)
        (sigma_est ** 2) * (S_path_sindy ** 2) * u_ss_sindy,  # Ito Correction (Vol² · Gamma)

        # --- B. Variance & Mean Reversion ---
        # (sigma_est ** 2) * S_path_sindy,  # Variance Bias (Vol² · S)
        # assumed_R * u_path_sindy,  # Mean Reversion (Rate · u)

        # --- C. Nonlinear / Distress Terms ---
        (sigma_est ** 2) * (u_path_sindy ** 2) / S_path_sindy,  # Inverse Price / Distress

        # --- D. Higher Order Interactions ---
        # Interaction: Rate x Convexity
        assumed_R * (S_path_sindy ** 2) * u_ss_sindy,  # Rate-Gamma Cross

        # Interaction: Hedging Flow (Delta x Gamma)
        # Note: Must divide by S to maintain units of [Currency]/[Time]
        # Unit Check: [1/T] * [C] * [1] * [C]^2 * [1/C] / [C] = [C]/[T] (Correct)
        (sigma_est ** 2) * S_path_sindy * u_s_sindy * (S_path_sindy ** 2) * u_ss_sindy / S_path_sindy
    ]).T

    f_candidate_feature_names = [
        "u_t",  # Time evolution
        "r·S·u_x",  # Delta drift
        "σ²·S²·u_xx",  # Ito/Gamma
        # "σ²·S",  # Linear Variance  # STRUGGLES WITH THIS TERM
        # "r·u",  # Mean Reversion  # IT ALWAYS GETS THIS TERM
        "σ²·u²/S",  # Inverse/Distress
        "r·S²·u_xx",  # Rate-Gamma Cross  # HIGHLY CORRELATED TERM
        "σ²·S²·u_x·u_xx"  # Delta-Gamma Interaction
    ]

    # --- 2. Diffusion Candidates (dB terms) ---
    # Intrinsic Units required: [Currency] / [Root-Time]
    # When multiplied by dB [Root-Time], the result is [Currency].

    Z_candidate_terms_matrix = np.vstack([
        # --- A. Standard Black-Scholes Diffusion ---
        sigma_est * S_path_sindy * u_s_sindy,  # Geometric Vol (Vol · Delta)

        # --- B. Alternative Volatility Models ---
        sigma_est * S_path_sindy,  # Bachelier / Normal Vol
        sigma_est * u_path_sindy,  # Value-Proportional Vol

        # --- C. Nonlinear Noise ---
        sigma_est * (S_path_sindy ** 2) * u_ss_sindy,  # Gamma-Driven Noise
        sigma_est * (u_path_sindy ** 2) / S_path_sindy  # Inverse/Distress Noise
    ]).T

    Z_candidate_feature_names = [
        "σ·S·u_x",  # Standard Diffusion
        "σ·S",  # Constant Vol
        "σ·u",  # Value-Proportional
        "σ·S²·u_xx",  # Gamma-Driven
        "σ·u²/S"  # Inverse/Distress
    ]

    """
    # --- Slightly more flexible library (SINDy test) ---
    rate_term = S_path_sindy * u_s_sindy
    f_candidate_terms_matrix = np.vstack([
        u_t_sindy,
        rate_term,
        S_path_sindy ** 2 * u_ss_sindy,

        # Noisy terms for BS
        # np.ones_like(S_path_sindy),
        S_path_sindy ** 3,
        u_path_sindy ** 2,
        S_path_sindy * u_path_sindy ** 2
    ]).T
    Z_candidate_terms_matrix = np.vstack([
        S_path_sindy * u_s_sindy,
    ]).T
    f_candidate_feature_names = ["u_t", "X * u_x", "X^2 * u_xx", "X^3", "u^2", "X * u^2"]
    Z_candidate_feature_names = ["X * u_x", "X", "u"]


    # --- Flexible Library ---
    # Candidate terms for the drift part, f(t, X, u, u_x, u_xx)
    f_candidate_terms_matrix = np.vstack([
        # np.ones_like(S_path_sindy),  # Bias/constant term
        u_path_sindy,  # For interest rate-like terms (e.g., r*u)
        u_s_sindy,  # For costs or drift related to delta
        S_path_sindy * u_s_sindy,  # Classic BS drift/hedging term
        S_path_sindy ** 2 * u_ss_sindy,  # Classic BS convexity/gamma term
    ]).T
    f_candidate_feature_names = [
        # "1",
        "u",
        "u_x",
        "X*u_x",
        "X^2*u_xx",
    ]
    # Candidate terms for the diffusion part, Z(t, X, u, u_x, u_xx)
    Z_candidate_terms_matrix = np.vstack([
        S_path_sindy,  # For volatility proportional to price (e.g., σX)
        S_path_sindy * u_s_sindy,  # Classic BS diffusion/hedging term
        u_path_sindy  # For volatility dependent on the option price itself
    ]).T
    Z_candidate_feature_names = [
        "X",
        "X*u_x",
        "u"
    ]

    # --- Extremely Flexible Library ---
    # Candidate terms for the drift part, f(t, X, u, u_x, u_xx)
    epsilon = 1e-8
    # --- 1. Drift Candidate Library (f) ---
    f_candidate_terms_matrix = np.vstack([
        # A) Core Polynomials & Derivatives
        np.ones_like(S_path_sindy),
        u_path_sindy,
        u_path_sindy ** 2,
        u_path_sindy ** 3,
        S_path_sindy,
        S_path_sindy ** 2,
        S_path_sindy ** 3,
        u_t_sindy,
        u_s_sindy,
        u_s_sindy ** 2,
        u_ss_sindy,
        u_ss_sindy ** 2,

        # B) Comprehensive Derivative Interactions (Polynomial)
        S_path_sindy * u_t_sindy,
        S_path_sindy ** 2 * u_t_sindy,
        u_path_sindy * u_t_sindy,
        S_path_sindy * u_s_sindy,  # Black-Scholes Drift
        S_path_sindy ** 2 * u_s_sindy,
        u_path_sindy * u_s_sindy,
        S_path_sindy * u_ss_sindy,
        S_path_sindy ** 2 * u_ss_sindy,  # Black-Scholes Gamma
        u_path_sindy * u_ss_sindy,

        # C) Derivative Cross-Multiplication (Path-Dependence)
        u_t_sindy * u_s_sindy,
        u_t_sindy * u_ss_sindy,
        u_s_sindy * u_ss_sindy,
        S_path_sindy * u_s_sindy * u_ss_sindy,

        # D) Rational Functions (Financial Ratios)
        u_path_sindy / (S_path_sindy + epsilon),  # Moneyness/Value Ratio
        u_s_sindy / (u_path_sindy + epsilon),  # Delta relative to price
        u_ss_sindy / (u_path_sindy + epsilon),  # Gamma relative to price
        (S_path_sindy * u_s_sindy) / (u_path_sindy + epsilon),  # BS Drift relative to price

        # E) Exotic Functions (Logarithmic & Exponential)
        np.log(S_path_sindy + epsilon),
        np.log(u_path_sindy + epsilon),
        np.exp(u_path_sindy),
        u_s_sindy * np.log(S_path_sindy + epsilon),
        u_ss_sindy * np.log(S_path_sindy + epsilon)
    ]).T

    f_candidate_feature_names = [
        # A
        "1", "u", "u^2", "u^3", "X", "X^2", "X^3",
        "u_t", "u_x", "u_x^2", "u_xx", "u_xx^2",
        # B
        "X*u_t", "X^2*u_t", "u*u_t",
        "X*u_x", "X^2*u_x", "u*u_x",
        "X*u_xx", "X^2*u_xx", "u*u_xx",
        # C
        "u_t*u_x", "u_t*u_xx", "u_x*u_xx", "X*u_x*u_xx",
        # D
        "u/X", "u_x/u", "u_xx/u", "(X*u_x)/u",
        # E
        "log(X)", "log(u)", "exp(u)", "u_x*log(X)", "u_xx*log(X)"
    ]

    # Candidate terms for the diffusion part, Z(t, X, u, u_x, u_xx)
    Z_candidate_terms_matrix = np.vstack([
        # A) Core Polynomials & Derivatives
        np.ones_like(S_path_sindy),
        S_path_sindy,
        S_path_sindy ** 2,
        u_path_sindy,
        u_path_sindy ** 2,
        u_s_sindy,
        u_s_sindy ** 2,
        u_ss_sindy,

        # B) Comprehensive Derivative Interactions
        S_path_sindy * u_path_sindy,
        S_path_sindy * u_s_sindy,  # Black-Scholes Volatility
        S_path_sindy * u_ss_sindy,
        u_path_sindy * u_s_sindy,
        u_path_sindy * u_ss_sindy,

        # C) Rational Functions (Financial Ratios)
        u_path_sindy / (S_path_sindy + epsilon),  # Moneyness
        S_path_sindy / (u_path_sindy + epsilon),  # Inverse Moneyness
        u_s_sindy / (u_path_sindy + epsilon),  # Relative Delta

        # D) Exotic Functions (CEV / Heston-like terms)
        np.sqrt(S_path_sindy + epsilon),
        np.sqrt(u_path_sindy + epsilon),
        np.log(S_path_sindy + epsilon),
        np.log(u_path_sindy + epsilon)
    ]).T

    Z_candidate_feature_names = [
        # A
        "1", "X", "X^2", "u", "u^2", "u_x", "u_x^2", "u_xx",
        # B
        "X*u", "X*u_x", "X*u_xx", "u*u_x", "u*u_xx",
        # C
        "u/X", "X/u", "u_x/u",
        # D
        "sqrt(X)", "sqrt(u)", "log(X)", "log(u)"
    ]
    """

    # Combine the drift and diffusion terms into the final Theta matrix
    # The structure of the equation is dY ≈ f*dt + Z*dB
    Theta_matrix = np.hstack([
        dt * f_candidate_terms_matrix,
        recovered_dB_sindy.reshape(-1, 1) * Z_candidate_terms_matrix
    ])

    # Create descriptive feature names for the final model
    feature_names = [f"dt*{name}" for name in f_candidate_feature_names] + \
                    [f"dB*{name}" for name in Z_candidate_feature_names]

    return Theta_matrix, dY, feature_names


def discover_equation(s_path, u_path, t_path, derivatives, assumed_R=0.1, uniform_t=False, trim_percent=None,
                      save_dir=None):
    """Discovers the PDE using SINDy on the current data history."""
    u_pred, u_t_pred, u_s_pred, u_ss_pred = derivatives

    # If t is uniform, then make certain assumptions about sigma and Brownian motion
    # Else make other assumptions
    if uniform_t:
        dt = t_path[1] - t_path[0]

        sigma_est = estimate_constant_sigma(s_path, dt)
        recovered_dB = extract_brownian(assumed_R, s_path, sigma_est, dt)

        t_sindy = dt
    else:
        dt = min(np.diff(t_path))

        # Time threshold gets rid of points just before a time skip in the dataset
        # Necessary for better estimation of Brownian
        s_grid, sigma_on_grid = estimate_diffusion_unprocessed(s_path, t_path, time_threshold=dt)
        sigma_est = np.interp(s_path, s_grid, sigma_on_grid)
        recovered_dB = extract_brownian(assumed_R, s_path, sigma_est, dt)

        t_sindy = t_path

    theta_matrix, dy, feature_names = prepare_theta_matrix(
        s_path, u_path, u_t_pred, u_s_pred, u_ss_pred, recovered_dB, dt, assumed_R, sigma_est, trim_percent=trim_percent
    )

    # Manually trim t_path
    # This follows exactly what happens in prepare_theta_matrix
    if not uniform_t:   # If t_sindy is not dt
        if trim_percent is not None:
            trim_size = int(len(u_path) * trim_percent)
            t_path = t_path[:trim_size]
        else:
            t_path = t_path[:-1]

    # Apply masks that get rid of big time jumps
    if not uniform_t:
        valid_indices = np.where(np.diff(t_path) <= dt)[0]
        theta_matrix = theta_matrix[valid_indices, :]
        dy = dy[valid_indices]
        t_sindy = t_path[valid_indices]
        recovered_dB = recovered_dB[valid_indices]

    if save_dir is not None:
        save_updated_brownian(recovered_dB, dt, save_dir)

    optimizer_lin_reg = ps.STLSQ(threshold=0, alpha=0, normalize_columns=False, max_iter=1)
    optimizer_STLSQ = ps.STLSQ(threshold=5e-2, alpha=0, normalize_columns=True)
    optimizer_SR3 = ps.SR3(reg_weight_lam=0.5, regularizer='L2', normalize_columns=True)

    sindy_model = ps.SINDy(
        optimizer=optimizer_STLSQ,
        feature_library=ps.IdentityLibrary(),
    )

    """
    # Stats for debugging
    print(f"dy stats: mean={np.mean(dy):.3e}, std={np.std(dy):.3e}" )
    for i in range(theta_matrix.shape[1]):
        print(f"Theta col {i} stats: mean={np.mean(theta_matrix[:, i]):.3e}, std={np.std(theta_matrix[:, i]):.3e}")
    """

    sindy_model.fit(theta_matrix, x_dot=dy.reshape(-1, 1), t=t_sindy, feature_names=feature_names)

    # Score for debugging
    logging.info(f"SINDy score (R^2): {sindy_model.score(x=theta_matrix, x_dot=dy.reshape(-1, 1), t=t_sindy)}")

    # Equation for debugging
    # sindy_model.print(lhs=["dY"])

    # Plot for debugging if not Black Scholes
    if save_dir is not None and not isinstance(t_sindy, float):
        plt.figure(figsize=(16, 8))
        plt.plot(t_sindy, dy, label="dY")
        plt.plot(t_sindy, sindy_model.predict(theta_matrix), label="predicted dY")
        plt.grid()
        plt.legend()
        plt.savefig(Path(save_dir) / "dy.png", bbox_inches="tight", dpi=300)
        plt.close()

    return sindy_model


def build_sindy_library_torch(S_path, u_path, u_t_pred, u_S_pred, u_SS_pred, recovered_dB, dt, assumed_R, sigma_est):
    """
    PyTorch implementation of the prepare_theta_matrix library to enable differentiability.
    Constructs the matrix Theta such that dY ~ Theta @ xi

    Args:
        S_path, u_path, u_t_pred, u_S_pred, u_SS_pred, recovered_dB, dt: Tensors of shape (N, 1) or (N,)
        recovered_dB: Tensor of shape (N, 1) or (N,) representing Brownian increments
        dt: float or scalar Tensor

    Returns:
        Theta: Tensor of shape (N, n_features)
    """
    """
    # PERFECT LIBRARY
    # Ensure shapes are compatible (flatten to column vectors)
    S = S_path.reshape(-1, 1)
    u = u_path.reshape(-1, 1)
    u_t = u_t_pred.reshape(-1, 1)
    u_S_pred = u_S_pred.reshape(-1, 1)
    u_SS_pred = u_SS_pred.reshape(-1, 1)
    recovered_dB = recovered_dB.reshape(-1, 1)

    # --- 1. Drift Candidate Library (f) ---
    # Matches: [u_t, X * u_x, X^2 * u_xx]
    f_term_1 = u_t
    f_term_2 = S * u_S_pred
    f_term_3 = (S ** 2) * u_SS_pred

    f_matrix = torch.cat([f_term_1, f_term_2, f_term_3], dim=1)

    # --- 2. Diffusion Candidate Library (Z) ---
    # Matches: [X * u_x]
    z_term_1 = S * u_S_pred

    z_matrix = z_term_1  # Shape (N, 1)

    # --- 3. Construct Theta ---
    # Theta = [dt * f, dB * Z]
    # Structure: dY = f*dt + Z*dB

    Theta = torch.cat([
        dt * f_matrix,
        recovered_dB * z_matrix
    ], dim=1)
    """
    # Ensure shapes are compatible (flatten to column vectors)
    S = S_path.reshape(-1, 1)
    u = u_path.reshape(-1, 1)
    u_t = u_t_pred.reshape(-1, 1)
    u_S = u_S_pred.reshape(-1, 1)
    u_SS = u_SS_pred.reshape(-1, 1)
    dB = recovered_dB.reshape(-1, 1)

    # --- 1. Drift Candidate Library (f) ---
    # Intrinsic Units: [Currency] / [Time]
    # When multiplied by dt, result is [Currency]

    # Term 1: Time Decay (Theta)
    f_term_1 = u_t

    # Term 2: Linear Drift (Rate * Delta * S)
    f_term_2 = assumed_R * S * u_S

    # Term 3: Convexity / Ito Correction (0.5 * Vol^2 * Gamma * S^2)
    f_term_3 = 0.5 * (sigma_est ** 2) * (S ** 2) * u_SS

    # Term 4: Variance Bias (Vol^2 * S)
    f_term_4 = (sigma_est ** 2) * S

    # Term 5: Mean Reversion (Rate * u)
    f_term_5 = assumed_R * u

    # Term 6: Inverse Price / Distress (Vol^2 * u^2 / S)
    f_term_6 = (sigma_est ** 2) * (u ** 2) / (S + 1e-6)  # Added epsilon for stability

    # Term 7: Rate-Gamma Cross (Rate * Gamma * S^2)
    f_term_7 = assumed_R * (S ** 2) * u_SS

    f_matrix = torch.cat([
        f_term_1, f_term_2, f_term_3,
        f_term_4, f_term_5, f_term_6, f_term_7
    ], dim=1)

    # --- 2. Diffusion Candidate Library (Z) ---
    # Intrinsic Units: [Currency] / [Root-Time]
    # When multiplied by dB, result is [Currency]

    # Term 1: Geometric Brownian Motion (Vol * Delta * S)
    z_term_1 = sigma_est * S * u_S

    # Term 2: Bachelier / Normal Vol (Vol * S)
    z_term_2 = sigma_est * S

    # Term 3: Value-Proportional Vol (Vol * u)
    z_term_3 = sigma_est * u

    # Term 4: Gamma-Driven Noise (Vol * Gamma * S^2)
    z_term_4 = sigma_est * (S ** 2) * u_SS

    # Term 5: Inverse Price Noise (Vol * u^2 / S)
    z_term_5 = sigma_est * (u ** 2) / (S + 1e-6)

    z_matrix = torch.cat([
        z_term_1, z_term_2, z_term_3,
        z_term_4, z_term_5
    ], dim=1)

    # --- 3. Construct Theta ---
    # Theta = [dt * f, dB * Z]
    # Note: We must broadcast dB against the Z matrix columns

    Theta = torch.cat([
        dt * f_matrix,
        dB * z_matrix
    ], dim=1)

    return Theta

    return Theta