import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps

from src.utils.sigma_estimation import estimate_constant_sigma, estimate_diffusion_unprocessed


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


def prepare_theta_matrix(S_path, u_path, u_t_pred, u_S_pred, u_SS_pred, recovered_dB, dt, trim_percent=None):

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
        rate_term,
        S_path_sindy ** 2 * u_ss_sindy,
    ]).T
    Z_candidate_terms_matrix = np.vstack([
        S_path_sindy * u_s_sindy
    ]).T
    f_candidate_feature_names = ["u_t", "X * u_x", "X^2 * u_xx"]
    Z_candidate_feature_names = ["X * u_x"]
    """

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
    """


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


    """
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
        s_path, u_path, u_t_pred, u_s_pred, u_ss_pred, recovered_dB, dt, trim_percent=trim_percent
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

    optimizer_lin_reg = ps.STLSQ(threshold=0, alpha=0, normalize_columns=True)
    optimizer_STLSQ = ps.STLSQ(threshold=1e-3, alpha=5e+2, normalize_columns=True)
    optimizer_SR3 = ps.SR3(reg_weight_lam=1e+1, regularizer='L2', max_iter=1000, normalize_columns=True)

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
    print(sindy_model.score(x=theta_matrix, x_dot=dy.reshape(-1, 1), t=t_sindy))

    sindy_model.print(lhs=["dY"])

    # Plot for debugging
    if save_dir is not None:
        plt.figure(figsize=(16, 8))
        plt.plot(t_sindy, dy, label="dY")
        plt.plot(t_sindy, sindy_model.predict(theta_matrix), label="predicted dY")
        plt.grid()
        plt.legend()
        plt.savefig(Path(save_dir) / "dy.png", bbox_inches="tight", dpi=300)
        plt.close()

    return sindy_model