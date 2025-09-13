import numpy as np
import pysindy as ps

from src.utils.sigma_estimation import estimate_constant_sigma, estimate_diffusion_unprocessed

def extract_brownian(assumed_r, S_path, sigma_estimate, dt):
    drift_term = assumed_r * S_path[:-1] * dt

    # If we have sigma values for all time points, then remove the last one
    # Just like for S_path
    if not isinstance(sigma_estimate, float):
        sigma_estimate = sigma_estimate[:-1]

    recovered_dB = (np.diff(S_path) - drift_term) / (sigma_estimate * S_path[:-1])

    return recovered_dB


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

    # --- Commented out the old Black-Scholes specific library ---
    # The original library was hard-coded for a Black-Scholes like structure.
    # It assumed f was a combination of u_t, X*u_x, and X^2*u_xx,
    # and Z was solely X*u_x.
    #
    # rate_term = S_path_sindy * u_s_sindy
    # f_candidate_terms_matrix = np.vstack([
    #     u_t_sindy,
    #     rate_term,
    #     S_path_sindy ** 2 * u_ss_sindy,
    # ]).T
    # Z_candidate_terms_matrix = np.vstack([
    #     S_path_sindy * u_s_sindy
    # ]).T
    # f_candidate_feature_names = ["u_t", "X * u_x", "X^2 * u_xx"]
    # Z_candidate_feature_names = ["X * u_x"]

    # --- New, Flexible Library for General BSDE Discovery ---
    # This library includes a wider range of candidate terms to discover
    # the drift (f) and diffusion (Z) terms for potentially non-BS models.

    # Candidate terms for the drift part, f(t, X, u, u_x, u_xx)
    f_candidate_terms_matrix = np.vstack([
        np.ones_like(S_path_sindy),  # Bias/constant term
        u_path_sindy,  # For interest rate-like terms (e.g., r*u)
        u_s_sindy,  # For costs or drift related to delta
        S_path_sindy * u_s_sindy,  # Classic BS drift/hedging term
        S_path_sindy ** 2 * u_ss_sindy,  # Classic BS convexity/gamma term
    ]).T
    f_candidate_feature_names = [
        "1",
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


def discover_equation(s_path, u_path, t_path, derivatives, assumed_R=0.1, uniform_t=False, trim_percent=None):
    """Discovers the PDE using SINDy on the current data history."""
    u_pred, u_t_pred, u_s_pred, u_ss_pred = derivatives

    # If t is uniform, then make certain assumptions
    # Else, make other assumptions
    if uniform_t:
        dt = t_path[1] - t_path[0]
        sigma_est = estimate_constant_sigma(s_path, dt)
        recovered_dB = extract_brownian(assumed_R, s_path, sigma_est, dt)

        t_sindy = dt
    else:
        dt = min(np.diff(t_path))

        # Time threshold gets rid of points just before a time skip in the dataset
        # Necessary for better estimation of Brownian
        sigma_est = estimate_diffusion_unprocessed(s_path, t_path, time_threshold=dt)
        recovered_dB = extract_brownian(assumed_R, s_path, sigma_est, dt)

        # Apply masks that get rid of big time jumps
        valid_indices = np.where(np.diff(t_path) <= dt)[0]
        s_path = s_path[valid_indices]
        u_path = u_path[valid_indices]
        u_t_pred = u_t_pred[valid_indices]
        u_s_pred = u_s_pred[valid_indices]
        u_ss_pred = u_ss_pred[valid_indices]
        # Ensure recovered_dB is 1 element less, required for prepare_theta_matrix
        recovered_dB = recovered_dB[valid_indices][:-1]
        t_sindy = t_path[valid_indices]

    # Manually trim t_sindy
    # This follows exactly what happens in prepare_theta_matrix
    if trim_percent is not None:
        trim_size = int(len(u_path) * trim_percent)
        t_sindy = t_sindy[:trim_size]
    else:
        t_sindy = t_sindy[:-1]

    theta_matrix, dy, feature_names = prepare_theta_matrix(
        s_path, u_path, u_t_pred, u_s_pred, u_ss_pred, recovered_dB, dt, trim_percent=trim_percent
    )

    sindy_model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=0, alpha=0, normalize_columns=True),
        feature_library=ps.IdentityLibrary(),
        feature_names=feature_names
    )

    sindy_model.fit(theta_matrix, x_dot=dy, t=t_sindy)

    return sindy_model