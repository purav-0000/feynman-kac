import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import scipy
import torch
import yaml
from tqdm import tqdm

# Assuming your utility functions are structured like this
from src.utils.common import apply_overrides
from src.utils.data_processing import prepare_dataset_for_model
from src.utils.model import load_model_and_xi
from src.utils.sigma_estimation import estimate_constant_sigma, estimate_diffusion_unprocessed
from src.utils.sindy import discover_equation, extract_brownian, prepare_theta_matrix
from src.utils.training import make_closure


# --- Configuration ---

@dataclass
class Config:
    """Configuration for the online prediction script."""

    # Directories
    data_dir: str = "black_scholes_simulated_data"
    model_dir: str = "run_?"  # Path to the trained model directory

    # Data options
    uniform_t: bool = True

    # Prediction & Update Control
    prediction_horizon: int = 1  # How many steps to predict into the future at each iteration.
    sindy_update_every: int = 50  # Number of steps before re-discovering the SINDy model.
    nn_update_every: int = 5000  # Number of steps before retraining the neural network.

    retrain_evals: int = 500  # Number of L-BFGS iterations for online updates

    # Confidence Interval Control
    calculate_intervals: bool = True
    confidence_level: float = 0.95

    # Data sampling hyperparameters
    N_u: int = 125000
    N_f: int = 87500

    # Training hyperparameters
    w_data: float = 1.0
    w_physics: float = 1.0
    w_l1: float = 0.0
    display_every: int = 100

    # Generation
    n_steps: int = 50_000
    n_paths: int = 100
    assumed_R: int = 0.1


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Online Predictor Class ---

class OnlinePredictor:
    """
    Manages the state and logic for online prediction and model updates.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_u, self.xi = None, None

        # Initialize empty state variables
        self.s_total, self.t_total, self.u_total = None, None, None
        self.s_history, self.t_history, self.u_history = None, None, None
        self.dt = None
        self.num_unhidden = 0

        # Get config file used for generation from data directory
        # Used only for setting assumed R to the R used for simulation for now
        with open("data" / Path(self.config.data_dir) / "config_used.yaml", "r") as file:
            self.bs_data_config = yaml.safe_load(file)

        # Pbar for the entire prediction process
        self.pbar = None

    def _load_models(self):
        """Loads the pre-trained neural network and xi parameter."""
        logging.info(f"Loading pre-trained model from {self.config.model_dir}")
        self.net_u, self.xi = load_model_and_xi("models" / Path(self.config.model_dir), self.device)

        # Initialize model bounds
        S_min, S_max = self.s_history.min(), self.s_history.max()
        t_min, t_max = self.t_history.min(), self.t_history.max()

        # For normalization
        lb = torch.tensor([S_min, t_min], device=self.device, dtype=torch.double)
        ub = torch.tensor([S_max, t_max], device=self.device, dtype=torch.double)

        self.net_u.lower_bound = lb
        self.net_u.upper_bound = ub

    def _load_and_prepare_data(self):
        """Loads the full dataset and initializes the history window."""
        data_path = "data" / Path(self.config.data_dir)
        unhidden_data = np.load(data_path / "unhidden_data.npz")
        hidden_data = np.load(data_path / "hidden_data.npz")

        self.s_total = np.concatenate([unhidden_data['S_PATH'], hidden_data['S_PATH']])
        self.t_total = np.concatenate([unhidden_data['T_PATH'], hidden_data['T_PATH']])

        if self.config.uniform_t:
            self.dt = self.t_total[1] - self.t_total[0]
        else:   # Assuming that there is one standard time skip and it is the smallest
            self.dt = min(np.diff(self.t_total))
        self.u_total = np.concatenate([unhidden_data['U_PATH'], hidden_data['U_PATH']])
        self.num_unhidden = len(unhidden_data['S_PATH'])

        # Initialize the sliding window with the visible "unhidden" data
        self.s_history = self.s_total[:self.num_unhidden].copy()
        self.t_history = self.t_total[:self.num_unhidden].copy()
        self.u_history = self.u_total[:self.num_unhidden].copy()
        logging.info(f"Data loaded. Initial history size: {len(self.s_history)}")

    def _discover_equation(self):
        """Discovers the PDE using SINDy on the current data history."""
        inputs = np.hstack((self.s_history.reshape(-1, 1), self.t_history.reshape(-1, 1)))
        inputs_t = torch.from_numpy(inputs).double().to(self.device).requires_grad_(True)

        u_pred, u_t_pred, u_s_pred, u_ss_pred = self.net_u.get_derivatives(inputs_t)

        # If Black-Scholes simulated data, set simulated R to what was actually used in the simulation
        if self.config.data_dir == "black_scholes_simulated_data":
            assumed_R = self.bs_data_config['R']
        else:
            assumed_R = 0.1

        # Discover equation
        sindy_model = discover_equation(
            s_path=self.s_history,
            u_path=self.u_history,
            t_path=self.t_history,
            derivatives=(u_pred, u_t_pred, u_s_pred, u_ss_pred),
            assumed_R=assumed_R,
            uniform_t=self.config.uniform_t,
            trim_percent=None
        )

        return sindy_model

    def _predict_next_steps(self, sindy_model, start_idx, num_steps):
        """Predicts future option prices using the discovered SINDy model."""
        pred_slice = slice(start_idx, start_idx + num_steps)
        s_future = self.s_total[pred_slice]
        t_future = self.t_total[pred_slice]

        # We need one point ahead in the future to extract Brownian
        s_for_db = np.append(s_future, self.s_total[start_idx + num_steps])

        inputs = np.hstack((s_future.reshape(-1, 1), t_future.reshape(-1, 1)))
        inputs_t = torch.from_numpy(inputs).double().to(self.device).requires_grad_(True)

        u_pred, u_t_pred, u_s_pred, u_ss_pred = self.net_u.get_derivatives(inputs_t)

        sigma_est = estimate_constant_sigma(self.s_history, self.dt)  # Use sigma from history
        recovered_dB = extract_brownian(assumed_r=self.bs_data_config['R'], S_path=s_for_db,
                                        sigma_estimate=sigma_est, dt=self.dt)

        # !!! NOTE: prepare_theta_matrix usually discards last value of each array. This is because it assumes
        # that recovered Brownian motion is always one less (which it is if we extract Brownian for the entire
        # array). But, during prediction, the size of recovered_dB is exactly that of the other arguments.
        # So we append all arrays with the value 0.
        s_future = np.append(s_future, 0)
        u_pred = np.append(u_pred.cpu().detach().numpy(), 0)
        zero_array = torch.tensor([[0.0]]).to(self.device)
        u_t_pred = torch.cat([u_t_pred, zero_array])
        u_s_pred = torch.cat([u_s_pred, zero_array])
        u_ss_pred = torch.cat([u_ss_pred, zero_array])

        # u_pred is passed but not utilized
        theta_matrix, _, _ = prepare_theta_matrix(
            s_future, u_pred, u_t_pred, u_s_pred, u_ss_pred, recovered_dB, self.dt, trim_percent=None
        )

        coeffs = sindy_model.coefficients()[0]
        increments = theta_matrix @ coeffs

        last_known_u = self.u_history[-1]
        predictions = last_known_u + np.cumsum(increments)
        return list(predictions)

    def _predict_intervals(self, sindy_model, recovered_dB, bounds):

        inputs = np.hstack((self.s_history[-1].reshape(-1, 1), self.t_history[-1].reshape(-1, 1)))
        inputs_t = torch.from_numpy(inputs).double().to(self.device).requires_grad_(True)

        u_pred, u_t_pred, u_s_pred, u_ss_pred = self.net_u.get_derivatives(inputs_t)

        # !!! NOTE: prepare_theta_matrix usually discards last value of each array. This is because it assumes
        # that recovered Brownian motion is always one less (which it is if we extract Brownian for the entire
        # array). But, during prediction, the size of recovered_dB is exactly that of the other arguments.
        # So we append all arrays with the value 0.
        s_theta = np.append(self.s_history[-1], 0)
        u_theta = np.append(self.u_history[-1], 0)

        zero_array = torch.tensor([[0.0]]).to(self.device)
        u_t_pred = torch.cat([u_t_pred, zero_array])
        u_s_pred = torch.cat([u_s_pred, zero_array])
        u_ss_pred = torch.cat([u_ss_pred, zero_array])

        # u_theta is passed but not utilized
        theta_matrix, _, _ = prepare_theta_matrix(
            s_theta, u_theta, u_t_pred, u_s_pred, u_ss_pred, recovered_dB, self.dt, trim_percent=None
        )

        coeffs = sindy_model.coefficients()[0]
        increments = theta_matrix @ coeffs

        last_known_u = self.u_history[-1]
        predictions = last_known_u + np.cumsum(increments)

        """
        # Now we calculate the bounds
        # Calculate alpha (total tail probability)
        alpha = 1 - self.config.confidence_level

        # Calculate the cumulative probability for the upper z-score
        # This is the area to the left of the upper bound
        cumulative_prob = 1 - (alpha / 2)

        # Use the Percent Point Function (ppf) to find the z-score
        z_score = scipy.stats.norm.ppf(cumulative_prob)

        # Mean of 0, std of np.sqrt(dt)
        bound = 0 + z_score * np.sqrt(self.dt)
        bounds = [-abs(bound), abs(bound)]
        """
        upper_bound_pred, lower_bound_pred = None, None
        for bound in bounds:
            # Set recovered dB to mean prediction
            recovered_dB = bound

            # u_pred is passed but not utilized
            theta_matrix, _, _ = prepare_theta_matrix(
                s_theta, u_theta, u_t_pred, u_s_pred, u_ss_pred, recovered_dB, self.dt, trim_percent=None
            )

            coeffs = sindy_model.coefficients()[0]
            increments = theta_matrix @ coeffs

            last_known_u = self.u_history[-1]

            if bound == min(bounds):
                lower_bound_pred = last_known_u + np.cumsum(increments)
            else:
                upper_bound_pred = last_known_u + np.cumsum(increments)

        return predictions, lower_bound_pred, upper_bound_pred

    def _update_history(self, current_idx, num_steps):
        """Updates the sliding window with new, true data."""
        true_slice = slice(current_idx + 1, current_idx + num_steps + 1)
        self.s_history = np.roll(self.s_history, -num_steps)
        self.s_history[-num_steps:] = self.s_total[true_slice]

        self.t_history = np.roll(self.t_history, -num_steps)
        self.t_history[-num_steps:] = self.t_total[true_slice]

        self.u_history = np.roll(self.u_history, -num_steps)
        self.u_history[-num_steps:] = self.u_total[true_slice]

    def _update_neural_net(self):
        """STEP 3: Retrains the PINN on the updated data history."""

        logging.info("Updating neural network parameters...")

        X_u_train_t, X_f_train_t, u_train_t = prepare_dataset_for_model(
            self.config.N_u, self.config.N_f, self.s_history, self.t_history, self.u_history, self.device
        )

        # Ensure normalization parameters are consistent
        S_min, S_max = self.s_history.min(), self.s_history.max()
        t_min, t_max = self.t_history.min(), self.t_history.max()

        # For normalization
        lb = torch.tensor([S_min, t_min], device=self.device, dtype=torch.double)
        ub = torch.tensor([S_max, t_max], device=self.device, dtype=torch.double)

        self.net_u.lower_bound = lb
        self.net_u.upper_bound = ub

        # Update optimizer with smaller number of iterations
        optimizer_lbfgs = torch.optim.LBFGS(
            list(self.net_u.parameters()) + [self.xi],
            max_iter=500,
            max_eval=500,
            tolerance_grad=np.finfo(float).eps,
            tolerance_change=np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        closure = make_closure(self.net_u, optimizer_lbfgs, X_u_train_t, X_f_train_t, u_train_t, self.xi,
                               self.config, self.pbar)

        optimizer_lbfgs.step(closure)

        pass

    def predict(self):
        """Runs the entire online prediction and retraining loop."""
        self._load_and_prepare_data()
        self._load_models()

        current_idx = self.num_unhidden - 1
        predicted_u_path = list(self.u_history)
        lower_bound_path = list(self.u_history)
        upper_bound_path = list(self.u_history)

        total_hidden = len(self.s_total) - self.num_unhidden
        self.pbar = tqdm(total=total_hidden, desc="Online Prediction")

        # Counters and cached model for fine-grained control
        steps_since_last_sindy_update = 0
        steps_since_last_nn_update = 0
        sindy_model = None  # This will hold the "cached" SINDy model
        recovered_dB = None # This will hold the "cached" Brownian motion
        bounds = None       # This will hold the "cached" bounds for interval prediction

        # Store minimum dt to skip predictions over time skips
        dt = min(np.diff(self.t_total))

        while current_idx < len(self.s_total) - 1:
            # Determine the number of steps for this prediction cycle based on the horizon
            num_steps = min(self.config.prediction_horizon, len(self.s_total) - 1 - current_idx)
            if num_steps <= 0:  # Safety break if we are at the very end
                break

            # Step 1: Discover/update the SINDy model only when scheduled
            if sindy_model is None or steps_since_last_sindy_update >= self.config.sindy_update_every:
                sindy_model = self._discover_equation()

                # Cache Brownian motion
                s_grid, sigma_on_grid = estimate_diffusion_unprocessed(self.s_history, self.t_history,
                                                                       time_threshold=self.dt)
                sigma_est = np.interp(self.s_history, s_grid, sigma_on_grid)
                assumed_R = 0.1
                recovered_dB = extract_brownian(assumed_R, self.s_history, sigma_est, self.dt)

                # Apply mask that get rid of big time jumps
                valid_indices = np.where(np.diff(self.t_history) <= self.dt)[0]
                recovered_dB = recovered_dB[valid_indices]

                # Calculate the lower and upper percentiles required for the confidence interval
                alpha = 1 - self.config.confidence_level  # e.g., 1 - 0.95 = 0.05
                lower_percentile = (alpha / 2) * 100  # e.g., 2.5
                upper_percentile = (1 - alpha / 2) * 100  # e.g., 97.5

                # Use numpy.percentile to find the actual values from your historical data
                bounds = np.percentile(recovered_dB, [lower_percentile, upper_percentile])

                steps_since_last_sindy_update = 0  # Reset counter

            # Step 2: Predict the next `num_steps` using the current SINDy model
            # If there is a time skip after this point, ignore
            if self.config.calculate_intervals:
                if (self.t_total[current_idx + 1] - self.t_total[current_idx]) <= dt + 1e-7:
                    predictions = self._predict_intervals(sindy_model, np.mean(recovered_dB), bounds)

                    # Returns mean, lower bound, upper bound
                    predicted_u_path.extend(predictions[0])
                    lower_bound_path.extend(predictions[1])
                    upper_bound_path.extend(predictions[2])
                else:
                    # This ignores the time skip
                    predicted_u_path.extend([self.u_total[current_idx + 1]])
                    lower_bound_path.extend([self.u_total[current_idx + 1]])
                    upper_bound_path.extend([self.u_total[current_idx + 1]])
            else:   # !!! IGNORING TIME SKIP NOT IMPLEMENTED FOR THE ELSE BLOCK BELOW
                predictions = self._predict_next_steps(sindy_model, current_idx, num_steps)
                predicted_u_path.extend(predictions)

            # Step 3: Observe the truth and update the history window
            self._update_history(current_idx, num_steps)

            # Step 4: Increment counters and check for model updates
            steps_since_last_sindy_update += num_steps

            # TEMP: IGNORE NEURAL NETWORK UPDATES FOR NOW
            """
            steps_since_last_nn_update += num_steps

            if (steps_since_last_nn_update >= self.config.nn_update_every and
                    current_idx + num_steps < len(self.s_total) - 1):  # Check against the end
                self._update_neural_net()
                steps_since_last_nn_update = 0  # Reset counter
            """

            # Advance the main index and progress bar
            current_idx += num_steps
            self.pbar.update(num_steps)

        self.pbar.close()

        if self.config.calculate_intervals:
            return (
                self.t_total,
                self.u_total,
                np.array(predicted_u_path[:len(self.t_total)]),
                np.array(lower_bound_path[:len(self.t_total)]),
                np.array(upper_bound_path[:len(self.t_total)])
            )

        return self.t_total, self.u_total, np.array(predicted_u_path[:len(self.t_total)])

    def generate_paths(self, n_steps=10_000, n_paths=1000, assumed_R=0.1):
        "Generates paths by sampling from the observed Brownian increments"

        # This is necessary if you've commented out predict
        self._load_and_prepare_data()
        self._load_models()

        # Learn the volatility function sigma(S) from historical data
        logging.info("Learning the volatility function sigma(S) from historical data...")
        s_grid, sigma_on_grid = estimate_diffusion_unprocessed(self.s_history, self.t_history)

        # Create a simple, callable interpolation function for the diffusion term
        get_diffusion_term = lambda s: np.interp(s, s_grid, sigma_on_grid)

        # Recover historical Brownian increments for sampling
        # (This part would be the same as in your previous 'predict' function)
        sigma_est = get_diffusion_term(self.s_history)
        recovered_dB = extract_brownian(assumed_R, self.s_history, sigma_est, self.dt)

        # Apply mask that get rid of big time jumps
        valid_indices = np.where(np.diff(self.t_history) <= self.dt)[0]
        recovered_dB = recovered_dB[valid_indices]

        # Discover equation
        sindy_model = self._discover_equation()

        def forward_SDE(S):
            "Take the stock one step forward in time"
            # Get the state-dependent diffusion term for the *current* stock price
            diffusion_val = get_diffusion_term(S)

            # Sample an increment from the historical distribution
            db_sample = np.random.choice(recovered_dB, len(S))

            # Apply the SDE formula: dS = r*S*dt + (sigma_func)*dB
            dS = assumed_R * S * self.dt + diffusion_val * db_sample

            return S + dS

        all_stocks = np.zeros((n_paths, n_steps))
        all_stocks[:, 0] = self.s_history[-1]

        all_options = np.zeros((n_paths, n_steps))
        all_options[:, 0] = self.u_history[-1]

        # Time array could be 1D but 2D allows for easy NN batching
        all_time = np.zeros((n_paths, n_steps))
        all_time[:, 0] = self.t_history[-1]

        current_idx = self.num_unhidden - 1

        # Now we generate
        i = 1
        self.pbar = tqdm(total=n_steps, desc="Generation")
        while current_idx < self.num_unhidden - 1 + (n_steps - 1):
            all_stocks[:, i] = forward_SDE(all_stocks[:, i - 1])
            all_time[:, i] = all_time[:, i - 1] + self.dt

            # Generate theta matrix for options prediction
            inputs = np.hstack((all_stocks[:, i - 1].reshape(-1, 1), all_time[:, i - 1].reshape(-1, 1)))
            inputs_t = torch.from_numpy(inputs).double().to(self.device).requires_grad_(True)

            u_pred, u_t_pred, u_s_pred, u_ss_pred = self.net_u.get_derivatives(inputs_t)

            # Explained below why this is necessary
            zero_array = torch.zeros_like(u_t_pred).to(self.device)
            u_t_pred = torch.cat([u_t_pred, zero_array], dim=1)
            u_s_pred = torch.cat([u_s_pred, zero_array], dim=1)
            u_ss_pred = torch.cat([u_ss_pred, zero_array], dim=1)

            # Unfortunate for loop because prepare_theta_matrix does not handle 2D arrays
            for j in range(n_paths):
                # !!! NOTE: prepare_theta_matrix usually discards last value of each array. This is because it assumes
                # that recovered Brownian motion is always one less (which it is if we extract Brownian for the entire
                # array). But, during prediction, the size of recovered_dB is exactly that of the other arguments.
                # So we append all arrays with the value 0.
                s_theta = np.append(all_stocks[j, i - 1], 0)
                u_theta = np.append(all_options[j, i - 1], 0)

                theta_matrix, _, _ = prepare_theta_matrix(
                    s_theta, u_theta, u_t_pred[j], u_s_pred[j], u_ss_pred[j], np.random.choice(recovered_dB), self.dt,
                    trim_percent=None
                )

                coeffs = sindy_model.coefficients()[0]
                increments = theta_matrix @ coeffs

                all_options[j, i] = all_options[j, i - 1] + increments.item()

            # Update counter
            i += 1
            current_idx += 1
            self.pbar.update()

        self.pbar.close()

        # --- 3. Plotting Phase ---
        # Create the time axis for the generated paths
        t_generated = all_time[0]

        self._plot_generated_paths(
            time_generated=t_generated,
            options_generated=all_options,
            stocks_generated=all_stocks
        )

        # Return the generated data
        return t_generated, all_stocks, all_options

    def save_results_plot(self, t_path, true_u_path, predicted_u_path, lower_path, upper_path):
        """Saves a publication-ready plot comparing ground truth and prediction, with a zoom-in inset."""
        logging.info("Saving final results plot with inset zoom...")

        save_dir = Path("models") / Path(self.config.model_dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "online_prediction_vs_truth.png"

        # --- Main Plot Setup ---
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot the full ground truth path
        ax.plot(t_path[self.num_unhidden - 1:], true_u_path[self.num_unhidden - 1:], label='Ground Truth',
                color='black', linewidth=2, zorder=2)

        # Plot the newly predicted points with a thinner line
        ax.plot(t_path[self.num_unhidden - 1:], predicted_u_path[self.num_unhidden - 1:],
                label='Prediction (Online)', color='red', linestyle='--', linewidth=2.0, zorder=3)

        # Plot the confidence interval as a shaded region
        if self.config.calculate_intervals:
            ax.fill_between(
                t_path[self.num_unhidden - 1:],
                lower_path[self.num_unhidden - 1:],
                upper_path[self.num_unhidden - 1:],
                color='red',
                alpha=0.2,
                label=f'{int(self.config.confidence_level * 100)}% Confidence Interval'
            )

        # Add a vertical line to mark where the prediction begins
        ax.axvline(x=t_path[self.num_unhidden - 1], color='gray', linestyle=':',
                   label='Prediction Start', zorder=1)

        # --- Inset Plot (Zoom-in) ---
        # Position: [left, bottom, width, height] -> moved to bottom right
        axins = ax.inset_axes([0.65, 0.15, 0.3, 0.28])

        # Define a smaller zoom region for a more detailed view
        zoom_start_t, zoom_end_t = 0.95595, 0.95600

        # Create masks to select data for the zoom
        valid_range_mask = t_path >= t_path[self.num_unhidden - 1]
        zoom_mask = (t_path >= zoom_start_t) & (t_path <= zoom_end_t) & valid_range_mask

        # If the default zoom region has no data, create a new one from the last 100 points
        if not np.any(zoom_mask):
            logging.warning(f"Default zoom range [{zoom_start_t}, {zoom_end_t}] is empty. Selecting last 100 points.")
            end_index = len(t_path)
            start_index = max(self.num_unhidden - 1, end_index - 100)

            zoom_mask = np.zeros_like(t_path, dtype=bool)
            zoom_mask[start_index:end_index] = True

            zoom_start_t = t_path[start_index]
            zoom_end_t = t_path[end_index - 1]

        # Plot the same data but for the zoomed region on the inset axes
        axins.plot(t_path[zoom_mask], true_u_path[zoom_mask], color='black', linewidth=1.5)
        axins.plot(t_path[zoom_mask], predicted_u_path[zoom_mask], color='red', linestyle='--', linewidth=2.0)
        if self.config.calculate_intervals:
            axins.fill_between(t_path[zoom_mask], lower_path[zoom_mask], upper_path[zoom_mask],
                               color='red', alpha=0.2)

        # Set the limits and labels for the inset plot
        axins.set_xlim(zoom_start_t, zoom_end_t)
        y_min_inset = min(np.min(lower_path[zoom_mask]), np.min(true_u_path[zoom_mask]))
        y_max_inset = max(np.max(upper_path[zoom_mask]), np.max(true_u_path[zoom_mask]))
        y_padding = (y_max_inset - y_min_inset) * 0.1
        axins.set_ylim(y_min_inset - y_padding, y_max_inset + y_padding)
        axins.set_title(f"Zoom in (t={zoom_start_t:.2f}-{zoom_end_t:.2f})")
        axins.grid(True, linestyle=':')

        # Draw a box indicating the zoom area on the main plot
        ax.indicate_inset_zoom(axins, edgecolor="black")

        # --- Final Touches for Main Plot ---
        ax.set_title('Online Prediction of Option Price vs. Ground Truth', fontsize=16)
        ax.set_xlabel('Time (t)', fontsize=12)
        ax.set_ylabel('Option Price (u)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle=':')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Plot saved successfully to {save_path}")

    def _plot_generated_paths(self, time_generated, options_generated, stocks_generated):
        """Saves a plot showing historical data and generated future paths for both stock and options."""
        logging.info("Saving generated paths plot...")

        save_dir = Path("models") / Path(self.config.model_dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / "monte_carlo_simulation.png"

        # --- 1. Set up the figure and subplots ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        fig.suptitle('Monte Carlo Simulation of Stock and Option Paths', fontsize=18)

        # --- 2. Limit the amount of history shown ---
        num_generated_steps = options_generated.shape[1]
        history_to_show = min(len(self.t_history), num_generated_steps)
        hist_start_idx = len(self.t_history) - history_to_show

        time_hist_sliced = self.t_history[hist_start_idx:]
        stock_hist_sliced = self.s_history[hist_start_idx:]
        option_hist_sliced = self.u_history[hist_start_idx:]

        # --- 3. Plot the known historical paths ---
        ax1.plot(time_hist_sliced, stock_hist_sliced, label='Historical Path', color='black', linewidth=3, zorder=5)
        ax2.plot(time_hist_sliced, option_hist_sliced, label='Historical Path', color='black', linewidth=3, zorder=5)

        # --- 4. Plot all the generated future paths with paired coloring ---
        num_to_plot = min(500, options_generated.shape[0])  # Limit plotted paths for clarity

        # Use a colormap to generate distinct colors for each path
        colors = plt.cm.viridis(np.linspace(0, 1, num_to_plot))

        for i in range(num_to_plot):
            label = 'Generated Paths' if i == 0 else None  # Add a single legend entry

            # Use the same color for the corresponding stock and option path
            path_color = colors[i]

            ax1.plot(time_generated, stocks_generated[i, :], color=path_color, alpha=0.6, linewidth=1.5, label=label)
            ax2.plot(time_generated, options_generated[i, :], color=path_color, alpha=0.6, linewidth=1.5)

        # --- 5. Add vertical lines to mark where the simulation begins ---
        simulation_start_time = self.t_history[-1]
        ax1.axvline(x=simulation_start_time, color='gray', linestyle=':', linewidth=2, label='Simulation Start',
                    zorder=4)
        ax2.axvline(x=simulation_start_time, color='gray', linestyle=':', linewidth=2)

        # --- 6. Finalize plot aesthetics ---
        ax1.set_title('Stock Price (S) Paths', fontsize=14)
        ax1.set_ylabel('Stock Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle=':')

        ax2.set_title('Option Price (u) Paths', fontsize=14)
        ax2.set_xlabel('Time (t)', fontsize=12)
        ax2.set_ylabel('Option Price', fontsize=12)
        # The legend from the first plot is sufficient

        ax2.grid(True, linestyle=':')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        logging.info(f"Plot saved successfully to {save_path}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="PINN Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="black_scholes_online",
        help="Config file name in configs/prediction/ (without .yaml extension)"
    )
    parser.add_argument("--overrides", nargs='*', help="...")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

    config_path = Path("configs/prediction") / f"{args.config}.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(str(config_path))

    if args.overrides:
        apply_overrides(config, args.overrides)

    predictor = OnlinePredictor(config)


    if config.calculate_intervals:
        t, u_true, u_pred, lb, ub = predictor.predict()
        predictor.save_results_plot(t, u_true, u_pred, lb, ub)
    else:
        t, u_true, u_pred = predictor.predict()
        predictor.save_results_plot(t, u_true, u_pred, None, None)


    predictor.generate_paths(n_steps=config.n_steps, n_paths=config.n_paths, assumed_R=config.assumed_R)


if __name__ == "__main__":
    main()