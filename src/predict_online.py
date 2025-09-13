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
                sigma_est = estimate_diffusion_unprocessed(self.s_history, self.t_history, time_threshold=self.dt)
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
                if self.t_total[current_idx + 1] - self.t_history[-1] <= dt:
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

    def save_results_plot(self, t_path, true_u_path, predicted_u_path, lower_path, upper_path):
        """Saves a plot comparing the ground truth and predicted option price paths."""
        logging.info("Saving final results plot...")

        save_dir = "models" / Path(self.config.model_dir)
        save_dir.mkdir(exist_ok=True) # Ensure the directory exists
        save_path = save_dir / "online_prediction_vs_truth.png"

        plt.figure(figsize=(15, 7))

        # Forcing limits for now
        plt.ylim(59.0, 62.0)

        # Plot the full ground truth path
        plt.plot(t_path[self.num_unhidden - 1:], true_u_path[self.num_unhidden - 1:], label='Ground Truth', color='black', linewidth=2, zorder=2)

        """
        # Plot the initial history that the prediction starts with (solid red)
        plt.plot(t_path[:self.num_unhidden], predicted_u_path[:self.num_unhidden],
                 label='Prediction (Known History)', color='red', linewidth=2, zorder=3)
        """

        # Plot the newly predicted points with a different style (dashed red)
        # We start from num_unhidden-1 to create a continuous line
        plt.plot(t_path[self.num_unhidden - 1:], predicted_u_path[self.num_unhidden - 1:],
                 label='Prediction (Online)', color='red', linestyle='--', linewidth=2.5, zorder=3)

        # Plot the confidence interval as a shaded region
        if self.config.calculate_intervals:
            plt.fill_between(
                t_path[self.num_unhidden - 1:],
                lower_path[self.num_unhidden - 1:],
                upper_path[self.num_unhidden - 1:],
                color='red',
                alpha=0.2,
                label=f'{int(self.config.confidence_level * 100)}% Confidence Interval'
            )

        # Add a vertical line to mark where the prediction begins
        plt.axvline(x=t_path[self.num_unhidden - 1], color='gray', linestyle=':',
                    label='Prediction Start', zorder=1)

        plt.title('Online Prediction of Option Price vs. Ground Truth', fontsize=16)
        plt.xlabel('Time (t)', fontsize=12)
        plt.ylabel('Option Price (u)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':')
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


if __name__ == "__main__":
    main()