import argparse
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from sklearn.preprocessing import MinMaxScaler
import torch
import yaml

from src.utils.common import apply_overrides
from src.utils.data_processing import prepare_dataset_for_model, build_library
from src.utils.model import prepare_model
from src.utils.sigma_estimation import estimate_constant_sigma, estimate_diffusion_unprocessed
from src.utils.sindy import discover_equation, extract_brownian, prepare_theta_matrix
from src.utils.training import make_closure
from src.utils.true_greeks import black_scholes_partial_t, black_scholes_partial_x, black_scholes_partial_xx


# --- Configuration ---

@dataclass
class Config:
    """Configuration schema for the training script."""
    # Data options
    data_dir: str = "black_scholes_simulated_data"
    uniform_t: bool = True

    # Data sampling hyperparameters
    N_u: int = 125000
    N_f: int = 87500

    # Model hyperparameters
    layers: list = field(default_factory=lambda: [10, 10])

    # Training hyperparameters
    # Loss weights
    w_data: float = 1.0
    w_physics: float = 1.0
    w_l1: float = 0.0

    # Optimizer
    max_eval: int = 3_000
    patience: int = 3

    display_every: int = 100

    # Reproducibility
    seed: int = 42


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Main Training Workflow Class ---

class PINNTrainer:
    """Encapsulates the entire model training workflow."""

    def __init__(self, config: Config):
        """Initializes the trainer with a configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

        # Load and process data
        self.s_train, self.t_train, self.u_train = self._load_data()

        # Use normalized data for training
        self.x_u_train_t, self.x_f_train_t, self.u_train_t = prepare_dataset_for_model(
            self.config.N_u, self.config.N_f, self.s_train, self.t_train, self.u_train, self.device
        )

        # Prepare model and optimizer
        self.net_u, self.xi, self.optimizer = self._prepare_model_and_optimizer()

        # Get config file used for generation from data directory
        # This gives us paramters like strike price, rate, etc. to calculate analytic derivatives
        if self.config.data_dir == "black_scholes_simulated_data":
            with open("data" / Path(self.config.data_dir) / "config_used.yaml", "r") as file:
                self.bs_data_config = yaml.safe_load(file)

        self.output_dir = Path("models") / f"run_{int(time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Main execution method to start the training process."""
        # Train the model
        self._train_model()

        # Check derivatives if using Black-Scholes model
        if self.config.data_dir == "black_scholes_simulated_data":
            self._check_derivatives()
        else:
            self._evaluate_on_test_data()

        # SINDy step
        self._sindy_eq()

        # Save artifacts
        self._save_artifacts()

    def _load_data(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Loads the training data from the .npz file specified in the config."""
        data_path = "data" / Path(self.config.data_dir) / "unhidden_data.npz"
        logging.info(f"Loading training data from: {data_path}")

        if not data_path.exists():
            logging.error(f"Data file not found at {data_path}!")
            raise FileNotFoundError(f"Data file not found at {data_path}!")

        data = np.load(data_path)
        s_train, t_train, u_train = data['S_PATH'], data['T_PATH'], data['U_PATH']
        logging.info(f"Successfully loaded data. S_PATH shape: {s_train.shape}")

        return s_train, t_train, u_train

    def _prepare_model_and_optimizer(self):
        """Initializes the neural network, sparse parameter (xi), and optimizer."""
        logging.info("Preparing model and optimizer...")

        net_u = prepare_model(self.s_train, self.t_train, self.config.layers, self.device)

        # Initialize shape for the sparse parameter xi
        _u, _u_t, _u_s, _u_ss = net_u.get_derivatives(self.x_f_train_t)
        s_f_t = self.x_f_train_t[:, 0]
        phi_dummy, _ = build_library(s_f_t, _u, _u_s, _u_ss)

        # Initialize xi
        xi = torch.nn.Parameter(torch.randn(phi_dummy.shape[1], 1, device=self.device, dtype=torch.double))
        # Initialize its value in-place, outside the computation graph
        with torch.no_grad():
            xi.data.mul_(0.1)

        # Initialize L-BFGS optimizer
        optimizer = torch.optim.LBFGS(
            list(net_u.parameters()) + [xi],
            max_iter=10_000,
            max_eval=self.config.max_eval,
            tolerance_grad=np.finfo(float).eps,
            tolerance_change=np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        return net_u, xi, optimizer

    def _train_model(self):
        """Performs the training loop. This function exists to account for more complex training steps."""
        self._train_lbfgs()

    def _train_lbfgs(self):
        logging.info("Starting L-BFGS training...")

        start_time = time()

        # The L-BFGS optimizer requires a "closure" function
        closure = make_closure(self.net_u, self.optimizer, self.x_u_train_t, self.x_f_train_t, self.u_train_t, self.xi,
                               self.config)

        # Training loop
        try:
            self.optimizer.step(closure)
        except Exception as e:
            logging.warning(e)

        duration = time() - start_time
        logging.info(f"L-BFGS training finished in {duration:.2f} seconds.")

    def _check_derivatives(self):
        """ Compare derivatives of model and analytic derivatives on a plot """
        logging.info("Generating derivative comparison plot...")

        u_pred, u_t_pred, u_S_pred, u_SS_pred = self._get_current_derivatives()

        # Move predicted tensors to CPU for plotting
        u_pred = u_pred.cpu().detach().numpy()
        u_t_pred = u_t_pred.cpu().detach().numpy()
        u_S_pred = u_S_pred.cpu().detach().numpy()
        u_SS_pred = u_SS_pred.cpu().detach().numpy()


        time_to_maturity = self.bs_data_config['T'] - self.t_train
        actual_theta = black_scholes_partial_t(
            self.s_train, self.bs_data_config['K'], time_to_maturity, self.bs_data_config['R'],
            self.bs_data_config['SIGMA_VAL']
        )
        actual_delta = black_scholes_partial_x(
            self.s_train, self.bs_data_config['K'], time_to_maturity, self.bs_data_config['R'],
            self.bs_data_config['SIGMA_VAL']
        )
        actual_gamma = black_scholes_partial_xx(
            self.s_train, self.bs_data_config['K'], time_to_maturity, self.bs_data_config['R'],
            self.bs_data_config['SIGMA_VAL']
        )

        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 18), sharex=True)
        fig.suptitle("Comparison of Model Derivatives and Analytical Greeks", fontsize=16)

        plot_data = [
            ("u (Option Price)", self.u_train, u_pred),
            ("u_t (Theta)", actual_theta, u_t_pred),
            ("u_S (Delta)", actual_delta, u_S_pred),
            ("u_SS (Gamma)", actual_gamma, u_SS_pred)
        ]

        for i, (title, truth, pred) in enumerate(plot_data):
            axs[i].plot(self.t_train, truth, label="Ground Truth (Analytical)", color='black', linestyle='--')
            axs[i].plot(self.t_train, pred, label="Model Prediction", color='red', alpha=0.7)
            axs[i].set_title(title)
            axs[i].grid(True, linestyle=':')
            axs[i].legend()

        plt.xlabel("Time (t)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        save_path = self.output_dir / "derivatives_comparison.png"
        plt.savefig(save_path)
        logging.info(f"Saved derivative plot to {save_path}")

        plt.close(fig)

    def _sindy_eq(self):
        logging.info("Performing SINDy step")

        u_pred, u_t_pred, u_s_pred, u_ss_pred = self._get_current_derivatives()

        # If Black-Scholes simulated data, set simulated R to what was actually used in the simulation
        if self.config.data_dir == "black_scholes_simulated_data":
            assumed_R = self.bs_data_config['R']
        else:
            assumed_R = 0.1

        # 1. With no trimming
        logging.info("Equation with no trimming")
        # Discover equation
        sindy_model = discover_equation(
            s_path=self.s_train,
            u_path=self.u_train,
            t_path=self.t_train,
            derivatives=(u_pred, u_t_pred, u_s_pred, u_ss_pred),
            assumed_R=assumed_R,
            uniform_t=self.config.uniform_t,
            trim_percent=None
        )
        sindy_model.print(lhs=["dY"])

        # 1. With 0.8 trimming
        logging.info("Equation with 0.8 trimming")
        # Discover equation
        sindy_model = discover_equation(
            s_path=self.s_train,
            u_path=self.u_train,
            t_path=self.t_train,
            derivatives=(u_pred, u_t_pred, u_s_pred, u_ss_pred),
            assumed_R=assumed_R,
            uniform_t=self.config.uniform_t,
            trim_percent=0.8
        )

        sindy_model.print(lhs=["dY"])

    def _get_current_derivatives(self):
        x_path = np.hstack((self.s_train.reshape(-1, 1), self.t_train.reshape(-1, 1)))
        u_pred, u_t_pred, u_S_pred, u_SS_pred = self.net_u.get_derivatives(
            x_f_t=torch.from_numpy(x_path).double().to(self.device).requires_grad_(True)
        )

        return u_pred, u_t_pred, u_S_pred, u_SS_pred

    def _save_artifacts(self):

        # Save model state and xi
        torch.save(self.net_u.state_dict(), self.output_dir / "net_u.pth")
        torch.save(self.xi, self.output_dir / "xi.pth")

        # Save the config file used for this run
        with open(self.output_dir / "config_used.yaml", "w") as f:
            yaml.dump(asdict(self.config), f)

    def _evaluate_on_test_data(self):
        """Loads the test data and evaluates the trained model's performance."""
        logging.info("Evaluating model on the test dataset...")
        test_data_path = "data" / Path(self.config.data_dir) / "hidden_data.npz"

        if not test_data_path.exists():
            logging.warning(f"Test data file not found at {test_data_path}. Skipping evaluation.")
            return

        # Load test data
        data = np.load(test_data_path)
        s_test, t_test, u_test = data['S_PATH'], data['T_PATH'], data['U_PATH']

        # Prepare input tensor for the model
        x_test_t = torch.from_numpy(
            np.hstack((s_test.reshape(-1, 1), t_test.reshape(-1, 1)))
        ).double().to(self.device)

        # Set the model to evaluation mode
        self.net_u.eval()

        # Get predictions
        with torch.no_grad():
            u_pred_test = self.net_u(x_test_t)

        # Move predictions to CPU and convert to numpy
        u_pred_test_np = u_pred_test.cpu().numpy()

        # Calculate Mean Squared Error
        mse = np.mean((u_test - u_pred_test_np.squeeze())**2)
        logging.info(f"Test MSE: {mse:.6f}")

        # --- Plotting ---
        plt.figure(figsize=(12, 6))
        plt.plot(t_test, u_test, label='Ground Truth (Test Data)', color='black', linestyle='--')
        plt.plot(t_test, u_pred_test_np, label='Model Prediction', color='red', alpha=0.8)
        plt.title(f"Model Performance on Test Data (MSE: {mse:.6f})", fontsize=16)
        plt.xlabel("Time (t)")
        plt.ylabel("Option Price (u)")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        save_path = self.output_dir / "test_data_evaluation.png"
        plt.savefig(save_path)
        logging.info(f"Saved test evaluation plot to {save_path}")
        plt.close()


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="PINN Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config file name in configs/training/ (without .yaml extension)"
    )
    parser.add_argument("--overrides", nargs='*', help="...")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

    config_path = Path("configs/training") / f"{args.config}.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(str(config_path))

    if args.overrides:
        apply_overrides(config, args.overrides)

    trainer = PINNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()