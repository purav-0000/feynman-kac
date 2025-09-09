import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import norm

from src.utils.common import apply_overrides

# --- Configuration ---

@dataclass
class Config:
    """Configuration schema for the data generation script."""
    # Simulation hyperparameters
    T: float = 1.0
    N_SIM: int = 250_000
    R: float = 0.3
    K: float = 102.0
    S0: float = 100.0
    SIGMA_VAL: float = 0.2

    # Data split point
    NUM_UNHIDDEN_POINTS: int = 150_000

    # Reproducibility
    seed: int = 42

    # Output configuration
    output_dir_name: str = "black_scholes_simulated_data"


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic ---

class DataGenerator:
    """Encapsulates the data generation and saving workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.output_path = Path("data") / self.config.output_dir_name
        np.random.seed(self.config.seed)

    def run(self):
        """Main execution method to generate and save all artifacts."""
        logging.info(f"Starting data generation with seed {self.config.seed}.")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate paths
        t_path, s_path, db_path, u_path = self._generate_paths()

        # 2. Split data into visible and hidden parts
        visible_data, hidden_data = self._split_data(t_path, s_path, db_path, u_path)

        # 3. Save data files
        self._save_data_files(visible_data, hidden_data)

        # 4. Generate and save plots
        self._generate_and_save_plots(t_path, s_path, u_path, visible_data)

        # 5. Save the configuration used
        self._save_config()

        logging.info(f"✅ Data generation complete. Artifacts saved to: {self.output_path}")

    def _generate_paths(self):
        """Generates the time, stock, Brownian motion, and option price paths."""
        cfg = self.config
        dt = cfg.T / cfg.N_SIM
        t_path = np.linspace(0, cfg.T, cfg.N_SIM, dtype=np.float64)

        # Initialize stock price path
        s_path = np.zeros(cfg.N_SIM, dtype=np.float64)
        s_path[0] = cfg.S0

        # Generate path using Euler-Maruyama for Geometric Brownian Motion
        db_path = np.random.normal(0, np.sqrt(dt), size=(cfg.N_SIM - 1))
        for i in range(cfg.N_SIM - 1):
            s_path[i + 1] = s_path[i] + cfg.R * s_path[i] * dt + cfg.SIGMA_VAL * s_path[i] * db_path[i]

        # Calculate option price path using the analytical Black-Scholes solution
        u_path = self._black_scholes_solution_vectorized(s_path, cfg.K, cfg.T - t_path, cfg.R, cfg.SIGMA_VAL)

        return t_path, s_path, db_path, u_path

    @staticmethod
    def _black_scholes_solution_vectorized(s_array, K, T_to_maturity, r, sigma_val):
        """Analytical solution for European call option price."""
        tau = np.maximum(T_to_maturity, 1e-9)
        d1 = (np.log(s_array / K) + (r + 0.5 * sigma_val ** 2) * tau) / (sigma_val * np.sqrt(tau))
        d2 = d1 - sigma_val * np.sqrt(tau)
        call_price = s_array * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        # At maturity (T_to_maturity <= 0), the price is the payoff
        return np.where(T_to_maturity <= 0, np.maximum(s_array - K, 0), call_price).astype(np.float64)

    def _split_data(self, t_path, s_path, db_path, u_path):
        """Splits the generated paths into unhidden (training) and hidden (prediction) sets."""
        n_unhidden = self.config.NUM_UNHIDDEN_POINTS

        visible_data = {
            "T_PATH": t_path[:n_unhidden],
            "S_PATH": s_path[:n_unhidden],
            "U_PATH": u_path[:n_unhidden],
            "DB": db_path[:n_unhidden - 1]  # Last dB point is not utilized in any calculation
        }
        hidden_data = {
            "T_PATH": t_path[n_unhidden:],
            "S_PATH": s_path[n_unhidden:],
            "U_PATH": u_path[n_unhidden:],
            "DB": db_path[n_unhidden - 1:]  # Same logic
        }
        return visible_data, hidden_data

    def _save_data_files(self, visible_data, hidden_data):
        """Saves the datasets to .npz files."""
        unhidden_file = self.output_path / "unhidden_data.npz"
        hidden_file = self.output_path / "hidden_data.npz"

        np.savez(unhidden_file, **visible_data)
        np.savez(hidden_file, **hidden_data)
        logging.info(f"Saved unhidden data to {unhidden_file}")
        logging.info(f"Saved hidden data to {hidden_file}")

    def _generate_and_save_plots(self, t_path, s_path, u_path, visible_data):
        """Generates and saves visualizations of the data."""
        # Plot 1: Full simulated path
        plt.figure(figsize=(12, 5))
        plt.plot(t_path, s_path, label='Simulated $X_t$', color="blue", linewidth=1.5)
        plt.plot(t_path, u_path, label='Call Option Price $U_t$', color='purple', linewidth=1.5)
        plt.xlabel("Time $t$")
        plt.ylabel("Value")
        plt.title("Full Simulated Stock and Call Option Price", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / "full_simulation.png")
        plt.close()

        # Plot 2: Visible vs. Hidden Data
        hidden_t = t_path[self.config.NUM_UNHIDDEN_POINTS:]
        hidden_s = s_path[self.config.NUM_UNHIDDEN_POINTS:]
        hidden_u = u_path[self.config.NUM_UNHIDDEN_POINTS:]

        plt.figure(figsize=(12, 5))
        # Plot visible parts
        plt.plot(visible_data["T_PATH"], visible_data["S_PATH"], label='Visible $X_t$', color="blue", linewidth=2)
        plt.plot(visible_data["T_PATH"], visible_data["U_PATH"], label='Visible Call Option Price', color='purple',
                 linewidth=2)
        # Plot hidden parts with transparency
        plt.plot(hidden_t, hidden_s, label='Hidden $X_t$', color="blue", linewidth=2, alpha=0.5, linestyle='--')
        plt.plot(hidden_t, hidden_u, label='Hidden Call Option Price', color='purple', linewidth=2, alpha=0.5,
                 linestyle='--')
        plt.xlabel("Time $t$")
        plt.ylabel("Value")
        plt.title("Visible (Training) and Hidden (Prediction) Data", fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path / "visible_vs_hidden.png")
        plt.close()

        logging.info("Saved plots to output directory.")

    def _save_config(self):
        """Saves the final configuration used for the run to a YAML file."""
        config_path = self.output_path / "config_used.yaml"
        with open(config_path, "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False, sort_keys=False)
        logging.info(f"Saved final configuration to {config_path}")


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Black-Scholes Data Generation Script")
    parser.add_argument(
        "--config",
        type=str,
        default="default_black_scholes",
        help="Config file name in configs/generators/ (without .yaml extension)"
    )
    parser.add_argument(
        "--overrides",
        nargs='*',
        help="Overrides in key=value format (e.g., N_SIM=500000 K=105.0)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config_path = Path("configs/generators") / f"{args.config}.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(str(config_path))

    if args.overrides:
        apply_overrides(config, args.overrides)

    generator = DataGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()