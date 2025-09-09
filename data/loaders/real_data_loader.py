import argparse
import logging
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.utils.common import apply_overrides

# --- Configuration ---

@dataclass
class Config:
    """Configuration schema for the AAPL data loading script."""
    # File paths
    input_stock_path: str
    input_options_path: str
    output_dir_name: str

    # Data processing hyperparameters
    num_points_to_use: int
    num_unhidden_points: int

    # Reproducibility
    seed: int


def load_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


# --- Core Logic ---

class DataLoader:
    """Encapsulates the data loading, processing, and saving workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.output_path = Path("data") / self.config.output_dir_name
        np.random.seed(self.config.seed)

    def run(self):
        """Main execution method to generate and save all artifacts."""
        logging.info(f"Starting data loading with seed {self.config.seed}.")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 1. Load and preprocess data
        stock_time, stock_mid, option_time, option_mid = self._load_and_preprocess()

        # 2. Slice data to the desired number of points from the end
        if self.config.num_points_to_use > 0:
            num_points = self.config.num_points_to_use
            logging.info(f"Using the last {num_points} data points.")
            stock_time = stock_time[-num_points:]
            stock_mid = stock_mid[-num_points:]
            option_time = option_time[-num_points:]
            option_mid = option_mid[-num_points:]
        else:
            logging.info("Using all available data points.")

        # 3. Split data into visible and hidden parts
        visible_data, hidden_data = self._split_data(stock_time, stock_mid, option_time, option_mid)

        # 4. Save data files
        self._save_data_files(visible_data, hidden_data)

        # 5. Generate and save plots
        self._generate_and_save_plots(visible_data, hidden_data)

        # 6. Save the configuration used
        self._save_config()

        logging.info(f"✅ Data processing complete. Artifacts saved to: {self.output_path}")

    def _load_and_preprocess(self):
        """Loads, preprocesses, and converts time data to seconds from start."""
        logging.info("Loading and preprocessing data using pandas...")

        def _load_and_clean_file(path):
            """Helper function to load a single CSV, clean it, and return arrays."""
            # Use low_memory=False to suppress the DtypeWarning
            df = pd.read_csv(path, usecols=['ms_of_day', 'date', 'bid', 'ask'], low_memory=False)

            # Force bid and ask to numeric, coercing errors to NaN
            df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
            df['ask'] = pd.to_numeric(df['ask'], errors='coerce')

            # Drop any rows with NaN values that resulted from coercion or were already present
            df.dropna(subset=['bid', 'ask', 'date', 'ms_of_day'], inplace=True)

            # Calculate mid-price
            df['mid_price'] = (df['bid'] + df['ask']) / 2

            # --- Handle zero-price artifacts ---
            # Replace values at or below a small threshold with NaN
            zero_price_mask = df['mid_price'] <= 1e-6
            num_zero_prices = zero_price_mask.sum()
            if num_zero_prices > 0:
                logging.info(f"Found and cleaning {num_zero_prices} zero-price data points in {Path(path).name}.")
                df.loc[zero_price_mask, 'mid_price'] = np.nan
                # Use backfill to replace NaNs with the next valid observation
                df['mid_price'] = df['mid_price'].bfill()
                # Drop any remaining NaNs if zeros occurred at the very end of the file
                df.dropna(subset=['mid_price'], inplace=True)

            # Extract numpy arrays after cleaning
            ms = df['ms_of_day'].values
            dates = df['date'].values
            mid_price = df['mid_price'].values

            dt_objects = np.array([
                datetime.datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M:%S.%f")
                for d, t in zip(dates, ms)
            ])

            return dt_objects, mid_price

        # --- Load and Clean Stock & Options Data ---
        stock_dt, stock_mid = _load_and_clean_file(self.config.input_stock_path)
        option_dt, option_mid = _load_and_clean_file(self.config.input_options_path)

        # --- Convert time to seconds from the start of the series ---
        if len(stock_dt) > 0:
            t0 = stock_dt[0]
            stock_time_seconds = np.array([(t - t0).total_seconds() for t in stock_dt])
            option_time_seconds = np.array([(t - t0).total_seconds() for t in option_dt])
        else:
            stock_time_seconds = np.array([])
            option_time_seconds = np.array([])

        # Basic alignment check
        if len(stock_time_seconds) != len(option_time_seconds):
            logging.warning("Stock and Option data have different lengths. This might cause issues.")

        return stock_time_seconds, stock_mid, option_time_seconds, option_mid

    def _split_data(self, stock_time, stock_mid, option_time, option_mid):
        """Splits the data into unhidden (visible) and hidden sets."""
        n_unhidden = self.config.num_unhidden_points
        if n_unhidden >= len(stock_time):
            raise ValueError(
                f"num_unhidden_points ({n_unhidden}) must be smaller than the "
                f"number of points being used ({len(stock_time)})."
            )

        visible_data = {
            # ASSUMING STOCK AND OPTIONS HAVE SAME TIME PATH IN THE DATASET
            "T_PATH": stock_time[:n_unhidden],
            "S_PATH": stock_mid[:n_unhidden],
            "U_PATH": option_mid[:n_unhidden],
            # For verification purposes
            "T_PATH_U": option_time[:n_unhidden],
        }
        hidden_data = {
            "T_PATH": stock_time[n_unhidden:],
            "S_PATH": stock_mid[n_unhidden:],
            "U_PATH": option_mid[n_unhidden:],
            # For verification purposes
            "T_PATH_U": option_time[n_unhidden:],
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

    def _generate_and_save_plots(self, visible_data, hidden_data):
        """Generates and saves visualizations of the visible vs. hidden data."""
        plt.figure(figsize=(15, 7))

        # Plot visible parts
        plt.plot(visible_data["T_PATH"], visible_data["S_PATH"], label='Visible Stock Price',
                 color="blue", linewidth=1.5)
        plt.plot(visible_data["T_PATH_U"], visible_data["U_PATH"], label='Visible Option Price',
                 color='purple', linewidth=1.5)

        # Plot hidden parts with transparency
        plt.plot(hidden_data["T_PATH"], hidden_data["S_PATH"], label='Hidden Stock Price',
                 color="blue", alpha=0.5, linestyle='--')
        plt.plot(hidden_data["T_PATH_U"], hidden_data["U_PATH"], label='Hidden Option Price',
                 color='purple', alpha=0.5, linestyle='--')

        plt.xlabel("Time (seconds from start)")
        plt.ylabel("Mid Price")
        plt.title("Visible and Hidden Data for AAPL Stock and Options", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plot_path = self.output_path / "visible_vs_hidden.png"
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"Saved plot to {plot_path}")

    def _save_config(self):
        """Saves the final configuration used for the run to a YAML file."""
        config_path = self.output_path / "config_used.yaml"
        # Convert Path objects to strings for clean YAML output
        config_dict = asdict(self.config)
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logging.info(f"Saved final configuration to {config_path}")


# --- Main Entry Point ---

def main():
    """Main function to run the data loading and processing."""
    parser = argparse.ArgumentParser(description="AAPL Data Loading and Processing Script")
    parser.add_argument(
        "--config",
        type=str,
        default="APPL",
        help="Config file name in config/loaders/ (without .yaml extension)"
    )
    parser.add_argument(
        "--overrides",
        nargs='*',
        help="Overrides in key=value format (e.g., N_SIM=500000 K=105.0)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config_path = Path("configs/loaders") / f"{args.config}.yaml"
    if not config_path.exists():
        logging.error(f"Configuration file not found at {config_path}")
        return

    config = load_config(str(config_path))

    if args.overrides:
        apply_overrides(config, args.overrides)

    loader = DataLoader(config)
    loader.run()


if __name__ == "__main__":
    main()
