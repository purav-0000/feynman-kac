import numpy as np
import torch

from src.utils.data_processing import build_library
from src.utils.sindy import build_sindy_library_torch


# Define a custom exception to signal early stopping
class EarlyStoppingException(Exception):
    pass


def make_closure(net_u, optimizer, X_u_train_t, X_f_train_t, u_train_t, xi, cfg, pbar=None):
    loss_data_fn = torch.nn.MSELoss()
    eval_counter = 0
    best_loss = float('inf')
    patience_counter = 0

    def closure():
        nonlocal eval_counter
        nonlocal best_loss
        nonlocal patience_counter

        optimizer.zero_grad()

        # --- Data loss ---
        u_pred_data = net_u(X_u_train_t)
        loss_data = loss_data_fn(u_pred_data, u_train_t)

        # --- Physics loss ---
        u_coll, u_t_coll, u_S_coll, u_SS_coll = net_u.get_derivatives(X_f_train_t)
        S_coll = X_f_train_t[:, 0]
        Phi, _ = build_library(S_coll, u_coll, u_S_coll, u_SS_coll)

        u_t_pred_physics = torch.matmul(Phi, xi)
        loss_physics = torch.mean((u_t_coll - u_t_pred_physics) ** 2)

        # --- L1 regularization ---
        loss_l1 = torch.norm(xi, p=1)

        # --- Total loss ---
        loss_total = cfg.w_data * loss_data + cfg.w_physics * loss_physics + cfg.w_l1 * loss_l1
        loss_total.backward()

        if (eval_counter + 1) % cfg.display_every == 0:
            message = (
                f"Eval: {eval_counter + 1} | "
                f"Total loss: {loss_total.item():.6f}, "
                f"Data: {loss_data.item():.6f}, "
                f"Physics: {loss_physics.item():.6f}, "
                f"L1: {loss_l1.item():.6f}"
            )
            if pbar is not None:
                pbar.write(message)
            else:
                print(message)

            # Patience implementation
            if loss_total.item() < best_loss - 1e-8:
                best_loss = loss_total.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    raise EarlyStoppingException(
                        f"Stopping early. No improvement in loss for {patience_counter} checks.")

        eval_counter += 1
        return loss_total

    return closure


def make_sindy_loss_closure(net_u, optimizer, s_train, t_train, u_train, recovered_dB, dt, assumed_R, sigma_est,
                            mask_indices, cfg, fixed_indices=None):
    """
    Closure for fine-tuning. Accepts 'fixed_indices' to ensure Adam and L-BFGS
    work on the exact same data subset.
    """
    # 1. Configuration
    BATCH_SIZE = 20000
    N_samples = len(recovered_dB)

    # 2. Fixed Subsampling
    # If indices are provided (shared between optimizers), use them.
    # Otherwise, generate new ones.
    if fixed_indices is None:
        indices = torch.randperm(N_samples, device=s_train.device)[:BATCH_SIZE]
    else:
        indices = fixed_indices

    # Prepare Fixed Input Tensors
    X_full = torch.cat([s_train.reshape(-1, 1), t_train.reshape(-1, 1)], dim=1)

    # Network Inputs
    X_curr = X_full[indices].clone()
    X_curr.requires_grad = True

    # Ground Truth Data
    u_true_curr = u_train[indices].reshape(-1, 1)
    u_true_next = u_train[indices + 1].reshape(-1, 1)

    dB_batch = recovered_dB[indices].reshape(-1, 1)

    eval_counter = 0
    best_loss = float('inf')
    patience_counter = 0

    def closure():
        nonlocal eval_counter
        nonlocal best_loss
        nonlocal patience_counter

        optimizer.zero_grad()

        # --- 1. Forward Pass ---
        u_pred_curr, u_t_pred, u_x_pred, u_xx_pred = net_u.get_derivatives(X_curr)

        # --- 2. Data Loss ---
        loss_data = torch.nn.MSELoss()(u_pred_curr, u_true_curr)

        # --- 3. SINDy Prep ---
        dY_target = u_true_next - u_true_curr

        S_batch = X_curr[:, 0].reshape(-1, 1)
        u_library = u_true_curr
        u_t_batch = u_t_pred.reshape(-1, 1)
        u_x_batch = u_x_pred.reshape(-1, 1)
        u_xx_batch = u_xx_pred.reshape(-1, 1)

        # --- 4. Library & Mask ---
        Theta = build_sindy_library_torch(
            S_batch, u_library, u_t_batch, u_x_batch, u_xx_batch,
            dB_batch, dt, assumed_R, sigma_est
        )
        Theta_active = Theta[:, mask_indices]

        # --- 5. Solve ---
        xi_optimal, _, _, _ = torch.linalg.lstsq(Theta_active, dY_target, driver='gels')

        dY_reconstructed = torch.matmul(Theta_active, xi_optimal)
        loss_sindy = torch.mean((dY_target - dY_reconstructed) ** 2) * cfg.w_sindy

        # --- 6. Total Loss ---
        loss_total = loss_data + loss_sindy

        loss_total.backward()

        # --- Logging ---
        # Log frequently for Adam (every 20 steps)
        if (eval_counter + 1) % 20 == 0:
            xi_vals = xi_optimal.detach().cpu().flatten().numpy()
            xi_str = ", ".join([f"{x:.4f}" for x in xi_vals])
            print(f"Fine-Tune Step {eval_counter + 1:04d} | Total: {loss_total.item():.2e} | "
                  f"SINDy: {loss_sindy.item():.2e} | Data: {loss_data.item():.2e} | Coeffs: [{xi_str}]")

            # Patience implementation
            if loss_total.item() < best_loss - 1e-8:
                best_loss = loss_total.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    raise EarlyStoppingException(
                        f"Stopping early. No improvement in loss for {patience_counter} checks.")

        eval_counter += 1
        return loss_total

    return closure