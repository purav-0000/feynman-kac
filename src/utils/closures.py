import numpy as np
import torch

from src.utils.dataclasses import CollocationData, DerivativeData, PathData, SINDyContext
from src.utils.data_processing import build_library
from src.utils.sindy import build_theta_matrix


# Define a custom exception to signal early stopping
class EarlyStoppingException(Exception):
    pass


def make_closure(net_u, optimizer, coll: CollocationData, xi, cfg, pbar=None):
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
        u_pred_data = net_u(coll.X_u)
        loss_data = loss_data_fn(u_pred_data, coll.u)

        # --- Physics loss ---
        u_coll, u_t_coll, u_S_coll, u_SS_coll = net_u.get_derivatives(coll.X_f)
        S_coll = coll.X_f[:, 0]
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
                f"Total: {loss_total.item():.2e} | "
                f"Data: {loss_data.item():.2e} | "
                f"Physics: {loss_physics.item():.2e} | "
                f"L1: {loss_l1.item():.2e}"
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


def make_sindy_loss_closure(net_u, optimizer, paths: PathData, ctx: SINDyContext, cfg):
    """
    Closure for fine-tuning. Accepts 'fixed_indices' to ensure Adam and L-BFGS
    work on the exact same data subset.
    """

    N_samples = len(ctx.recovered_dB)

    # Fixed Subsampling (L-BFGS requires loss landscape to stay constant
    # and we do not have compute to train on the entire dataset)
    # Note: we add 1 to subsample_size because build_theta_matrix trims off 1 array element
    indices = torch.randperm(N_samples, device=paths.s.device)[:cfg.subsample_size + 1]

    # Prepare Fixed Input Tensors
    X_full = torch.cat([paths.s.reshape(-1, 1), paths.t.reshape(-1, 1)], dim=1)

    # Network Inputs
    X_curr = X_full[indices].clone()
    X_curr.requires_grad = True

    # Ground Truth Data
    u_true_curr = paths.u[indices].reshape(-1, 1)
    u_true_next = paths.u[indices + 1].reshape(-1, 1)

    eval_counter = 0
    best_loss = float('inf')
    patience_counter = 0

    def closure():
        nonlocal eval_counter
        nonlocal best_loss
        nonlocal patience_counter

        optimizer.zero_grad()

        # --- Forward Pass ---
        u_pred_curr, u_t_pred, u_x_pred, u_xx_pred = net_u.get_derivatives(X_curr)

        # --- Data Loss ---
        loss_data = torch.nn.MSELoss()(u_pred_curr, u_true_curr)

        # --- SINDy Prep ---
        # We slice off the last one because build_theta_matrix slices 1 element
        dY_target = (u_true_next - u_true_curr)[:-1]
        u_library = u_true_curr
        u_t_sample = u_t_pred.reshape(-1, 1)
        u_x_sample = u_x_pred.reshape(-1, 1)
        u_xx_sample = u_xx_pred.reshape(-1, 1)

        # --- Library & Mask ---
        paths_sample = PathData(
            s=X_curr[:, 0].reshape(-1, 1),
            t=X_curr[:, 1].reshape(-1, 1),
            u=u_library
        )
        derivs_sample = DerivativeData(
            u=None,
            u_t=u_t_sample,
            u_s=u_x_sample,
            u_ss=u_xx_sample
        )

        ctx_sample = SINDyContext(
            assumed_R=ctx.assumed_R,
            recovered_dB=ctx.recovered_dB[indices].reshape(-1, 1),
            sigma_est=ctx.sigma_est[indices].reshape(-1, 1)[:-1],   # Trim off one element for build_theta_matrix
            dt=ctx.dt
        )

        Theta, _, _ = build_theta_matrix(paths_sample, derivs_sample, ctx_sample, return_tensors=True)
        Theta_active = Theta[:, ctx.mask_indices]

        # --- Solve ---
        xi_optimal, _, _, _ = torch.linalg.lstsq(Theta_active, dY_target, driver='gels')

        dY_reconstructed = torch.matmul(Theta_active, xi_optimal)
        loss_sindy = torch.mean((dY_target - dY_reconstructed) ** 2) * cfg.w_sindy

        # --- Total Loss ---
        loss_total = loss_data + loss_sindy

        loss_total.backward()

        # --- Logging ---
        # TODO: Pbar
        if (eval_counter + 1) % 20 == 0:
            xi_vals = xi_optimal.detach().cpu().flatten().numpy()
            xi_str = ", ".join([f"{x:.4f}" for x in xi_vals])
            print(f"Eval {eval_counter + 1:04d} | Total: {loss_total.item():.2e} | "
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