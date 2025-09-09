import numpy as np
import torch

from src.utils.data_processing import build_library


# Define a custom exception to signal early stopping
class EarlyStoppingException(Exception):
    pass


def make_closure(net_u, optimizer, X_u_train_t, X_f_train_t, u_train_t, xi, cfg, pbar=None, plotting_func=None):
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

            if plotting_func is not None:
                plotting_func(eval_counter=eval_counter)

            # Only consider after pretraining
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
