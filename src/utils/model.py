from pathlib import Path
import torch
from torch import nn
import yaml

class NetU(nn.Module):

    def __init__(self, layers, lb, ub):
        super(NetU, self).__init__()
        self.lower_bound = lb
        self.upper_bound = ub
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.network = nn.Sequential(*modules)

    def forward(self, x_in):
        x_normalized = 2.0 * (x_in - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0
        return self.network(x_normalized)

    def get_derivatives(self, x_f_t):
        u = self(x_f_t)

        grad_u = torch.autograd.grad(u, x_f_t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grad_u[:, 1].unsqueeze(-1)
        u_S = grad_u[:, 0].unsqueeze(-1)

        grad_u_S = torch.autograd.grad(u_S, x_f_t, grad_outputs=torch.ones_like(u_S), create_graph=True)[0]
        u_SS = grad_u_S[:, 0].unsqueeze(-1)

        return u, u_t, u_S, u_SS


def prepare_model(S_path, t_path, layers, device):
    S_min, S_max = S_path.min(), S_path.max()
    t_min, t_max = t_path.min(), t_path.max()

    # For normalization
    lb = torch.tensor([S_min, t_min], device=device, dtype=torch.double)
    ub = torch.tensor([S_max, t_max], device=device, dtype=torch.double)

    # Model
    net_u = NetU(layers, lb, ub).to(device)

    return net_u.to(torch.double)


def load_model_and_xi(model_dir, device):
    model_dir = Path(model_dir)

    # Load the config used for training
    # The config is needed to reconstruct the model with the correct architecture
    config_path = model_dir / "config_used.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Re-create the model architecture
    # Lb and ub can be dynamically modified by the script using the model itself
    net_u = NetU(layers=config['layers'], lb=0, ub=0).to(device)
    net_u = net_u.to(torch.double)

    # Load the Saved Weights (state_dict)
    state_dict_path = model_dir / "net_u.pth"
    net_u.load_state_dict(torch.load(state_dict_path, map_location=device))

    # Load the xi Tensor
    xi_path = model_dir / "xi.pth"
    xi = torch.load(xi_path, map_location=device)

    return net_u, xi