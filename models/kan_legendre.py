import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinearHermite(nn.Module):
    """
    Hermite polynomial based KAN layer.
    """
    def __init__(
        self,
        in_features,
        out_features,
        max_degree=3,
        scale_noise=0.1,
        scale_coeff=1.0,
        enable_standalone_scale_coeff=True,
        base_activation=nn.SiLU,
        domain=[-3, 3],
    ):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.max_degree   = max_degree
        self.num_basis    = max_degree + 1

        self.base_weight  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.coeff_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, self.num_basis)
        )

        if enable_standalone_scale_coeff:
            self.coeff_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.coeff_scaler = None

        self.base_activation = base_activation()
        self.scale_noise = scale_noise
        self.scale_coeff = scale_coeff

        self.domain = torch.tensor(domain, dtype=torch.float32)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight)

        with torch.no_grad():
            noise = (torch.rand_like(self.coeff_weight) - 0.5) * self.scale_noise
            self.coeff_weight.data.copy_(noise)
            if self.coeff_scaler is not None:
                nn.init.kaiming_uniform_(self.coeff_scaler)

    def hermite_basis(self, x):
        a, b = self.domain
        x_scaled = 2 * (x - a) / (b - a) - 1

        H = [torch.ones_like(x_scaled)]
        if self.max_degree >= 1:
            H.append(2 * x_scaled)

        for n in range(2, self.num_basis):
            Hn = 2 * x_scaled * H[-1] - 2 * (n - 1) * H[-2]
            H.append(Hn)

        return torch.stack(H, dim=2)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, self.in_features)

        base_out = F.linear(self.base_activation(x), self.base_weight)

        H = self.hermite_basis(x)
        if self.coeff_scaler is not None:
            C = self.coeff_weight * self.coeff_scaler.unsqueeze(-1)
        else:
            C = self.coeff_weight

        hermite_out = torch.einsum("bik,oik->bo", H, C)

        return (base_out + hermite_out).view(*shape[:-1], self.out_features)


class KANHermite(nn.Module):
    def __init__(self, layer_sizes, max_degree=3, **kwargs):
        super().__init__()
        layers = []
        for a, b in zip(layer_sizes, layer_sizes[1:]):
            layers.append(
                KANLinearHermite(
                    in_features=a,
                    out_features=b,
                    max_degree=max_degree,
                    **kwargs
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
