import torch
import torch.nn as nn
import math


class KANLinearBSpline(nn.Module):
    """
    B-Spline based KAN layer.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-3, 3],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Build grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Base + spline weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.spline_scaler = None

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        with torch.no_grad():
            noise = torch.rand(
                self.grid_size + 1,
                self.in_features,
                self.out_features
            ) - 0.5
            noise *= (self.scale_noise / self.grid_size)

            coeffs = self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order],
                noise,
            )

            self.spline_weight.data.copy_(coeffs)
            if self.spline_scaler is not None:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).float()

        for k in range(1, self.spline_order + 1):
            left_num  = x - self.grid[:, :-(k + 1)]
            left_den  = self.grid[:, k:-1] - self.grid[:, :-(k + 1)]
            right_num = self.grid[:, k+1:] - x
            right_den = self.grid[:, k+1:] - self.grid[:, 1:-k]

            bases = (left_num / left_den) * bases[:, :, :-1] + \
                    (right_num / right_den) * bases[:, :, 1:]

        return bases

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        sol = torch.linalg.lstsq(A, B).solution
        return sol.permute(2, 0, 1)

    @property
    def scaled_spline_weight(self):
        if self.spline_scaler is not None:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.in_features)

        base_out = nn.functional.linear(self.base_activation(x), self.base_weight)

        b = self.b_splines(x)
        b = b.view(b.size(0), -1)
        spline_out = nn.functional.linear(
            b,
            self.scaled_spline_weight.view(self.out_features, -1)
        )

        y = base_out + spline_out
        return y.view(*orig_shape[:-1], self.out_features)


class KAN(nn.Module):
    """
    Full KAN model using B-spline layers.
    Use: KAN(layer_sizes=[2,64,64,64,64,1], spline_order=5)
    """
    def __init__(
        self,
        layer_sizes,
        grid_size=5,
        spline_order=3,
        **kwargs
    ):
        super().__init__()

        layers = []
        for inp, outp in zip(layer_sizes, layer_sizes[1:]):
            layers.append(
                KANLinearBSpline(
                    in_features=inp,
                    out_features=outp,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    **kwargs
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
