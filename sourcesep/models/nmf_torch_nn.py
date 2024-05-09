import torch
import torch.nn as nn
import numpy as np


def parameter(x):
    return torch.tensor(x, requires_grad=True, dtype=torch.float32)


class AHX(nn.Module):
    def __init__(self, n_sources, n_channels, n_wavelengths, n_timepoints, s, se, X):
        """
        Args:
            s (list): elements are source spectra
            se (list): elements are source spectra
            n_sources (int): number of sources
            n_channels (int): number of channels
            n_wavelengths (int): number of wavelengths
            n_timepoints (int): number of timepoints
        """

        super().__init__()

        self.n_sources = n_sources
        self.n_wavelengths = n_wavelengths
        self.n_timepoints = n_timepoints
        self.n_channels = n_channels
        self.min_value = 1e-8

        # checks
        assert len(s) == n_sources, f"s must have {n_sources} elements"
        assert len(se) == n_sources, f"se must have {n_sources} elements"
        for i in range(n_sources):
            assert s[i].size == n_wavelengths, f"s[{i}] must have shape {n_wavelengths}"
            assert se[i].size == n_channels, f"se[{i}] must have shape {n_channels}"

        # fmt: off
        self.a = nn.ParameterDict({str(i): parameter(s[i].reshape(n_wavelengths, 1)) for i in range(n_sources)})
        self.a_coef = nn.ParameterDict({str(i): parameter(se[i]) for i in range(n_sources)})
        # fmt: on

        self.sigma_h = 0.03
        self.mu_h = 0.3
        self.set_fac(X)

        # fmt: off
        self.h_coef = parameter([1/(self.fac*self.mu_h), 1/(self.fac*self.mu_h)])
        self.h01 = parameter(self.sigma_h * np.random.randn(1, n_timepoints) + self.mu_h)  # req. positive entries
        self.h2 = parameter(self.sigma_h * np.random.randn(1, n_timepoints) + self.mu_h)  # req. positive entries
        # fmt: on

        self.rescale_coef = 0.90
        self.prox_plus = torch.nn.Threshold(0, self.min_value)

    def set_fac(self, X=None):
        if X is None:
            self.fac = 1
        else:
            A_ = self.get_A()
            AA = A_.detach().numpy().copy()
            X_init = AA @ np.ones((AA.shape[1], X.shape[1]))
            self.fac = np.median(X_init[X > 0] / X[X > 0])
        return

    def get_A(self):
        with torch.no_grad():
            a_ = {}
            for src in range(self.n_sources):
                a_[str(src)] = torch.cat(
                    list(
                        self.a_coef[str(src)][ch] * self.a[str(src)]
                        for ch in range(self.n_channels)
                    ),
                    dim=0,
                )
        A_ = torch.cat((a_["0"], a_["1"], a_["2"]), dim=1)
        return A_

    def get_H(self):
        h0_ = self.h_coef[0] * self.h01
        h1_ = self.prox_plus(self.h_coef[1] * (1 - self.h01))
        H_ = torch.cat((h0_, h1_, self.h2), dim=0)
        return H_

    def forward(self):
        A_ = self.get_A()
        H_ = self.get_H()
        X_ = torch.matmul(A_, H_)
        return X_

    def rescale(self):
        while torch.sum(self.h01 > 1) > (0.05 * self.h01.numel()):
            self.h01.data = self.h01 * self.rescale_coef
            self.h_coef.data = self.h_coef / self.rescale_coef

    def fit(self, X):
        # observation
        lr = 1e-4
        n_steps = 4_000

        loss_list = []
        for i in range(n_steps + 1):
            X_ = self.forward()
            loss = torch.linalg.matrix_norm(X_ - X, ord="fro") ** 2
            loss.backward()
            loss_list.append(loss.item())
            param_list = [self.h01, self.h2, self.h_coef]

            for p in param_list:
                p.data -= lr * p.grad
                p.grad.zero_()
                p.data = self.prox_plus(p)

            self.rescale()

            print(f"Step: {i}, Loss: {loss.item()}")

        return


if __name__ == "__main__":
    from sourcesep.models.datasets import test_data_v1

    dat, sim, S, S_autofl, xdat, Y, W_df = test_data_v1()
    laser_names = W_df.columns.tolist()
    source_names = ["Wb", "Wf", "FAD"]
    n_sources = len(source_names)
    n_channels = len(laser_names)
    n_timepoints = Y.shape[1]
    n_wavelengths = Y.shape[0] // n_channels

    s = [None] * n_sources
    s[0] = S["bound_em"].reshape(-1, 1)
    s[1] = S["free_em"].reshape(-1, 1)
    s[2] = S_autofl[0, :].reshape(-1, 1)
    se = [None] * n_sources
    for i, src in enumerate(source_names):
        se[i] = W_df.loc[src, laser_names].values

    model = AHX(n_sources, n_channels, n_wavelengths, n_timepoints, s, se, Y)
    X = torch.as_tensor(Y, dtype=torch.float32).to("cpu")
    model.to("cpu")
    model.fit(X)
