import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import nnls


def parameter(x):
    return torch.tensor(x, requires_grad=True, dtype=torch.float32)


def const(x):
    return torch.tensor(x, requires_grad=False, dtype=torch.float32)


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
        self.a_norm = nn.ParameterDict({str(i): const(np.linalg.norm(s[i])) for i in range(n_sources)})

        # fmt: on

        H_init = self.get_H_init(X)
        self.h_coef = parameter([1.0, 1.0])
        self.h01 = parameter(H_init[[0], :])
        self.h2 = parameter(H_init[[2], :])
        self.rescale_coef = 0.90
        self.rescale_H()

        self.prox_plus = torch.nn.Threshold(0, self.min_value)

    def get_H_init(self, X=None):
        A_ = self.get_A()
        # if A_ is a tensor convert ot a numpy array
        if isinstance(A_, torch.Tensor):
            A_ = A_.detach().cpu().numpy()
        if isinstance(A_, torch.Tensor):
            X = X.detach().cpu().numpy()
        H_init = np.zeros((self.n_sources, self.n_timepoints))
        for i in range(self.n_timepoints):
            H_init[:, i], _ = nnls(A_, X[:, i])

        return H_init

    def get_A(self):
        a_ = {}
        for src in range(self.n_sources):
            a_[str(src)] = torch.cat(
                list(self.a_coef[str(src)][ch] * self.a[str(src)] for ch in range(self.n_channels)),
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

    def rescale_H(self):
        with torch.no_grad():
            while torch.sum(self.h01 > 1) > (0.05 * self.h01.numel()):
                self.h01.data = self.h01 * self.rescale_coef
                self.h_coef.data[0] = self.h_coef[0] / self.rescale_coef
            while torch.sum(self.h2 > 1) > (0.05 * self.h2.numel()):
                self.h2.data = self.h2 * self.rescale_coef
                self.h_coef.data[1] = self.h_coef[1] / self.rescale_coef

    def rescale_a(self):
        """Spectra in self.a have the same norm throughout.
        This operation keeps self.A the same, only rescales the constituents.
        """
        with torch.no_grad():
            for src in range(self.n_sources):
                rescale_factor = self.a_norm[str(src)] / torch.linalg.norm(self.a[str(src)])
                self.a[str(src)].data = self.a[str(src)] * rescale_factor
                self.a_coef[str(src)].data = self.a_coef[str(src)] / rescale_factor
        return

    def fit(self, X):
        # observation
        lr = 1e-4
        n_steps = 8_000

        loss_list = []
        for i in range(n_steps + 1):
            X_ = self.forward()
            loss = torch.linalg.matrix_norm(X_ - X, ord="fro") ** 2
            loss.backward()
            loss_list.append(loss.item())

            param_list = [
                self.h01,
                self.h2,
                self.h_coef,
                self.a["0"],
                self.a["1"],
                self.a["2"],
                self.a_coef["0"],
                self.a_coef["1"],
                self.a_coef["2"],
            ]

            torch.nn.utils.clip_grad_norm_(param_list, 1.0)
            if i % 1000 == 0:
                print("grad: ", lr * self.h_coef.grad)
                print("value: ", self.h_coef)

            for p in param_list:
                p.data -= lr * p.grad
                torch.mean(p.grad)
                p.grad.zero_()
                p.data = self.prox_plus(p)

            self.rescale_H()
            self.rescale_a()

            if i % 1000 == 0:
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
