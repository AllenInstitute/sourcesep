import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,T=10, I=3, L=20, J=2, S=None, E=None, Mu_ox=None, Mu_dox=None):
        super(Model, self).__init__()
        self.T = T  # n_samples in time window
        self.I = I  # n_indicators
        self.L = L  # n_pixels (wavelengths)
        self.J = J  # n_laser channels

        # Constants
        self.S = torch.tensor(S).to(torch.float)             # shape = (I,L)
        self.E = torch.tensor(E).to(torch.float)             # shape = (J,L)
        self.Mu_ox = torch.tensor(Mu_ox).to(torch.float)     # shape=(L,)
        self.Mu_dox = torch.tensor(Mu_dox).to(torch.float)   # shape=(L,)

        self.A = torch.nn.Parameter(torch.empty((T,I)))
        self.W = torch.nn.Parameter(torch.empty((I,J)))
        self.N = torch.nn.Parameter(torch.empty((T,J)))
        self.B = torch.nn.Parameter(torch.empty((J,L)))
        self.M = torch.nn.Parameter(torch.empty((T,)))
        self.H_ox = torch.nn.Parameter(torch.empty((T,)))
        self.H_dox = torch.nn.Parameter(torch.empty((T,)))
        self.shape_checks()
        self.param_init()

    def shape_checks(self):
        assert self.S.shape==(self.I,self.L), 'S shape mismatch'
        assert self.E.shape==(self.J,self.L), 'E shape mismatch'
        assert self.Mu_ox.shape==(self.L,), 'Mu_ox shape mismatch'
        assert self.Mu_dox.shape==(self.L,), 'Mu_dox shape mismatch'

    def param_init(self):
        nn.init.normal_(self.A)
        nn.init.normal_(self.W)
        nn.init.normal_(self.N)
        nn.init.normal_(self.B)
        nn.init.normal_(self.M)
        nn.init.normal_(self.H_ox)
        nn.init.normal_(self.H_dox)

    def forward(self, O):
        # 1st term
        AS = torch.einsum('ti,il->til', self.A, self.S)
        ASW = torch.einsum('til,ij->tjl', AS, self.W)
        E = torch.einsum('td,djl -> tjl', torch.ones((self.T, 1)), self.E[None, ...])
        ASWE = ASW + E

        # 2nd term
        HD = torch.einsum('td,dl -> tl', self.H_ox[..., None], self.Mu_ox[None, ...]) \
            + torch.einsum('td,dl -> tl', self.H_dox[..., None], self.Mu_dox[None, ...])
        HD = torch.einsum('j,tl -> tjl', torch.ones((self.J,)), HD)
        HDM = torch.einsum('tjl,t -> tjl', HD, self.M)
        B = torch.einsum('t,jl -> tjl', torch.ones((self.T,)), self.B)
        H = HDM + B

        # 3rd term
        N = torch.einsum('l,tj -> tjl', torch.ones((self.L,)), self.N)

        # Combining all the terms
        O_pred = torch.einsum('tjl,tjl -> tjl', ASWE, H)
        O_pred = torch.einsum('tjl,tjl -> tjl', O_pred, N)
        return O_pred
