import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class BaseUnet(nn.Module):
    """Base Unet model for comparisons

    Args:
        in_channels (int, optional): Time axis.
        out_channels (int, optional): Defaults to 8.
    """
    def __init__(self, in_channels=60, out_channels=8, cfg=None, **kwargs):
        super().__init__()
        self.conv_0 = self.double_conv(in_channels=in_channels, out_channels=16, kernel_size=16)
        self.mp_0 = nn.MaxPool1d(kernel_size=4)
        self.conv_1 = self.double_conv(16, 32, kernel_size=15)
        self.mp_1 = nn.MaxPool1d(kernel_size=2)
        self.conv_2 = self.double_conv(32, 64, kernel_size=16)
        self.mp_2 = nn.MaxPool1d(kernel_size=2)
        self.conv_3 = self.double_conv(64, 64, kernel_size=8)

        self.up_conv_2 = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2))
        self.conv_2_ = self.double_conv(96, 32, kernel_size=3)

        self.up_conv_1 = nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2)
        self.conv_1_ = self.double_conv(48, 16, kernel_size=3)

        self.up_conv_0 = nn.ConvTranspose1d(16, 8, kernel_size=8, stride=4)
        self.conv_0_ = self.double_conv(24, 8, kernel_size=3)
        self.out_conv = nn.Conv1d(8, 8, 3, padding='valid')
        self.lrelu = nn.LeakyReLU()
        return

    def forward(self, input):
        x0 = self.conv_0(input)
        #print('x0', x0.shape)
        x0_mp = self.mp_0(x0)
        #print('x0_mp', x0_mp.shape)
        x1 = self.conv_1(x0_mp)
        #print('x1', x1.shape)
        x1_mp = self.mp_1(x1)
        #print('x1_mp', x1_mp.shape)
        x2 = self.conv_2(x1_mp)
        #print('x2', x2.shape)
        x2_mp = self.mp_2(x2)
        #print('x2_mp', x2_mp.shape)
        x3 = self.conv_3(x2_mp)
        #print('x3', x3.shape)
        x2_ = self.lrelu(self.up_conv_2(x3))
        #print('x2_', x2_.shape)
        x2_cat = torch.cat([self.crop(input=x2,  target=x2_, dim=2), x2_], dim=1)
        #print('x2_cat', x2_cat.shape)
        x2_ = self.conv_2_(x2_cat)
        #print('x2_ conv', x2_.shape)
        x1_ = self.lrelu(self.up_conv_1(x2_))
        #print('x1_', x1_.shape)
        x1_cat = torch.cat([self.crop(input=x1,  target=x1_, dim=2), x1_], dim=1)
        #print('x1_cat', x1_cat.shape)
        x1_ = self.conv_1_(x1_cat)
        #print('x1_', x1_.shape)

        x0_ = self.lrelu(self.up_conv_0(x1_))
        #print('x0_', x0_.shape)

        x0_cat = torch.cat([self.crop(input=x0,  target=x0_, dim=2), x0_], dim=1)
        #print('x0_cat', x0_cat.shape)
        x0_ = self.conv_0_(x0_cat)
        #print('x0_', x0_.shape)

        output = self.lrelu(self.out_conv(x0_))
        return output
    
    def test(self):
        # make random input
        input = torch.rand(20, 60, 2048)
        output = self.forward(input)
        return

    @staticmethod
    def double_conv(in_channels, out_channels, kernel_size):
        conv = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      padding='valid'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size,
                      padding='valid'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        return conv
    
    def crop(self, input, target, dim):
        """Crop input to match target size along a given dimension
        """
        delta = input.shape[dim] - target.shape[dim]
        delta = delta // 2
        index = torch.arange(delta, input.shape[dim]-delta, 1, device=input.device)
        return torch.index_select(input, dim, index)


class Compose(nn.Module):
    def __init__(self, S, E, W, Mu_ox, Mu_dox, B, **kwargs):
        super().__init__()
        self.register_buffer('S', torch.tensor(S), persistent=True)             # shape = (I,L)
        self.register_buffer('E', torch.tensor(E), persistent=True)             # shape = (J,L)
        self.register_buffer('W', torch.tensor(W), persistent=True)             # shape = (I,J)
        self.register_buffer('B', torch.tensor(B), persistent=True)             # shape = (J,L)
        self.register_buffer('Mu_ox', torch.tensor(Mu_ox), persistent=True)     # shape=(L,)
        self.register_buffer('Mu_dox', torch.tensor(Mu_dox), persistent=True)   # shape=(L,)

    def forward(self, A, H_ox, H_dox, M, N):
        device = self.S.device
        T = A.shape[-1]
        J = self.E.shape[0]
        L = self.E.shape[1]
        batch_size = A.shape[0]

        AS = torch.einsum('bit,il -> btil', A, self.S)
        ASW = torch.einsum('btil,ij -> btjl', AS, self.W)
        ASWE = ASW + self.E[None, ...].expand(batch_size, T, -1, -1)

        HD = torch.exp(-( \
             torch.einsum('btd,dl -> btl', H_ox[..., None], self.Mu_ox[None, ...]) \
             + torch.einsum('btd,dl -> btl', H_dox[..., None], self.Mu_dox[None, ...])))

        HDM = torch.einsum('btjl,bt -> btjl', HD[:, :, None, :].expand(-1, -1, J, -1), M)
        H = HDM + self.B[None, None, ...].expand_as(HDM)

        N = N[:,:,:,None].expand(-1, -1, -1, L)
        N = torch.einsum('bjtl -> btjl', N)

        # Combining all the terms
        O_pred = torch.einsum('btjl,btjl -> btjl', ASWE, H)
        O_pred = torch.einsum('btjl,btjl -> btjl', O_pred, N)
        O_pred = O_pred.reshape(batch_size, T, J*L)
        O_pred = torch.einsum('btc -> bct', O_pred)
        return O_pred


class LitBaseUnet(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = BaseUnet(**kwargs)
        self.compose = Compose(**kwargs)

        self.A_loss = nn.L1Loss(reduction='mean')
        self.H_loss = nn.L1Loss(reduction='mean')
        self.pad = 293 # to sidestep boundary issues for now
        self.save_hyperparameters() # saves all kwargs passed to the model

    def loss_A(self, input, target):
        return nn.functional.l1_loss(input, target, reduction='mean')

    def loss_H(self, input, target):
        return nn.functional.l1_loss(input, target, reduction='mean')
    
    def loss_recon(self, input, target):
        return nn.functional.l1_loss(input, target, reduction='mean')

    def training_step(self, batch, batch_idx):
        output = self.model(batch['O'])
        Ar = torch.squeeze(output[:, 0:3, ...])
        H_oxr = torch.squeeze(output[:, 3, ...])
        H_doxr = torch.squeeze(output[:, 4, ...]
        # cropped ground truth
        A = batch['A'][:,:,self.pad:-self.pad])
        H_ox = torch.squeeze(batch['H_ox'][:,:,self.pad:-self.pad]))
        H_dox = torch.squeeze(batch['H_dox'][:,:,self.pad:-self.pad])
        M = torch.squeeze(batch['M'][:,:,self.pad:-self.pad])
        N = batch['N'][:,:,self.pad:-self.pad]

        # reconstruct with phenomenological model
        Or = self.compose(Ar, H_oxr, H_doxr, M, N)
        # reconstruct with ground truth
        # Ox = self.compose(A, H_ox, H_dox, M, N)
        # assert torch.allclose(Ox, O), 'compose test failed'
        
        O = batch['O'][:,:,self.pad:-self.pad]
        loss = self.loss_A(Ar[:,0,:], A[:,0,:]) 
            #+ self.loss_A(Ar[:,1,:], A[:,1,:]) \
            #+ self.loss_A(Ar[:,2,:], A[:,2,:])
            #+ self.loss_O(Or, O)

        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.model(batch['O'])
        Ar = torch.squeeze(output[:, 0:3, ...])
        A = batch['A'][:,:,self.pad:-self.pad]
        H_oxr = torch.squeeze(output[:, 3, ...])
        H_doxr = torch.squeeze(output[:, 4, ...])

        loss = self.loss_A(Ar[:,0,:], A[:,0,:]) \
            #+ self.loss_A(Ar[:,1,:], A[:,1,:]) \
            #+ self.loss_A(Ar[:,2,:], A[:,2,:])

        f, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(torch.squeeze(A[0].T[:,0]).to('cpu'), label='data')
        ax.plot(torch.squeeze(Ar[0].T[:,0]).to('cpu'), label='recon')
        ax.legend()
        self.logger.experiment.add_figure('A reconstruction', f, self.current_epoch)
        self.log('val_loss', loss)
        return
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
