import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics import CosineSimilarity, ExplainedVariance, MeanAbsoluteError, MeanSquaredError


class BaseUnet(nn.Module):
    """Base Unet model for comparisons

    Args:
        in_channels (int, optional): Time axis.
        out_channels (int, optional): Defaults to 8.
    """

    def __init__(self, in_channels=60, cfg=None, **kwargs):
        super().__init__()

        # input: torch.Size([batch, 300, 2048])
        self.norm_0 = nn.LayerNorm((2048))
        self.conv_0 = self.double_conv(in_channels=in_channels, out_channels=16, kernel_size=16)
        self.pool_0 = nn.MaxPool1d(kernel_size=4)

        # x0: torch.Size([batch, 16, 504])
        self.norm_1 = nn.LayerNorm((504))
        self.conv_1 = self.double_conv(16, 32, kernel_size=15)
        self.pool_1 = nn.MaxPool1d(kernel_size=2)

        # x1: torch.Size([batch, 32, 238])
        self.norm_2 = nn.LayerNorm((238))
        self.conv_2 = self.double_conv(32, 64, kernel_size=16)
        self.pool_2 = nn.MaxPool1d(kernel_size=2)

        # x2: torch.Size([batch, 64, 104])
        self.norm_3 = nn.LayerNorm((104))
        self.conv_3 = self.double_conv(64, 64, kernel_size=8)

        self.up_conv_2 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=2)
        self.conv_2_ = self.double_conv(96, 32, kernel_size=3)

        self.up_conv_1 = nn.ConvTranspose1d(32, 16, kernel_size=8, stride=2)
        self.conv_1_ = self.double_conv(48, 16, kernel_size=3)

        self.up_conv_0 = nn.ConvTranspose1d(16, 8, kernel_size=8, stride=4)

        self.outheads = nn.ModuleDict(
            {"A0": self.out_head(24, 8, kernel_size=3), "A1": self.out_head(24, 8, kernel_size=3)}
        )
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        return

    def forward(self, input):
        # print(f'input: {input.shape}')
        x0 = self.conv_0(self.norm_0(input))
        x0_mp = self.pool_0(x0)

        # print(f'x0: {x0_mp.shape}')
        x1 = self.conv_1(self.norm_1(x0_mp))
        x1_mp = self.pool_1(x1)

        # print(f'x1: {x1_mp.shape}')
        x2 = self.conv_2(self.norm_2(x1_mp))
        x2_mp = self.pool_2(x2)

        # print(f'x2: {x2_mp.shape}')
        x3 = self.conv_3(self.norm_3(x2_mp))

        x2_ = self.tanh(self.up_conv_2(x3))
        x2_cat = torch.cat([self.crop(input=x2, target=x2_, dim=2), x2_], dim=1)
        x2_ = self.conv_2_(x2_cat)

        x1_ = self.tanh(self.up_conv_1(x2_))
        x1_cat = torch.cat([self.crop(input=x1, target=x1_, dim=2), x1_], dim=1)
        x1_ = self.conv_1_(x1_cat)

        x0_ = self.tanh(self.up_conv_0(x1_))
        x0_cat = torch.cat([self.crop(input=x0, target=x0_, dim=2), x0_], dim=1)

        A0 = self.outheads["A0"](x0_cat)
        A1 = self.outheads["A1"](x0_cat)
        output = torch.cat([A0, A1], dim=1)

        return output

    def test(self):
        # make random input
        input = torch.rand(20, 60, 2048)
        output = self.forward(input)
        return

    @staticmethod
    def out_head(in_channels, int_channels, kernel_size):
        conv = nn.Sequential(
            nn.Conv1d(in_channels, int_channels, kernel_size, padding="valid"),
            nn.Tanh(),
            nn.Conv1d(int_channels, int_channels, kernel_size, padding="valid"),
            nn.Tanh(),
            nn.Conv1d(int_channels, 1, 3, padding="valid"),
        )
        return conv

    @staticmethod
    def double_conv(in_channels, out_channels, kernel_size):
        conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="valid"),
            nn.Tanh(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="valid"),
            nn.Tanh(),
        )
        return conv

    def crop(self, input, target, dim):
        """Crop input to match target size along a given dimension"""
        delta = (input.shape[dim] - target.shape[dim]) // 2
        index = torch.arange(delta, input.shape[dim] - delta, 1, device=input.device, requires_grad=False)
        return torch.index_select(input, dim, index)


class Compose(nn.Module):
    def __init__(self, S, E, W, Mu_ox, Mu_dox, B, **kwargs):
        super().__init__()
        self.register_buffer("S", torch.tensor(S), persistent=True)  # shape = (I,L)
        self.register_buffer("E", torch.tensor(E), persistent=True)  # shape = (J,L)
        self.register_buffer("W", torch.tensor(W), persistent=True)  # shape = (I,J)
        self.register_buffer("B", torch.tensor(B), persistent=True)  # shape = (J,L)
        self.register_buffer("Mu_ox", torch.tensor(Mu_ox), persistent=True)  # shape=(L,)
        self.register_buffer("Mu_dox", torch.tensor(Mu_dox), persistent=True)  # shape=(L,)

    def forward(self, A, H_ox, H_dox, M, N):
        T = A.shape[-1]
        J = self.E.shape[0]
        L = self.E.shape[1]
        batch_size = A.shape[0]

        AS = torch.einsum("bit,il -> btil", A, self.S)
        ASW = torch.einsum("btil,ij -> btjl", AS, self.W)
        ASWE = ASW + self.E[None, ...].expand(batch_size, T, -1, -1)

        HD = torch.exp(
            -(
                torch.einsum("btd,dl -> btl", H_ox[..., None], self.Mu_ox[None, ...])
                + torch.einsum("btd,dl -> btl", H_dox[..., None], self.Mu_dox[None, ...])
            )
        )

        HDM = torch.einsum("btjl,bt -> btjl", HD[:, :, None, :].expand(-1, -1, J, -1), M)
        H = HDM + self.B[None, None, ...].expand_as(HDM)

        N = N[:, :, :, None].expand(-1, -1, -1, L)
        N = torch.einsum("bjtl -> btjl", N)

        # Combining all the terms
        O_pred = torch.einsum("btjl,btjl -> btjl", ASWE, H)
        O_pred = torch.einsum("btjl,btjl -> btjl", O_pred, N)
        O_pred = O_pred.reshape(batch_size, T, J * L)
        O_pred = torch.einsum("btc -> bct", O_pred)
        return O_pred


class LitBaseUnet(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = BaseUnet(**kwargs)
        self.compose = Compose(**kwargs)
        self.pad = 293  # to sidestep boundary issues for now
        self.save_hyperparameters()  # saves all kwargs passed to the model

        # torchmetrics. These are reset implicitly at each epoch end.
        self.metric_cos_sim = CosineSimilarity(reduction="mean", is_differentiable=False)
        self.metric_exp_var = ExplainedVariance(multioutput="uniform_average", is_differentiable=False)
        self.metric_mean_l1 = MeanAbsoluteError(is_differentiable=False)
        self.metric_mean_l2 = MeanSquaredError(is_differentiable=False)
        self.getx = lambda x: x.detach().cpu().numpy()

        # nn losses
        self.smoothl1 = nn.SmoothL1Loss(beta=0.1, reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

        # initialize
        for name in ["A0", "A1"]:
            layer = list(self.model.outheads[name].children())[-1]
            layer.bias.data.fill_(0.0)  # 0 is transformed activity mean

    @staticmethod
    def A_transform(x):
        return 5.0 * (x - 1.0)

    @staticmethod
    def A_transform_inv(x):
        return (1.0 / 5.0) * (x) + 1.0

    @staticmethod
    def crop_batch(batch, pad):
        batch_cropped = {}
        batch_cropped["A"] = batch["A"][:, :, pad:-pad]
        batch_cropped["H_ox"] = torch.squeeze(batch["H_ox"][:, :, pad:-pad])
        batch_cropped["H_dox"] = torch.squeeze(batch["H_dox"][:, :, pad:-pad])
        batch_cropped["M"] = torch.squeeze(batch["M"][:, :, pad:-pad])
        batch_cropped["N"] = batch["N"][:, :, pad:-pad]
        batch_cropped["O"] = batch["O"][:, :, pad:-pad]
        return batch_cropped

    def forward(self, batch):
        Ar = self.model(batch["O"])
        batch_cropped = self.crop_batch(batch, self.pad)
        Or = self.compose(
            self.A_transform_inv(Ar),
            batch_cropped["H_ox"],
            batch_cropped["H_dox"],
            batch_cropped["M"],
            batch_cropped["N"],
        )

        # reconstruct with ground truth
        # Ox = self.compose(A, H_ox, H_dox, M, N)
        # assert torch.allclose(Ox, O), 'compose test failed'
        return Ar, Or, batch_cropped

    def update_metrics(self, true, pred):
        """Metrics are accumulated over batches and then averaged over epochs"""
        self.metric_mean_l1.update(true, pred)
        self.metric_mean_l2.update(true, pred)
        self.metric_exp_var.update(true, pred)
        self.metric_cos_sim.update(true[:, 0, :], pred[:, 0, :])
        self.metric_cos_sim.update(true[:, 1, :], pred[:, 1, :])
        return

    def reset_metrics(self):
        self.metric_mean_l1.reset()
        self.metric_mean_l2.reset()
        self.metric_exp_var.reset()
        self.metric_cos_sim.reset()
        self.metric_cos_sim.reset()
        self.metric_cos_sim.reset()
        return

    def get_losses(self, A_true, A_pred, O_true, O_pred):
        """Losses are not averaged over batches in the current form."""
        loss_activity_smoothL1 = self.smoothl1(A_pred[:, 0, :], A_true[:, 0, :]) + self.smoothl1(
            A_pred[:, 1, :], A_true[:, 1, :]
        )
        loss_reconstruction_mse = self.mse(O_pred, O_true)

        # lightning implicitly uses key 'loss' for backprop
        return {
            "loss": loss_activity_smoothL1 + loss_reconstruction_mse,
            "loss_activity_smoothL1": loss_activity_smoothL1,
            "loss_reconstruction_mse": loss_reconstruction_mse,
        }

    def make_plot(self, true, pred):
        f, ax = plt.subplots(2, 1, figsize=(10, 8))
        for ch in range(2):
            sample_idx = 0
            true_arr = self.getx(torch.squeeze(true[sample_idx, ch, :]))
            pred_arr = self.getx(torch.squeeze(pred[sample_idx, ch, :]))
            ax[ch].plot(true_arr, label="true", c="dodgerblue", alpha=0.8)
            ax[ch].plot(pred_arr, label="pred", c="crimson", alpha=0.5)
            ax[ch].legend()
        return f

    def training_step(self, batch, batch_idx):
        Ar, Or, batch_cropped = self(batch)
        A_transformed = self.A_transform(batch_cropped["A"])

        # separate loss components
        losses_metrics = self.get_losses(A_true=A_transformed, A_pred=Ar, O_pred=Or, O_true=batch_cropped["O"])
        self.update_metrics(true=A_transformed, pred=Ar)
        losses_metrics.update(
            {
                "metric_mean_l1": self.metric_mean_l1.compute(),
                "metric_mean_l2": self.metric_mean_l2.compute(),
                "metric_exp_var": self.metric_exp_var.compute(),
                "metric_cos_sim": self.metric_cos_sim.compute(),
            }
        )
        self.log_dict(losses_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        if batch_idx == 0:
            f = self.make_plot(true=A_transformed, pred=Ar)
            self.logger.experiment.add_figure("Training", f, self.current_epoch)

        # implicitly uses key 'loss' for backprop
        return losses_metrics

    def on_train_epoch_end(self):
        self.reset_metrics()
        return

    def validation_step(self, batch, batch_idx):
        Ar, Or, batch_cropped = self(batch)
        A_transformed = self.A_transform(batch_cropped["A"])

        losses_metrics = self.get_losses(A_true=A_transformed, A_pred=Ar, O_pred=Or, O_true=batch_cropped["O"])
        self.update_metrics(true=A_transformed, pred=Ar)
        losses_metrics.update(
            {
                "metric_mean_l1": self.metric_mean_l1.compute(),
                "metric_mean_l2": self.metric_mean_l2.compute(),
                "metric_exp_var": self.metric_exp_var.compute(),
                "metric_cos_sim": self.metric_cos_sim.compute(),
            }
        )

        losses_metrics = {"val_" + k: v for k, v in losses_metrics.items()}
        self.log_dict(losses_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        if batch_idx == 0:
            f = self.make_plot(true=A_transformed, pred=Ar)
            self.logger.experiment.add_figure("Validation", f, self.current_epoch)

        return losses_metrics

    def on_validation_epoch_end(self):
        self.reset_metrics()
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())  # default is 1e-3
