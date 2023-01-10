import torch
import torch.nn as nn

# input is (T=1024, J=5, L=300) (time, laser, lambda)
# input should be re-organized as (batch, J * L, T) (batch, channels, time)
# output is of shape (T=1024, I=8) (time, sources={indicators, hemodynamics, noise})

# T = 1024
# J = 5
# L = 300

class BaseUnet(nn.Module):
    """Base Unet model for comparisons

    Args:
        in_channels (int, optional): Defaults to 1500.
        out_channels (int, optional): Defaults to 8.
    """
    def __init__(self, in_channels=1500, out_channels=8):
        super().__init__()
        self.conv_0 = self.double_conv(in_channels, 8, kernel_size=8)
        self.mp_0 = nn.MaxPool1d(kernel_size=4)
        self.conv_1 = self.double_conv(8, 16, kernel_size=8)
        self.mp_1 = nn.MaxPool1d(kernel_size=2)
        self.conv_2 = self.double_conv(16, 32, kernel_size=8)
        self.mp_2 = nn.MaxPool1d(kernel_size=2)
        self.up_conv_2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.up_conv_1 = nn.ConvTranspose1d(32, 8, kernel_size=2, stride=2)
        self.up_conv_0 = nn.ConvTranspose1d(16, out_channels, kernel_size=4, stride=4)
        self.out_conv = nn.Conv1d(out_channels, out_channels, 3, padding='same', padding_mode='reflect')
        return

    def forward(self, input):
        x0 = self.mp_0(self.conv_0(input))
        x1 = self.mp_1(self.conv_1(x0))
        x2 = self.mp_2(self.conv_2(x1))

        x1_ = self.up_conv_2(x2)
        x1_cat = torch.cat([x1, x1_], dim=1)

        x0_ = self.up_conv_1(x1_cat)
        x0_cat = torch.cat([x0, x0_], dim=1)
        x0_pre = self.up_conv_0(x0_cat)
        output = self.out_conv(x0_pre)
        return output

    @staticmethod
    def double_conv(in_channels, out_channels, kernel_size):
        """Performm 2 successive convolution operations.

        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (int)

        Returns:
            nn.Seqeuntial object
        """
        conv = nn.Sequential(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      padding='same',
                      padding_mode='reflect'),
            #nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels,
                      out_channels,
                      kernel_size,
                      padding='same',
                      padding_mode='reflect'),
            #nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        return conv
