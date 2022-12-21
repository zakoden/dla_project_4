from torch import nn

from hw_asr.base import BaseModel


class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size=3, dilations=[[1, 1], [3, 1], [5, 1]]):
        super(ResBlock, self).__init__()

        self.outer_steps = len(dilations)
        self.inner_steps = len(dilations[0])
        modules_list = []
        for m in range(self.outer_steps):
            cur_list = []
            for l in range(self.inner_steps):
                dilation = dilations[m][l]  # D_r[n, m, l] in paper
                cur_list.append(
                    nn.LeakyReLU(0.1)
                )
                cur_list.append(
                    nn.Conv1d(in_channels=n_channels,
                              out_channels=n_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              dilation=dilation,
                              padding=int(dilation * (kernel_size - 1) / 2))
                )

            modules_list.append(nn.Sequential(*cur_list))
        self.conv_blocks = nn.ModuleList(modules_list)

    def forward(self, x):
        for m in range(self.outer_steps):
            x = x + self.conv_blocks[m](x)
        return x


class MRF(nn.Module):
    def __init__(self, n_channels, kernel_sizes=[3, 7, 11], dilations=[[1, 1], [3, 1], [5, 1]]):
        super(MRF, self).__init__()

        self.n_resblock = len(kernel_sizes)  # |k_r|
        modules_list = []
        for n in range(self.n_resblock):
            modules_list.append(
                ResBlock(n_channels, kernel_size=kernel_sizes[n], dilations=dilations)
            )
        self.resblocks = nn.ModuleList(modules_list)

    def forward(self, x):
        x_sum = None
        for n in range(self.n_resblock):
            if n == 0:
                x_sum = self.resblocks[n](x)
            else:
                x_sum = x_sum + self.resblocks[n](x)
        return x_sum


class Generator(BaseModel):
    def __init__(self, in_channels=80, hidden_dim=512, upsample_blocks_vals=[16, 16, 4, 4],
                 resblock_kernels=[3, 7, 11], dilations=[[1, 1], [3, 1], [5, 1]], **batch):
        super().__init__(**batch)

        modules_list = []
        self.in_channels = in_channels

        self.hidden_dim = hidden_dim  # h_u in paper

        self.n_upsample_blocks = len(upsample_blocks_vals)  # |k_u|
        self.upsample_blocks_vals = upsample_blocks_vals  # k_u

        self.resblock_kernels = resblock_kernels  # k_r

        self.dilations = dilations

        modules_list.append(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=hidden_dim,
                      kernel_size=7,
                      stride=1,
                      padding=3)
        )

        for ind_upsample_block in range(self.n_upsample_blocks):
            l = ind_upsample_block + 1  # l from paper

            modules_list.append(nn.LeakyReLU(0.1))

            # conv part
            kernel_size = self.upsample_blocks_vals[l - 1]
            stride = self.upsample_blocks_vals[l - 1] // 2
            modules_list.append(
                nn.ConvTranspose1d(in_channels=self.hidden_dim // (2 ** (l - 1)),
                                   out_channels=self.hidden_dim // (2 ** l),
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=(kernel_size - stride) // 2)
            )
            # MRF part
            modules_list.append(
                MRF(self.hidden_dim // (2 ** l), kernel_sizes=self.resblock_kernels, dilations=self.dilations)
            )

        modules_list.append(nn.LeakyReLU(0.1))
        modules_list.append(
            nn.Conv1d(in_channels=self.hidden_dim // (2 ** self.n_upsample_blocks),
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=3)
        )
        modules_list.append(nn.Tanh())

        self.layers = nn.Sequential(*modules_list)

    def forward(self, spectrogram, **batch):
        return {"wave_pred": self.layers(spectrogram)}

