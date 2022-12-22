import torch
from torch import nn

from hw_asr.base import BaseModel
from torch.nn.utils import weight_norm, spectral_norm


class sub_MPD(nn.Module):
    def __init__(self, period):
        super(sub_MPD, self).__init__()

        self.period = period
        self.relu = nn.LeakyReLU(0.1)
        modules_list = []

        for l in range(4):
            modules_list.append(
                weight_norm(
                nn.Conv2d(in_channels=1 if l == 0 else 2 ** (5 + l),
                          out_channels=2 ** (5 + l + 1),
                          kernel_size=(5, 1),
                          stride=(3, 1),
                          padding=(2, 0)))
            )
        modules_list.append(
            weight_norm(
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=(5, 1),
                      stride=(1, 1),
                      padding=(2, 0)))
        )
        modules_list.append(
            weight_norm(
            nn.Conv2d(in_channels=1024,
                      out_channels=1,
                      kernel_size=(3, 1),
                      stride=(1, 1),
                      padding=(1, 0)))
        )

        self.layers = nn.ModuleList(modules_list)

    def forward(self, x):
        # reshape + pad
        batch_size, n_channels, time_dim = x.shape
        pad_add = time_dim % self.period
        if pad_add > 0:
            pad_add = self.period - pad_add
        x_padded = torch.zeros(batch_size, n_channels, time_dim + pad_add, device=x.device)
        x_padded[:, :, :time_dim] = x
        time_dim += pad_add
        x = x_padded.view(batch_size, n_channels, time_dim // self.period, self.period)

        # forward
        features = []
        for i in range(6):
            x = self.layers[i](x)
            if i < 5:
                x = self.relu(x)
            features.append(x)
        return torch.flatten(x, start_dim=1, end_dim=-1), features


class MPD(BaseModel):
    def __init__(self, **batch):
        super().__init__(**batch)

        self.sub_MPD_list = nn.ModuleList([
            sub_MPD(2), sub_MPD(3), sub_MPD(5), sub_MPD(7), sub_MPD(11)
        ])

    def forward(self, wave, **batch):
        logits = []
        features = []
        for layer in self.sub_MPD_list:
            cur_logits, cur_features = layer(wave)
            logits.append(cur_logits)
            features.append(cur_features)
        return logits, features


class sub_MSD(nn.Module):
    def __init__(self, is_raw=False):
        super(sub_MSD, self).__init__()

        self.relu = nn.LeakyReLU(0.1)

        if is_raw:
            cur_norm = spectral_norm
        else:
            cur_norm = weight_norm

        self.layers = nn.ModuleList([
            cur_norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
            cur_norm(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
            cur_norm(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
            cur_norm(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            cur_norm(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            cur_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
            cur_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
            cur_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        ])

    def forward(self, x):
        features = []
        for i in range(8):
            x = self.layers[i](x)
            if i < 7:
                x = self.relu(x)
            features.append(x)
        return torch.flatten(x, start_dim=1, end_dim=-1), features


class MSD(BaseModel):
    def __init__(self, **batch):
        super().__init__(**batch)

        self.sub_MSD_list = nn.ModuleList([
            sub_MSD(is_raw=True), sub_MSD(is_raw=False), sub_MSD(is_raw=False)
        ])

        self.poolings = nn.ModuleList([
            nn.Identity(), nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, wave, **batch):
        logits = []
        features = []
        for ind, layer in enumerate(self.sub_MSD_list):
            # ind == 0: raw audio
            # ind == 1: x2 pooled
            # ind == 2: x4 (x2 x2) pooled
            wave = self.poolings[ind](wave)

            cur_logits, cur_features = layer(wave)
            logits.append(cur_logits)
            features.append(cur_features)
        return logits, features
