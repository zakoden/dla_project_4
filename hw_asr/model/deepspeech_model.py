from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeechModel(BaseModel):
    def __init__(self, n_feats, n_class, rnn_hidden=64, seq_batch_norm=False, **batch):
        super().__init__(n_feats, n_class, **batch)

        # From paper:
        # Architecture | Channels   | Filter dimension    | Stride
        # 2-layer 2D   | 32, 32     | 41x11, 21x11        | 2x2, 2x1
        # 3-layer 2D   | 32, 32, 96 | 41x11, 21x11, 21x11 | 2x2, 2x1, 2x1
        #
        # 2 layers model:
        self.conv_block = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        rnn_input_size = 32 * (n_feats // 4)
        self.rnn_block = nn.GRU(rnn_input_size, hidden_size=rnn_hidden, num_layers=7,
                                batch_first=True, bidirectional=True)

        if seq_batch_norm:
            self.seq_batch_norm = nn.BatchNorm1d(2 * rnn_hidden)
        else:
            self.seq_batch_norm = None

        self.fc = Sequential(
            nn.Linear(in_features=2 * rnn_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):
        x = spectrogram[:, None, :, :]
        # x (batch_size, n_channels, feature_length, max_time)

        x = self.conv_block(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        # x (batch_size, new_feature_length, max_time)

        x = x.transpose(1, 2)
        # x (batch_size, max_time, new_feature_length)

        x = self.rnn_block(x)[0]
        # x (batch_size, max_time, 2 * rnn_hidden)

        if self.seq_batch_norm is not None:
            a, b, c = x.shape[0], x.shape[1], x.shape[2]
            x = x.reshape((a * b, c))
            x = self.seq_batch_norm(x)
            x = x.reshape((a, b, c))

        x = self.fc(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - 1) // 2 + 1
