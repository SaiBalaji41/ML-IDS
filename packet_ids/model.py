"""Hybrid 1D convolution + bidirectional LSTM classifier implemented in PyTorch."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from packet_ids import BYTE_LENGTH


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    conv_channels: int = 32
    lstm_hidden: int = 48
    dropout: float = 0.2
    byte_length: int = BYTE_LENGTH

    def __post_init__(self):
        if self.byte_length != BYTE_LENGTH:
            raise ValueError('The project contract requires exactly 1,024 input bytes.')
        if self.num_classes < 2 or self.conv_channels < 1 or self.lstm_hidden < 1:
            raise ValueError('Invalid model dimensions.')
        if not 0 <= self.dropout < 1:
            raise ValueError('Dropout must be in [0, 1).')

    def to_dict(self):
        return asdict(self)


class PacketCNNBiLSTM(nn.Module):
    """(batch, 1, 1024) -> 1D CNN -> (batch, 64, channels) -> BiLSTM -> logits.

    The LSTM processes ordered regions of the same packet. This is not an
    inter-packet temporal model. Bytes are never converted to 2D images.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config.conv_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(c, c * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.sequence = nn.LSTM(c * 2, config.lstm_hidden, batch_first=True,
                                bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1:] != (1, BYTE_LENGTH):
            raise ValueError('Expected a 1D byte signal tensor shaped (batch, 1, 1024).')
        encoded = self.encoder(inputs).transpose(1, 2)
        _, (hidden, _) = self.sequence(encoded)
        context = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(context)
