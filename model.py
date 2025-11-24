import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import model_config as config

class MelodyCRNN(nn.Module):
    def __init__(self):
        super(MelodyCRNN, self).__init__()

        # cnn modules
        # pooling added to reduce output size
        self.cnn_modules = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # bi-lstm module
        self.lstm_input_features = (config['cqt_bins'] // 4) * 64
        self.lstm_output_size = config['lstm_hidden_size'] * 2

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=config['lstm_hidden_size'],
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3 output heads for chroma, octave and svd classification
        def output_head(output_classes: int, hidden_size: int):
            return nn.Sequential(
                nn.Linear(self.lstm_output_size, hidden_size),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(hidden_size, output_classes)
            )

        self.chroma_head = output_head(config['chroma_classes'], config['fc_hidden_size'])
        self.octave_head = output_head(config['octave_classes'], config['fc_hidden_size'])
        self.voicing_head = output_head(config['voicing_classes'], config['fc_hidden_size'])


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = x.shape[0]

        # (b, 64, 91, 517)
        cnn_out = self.cnn_modules(x)

        # collapse along frequency
        # (b, 517, 64 * 91)
        lstm_input = cnn_out.permute(0, 3, 1, 2).contiguous()
        lstm_input = lstm_input.view(batch_size, lstm_input.shape[1], self.lstm_input_features)

        # (b, 517, 2 * 128)
        lstm_out, _ = self.lstm(lstm_input)
        
        # (b, 517, 12; 4; 2)
        chroma_logits = self.chroma_head(lstm_out)
        octave_logits = self.octave_head(lstm_out)
        voicing_logits = self.voicing_head(lstm_out)

        return chroma_logits, octave_logits, voicing_logits
