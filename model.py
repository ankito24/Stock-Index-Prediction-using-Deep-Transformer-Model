"""
model.py
Implements the Transformer encoder-decoder model for stock index prediction
(Section 4 of the paper), plus CNN / RNN / LSTM baselines used for comparison
(Section 3 of the paper, Table 2/3/4/5 comparisons).
"""

import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Positional Encoding (Eq. 3 in the paper)
# --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


# --------------------------------------------------------------------------
# Transformer model (Section 4, Fig. 3 in the paper)
# --------------------------------------------------------------------------
class StockTransformer(nn.Module):
    """
    Encoder-decoder Transformer for next-day closing price prediction.

    Encoder input: previous T closing prices.
    Decoder input: last N (< T) closing prices (paper uses N = 2), no mask
                   since decoder inputs are historical/observed data only.
    Output: predicted price at time T+1.
    """

    def __init__(self, T=9, N=2, d_model=32, n_heads=4, num_layers=2,
                 dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.T = T
        self.N = N
        self.d_model = d_model

        self.input_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, T)  -- previous T closing prices
        enc_in = x.unsqueeze(-1)                      # (batch, T, 1)
        enc_in = self.input_embedding(enc_in)          # (batch, T, d_model)
        enc_in = self.pos_encoding(enc_in)
        memory = self.encoder(enc_in)                  # (batch, T, d_model)

        # Decoder input = last N observed prices (paper: N = 2, no mask used)
        dec_in = x[:, -self.N:].unsqueeze(-1)           # (batch, N, 1)
        dec_in = self.input_embedding(dec_in)
        dec_in = self.pos_encoding(dec_in)

        out = self.decoder(tgt=dec_in, memory=memory)   # (batch, N, d_model)
        out = self.output_layer(out[:, -1, :])           # (batch, 1) -> use last step
        return out.squeeze(-1)


# --------------------------------------------------------------------------
# CNN baseline (Section 3.1 / Fig. 1)
# --------------------------------------------------------------------------
class StockCNN(nn.Module):
    def __init__(self, T=9, d=64, k=4, kernel_size=3):
        super().__init__()
        self.fc_in = nn.Linear(1, d)
        self.conv = nn.Conv1d(in_channels=d, out_channels=k, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        conv_out_len = T - kernel_size + 1
        pool_out_len = conv_out_len // 2
        self.fc_out = nn.Linear(k * pool_out_len, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (batch, T)
        h = self.fc_in(x.unsqueeze(-1))         # (batch, T, d)
        h = h.permute(0, 2, 1)                  # (batch, d, T)
        h = self.act(self.conv(h))              # (batch, k, T-2)
        h = self.pool(h)                        # (batch, k, (T-2)//2)
        h = h.flatten(1)
        out = self.fc_out(h)
        return out.squeeze(-1)


# --------------------------------------------------------------------------
# RNN baseline
# --------------------------------------------------------------------------
class StockRNN(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h, _ = self.rnn(x.unsqueeze(-1))
        out = self.fc(h[:, -1, :])
        return out.squeeze(-1)


# --------------------------------------------------------------------------
# LSTM baseline (Section 3.2, Eq. 2)
# --------------------------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h, _ = self.lstm(x.unsqueeze(-1))
        out = self.fc(h[:, -1, :])
        return out.squeeze(-1)


def build_model(name: str, T: int = 9, N: int = 2):
    name = name.lower()
    if name == "transformer":
        return StockTransformer(T=T, N=N)
    elif name == "cnn":
        return StockCNN(T=T)
    elif name == "rnn":
        return StockRNN()
    elif name == "lstm":
        return StockLSTM()
    else:
        raise ValueError(f"Unknown model name: {name}")
