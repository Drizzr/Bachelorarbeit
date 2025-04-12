import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Conv1D_BiLSTM_Segmenter(nn.Module):
    def __init__(self, num_classes=4,
                input_channels=1,
                cnn_filters=(16, 32, 64),
                cnn_kernel_size=3,
                lstm_units=(100, 50),
                dropout_rate=0.55,
                max_seq_len=5000):
        
        super().__init__()
        self.num_classes = num_classes
        current_channels = input_channels

        cnn_layers = []
        padding = cnn_kernel_size // 2
        for num_filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(current_channels, num_filters,
                        kernel_size=cnn_kernel_size, padding=padding, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            current_channels = num_filters
        self.cnn_base = nn.Sequential(*cnn_layers)

        self.pos_encoder = PositionalEncoding(d_model=current_channels, max_len=max_seq_len)

        self.bilstm1 = nn.LSTM(input_size=current_channels,
                               hidden_size=lstm_units[0],
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

        self.bilstm2 = nn.LSTM(input_size=lstm_units[0] * 2,
                               hidden_size=lstm_units[1],
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

        final_features = lstm_units[1] * 2
        self.classifier = nn.Linear(final_features, num_classes)

        print(f"--- Initialized SMALLER Conv1D_BiLSTM_Segmenter with Positional Encoding ---")
        print(f"  Input Channels: {input_channels}")
        print(f"  CNN Filters: {cnn_filters}, Kernel: {cnn_kernel_size}")
        print(f"  BiLSTM Units: {lstm_units}")
        print(f"  Dropout Rate: {dropout_rate}")
        print(f"  Output Classes: {num_classes}")
        print(f"  Max Seq Length: {max_seq_len}")
        print(f"---------------------------------------------------------")

    def forward(self, x):
        x = self.cnn_base(x)                # (batch, channels, seq_len)
        x = x.permute(0, 2, 1)              # (batch, seq_len, features)
        x = self.pos_encoder(x)             # Add positional info
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        x = self.dropout(x)
        logits = self.classifier(x)         # (batch, seq_len, num_classes)
        return logits


if __name__ == "__main__":
    # Simple test
    batch_size = 2
    seq_len = 300
    input_channels = 1
    model = Conv1D_BiLSTM_Segmenter(num_classes=4,
                                    input_channels=input_channels,
                                    max_seq_len=seq_len)

    dummy_input = torch.randn(batch_size, input_channels, seq_len)  # (B, C, L)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expecting (B, L, num_classes)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {parameters}")