import torch
import torch.nn as nn
import math


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


# --- Transformer Encoder Block ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# --- New Model with Attention ---
class Conv1D_Attention_Segmenter(nn.Module):
    def __init__(self, num_classes=4,
                 input_channels=1,
                 cnn_filters=(16, 32, 64),  # Reduced CNN filters
                 cnn_kernel_size=3,
                 attention_dim=128,         # Reduced attention dimension
                 num_heads=4,
                 num_transformer_layers=2, # Reduced number of transformer layers
                 dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        current_channels = input_channels

        # --- CNN Layers ---
        cnn_layers = []
        padding = cnn_kernel_size // 2
        for num_filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(current_channels, num_filters, kernel_size=cnn_kernel_size, padding=padding),
                nn.ReLU()
            ])
            current_channels = num_filters
        self.cnn_base = nn.Sequential(*cnn_layers)

        # --- Projection to Attention Dim ---
        self.project_to_attention = nn.Linear(current_channels, attention_dim)

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(attention_dim)

        # --- Transformer Blocks ---
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(d_model=attention_dim, num_heads=num_heads, dropout=dropout_rate)
            for _ in range(num_transformer_layers)
        ])

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout_rate)

        # --- Classifier ---
        self.classifier = nn.Linear(attention_dim, num_classes)

        print(f"--- Initialized Conv1D_Attention_Segmenter ---")
        print(f"  CNN Filters: {cnn_filters}, Kernel: {cnn_kernel_size}")
        print(f"  Attention Dim: {attention_dim}, Heads: {num_heads}, Layers: {num_transformer_layers}")
        print(f"  Output Classes: {num_classes}")
        print(f"---------------------------------------------")

    def forward(self, x):
        # Input shape: (B, C=1, T)
        x = self.cnn_base(x)  # -> (B, C_out, T)
        x = x.permute(0, 2, 1)  # -> (B, T, Features)

        # Project to attention dimension
        x = self.project_to_attention(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder blocks
        x = self.transformer_blocks(x)

        # Dropout + Classifier
        x = self.dropout(x)
        logits = self.classifier(x)  # -> (B, T, num_classes)
        return logits


# --- Count Parameters ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Test ---
if __name__ == "__main__":
    print("\n--- Testing Conv1D_Attention_Segmenter ---")
    NUM_CLASSES = 4
    TIME_STEPS = 250
    BATCH_SIZE = 16
    INPUT_CHANNELS = 1

    model = Conv1D_Attention_Segmenter(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        cnn_filters=(16, 32, 64),
        cnn_kernel_size=3,
        attention_dim=128,
        num_heads=4,
        num_transformer_layers=2,
        dropout_rate=0.2
    )

    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Output Shape: {logits.shape}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print("-" * 40)
