# model/model.py

import torch
import torch.nn as nn
import math


# --- Positional Encoding remains the same ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Input x shape: (Batch, Time, Features=d_model)
        # pe shape: (1, max_len, d_model)
        # We add positional encoding to the feature dimension
        # Ensure x and pe have compatible dimensions for broadcasting if necessary
        # Slicing pe ensures we only add encodings up to the sequence length of x
        return x + self.pe[:, :x.size(1)]


# --- Residual Attention Block - Modified FFN Width ---
class ResidualAttentionBlock(nn.Module):
    # Added ffn_multiplier argument
    def __init__(self, embed_dim, num_heads, dropout_rate, ffn_multiplier=4): # Increased default multiplier
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Make FFN hidden dimension controllable
        ffn_hidden_dim = int(embed_dim * ffn_multiplier)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim), # Wider intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, embed_dim), # Back to embed_dim
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (Batch, Time, Features=embed_dim)
        attn_output, _ = self.attn(x, x, x) # Query, Key, Value are all x
        # Residual connection 1 (Add & Norm)
        x = self.norm1(x + attn_output)
        # Feed Forward Network
        ffn_output = self.ffn(x)
        # Residual connection 2 (Add & Norm)
        x = self.norm2(x + ffn_output)
        return x


# --- WaveletCNNClassifier - MODIFIED CNN Width and uses updated Attention Block ---
class WaveletCNNClassifier(nn.Module):
    # Added cnn_channels and ffn_multiplier parameters
    def __init__(self, num_classes=4, dropout_rate=0.3, num_heads=4, # Kept num_heads=4 default
                 cnn_channels=(48, 96), # Example: Increased channels (1->48->96)
                 ffn_multiplier=3): # Example: Using 3x multiplier in Attention FFN
        super().__init__()

        # --- Parameter Sanity Check ---
        if not isinstance(cnn_channels, (list, tuple)) or len(cnn_channels) != 2:
             raise ValueError("cnn_channels must be a list or tuple of length 2, e.g., (48, 96)")
        ch1, ch2 = cnn_channels
        # --- End Check ---

        # Input CWT map shape: (B, 1, H=NumScales, T=TimeSteps)
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, ch1, kernel_size=(3, 3), padding=1), # Input channels = 1
            nn.ReLU(),
            nn.BatchNorm2d(ch1),
            # Layer 2
            nn.Conv2d(ch1, ch2, kernel_size=(3, 3), padding=1), # Input channels = ch1
            nn.ReLU(),
            nn.BatchNorm2d(ch2),
            # Pooling & Dropout
            nn.AdaptiveAvgPool2d((1, None)), # Average pool across the scale dimension -> (B, ch2, 1, T)
            nn.Dropout(dropout_rate) # Dropout after pooling
        )

        # The embedding dimension is now the output channel count of the CNN
        self.embed_dim = ch2
        self.pos_encoding = PositionalEncoding(d_model=self.embed_dim)

        # Instantiate the Attention Block with the potentially wider FFN
        self.attn_block = ResidualAttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            ffn_multiplier=ffn_multiplier # Pass the multiplier
        )

        # Classifier Head - input dimension matches embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2), # Example: hidden layer size relative to embed_dim
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim // 2, num_classes) # Output layer
        )

    def forward(self, x):
        # x: (B, 1, H, T) - Batch, Channel, Height(Scales), Width(Time)
        x = self.cnn(x)  # Output: (B, ch2, 1, T)
        # Squeeze the height dimension and permute to (B, T, Features=ch2)
        x = x.squeeze(2).permute(0, 2, 1) # Shape: (B, T, ch2) where ch2 = embed_dim

        # Add positional encoding
        x = self.pos_encoding(x) # Shape remains (B, T, embed_dim)

        # Pass through attention block
        x = self.attn_block(x) # Shape remains (B, T, embed_dim)

        # Classify each time step
        out = self.classifier(x)  # Output shape: (B, T, num_classes)
        return out

