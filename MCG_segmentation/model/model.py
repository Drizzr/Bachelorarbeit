# model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#########################
## ECG_segmentator architecture
#########################


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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class ECGSegmenter(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, hidden_channels=32,  # 32, 64 for xl
                lstm_hidden=64, dropout_rate=0.3, max_seq_len=2000):
        super().__init__()
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=input_channels, max_len=max_seq_len)
        
        # Initial convolution to increase channels
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-scale feature extraction with different kernel sizes
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=k, padding=k//2)
            for k in [3, 7, 15]  # Different receptive fields
        ])
        
        # Combine multi-scale features
        self.combine_scales = nn.Sequential(
            nn.Conv1d(hidden_channels * 3, hidden_channels * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks with increasing dilation
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels * 2, kernel_size=3, dilation=2**i, dropout=dropout_rate)
            for i in range(4)  # Dilations: 1, 2, 4, 8
        ])
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=hidden_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Self-attention
        self.self_attn = TransformerEncoderLayer(
            d_model=lstm_hidden * 2,
            nhead=8,
            dropout=dropout_rate
        )
        self.transformer = TransformerEncoder(self.self_attn, num_layers=1)
        
        # Skip connection from earlier in the network
        self.skip_connection = nn.Linear(hidden_channels, lstm_hidden * 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden * 2, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch_size, channels, seq_len]
        batch_size, _, seq_len = x.shape
        
        # Positional encoding
        x_pos = x.permute(0, 2, 1)  # [B, L, C]
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.permute(0, 2, 1)  # [B, C, L]

        
        # Initial convolution
        x = self.initial_conv(x)

        # Save for skip connection
        skip_features = x
        
        # Multi-scale feature extraction
        multi_scale_outputs = [conv(x) for conv in self.multi_scale]
        x = torch.cat(multi_scale_outputs, dim=1)

        # Combine scales
        x = self.combine_scales(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        
        # BiLSTM expects [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [B, L, C]
        
        # Apply BiLSTM
        x, _ = self.bilstm(x)
        
  
        # Self-attention
        x = self.transformer(x)
        
        # Skip connection from earlier features
        skip_features = skip_features.permute(0, 2, 1)  # [B, L, C]
        skip_features = self.skip_connection(skip_features)
        
        # Combine with skip connection
        x = x + skip_features
        
        # Final classifier
        logits = self.classifier(x)
        
        return logits


############################
# DENS_ECG_segmenter architecture
############################


class DENS_ECG_segmenter(nn.Module):
    def __init__(self, input_channels=1, num_classes=4):
        super(DENS_ECG_segmenter, self).__init__()

        # 1D Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # BiLSTM Layers
        self.bilstm1 = nn.LSTM(input_size=128, hidden_size=250, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=500, hidden_size=125, batch_first=True, bidirectional=True)

        # Dropout
        self.dropout = nn.Dropout(p=0.2)

        # TimeDistributed Dense Layer (implemented as a Linear layer over each time step)
        self.classifier = nn.Linear(250, num_classes)  # 2*125 from BiLSTM2

    def forward(self, x):
        # x: (batch_size, time_steps, features)
        # For Conv1D: (batch_size, channels, time_steps)
        x = F.relu(self.conv1(x))  # -> (batch, 32, time)
        x = F.relu(self.conv2(x))  # -> (batch, 64, time)
        x = F.relu(self.conv3(x))  # -> (batch, 128, time)

        x = x.permute(0, 2, 1)  # Convert to (batch_size, channels, time_steps) => (B, C, T)

        x, _ = self.bilstm1(x)  # -> (batch, time, 500)
        x, _ = self.bilstm2(x)  # -> (batch, time, 250)

        x = self.dropout(x)

        x = self.classifier(x)  # -> (batch, time, num_classes)
        x = F.softmax(x, dim=-1)

        return x


#############################
# UNet1D architecture with Multi-Head Self-Attention
##############################


class ConvBlock1D(nn.Module):
    """1D Convolutional block with two conv layers, batch norm, and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class MHSA1D(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)     # drop out nur für xl version
    
    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        return x.permute(0, 2, 1)  # Back to (B, C, T)


class UNet1D(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, features=[64, 128, 256, 512], dropout=0.4, num_heads=8):  # 900k dropout = 0. , num heads = 4 f, [32, 64, 128]
        super().__init__()
        
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(2)
        
        # Encoder
        in_ch = input_channels
        for feat in features:
            self.encoder_blocks.append(ConvBlock1D(in_ch, feat))
            in_ch = feat
        
        # Bottleneck + MHSA
        self.bottleneck_conv = ConvBlock1D(features[-1], features[-1] * 2)
        self.mhsa = MHSA1D(features[-1] * 2, num_heads=num_heads, dropout=dropout)  
        
        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = features[-1] * 2  # Start from bottleneck output channels
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose1d(in_ch, feat, kernel_size=2, stride=2))
            # After concatenation with skip connection: feat + feat = feat * 2
            self.decoder_blocks.append(ConvBlock1D(feat * 2, feat))
            in_ch = feat
        
        # Final output layer
        self.final_conv = nn.Conv1d(features[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        
        # Bottleneck with attention
        x = self.bottleneck_conv(x)
        x = self.mhsa(x)

        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        

        for idx in range(len(self.upconvs)):
            
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            # Handle size mismatch by padding the smaller tensor
            if x.shape[-1] != skip.shape[-1]:
                diff = abs(x.shape[-1] - skip.shape[-1])
                if x.shape[-1] < skip.shape[-1]:
                    x = F.pad(x, (0, diff))
                else:
                    skip = F.pad(skip, (0, diff))
            
            # Concatenate skip connection
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_blocks[idx](x)
        
        # Final output
        x = self.final_conv(x)
        return x.permute(0, 2, 1)  # (B, C, T) → (B, T, C)


if __name__ == "__main__":
    # Test code
    batch_size = 2
    seq_len = 100
    input_channels = 1
    num_classes = 4
    
    model = UNet1D(num_classes=num_classes, input_channels=input_channels) #32, 64, 128
    #model = ECGSegmenter(num_classes = 4, input_channels = 1, hidden_channels = 16, lstm_hidden = 20, dropout_rate = 0.3, max_seq_len = 2000)
    # Test input
    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [B, T, num_classes]
    
    # Parameter count
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {parameters:,}")