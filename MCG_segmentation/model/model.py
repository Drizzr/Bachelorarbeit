import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
    def __init__(self, num_classes=4, input_channels=1, hidden_channels=32, 
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class DENS_ECG_segmenter(nn.Module):
    def __init__(self, input_channels=1, num_classes=4, dropout_rate=0.2):
        super(DENS_ECG_segmenter, self).__init__()
        
        # CNN layer parameters
        kernel_size = 3
        cnn_filters = [32, 64, 128]
        
        # BiLSTM layer parameters
        lstm_units = [250, 125]
        
        # First 1D Convolutional layer: input_channels -> 32 filters
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=cnn_filters[0],
            kernel_size=kernel_size,
            padding=kernel_size//2  # Zero padding to maintain input dimensions
        )
        self.relu1 = nn.ReLU()
        
        # Second 1D Convolutional layer: 32 -> 64 filters
        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters[0],
            out_channels=cnn_filters[1],
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.relu2 = nn.ReLU()
        
        # Third 1D Convolutional layer: 64 -> 128 filters
        self.conv3 = nn.Conv1d(
            in_channels=cnn_filters[1],
            out_channels=cnn_filters[2],
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        self.relu3 = nn.ReLU()
        
        # First BiLSTM layer: 128 -> 250 hidden units (bidirectional, so 125 per direction)
        self.bilstm1 = nn.LSTM(
            input_size=cnn_filters[2],
            hidden_size=lstm_units[0] // 2,  # Divide by 2 because bidirectional doubles it
            batch_first=True,
            bidirectional=True
        )
        
        # Second BiLSTM layer: 250 -> 125 hidden units (bidirectional, so 62/63 per direction)
        self.bilstm2 = nn.LSTM(
            input_size=lstm_units[0],
            hidden_size=lstm_units[1] // 2,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layer (time distributed)
        self.dense = nn.Linear(lstm_units[1], num_classes)
        
        # Softmax activation
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # Input shape: [batch_size, channels, sequence_length]
        
        # Apply CNN layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # BiLSTM expects input as [batch_size, sequence_length, features]
        x = x.permute(0, 2, 1)
        
        # Apply BiLSTM layers
        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply the dense layer (time distributed)
        # This is automatically applied to every time step thanks to PyTorch's broadcasting
        x = self.dense(x)
        
        # Apply softmax to obtain probabilities
        x = self.softmax(x)
        
        return x



if __name__ == "__main__":
    # Test-Code
    batch_size = 2
    seq_len = 100
    input_channels = 1
    num_classes = 4
    
    model = ECGSegmenter(
        num_classes=num_classes,
        input_channels=input_channels,
    )
    
    # Test-Input
    dummy_input = torch.randn(batch_size, input_channels, seq_len)
    
    # Forward Pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Sollte [B, L, num_classes] sein
    
    # Anzahl der Parameter
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Gesamtzahl Parameter: {parameters}")
    
    # Test mit Focal Loss
    criterion = FocalLoss(gamma=2.0)
    target = torch.randint(0, num_classes, (batch_size, seq_len))
    loss = criterion(output.view(-1, num_classes), target.view(-1))
    print(f"Loss: {loss.item()}")