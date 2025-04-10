# model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F # Keep for potential use
import math

# --- Keep other models if needed for reference ---
# class PositionalEncoding...
# class ResidualAttentionBlock...
# class WaveletCNNClassifier...
# class WaveletCNN_LSTM_Segmenter...

# --- Model based on the text description (1D CNN -> BiLSTM -> Dense) ---
class Conv1D_BiLSTM_Segmenter(nn.Module):
    def __init__(self, num_classes=4, # Number of output classes
                 input_channels=1, # Usually 1 for single-lead ECG segment
                 cnn_filters=(32, 64, 128), # Filters for the 3 conv layers
                 cnn_kernel_size=3,
                 lstm_units=(250, 125), # Hidden units for the 2 BiLSTM layers
                 dropout_rate=0.2): # Dropout probability after LSTMs
        super().__init__()
        self.num_classes = num_classes
        current_channels = input_channels

        # --- 1D Convolutional Layers ---
        cnn_layers = []
        padding = cnn_kernel_size // 2 # Maintain length with stride 1
        for num_filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(current_channels, num_filters,
                          kernel_size=cnn_kernel_size, padding=padding, bias=True), # Bias usually True for Conv1d
                # Consider adding BatchNorm1d here? Description doesn't explicitly mention it, but often beneficial.
                # nn.BatchNorm1d(num_filters),
                nn.ReLU() # Standard activation, description doesn't specify one for CNN
            ])
            current_channels = num_filters
        self.cnn_base = nn.Sequential(*cnn_layers)
        # --- End CNN ---

        # --- BiLSTM Layers ---
        lstm_input_size = current_channels # Output channels from last CNN layer
        self.bilstm1 = nn.LSTM(input_size=lstm_input_size,
                               hidden_size=lstm_units[0],
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

        self.bilstm2 = nn.LSTM(input_size=lstm_units[0] * 2, # Input is output of first BiLSTM
                               hidden_size=lstm_units[1],
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)
        # --- End BiLSTM ---

        # --- Dropout Layer ---
        # Applied after the BiLSTM layers before the final classifier
        self.dropout = nn.Dropout(dropout_rate)
        # --- End Dropout ---

        # --- Time Distributed Dense Layer ---
        # Apply a Linear layer to each time step of the final LSTM output
        final_lstm_output_features = lstm_units[1] * 2 # Output of second BiLSTM
        self.classifier = nn.Linear(final_lstm_output_features, num_classes)
        # --- End Classifier ---

        print(f"--- Initialized Conv1D_BiLSTM_Segmenter ---")
        print(f"  Input Channels: {input_channels}")
        print(f"  CNN Filters: {cnn_filters}, Kernel: {cnn_kernel_size}")
        print(f"  BiLSTM Units: {lstm_units}")
        print(f"  Dropout Rate: {dropout_rate}")
        print(f"  Output Classes: {num_classes}")
        print(f"---------------------------------------------")


    def forward(self, x):
        # Input x shape: (Batch, Channels=1, Time)
        # print(f"Input: {x.shape}")

        # CNN Base
        x = self.cnn_base(x) # Output: (Batch, last_cnn_filter, Time)
        # print(f"After CNN: {x.shape}")

        # Prepare for LSTM: (Batch, Time, Features)
        x = x.permute(0, 2, 1) # (Batch, Time, last_cnn_filter)
        # print(f"Before LSTM1: {x.shape}")

        # BiLSTM Layers
        x, _ = self.bilstm1(x) # Output: (Batch, Time, lstm_units[0]*2)
        # print(f"After LSTM1: {x.shape}")
        x, _ = self.bilstm2(x) # Output: (Batch, Time, lstm_units[1]*2)
        # print(f"After LSTM2: {x.shape}")

        # Dropout
        x = self.dropout(x) # Apply dropout to LSTM output features

        # Time Distributed Classifier
        # Apply linear layer to each time step's feature vector
        # Input: (Batch, Time, lstm_units[1]*2) -> Output: (Batch, Time, num_classes)
        logits = self.classifier(x)
        # print(f"Logits: {logits.shape}")

        # Softmax is typically applied in the loss function (CrossEntropyLoss)
        return logits

# --- Function to count parameters ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Testing Conv1D_BiLSTM_Segmenter ---")
    NUM_CLASSES = 4
    TIME_STEPS = 250 # Example sequence length
    BATCH_SIZE = 16
    INPUT_CHANNELS = 1 # Single channel (lead)

    # Instantiate the model using defaults from description
    model_test = Conv1D_BiLSTM_Segmenter(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        cnn_filters=(32, 64, 128),
        cnn_kernel_size=3,
        lstm_units=(250, 125),
        dropout_rate=0.2
    )

    dummy_input_1d = torch.randn(BATCH_SIZE, INPUT_CHANNELS, TIME_STEPS)

    model_test.eval()
    with torch.no_grad():
        output_logits = model_test(dummy_input_1d)

    params_count = count_parameters(model_test)
    print(f"\nInput Shape (1D Signal): {dummy_input_1d.shape}")
    print(f"Output Logits Shape: {output_logits.shape}") # Should be (BATCH_SIZE, TIME_STEPS, NUM_CLASSES)
    print(f"Total Trainable Parameters: {params_count:,}") # Should be >> 1M

    print("-" * 40)