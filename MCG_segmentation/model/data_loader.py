# model/data_loader.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D

# Added for type hinting clarity
from typing import Tuple

class ECGFullDataset(Dataset):
    """
    Dataset that loads ECG data, applies bandpass filtering, adds noise,
    and returns 1D sequences with labels for 1D CNN/RNN models.
    
    This version includes improved and additional augmentations while maintaining
    backward compatibility.
    """
    def __init__(
        self,
        data_dir: str,
        label_column: str = "train_label",
        file_extension: str = ".csv",
        sequence_length: int = 512,
        overlap: int = 256,
        # --- Original Augmentation Parameters (for backward compatibility) ---
        sinusoidal_noise_mag: float = 0.05,
        gaussian_noise_std: float = 0.02,
        baseline_wander_mag: float = 0.05,
        baseline_wander_freq_max: float = 0.5,
        augmentation_prob: float = 0.5,
        # --- New Augmentation Parameters (disabled by default) ---
        amplitude_scale_range: Tuple[float, float] = (1.0, 1.0),
        time_shift_ratio: float = 0.0,
    ):
        self.data_dir = data_dir
        self.label_column = label_column
        self.file_extension = file_extension
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Augmentation parameters
        self.sinusoidal_noise_mag = sinusoidal_noise_mag
        self.gaussian_noise_std = gaussian_noise_std
        self.baseline_wander_mag = baseline_wander_mag
        self.baseline_wander_freq_max = baseline_wander_freq_max
        self.augmentation_prob = augmentation_prob
        self.amplitude_scale_range = amplitude_scale_range
        self.time_shift_ratio = time_shift_ratio

        self.sequences = [] # List to store 1D sequences
        self.sequence_labels = [] # List to store labels for each sequence

        self._load_and_slice_all()

        if not self.sequences:
            logging.warning(f"Dataset initialization resulted in zero sequences. Check data_dir ('{data_dir}') and file contents.")

    def _load_and_slice_all(self):
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        files = [f for f in os.listdir(self.data_dir) if f.endswith(self.file_extension) and not f.startswith('.')]
        if not files:
            logging.error(f"No '{self.file_extension}' files found in {self.data_dir}")
            raise ValueError(f"No '{self.file_extension}' files found in {self.data_dir}")
        num_processed = 0
        for fname in files:
            try:
                file_path = os.path.join(self.data_dir, fname)
                df = pd.read_csv(file_path)
                
                labels = torch.tensor(df[self.label_column].values, dtype=torch.long)
                signal_pd = pd.to_numeric(df["wave_form"], errors='coerce')
                signal = torch.tensor(signal_pd.values, dtype=torch.float32)
                self._slice_channel_sequences(signal, labels)
                num_processed += 1
            except pd.errors.EmptyDataError: 
                logging.warning(f"Skipping {fname}: Empty file.")
            except Exception as e: 
                logging.error(f"Error loading/processing {fname}: {e}", exc_info=True)

        logging.info(f"Loaded and sliced {len(self.sequences)} sequences from {num_processed} files.")

    def _slice_channel_sequences(self, signal: torch.Tensor, labels: torch.Tensor):
        stride = self.sequence_length - self.overlap
        if stride <= 0:
            logging.error(f"Stride must be positive. Current stride: {stride}")
            return
        total_length = signal.shape[0]
        if total_length < self.sequence_length: 
            return
        for start in range(0, total_length - self.sequence_length + 1, stride):
            end = start + self.sequence_length
            seq = signal[start:end]
            seq_label = labels[start:end]
            self.sequences.append(seq)
            self.sequence_labels.append(seq_label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # --- 1. INITIALIZATION ---
        # Clone the signal to prevent augmentations from modifying the original dataset
        signal = self.sequences[idx].clone()
        labels = self.sequence_labels[idx]
        
        # --- 2. AUGMENTATIONS ---
        # Each augmentation is applied independently based on the augmentation probability.
        
        # --- Amplitude Scaling ---
        if self.amplitude_scale_range[0] != self.amplitude_scale_range[1] and torch.rand(1).item() < self.augmentation_prob:
            scale = torch.rand(1).item() * (self.amplitude_scale_range[1] - self.amplitude_scale_range[0]) + self.amplitude_scale_range[0]
            signal = signal * scale

        # --- Time Shifting ---
        if self.time_shift_ratio > 0 and torch.rand(1).item() < self.augmentation_prob:
            seq_len = signal.size(0)
            shift = int(torch.randint(-int(self.time_shift_ratio * seq_len), int(self.time_shift_ratio * seq_len) + 1, (1,)).item())
            signal = torch.roll(signal, shifts=shift, dims=0)

        # Time vector is created only if needed for time-based noise
        time = None

        # --- Baseline Wander (Original Implementation) ---
        if self.baseline_wander_mag > 0 and torch.rand(1).item() < self.augmentation_prob:
            if time is None: time = torch.linspace(0, 1, signal.size(0), device=signal.device)
            bw_noise = torch.zeros_like(signal)
            num_bw_components = torch.randint(1, 3, (1,)).item()
            for _ in range(num_bw_components):
                freq = torch.rand(1).item() * self.baseline_wander_freq_max
                amplitude = (torch.rand(1).item() - 0.5) * 2 * self.baseline_wander_mag
                phase = torch.rand(1).item() * 2 * math.pi
                bw_noise += amplitude * torch.sin(2 * math.pi * freq * time + phase)
            signal = signal + bw_noise

        # --- Gaussian Noise (Original Implementation) ---
        if self.gaussian_noise_std > 0 and torch.rand(1).item() < self.augmentation_prob:
            noise = torch.randn_like(signal) * self.gaussian_noise_std
            signal = signal + noise

        # --- Sinusoidal Noise (Original Implementation) ---
        if self.sinusoidal_noise_mag > 0 and torch.rand(1).item() < self.augmentation_prob:
            if time is None: time = torch.linspace(0, 1, signal.size(0), device=signal.device)
            sinusoidal_noise = torch.zeros_like(signal)
            num_components = torch.randint(2, 5, (1,)).item()
            for _ in range(num_components):
                freq = torch.randint(1, 30, (1,)).item()
                amplitude = torch.rand(1).item() * self.sinusoidal_noise_mag
                phase = torch.rand(1).item() * 2 * math.pi
                sinusoidal_noise += amplitude * torch.sin(2 * math.pi * freq * time + phase)
            signal = signal + sinusoidal_noise

        # --- 3. NORMALIZATION ---
        # Applied *after* all augmentations to the final signal. This is a key correction.
        
        # Normalize to zero mean
        signal_mean = signal.mean()
        if not torch.isnan(signal_mean) and not torch.isinf(signal_mean): 
            signal = signal - signal_mean

        # Normalize to [-1, 1] range for numerical stability
        max_val = signal.abs().max()
        if max_val > 1e-6: # Avoid division by zero
            signal = signal / max_val

        # --- 4. FINAL FORMATTING ---
        # Add channel dimension: (T) -> (1, T) for Conv1d layers
        final_signal_output = signal.unsqueeze(0)

        return final_signal_output, labels


# --- Updated __main__ block to demonstrate new features ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Colors for plotting labels on signal
    TEST_CLASS_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}

    try:
        print("--- Testing ECGFullDataset for 1D Output with Enhanced Augmentations ---")
        test_dataset = ECGFullDataset(
            data_dir="MCG_segmentation/Datasets/val", # Adjust path if needed
            overlap=400,
            sequence_length=500,
            augmentation_prob=1.0, # Set to 1.0 to guarantee augmentations for visualization
            # Original augmentations
            sinusoidal_noise_mag=0.05,
            gaussian_noise_std=0.04,
            baseline_wander_mag=0.1,
            # New augmentations
            amplitude_scale_range=(0.8, 1.2), # Enable amplitude scaling
            time_shift_ratio=0.1, # Enable time shifting by up to 10%
        )

        if len(test_dataset) == 0:
            print("Dataset is empty. Check path and data.")
        else:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            print(f"Dataset loaded with {len(test_dataset)} samples.")

            # Expecting 2 items: signal, labels
            signal_batch, labels_batch = next(iter(test_loader))

            print("\nShapes of first batch element:")
            print("Processed Signal Shape:", signal_batch.shape) # Should be (1, 1, T)
            print("Label Shape:", labels_batch.shape) # Should be (1, T)

            # Prepare for plotting
            signal_single = signal_batch.squeeze(0).squeeze(0).cpu().numpy()
            labels_single = labels_batch.squeeze(0).cpu().numpy()
            time_axis = np.arange(signal_single.shape[0])

            # Create plot
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fig.suptitle(f"Example 1D Output from ECGFullDataset (Seq Length: {test_dataset.sequence_length})", fontsize=14)

            # Plot 1: Processed Signal with Labels
            ax[0].plot(time_axis, signal_single, color='black', linewidth=1.0, label='Processed Signal')
            for t in range(signal_single.shape[0]):
                color = TEST_CLASS_COLORS.get(labels_single[t], 'magenta')
                ax[0].scatter(t, signal_single[t], color=color, s=15, zorder=3)

            ax[0].set_ylabel("Amplitude (Normalized)")
            ax[0].set_title("Augmented & Normalized Signal with Ground Truth Labels")
            ax[0].grid(True, linestyle=':', alpha=0.7)
            legend_elements = [Line2D([0], [0], color='black', lw=1, label='Signal')]
            
            for lbl, col in TEST_CLASS_COLORS.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Label {lbl}',
                                    markerfacecolor=col, markersize=8))
            ax[0].legend(handles=legend_elements, loc='upper right', fontsize='small')

            # Plot 2: Labels
            ax[1].plot(time_axis, labels_single, drawstyle='steps-post', label='Label Sequence', color='darkorange')
            ax[1].set_xlabel("Time Step (Sample Index)")
            ax[1].set_ylabel("Label Index")
            ax[1].set_title("Ground Truth Label Sequence")
            ax[1].grid(True, linestyle=':', alpha=0.7)
            ax[1].legend(loc='upper right', fontsize='small')
            ax[1].set_yticks(np.unique(labels_single))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    except FileNotFoundError as e: 
        print(f"Error: {e}. Check data directory path.")
    except ValueError as e: 
        print(f"ValueError: {e}.")
    except ImportError as e: 
        print(f"ImportError: {e}. Ensure required libraries are installed (pandas, torch, matplotlib, numpy).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()