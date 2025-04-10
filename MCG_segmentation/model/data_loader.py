# model/data_loader.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Optional
import matplotlib.pyplot as plt
# import pywt # <-- REMOVED CWT related import
import math
from scipy.signal import butter, filtfilt
import torch.nn.functional as F
from matplotlib.lines import Line2D

# --- REMOVED compute_wavelet function ---

class ECGFullDataset(Dataset):
    """
    Dataset that loads ECG data, applies bandpass filtering, adds noise,
    and returns 1D sequences with labels for 1D CNN/RNN models.
    """
    def __init__(
        self,
        data_dir: str,
        channel_names: List[str] = ["ch1", "ch2"], # Still loads specified channels
        label_column: str = "train_label",
        file_extension: str = ".csv",
        sequence_length: int = 512,
        overlap: int = 256,
        # Noise parameters
        sinusoidal_noise_mag: float = 0.05, # Default from train.py args
        gaussian_noise_std: float = 0.02,
        baseline_wander_mag: float = 0.05,
        baseline_wander_freq_max: float = 0.5,
        amplitude_scale_range: float = 0.1,
        max_time_shift: int = 10,
        augmentation_prob: float = 0.5,
        # Filter parameters
        apply_filter: bool = True,
        filter_lowcut: float = 0.5,
        filter_highcut: float = 40.0,
        filter_order: int = 3,
        # --- REMOVED CWT parameters ---
        # transform=None,
        # wavelet: str = 'mexh',
        # wavelet_scales = ...
    ):
        self.data_dir = data_dir
        self.channel_names = channel_names # Which channels to load from CSV
        self.label_column = label_column
        self.file_extension = file_extension
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Augmentation parameters
        self.sinusoidal_noise_mag = sinusoidal_noise_mag
        self.gaussian_noise_std = gaussian_noise_std
        self.baseline_wander_mag = baseline_wander_mag
        self.baseline_wander_freq_max = baseline_wander_freq_max
        self.amplitude_scale_range = amplitude_scale_range
        self.max_time_shift = max_time_shift
        self.augmentation_prob = augmentation_prob

        # Filter parameters
        self.apply_filter = apply_filter
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_order = filter_order

        # Storage
        # Each list item will be a dict {'signal': tensor, 'label': tensor, 'fs': float}
        # Store data per channel separately initially before slicing maybe? Or handle channel selection later?
        # Let's stick to the current slicing approach but store fs.
        self.sequences = []         # List of Tensors of shape (T,) - NOW the signal itself
        self.sequence_labels = []   # List of Tensors of shape (T,)
        self.sequence_fs = []       # List of sampling frequencies

        self._load_and_slice_all()

        if not self.sequences:
            logging.warning(f"Dataset initialization resulted in zero sequences. Check data_dir ('{data_dir}') and file contents.")


    # _load_and_slice_all & _slice_channel_sequences remain the same as the previous version
    # They correctly store 1D signal segments in self.sequences now.
    def _load_and_slice_all(self):
        # ... (Identical to previous version - reads CSVs, infers fs, calls slice) ...
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        files = [f for f in os.listdir(self.data_dir) if f.endswith(self.file_extension) and not f.startswith('.')]
        if not files:
             logging.error(f"No '{self.file_extension}' files found in {self.data_dir}")
             raise ValueError(f"No {self.file_extension}' files found in {self.data_dir}")
        num_processed = 0
        for fname in files:
            try:
                file_path = os.path.join(self.data_dir, fname)
                df = pd.read_csv(file_path)
                fs = None
                if 'time' in df.columns and len(df['time']) > 1:
                    time_diffs = np.diff(df['time'].unique())
                    if len(time_diffs) > 0:
                         median_dt = np.median(time_diffs)
                         if median_dt > 1e-9: fs = 1.0 / median_dt
                         logging.debug(f"Inferred fs={fs:.2f} Hz for {fname}")
                if fs is None:
                     logging.warning(f"Could not infer fs for {fname}. Skipping."); continue
                fs = round(fs)
                required_cols = self.channel_names + [self.label_column]
                if not all(col in df.columns for col in required_cols):
                    missing_cols=[col for col in required_cols if col not in df.columns]
                    logging.warning(f"Skipping {fname}: Missing {missing_cols}"); continue
                labels = torch.tensor(df[self.label_column].values, dtype=torch.long)
                for ch in self.channel_names:
                    signal_pd = pd.to_numeric(df[ch], errors='coerce')
                    if signal_pd.isnull().any(): logging.warning(f"Skipping {ch} in {fname}: Non-numeric."); continue
                    signal = torch.tensor(signal_pd.values, dtype=torch.float32)
                    if signal.shape[0]!= labels.shape[0]: logging.warning(f"Skipping {ch} in {fname}: Length mismatch"); continue
                    self._slice_channel_sequences(signal, labels, fs)
                num_processed += 1
            except pd.errors.EmptyDataError: logging.warning(f"Skipping {fname}: Empty file.")
            except Exception as e: logging.error(f"Error loading/processing {fname}: {e}", exc_info=True)
        logging.info(f"Loaded and sliced {len(self.sequences)} sequences from {num_processed} files.")

    def _slice_channel_sequences(self, signal: torch.Tensor, labels: torch.Tensor, fs: float):
        """Slices 1D signal and labels, stores fs."""
        stride = self.sequence_length - self.overlap
        total_length = signal.shape[0]
        if total_length < self.sequence_length: return
        for start in range(0, total_length - self.sequence_length + 1, stride):
            end = start + self.sequence_length
            seq = signal[start:end]
            seq_label = labels[start:end]
            self.sequences.append(seq)
            self.sequence_labels.append(seq_label)
            self.sequence_fs.append(fs)

    def __len__(self):
        return len(self.sequences)

    # _apply_butter_bandpass_filter remains the same
    def _apply_butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        """Applies Butterworth bandpass filter."""
        nyq = 0.5 * fs; low = lowcut / nyq; high = highcut / nyq
        if low <= 0 or high >= 1.0: logging.warning(f"Invalid filter freq for fs={fs}. Skip."); return data
        try:
            b, a = butter(order, [low, high], btype='bandpass')
            data_np = data.numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
            padlen = 3 * order
            if len(data_np) <= padlen: logging.warning(f"Signal too short ({len(data_np)}) for filter order {order}. Skip."); return data
            y = filtfilt(b, a, data_np)
            return torch.from_numpy(y.copy()).to(dtype=data.dtype)
        except Exception as e: logging.error(f"Filter error: {e}"); return data

    # Modified __getitem__ to return processed 1D signal
    def __getitem__(self, idx):

        signal = self.sequences[idx] # This is the raw 1D sequence now
        labels = self.sequence_labels[idx]
        fs = self.sequence_fs[idx]

        # --- Apply Bandpass Filter ---
        if self.apply_filter:
            signal = self._apply_butter_bandpass_filter(
                signal, self.filter_lowcut, self.filter_highcut, fs, self.filter_order
            )

        # --- Apply Augmentations (Identical logic as before, applied to 1D signal) ---
        signal_processed = signal
        labels_processed = labels
        # Time Shift
        if self.max_time_shift > 0 and torch.rand(1).item() < self.augmentation_prob:
             shift = torch.randint(-self.max_time_shift, self.max_time_shift + 1, (1,)).item()
             if shift > 0:
                 signal_processed = F.pad(signal[:-shift], (shift, 0), value=signal[0].item())
                 labels_processed = F.pad(labels[:-shift], (shift, 0), value=labels[0].item())
             elif shift < 0:
                 shift_abs = -shift
                 signal_processed = F.pad(signal[shift_abs:], (0, shift_abs), value=signal[-1].item())
                 labels_processed = F.pad(labels[shift_abs:], (0, shift_abs), value=labels[-1].item())
        signal = signal_processed; labels = labels_processed # Update signal/labels if shifted

        # DC Offset
        signal_mean = signal.mean()
        if not torch.isnan(signal_mean) and not torch.isinf(signal_mean): signal = signal - signal_mean

        # Amplitude Scaling
        if self.amplitude_scale_range > 0 and torch.rand(1).item() < self.augmentation_prob:
            scale_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.amplitude_scale_range
            signal = signal * scale_factor

        # Normalize [-1, 1]
        max_val = signal.abs().max()
        if max_val > 1e-6: signal = signal / max_val

        # Baseline Wander
        noisy_signal = signal # Start noise addition here
        if self.baseline_wander_mag > 0 and torch.rand(1).item() < self.augmentation_prob:
             time = torch.linspace(0, 1, signal.size(0), device=signal.device)
             bw_noise = torch.zeros_like(signal)
             num_bw_components = torch.randint(1, 3, (1,)).item()
             for _ in range(num_bw_components):
                  freq = torch.rand(1).item() * self.baseline_wander_freq_max
                  amplitude = (torch.rand(1).item() - 0.5) * 2 * self.baseline_wander_mag
                  phase = torch.rand(1).item() * 2 * math.pi
                  bw_noise += amplitude * torch.sin(2 * math.pi * freq * time + phase)
             noisy_signal = noisy_signal + bw_noise

        # Gaussian Noise
        if self.gaussian_noise_std > 0 and torch.rand(1).item() < self.augmentation_prob:
            noise = torch.randn_like(noisy_signal) * self.gaussian_noise_std
            noisy_signal = noisy_signal + noise

        # Sinusoidal Noise
        if self.sinusoidal_noise_mag > 0 and torch.rand(1).item() < self.augmentation_prob:
            if 'time' not in locals(): time = torch.linspace(0, 1, signal.size(0), device=signal.device)
            sinusoidal_noise = torch.zeros_like(noisy_signal)
            num_components = torch.randint(2, 5, (1,)).item()
            for _ in range(num_components):
                freq = torch.randint(1, 30, (1,)).item()
                amplitude = torch.rand(1).item() * self.sinusoidal_noise_mag
                phase = torch.rand(1).item() * 2 * math.pi
                sinusoidal_noise += amplitude * torch.sin(2 * math.pi * freq * time + phase)
            noisy_signal = noisy_signal + sinusoidal_noise
        # --- End Augmentations ---


        # --- Prepare Output for 1D CNN Model ---
        # Add channel dimension: (1, T) - Conv1d expects (Batch, Channels, Length)
        final_signal_output = noisy_signal.unsqueeze(0)
        # --- Output is now (1, T), labels is (T,) ---

        # --- REMOVED CWT Calculation and related steps ---
        # tf_map = compute_wavelet(...)
        # tf_map_unsqueezed = tf_map.unsqueeze(0)
        # if self.transform: ...

        # Return the processed 1D signal (with channel dim) and labels
        return final_signal_output, labels


# --- if __name__ == "__main__": block needs updating ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Colors for plotting labels on signal
    TEST_CLASS_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}

    try:
        print("--- Testing ECGFullDataset for 1D Output ---")
        test_dataset = ECGFullDataset(
            data_dir="MCG_segmentation/qtdb/processed/val", # Adjust path
            overlap=125,
            sequence_length=250,
            # Enable augmentations for testing
            apply_filter=True,
            sinusoidal_noise_mag=0.05,
            gaussian_noise_std=0.02,
            baseline_wander_mag=0.05,
            amplitude_scale_range=0.1,
            max_time_shift=5,
            augmentation_prob=0.8,
            # CWT params are removed from __init__
        )

        if len(test_dataset) == 0:
            print("Dataset is empty.")
        else:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            print(f"Dataset loaded with {len(test_dataset)} samples.")

            # Expecting 2 items now: signal, labels
            signal_batch, labels_batch = next(iter(test_loader))

            print("\nShapes of first batch element:")
            print("Processed Signal Shape:", signal_batch.shape) # Should be (1, 1, T)
            print("Label Shape:", labels_batch.shape) # Should be (1, T)

            # Prepare for plotting
            signal_single = signal_batch.squeeze(0).squeeze(0).cpu().numpy() # Remove batch & channel dim
            labels_single = labels_batch.squeeze(0).cpu().numpy() # Remove batch dim
            time_axis = np.arange(signal_single.shape[0])

            # Create plot (simplified: Signal with Labels, Label Sequence)
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fig.suptitle(f"Example 1D Output from ECGFullDataset (Seq Length: {test_dataset.sequence_length})", fontsize=14)

            # Plot 1: Processed Signal with Labels
            ax[0].plot(time_axis, signal_single, color='black', linewidth=1.0, label='Processed Signal')
            for t in range(signal_single.shape[0]):
                color = TEST_CLASS_COLORS.get(labels_single[t], 'magenta')
                ax[0].scatter(t, signal_single[t], color=color, s=15, zorder=3)
            ax[0].set_ylabel("Amplitude (Normalized)")
            ax[0].set_title("Processed Signal with Ground Truth Labels (Color Dots)")
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

    except FileNotFoundError as e: print(f"Error: {e}. Check data directory path.")
    except ValueError as e: print(f"ValueError: {e}.")
    except ImportError as e: print(f"ImportError: {e}. Ensure required libraries are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()