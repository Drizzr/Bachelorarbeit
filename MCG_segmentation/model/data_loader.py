# model/data_loader.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Optional
import matplotlib.pyplot as plt
import pywt
import math
from scipy.signal import butter, filtfilt # <-- Import SciPy filter functions
import torch.nn.functional as F # Needed for time shift padding
from matplotlib.lines import Line2D

# --- 1. Wavelet Transform Function ---
# (compute_wavelet function remains the same)
def compute_wavelet(signal, wavelet='morl', scales=np.arange(1, 64)):
    """
    Compute Continuous Wavelet Transform for 1D ECG signal.
    signal: (T,) tensor or numpy array
    Returns: (num_scales, T) time-frequency map
    """
    # Ensure signal is numpy array on CPU for pywt
    if isinstance(signal, torch.Tensor):
        signal_np = signal.detach().cpu().numpy()
    else:
        signal_np = np.asarray(signal) # Ensure it's a numpy array

    if signal_np.ndim != 1:
        raise ValueError(f"Input signal must be 1D, but got shape {signal_np.shape}")
    if len(signal_np) < 2:
         # pywt.cwt might fail or give meaningless results on very short signals
         # Return zeros or handle as appropriate
        num_scales = len(scales)
        return torch.zeros((num_scales, len(signal_np)), dtype=torch.float32)

    try:
        coef, _ = pywt.cwt(signal_np, scales, wavelet)
        return torch.from_numpy(coef.astype(np.float32)) # Ensure float32
    except Exception as e:
        logging.error(f"Error during CWT computation: {e}")
        # Return zeros or raise error depending on desired behavior
        num_scales = len(scales)
        return torch.zeros((num_scales, len(signal_np)), dtype=torch.float32)


class ECGFullDataset(Dataset):
    """
    Dataset that loads ECG data, applies bandpass filtering, adds noise,
    computes CWT, and returns sequences with labels.
    """
    def __init__(
        self,
        data_dir: str,
        channel_names: List[str] = ["ch1", "ch2"],
        label_column: str = "train_label",
        file_extension: str = ".csv",
        sequence_length: int = 512,
        overlap: int = 256,
        # Noise parameters
        sinusoidal_noise_mag: float = 0.01,
        gaussian_noise_std: float = 0.02, # Added Gaussian noise control
        baseline_wander_mag: float = 0.05, # Added Baseline Wander control
        baseline_wander_freq_max: float = 0.5, # Max freq (relative if time=0->1)
        amplitude_scale_range: float = 0.1, # Added Amplitude Scaling control
        max_time_shift: int = 10, # Added Time Shift control
        augmentation_prob: float = 0.5, # Probability to apply each augmentation
        # Filter parameters
        apply_filter: bool = True, # Control whether to apply the filter
        filter_lowcut: float = 0.5,
        filter_highcut: float = 40.0,
        filter_order: int = 3,
        # CWT parameters
        transform=None, # Transform applied to the final CWT map
        wavelet: str = 'mexh',
        wavelet_scales = np.logspace(np.log2(2), np.log2(128), num=64, base=2.0)
    ):
        self.data_dir = data_dir
        self.channel_names = channel_names
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
        self.augmentation_prob = augmentation_prob # Store probability

        # Filter parameters
        self.apply_filter = apply_filter
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_order = filter_order

        # CWT parameters
        self.transform = transform
        self.wavelet = wavelet
        self.wavelet_scales = wavelet_scales

        self.sequences = []         # List of Tensors of shape (T,)
        self.sequence_labels = []   # List of Tensors of shape (T,)
        self.sequence_fs = []       # <-- Store sampling frequency for each sequence

        self._load_and_slice_all()

        if not self.sequences:
            logging.warning(f"Dataset initialization resulted in zero sequences. Check data_dir ('{data_dir}') and file contents.")


    def _load_and_slice_all(self):
        if not os.path.exists(self.data_dir):
            # Use logging instead of raising error immediately? Or keep error?
            logging.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        files = [f for f in os.listdir(self.data_dir) if f.endswith(self.file_extension) and not f.startswith('.')]
        if not files:
             logging.error(f"No '{self.file_extension}' files found in {self.data_dir}")
             raise ValueError(f"No {self.file_extension} files found in {self.data_dir}")

        num_processed = 0
        for fname in files:
            try:
                file_path = os.path.join(self.data_dir, fname)
                df = pd.read_csv(file_path)

                # Infer sampling frequency (fs) from the time column if possible
                fs = None
                if 'time' in df.columns and len(df['time']) > 1:
                    # Calculate median diff, handle potential NaNs or duplicates
                    time_diffs = np.diff(df['time'].unique())
                    if len(time_diffs) > 0:
                        median_dt = np.median(time_diffs)
                        if median_dt > 1e-9: # Avoid division by zero
                            fs = 1.0 / median_dt
                            logging.debug(f"Inferred fs={fs:.2f} Hz from time column for {fname}")

                if fs is None:
                    logging.warning(f"Could not infer sampling frequency (fs) from time column for {fname}. Skipping file.")
                    continue # Cannot proceed without fs for filtering

                # Round fs to avoid precision issues if necessary, e.g., 249.99 -> 250
                fs = round(fs)


                # Check for necessary columns
                required_cols = self.channel_names + [self.label_column]
                if not all(col in df.columns for col in required_cols):
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    logging.warning(f"Skipping {fname}: Missing required columns: {missing_cols}")
                    continue

                labels = torch.tensor(df[self.label_column].values, dtype=torch.long)

                for ch in self.channel_names:
                    # Ensure signal is numeric, handle potential non-numeric entries
                    signal_pd = pd.to_numeric(df[ch], errors='coerce')
                    if signal_pd.isnull().any():
                        logging.warning(f"Skipping channel {ch} in {fname}: Contains non-numeric values.")
                        continue
                    signal = torch.tensor(signal_pd.values, dtype=torch.float32)

                    if signal.shape[0] != labels.shape[0]:
                        logging.warning(f"Skipping channel {ch} in {fname}: Signal/label length mismatch ({signal.shape[0]} vs {labels.shape[0]})")
                        continue

                    # Pass fs to slicing function
                    self._slice_channel_sequences(signal, labels, fs)
                num_processed += 1

            except pd.errors.EmptyDataError:
                logging.warning(f"Skipping {fname}: File is empty.")
            except Exception as e:
                logging.error(f"Error loading or processing {fname}: {e}", exc_info=True) # Log traceback

        logging.info(f"Loaded and sliced {len(self.sequences)} sequences from {num_processed} files.")

    # Modified to accept and store fs
    def _slice_channel_sequences(self, signal: torch.Tensor, labels: torch.Tensor, fs: float):
        """
        Slices one 1D signal and its labels into overlapping segments and stores fs.
        """
        stride = self.sequence_length - self.overlap
        total_length = signal.shape[0]

        # Check if signal is long enough for at least one sequence
        if total_length < self.sequence_length:
            logging.debug(f"Signal length ({total_length}) shorter than sequence_length ({self.sequence_length}). Cannot slice.")
            return

        for start in range(0, total_length - self.sequence_length + 1, stride):
            end = start + self.sequence_length
            seq = signal[start:end]
            seq_label = labels[start:end]
            self.sequences.append(seq)
            self.sequence_labels.append(seq_label)
            self.sequence_fs.append(fs) # <-- Store fs for this sequence

    def __len__(self):
        return len(self.sequences)

    def _apply_butter_bandpass_filter(self, data, lowcut, highcut, fs, order):
        """Applies Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Check for invalid frequency limits
        if low <= 0 or high >= 1.0:
            logging.warning(f"Invalid filter frequencies for fs={fs}: low={low*nyq:.2f}Hz (norm={low:.3f}), high={high*nyq:.2f}Hz (norm={high:.3f}). Skipping filter.")
            return data # Return original data if frequencies are invalid
        try:
            b, a = butter(order, [low, high], btype='bandpass')
            # Ensure data is numpy array for filtfilt
            data_np = data.numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
            # Filtfilt requires signal length > padlen (usually 3 * order)
            padlen = 3 * order
            if len(data_np) <= padlen:
                logging.warning(f"Signal length ({len(data_np)}) too short for filter order ({order}). Required > {padlen}. Skipping filter.")
                return data # Return original data
            y = filtfilt(b, a, data_np)
            # Return as tensor with same dtype as input
            return torch.from_numpy(y.copy()).to(dtype=data.dtype)
        except Exception as e:
            logging.error(f"Error applying Butterworth filter: {e}")
            return data # Return original data on error


    def __getitem__(self, idx):

        signal = self.sequences[idx]
        labels = self.sequence_labels[idx]
        fs = self.sequence_fs[idx] # <-- Retrieve fs for this sequence

        # --- Apply Bandpass Filter (Early Step) ---
        if self.apply_filter:
            signal = self._apply_butter_bandpass_filter(
                signal, self.filter_lowcut, self.filter_highcut, fs, self.filter_order
            )
        # --- End Filter ---

        # --- Apply Random Time Shift ---
        signal_processed = signal
        labels_processed = labels
        if self.max_time_shift > 0 and torch.rand(1).item() < self.augmentation_prob:
            shift = torch.randint(-self.max_time_shift, self.max_time_shift + 1, (1,)).item()
            if shift > 0:
                signal_processed = F.pad(signal[:-shift], (shift, 0), value=signal[0].item())
                labels_processed = F.pad(labels[:-shift], (shift, 0), value=labels[0].item())
            elif shift < 0:
                shift_abs = -shift
                signal_processed = F.pad(signal[shift_abs:], (0, shift_abs), value=signal[-1].item())
                labels_processed = F.pad(labels[shift_abs:], (0, shift_abs), value=labels[-1].item())
            # else: no shift
        # Use the potentially shifted signal/labels for subsequent steps
        signal = signal_processed
        labels = labels_processed
        # --- End Random Time Shift ---


        # --- Remove DC offset ---
        signal_mean = signal.mean()
        if not torch.isnan(signal_mean) and not torch.isinf(signal_mean):
            signal = signal - signal_mean

        # --- Apply Amplitude Scaling ---
        if self.amplitude_scale_range > 0 and torch.rand(1).item() < self.augmentation_prob:
            scale_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.amplitude_scale_range
            signal = signal * scale_factor
        # --- End Amplitude Scaling ---


        # --- Normalize amplitude to [-1, 1] (Do this *after* scaling) ---
        max_val = signal.abs().max()
        # Add small epsilon to prevent division by zero if signal is flat zero
        if max_val > 1e-6:
            signal = signal / max_val
        # --- End Normalization ---


        # --- Apply Baseline Wander ---
        noisy_signal = signal # Start with current processed signal
        if self.baseline_wander_mag > 0 and torch.rand(1).item() < self.augmentation_prob:
            time = torch.linspace(0, 1, signal.size(0), device=signal.device) # Assumes time 0->1 represents window duration
            bw_noise = torch.zeros_like(signal)
            num_bw_components = torch.randint(1, 3, (1,)).item()
            for _ in range(num_bw_components):
                freq = torch.rand(1).item() * self.baseline_wander_freq_max # Low relative freq
                amplitude = (torch.rand(1).item() - 0.5) * 2 * self.baseline_wander_mag
                phase = torch.rand(1).item() * 2 * math.pi
                bw_noise += amplitude * torch.sin(2 * math.pi * freq * time + phase)
            noisy_signal = noisy_signal + bw_noise # Add baseline wander
        # --- End Baseline Wander ---


        # --- Add Gaussian Noise ---
        if self.gaussian_noise_std > 0 and torch.rand(1).item() < self.augmentation_prob:
            noise = torch.randn_like(noisy_signal) * self.gaussian_noise_std
            noisy_signal = noisy_signal + noise # Add to potentially baseline-wandered signal
        # --- End Gaussian Noise ---


        # --- Apply Sinusoidal Noise ---
        if self.sinusoidal_noise_mag > 0 and torch.rand(1).item() < self.augmentation_prob:
            if 'time' not in locals(): # Create time if not created for baseline wander
                time = torch.linspace(0, 1, signal.size(0), device=signal.device)
            sinusoidal_noise = torch.zeros_like(noisy_signal)
            num_components = torch.randint(2, 5, (1,)).item()
            for _ in range(num_components):
                freq = torch.randint(1, 30, (1,)).item() # Freq relative to window duration if time=0->1
                amplitude = torch.rand(1).item() * self.sinusoidal_noise_mag
                phase = torch.rand(1).item() * 2 * math.pi
                sinusoidal_noise += amplitude * torch.sin(2 * math.pi * freq * time + phase)
            noisy_signal = noisy_signal + sinusoidal_noise # Add sinusoidal noise
        # --- End Sinusoidal Noise ---


        # --- Compute Wavelet Transform ---
        tf_map = compute_wavelet(noisy_signal, wavelet=self.wavelet, scales=self.wavelet_scales)
        tf_map_unsqueezed = tf_map.unsqueeze(0) # Add channel dim: (1, num_scales, T)

        # --- Apply Transform to CWT Map (Optional) ---
        if self.transform:
            tf_map_unsqueezed = self.transform(tf_map_unsqueezed)
        # --- End Transform ---

        # Return the potentially augmented signal (for plotting/debugging), the final CWT map, and labels
        # The 'noisy_signal' returned here is the one fed into CWT
        return noisy_signal, tf_map_unsqueezed, labels


# --- if __name__ == "__main__": block remains largely the same ---
# --- Update instantiation if needed, e.g., ECGFullDataset(..., gaussian_noise_std=0.02, ...) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    TEST_CLASS_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}

    try:
        print("--- Testing ECGFullDataset with Filtering and Augmentations ---")
        # Instantiate with filter enabled and some augmentation
        test_dataset = ECGFullDataset(
            data_dir="MCG_segmentation/qtdb/processed/val", # Adjust path
            overlap=125,
            sequence_length=250,
            # Enable various augmentations for testing
            sinusoidal_noise_mag=0.00,
            gaussian_noise_std=0.00,
            baseline_wander_mag=0.05,
            amplitude_scale_range=0.1,
            max_time_shift=5,
            augmentation_prob=0.8, # Apply augmentations most of the time for test visibility
            # Filter enabled by default
            apply_filter=True,
            # CWT settings
            wavelet='mexh',
            wavelet_scales = np.logspace(np.log2(2), np.log2(128), num=64, base=2.0)
        )

        if len(test_dataset) == 0:
            print("Dataset is empty. Check path and file contents.")
        else:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            print(f"Dataset loaded with {len(test_dataset)} samples.")

            signal_batch, tf_map_batch, labels_batch = next(iter(test_loader))

            print("\nShapes of first batch element:")
            print("Input Signal (Post-Augmentation) Shape:", signal_batch.shape) # (1, T)
            print("Wavelet Map Shape:", tf_map_batch.shape) # (1, 1, num_scales, T)
            print("Label Shape:", labels_batch.shape) # (1, T)

            # Prepare for plotting
            signal_single = signal_batch.squeeze(0).cpu().numpy()
            tf_map_single = tf_map_batch.squeeze(0).squeeze(0).cpu().numpy()
            labels_single = labels_batch.squeeze(0).cpu().numpy()
            time_axis = np.arange(signal_single.shape[0])
            # We don't have the original fs easily here, use sample index for time axis

            # Create plot
            fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2, 1]})
            fig.suptitle(f"Example from ECGFullDataset (Seq Length: {test_dataset.sequence_length})", fontsize=14)

            # Plot 1: Wavelet Scalogram
            im = ax[0].imshow(tf_map_single, aspect='auto', cmap='viridis',
                            extent=[time_axis[0], time_axis[-1], test_dataset.wavelet_scales[-1], test_dataset.wavelet_scales[0]]) # Use actual scales on y-axis
            ax[0].set_ylabel("Wavelet Scale (Approx.)")
            ax[0].set_title(f"Wavelet Transform ('{test_dataset.wavelet}') - Input to CNN")
            plt.colorbar(im, ax=ax[0], label='Magnitude', pad=0.02)

            # Plot 2: Augmented Signal with Labels
            ax[1].plot(time_axis, signal_single, color='black', linewidth=1.0, label='Augmented Signal')
            for t in range(signal_single.shape[0]):
                color = TEST_CLASS_COLORS.get(labels_single[t], 'magenta')
                ax[1].scatter(t, signal_single[t], color=color, s=15, zorder=3)
            ax[1].set_ylabel("Amplitude (Normalized)")
            ax[1].set_title("Augmented Signal with Ground Truth Labels (Color Dots)")
            ax[1].grid(True, linestyle=':', alpha=0.7)
            legend_elements = [Line2D([0], [0], color='black', lw=1, label='Signal')]
            for lbl, col in TEST_CLASS_COLORS.items():
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Label {lbl}',
                                    markerfacecolor=col, markersize=8))
            ax[1].legend(handles=legend_elements, loc='upper right', fontsize='small')

            # Plot 3: Labels
            ax[2].plot(time_axis, labels_single, drawstyle='steps-post', label='Label Sequence', color='darkorange')
            ax[2].set_xlabel("Time Step (Sample Index)")
            ax[2].set_ylabel("Label Index")
            ax[2].set_title("Ground Truth Label Sequence")
            ax[2].grid(True, linestyle=':', alpha=0.7)
            ax[2].legend(loc='upper right', fontsize='small')
            ax[2].set_yticks(np.unique(labels_single))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    except FileNotFoundError as e: print(f"Error: {e}. Check data directory path.")
    except ValueError as e: print(f"ValueError: {e}.")
    except ImportError as e: print(f"ImportError: {e}. Ensure required libraries are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()