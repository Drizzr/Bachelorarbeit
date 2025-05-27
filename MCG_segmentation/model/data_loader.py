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
import torch.nn.functional as F
from matplotlib.lines import Line2D


class ECGAugmentations:
    """
    Essential ECG-specific augmentations for improved model robustness.
    """
    
    def __init__(
        self,
        # Core augmentation parameters
        amplitude_scale_range: float = 0.1,
        max_time_shift: int = 10,
        augmentation_prob: float = 0.5,
        baseline_wander_mag: float = 0.05,
        baseline_wander_freq_max: float = 0.5,
        
        # New ECG-specific parameters
        powerline_freq: float = 50.0,  # 50Hz or 60Hz powerline interference
        powerline_mag: float = 0.03,
        respiratory_artifact_prob: float = 0.4,
        respiratory_freq_range: Tuple[float, float] = (0.1, 0.5),
        respiratory_mag: float = 0.04,
        heart_rate_variability_prob: float = 0.3,
        hrv_scale_factor: float = 0.05,
        morphology_warp_prob: float = 0.2,
        morphology_warp_strength: float = 0.1,
    ):
        self.amplitude_scale_range = amplitude_scale_range
        self.max_time_shift = max_time_shift
        self.augmentation_prob = augmentation_prob
        self.baseline_wander_mag = baseline_wander_mag
        self.baseline_wander_freq_max = baseline_wander_freq_max
        self.powerline_freq = powerline_freq
        self.powerline_mag = powerline_mag
        self.respiratory_artifact_prob = respiratory_artifact_prob
        self.respiratory_freq_range = respiratory_freq_range
        self.respiratory_mag = respiratory_mag
        self.heart_rate_variability_prob = heart_rate_variability_prob
        self.hrv_scale_factor = hrv_scale_factor
        self.morphology_warp_prob = morphology_warp_prob
        self.morphology_warp_strength = morphology_warp_strength

    def apply_augmentations(self, signal: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply essential ECG augmentations to the signal.
        
        Args:
            signal: 1D ECG signal tensor
            labels: Corresponding labels tensor
            
        Returns:
            Tuple of (augmented_signal, augmented_labels)
        """
        # Make copies to avoid modifying originals
        aug_signal = signal.clone()
        aug_labels = labels.clone()
        
        # 2. Normalize to zero mean (before other augmentations)
        aug_signal = self._normalize_zero_mean(aug_signal)
        
        # 3. Amplitude scaling
        aug_signal = self._apply_amplitude_scaling(aug_signal)
        
        # 4. Heart rate variability simulation
        aug_signal = self._apply_heart_rate_variability(aug_signal)
        
        # 5. Morphology warping (subtle shape changes)
        aug_signal = self._apply_morphology_warping(aug_signal)
        
        # Normalize to [-1, 1] before adding noise
        aug_signal = self._normalize_amplitude(aug_signal)
        
        # 6. Baseline wander (improved)
        aug_signal = self._apply_baseline_wander(aug_signal)
        
        # 7. Powerline interference
        aug_signal = self._apply_powerline_interference(aug_signal)
        
        # 8. Respiratory artifacts
        aug_signal = self._apply_respiratory_artifacts(aug_signal)
        
        return aug_signal, aug_labels

    def _normalize_zero_mean(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalize signal to zero mean."""
        signal_mean = signal.mean()
        if not torch.isnan(signal_mean) and not torch.isinf(signal_mean):
            return signal - signal_mean
        return signal

    def _apply_amplitude_scaling(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply amplitude scaling with improved distribution."""
        if self.amplitude_scale_range <= 0 or torch.rand(1).item() >= self.augmentation_prob:
            return signal
            
        # Use log-normal distribution for more realistic scaling
        scale_factor = torch.exp(torch.randn(1) * self.amplitude_scale_range).item()
        return signal * scale_factor

    def _apply_heart_rate_variability(self, signal: torch.Tensor) -> torch.Tensor:
        """Simulate heart rate variability by subtle time warping."""
        if torch.rand(1).item() >= self.heart_rate_variability_prob:
            return signal
            
        length = signal.size(0)
        # Create smooth random time warping
        x_original = torch.linspace(0, 1, length)
        noise = torch.randn(length // 8) * self.hrv_scale_factor
        noise_interp = F.interpolate(noise.unsqueeze(0).unsqueeze(0), 
                                   size=length, mode='linear', align_corners=True).squeeze()
        x_warped = x_original + noise_interp
        x_warped = torch.clamp(x_warped, 0, 1)
        
        # Interpolate signal at warped time points
        indices = x_warped * (length - 1)
        return self._interpolate_1d(signal, indices)

    def _apply_morphology_warping(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply subtle morphology changes to ECG waveforms."""
        if torch.rand(1).item() >= self.morphology_warp_prob:
            return signal
            
        length = signal.size(0)
        # Create localized warping
        warp_centers = torch.randint(length // 4, 3 * length // 4, (2,))
        warped_signal = signal.clone()
        
        for center in warp_centers:
            width = length // 8
            start = max(0, center - width)
            end = min(length, center + width)
            
            if end > start:
                local_signal = signal[start:end]
                warp_strength = (torch.rand(1) - 0.5) * self.morphology_warp_strength
                # Apply local scaling
                warped_signal[start:end] = local_signal * (1 + warp_strength)
                
        return warped_signal

    def _normalize_amplitude(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalize signal amplitude to [-1, 1]."""
        max_val = signal.abs().max()
        if max_val > 1e-6:
            return signal / max_val
        return signal

    def _apply_baseline_wander(self, signal: torch.Tensor) -> torch.Tensor:
        """Enhanced baseline wander with more realistic patterns."""
        if self.baseline_wander_mag <= 0 or torch.rand(1).item() >= self.augmentation_prob:
            return signal
            
        length = signal.size(0)
        time = torch.linspace(0, 1, length, device=signal.device)
        bw_noise = torch.zeros_like(signal)
        
        # Use multiple frequency components with different characteristics
        num_components = torch.randint(1, 4, (1,)).item()
        for _ in range(num_components):
            freq = torch.rand(1).item() * self.baseline_wander_freq_max
            amplitude = torch.rand(1).item() * self.baseline_wander_mag
            phase = torch.rand(1).item() * 2 * math.pi
            
            # Add some frequency modulation for more realistic wander
            freq_mod = 1 + 0.1 * torch.sin(2 * math.pi * 0.1 * time)
            bw_noise += amplitude * torch.sin(2 * math.pi * freq * time * freq_mod + phase)
            
        return signal + bw_noise

    def _apply_powerline_interference(self, signal: torch.Tensor) -> torch.Tensor:
        """Add powerline interference (50/60 Hz)."""
        if self.powerline_mag <= 0 or torch.rand(1).item() >= self.augmentation_prob:
            return signal
            
        length = signal.size(0)
        # Assume 500 Hz sampling rate (adjust based on your data)
        sampling_rate = 250.0
        time = torch.arange(length, dtype=torch.float32, device=signal.device) / sampling_rate
        
        # Add powerline interference with harmonics
        interference = torch.zeros_like(signal)
        for harmonic in [1, 2]:  # Fundamental and first harmonic
            freq = self.powerline_freq * harmonic
            amplitude = self.powerline_mag / harmonic
            phase = torch.rand(1).item() * 2 * math.pi
            interference += amplitude * torch.sin(2 * math.pi * freq * time + phase)
            
        return signal + interference

    def _apply_respiratory_artifacts(self, signal: torch.Tensor) -> torch.Tensor:
        """Add respiratory-related artifacts."""
        if torch.rand(1).item() >= self.respiratory_artifact_prob:
            return signal
            
        length = signal.size(0)
        time = torch.linspace(0, 1, length, device=signal.device)
        
        # Respiratory frequency
        resp_freq = (torch.rand(1).item() * 
                    (self.respiratory_freq_range[1] - self.respiratory_freq_range[0]) + 
                    self.respiratory_freq_range[0])
        
        # Create respiratory artifact
        resp_artifact = self.respiratory_mag * torch.sin(2 * math.pi * resp_freq * time)
        
        return signal + resp_artifact

    def _interpolate_1d(self, signal: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Interpolate 1D signal at given indices."""
        length = signal.size(0)
        indices = torch.clamp(indices, 0, length - 1)
        
        # Linear interpolation
        indices_floor = torch.floor(indices).long()
        indices_ceil = torch.ceil(indices).long()
        
        # Handle edge case where indices_floor == indices_ceil
        weights = indices - indices_floor.float()
        
        # Clamp indices to valid range
        indices_floor = torch.clamp(indices_floor, 0, length - 1)
        indices_ceil = torch.clamp(indices_ceil, 0, length - 1)
        
        interpolated = (1 - weights) * signal[indices_floor] + weights * signal[indices_ceil]
        
        return interpolated


# in the QTDB database only about a third of the files are of the form (p)(QRS)(T) (most of the time: (p)(QRS)T))
# the compatible files are determined manually


class ECGFullDataset(Dataset):
    """
    Dataset that loads ECG data, applies enhanced augmentations,
    and returns 1D sequences with labels for 1D CNN/RNN models.
    """
    def __init__(
        self,
        data_dir: str,
        label_column: str = "train_label",
        file_extension: str = ".csv",
        sequence_length: int = 512,
        overlap: int = 256,
        # Enhanced augmentation parameters
        amplitude_scale_range: float = 0.1,
        max_time_shift: int = 10,
        augmentation_prob: float = 0.5,
        baseline_wander_mag: float = 0.05,
        baseline_wander_freq_max: float = 0.5,
        powerline_freq: float = 50.0,
        powerline_mag: float = 0.03,
        respiratory_artifact_prob: float = 0.4,
        respiratory_freq_range: Tuple[float, float] = (0.1, 0.5),
        respiratory_mag: float = 0.04,
        heart_rate_variability_prob: float = 0.3,
        hrv_scale_factor: float = 0.05,
        morphology_warp_prob: float = 0.2,
        morphology_warp_strength: float = 0.1,
        # Legacy parameters for backwards compatibility
        sinusoidal_noise_mag: float = 0.05,
        gaussian_noise_std: float = 0.02,
    ):
        self.data_dir = data_dir
        self.label_column = label_column
        self.file_extension = file_extension
        self.sequence_length = sequence_length
        self.overlap = overlap

        # Initialize enhanced augmentations
        self.augmentations = ECGAugmentations(
            amplitude_scale_range=amplitude_scale_range,
            max_time_shift=max_time_shift,
            augmentation_prob=augmentation_prob,
            baseline_wander_mag=baseline_wander_mag,
            baseline_wander_freq_max=baseline_wander_freq_max,
            powerline_freq=powerline_freq,
            powerline_mag=powerline_mag,
            respiratory_artifact_prob=respiratory_artifact_prob,
            respiratory_freq_range=respiratory_freq_range,
            respiratory_mag=respiratory_mag,
            heart_rate_variability_prob=heart_rate_variability_prob,
            hrv_scale_factor=hrv_scale_factor,
            morphology_warp_prob=morphology_warp_prob,
            morphology_warp_strength=morphology_warp_strength,
        )

        # Legacy noise parameters (for backward compatibility if needed)
        self.sinusoidal_noise_mag = sinusoidal_noise_mag
        self.gaussian_noise_std = gaussian_noise_std

        self.sequences = [] # List to store 1D sequences
        self.sequence_labels = [] # List to store labels for each sequence

        self._load_and_slice_all()

        if not self.sequences:
            logging.warning(f"Dataset initialization resulted in zero sequences. Check data_dir ('{data_dir}') and file contents.")

    def _load_and_slice_all(self):
        """Load and slice all ECG data files."""
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
        """Slice 1D signal and labels into overlapping sequences."""
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
        """Get item with enhanced augmentations applied."""
        signal = self.sequences[idx] # Raw 1D sequence
        labels = self.sequence_labels[idx]

        # Apply enhanced augmentations
        augmented_signal, augmented_labels = self.augmentations.apply_augmentations(signal, labels)


        # Add channel dimension: (1, T) - Conv1d expects (Batch, Channels, Length)
        final_signal_output = augmented_signal.unsqueeze(0)

        return final_signal_output, augmented_labels


# --- if __name__ == "__main__": block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Colors for plotting labels on signal
    TEST_CLASS_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}

    try:
        print("--- Testing ECGFullDataset with Enhanced Augmentations ---")
        test_dataset = ECGFullDataset(
            data_dir="MCG_segmentation/Datasets/train", # Adjust path
            overlap=125,
            sequence_length=500,
            # Enhanced augmentation parameters
            amplitude_scale_range=0.1,
            max_time_shift=5,
            augmentation_prob=1.0,
            baseline_wander_mag=0.2,
            powerline_mag=0.05,
            respiratory_artifact_prob=0.5,
            heart_rate_variability_prob=0.4,
            morphology_warp_prob=0.3,
            # Legacy parameters
            sinusoidal_noise_mag=0.05,
            gaussian_noise_std=0.04,
        )

        if len(test_dataset) == 0:
            print("Dataset is empty.")
        else:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            print(f"Dataset loaded with {len(test_dataset)} samples.")

            # Get first batch
            signal_batch, labels_batch = next(iter(test_loader))

            print("\nShapes of first batch element:")
            print("Processed Signal Shape:", signal_batch.shape) # Should be (1, 1, T)
            print("Label Shape:", labels_batch.shape) # Should be (1, T)

            # Prepare for plotting
            signal_single = signal_batch.squeeze(0).squeeze(0).cpu().numpy() # Remove batch & channel dim
            labels_single = labels_batch.squeeze(0).cpu().numpy() # Remove batch dim
            time_axis = np.arange(signal_single.shape[0])

            # Create plot
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            fig.suptitle(f"Enhanced ECG Augmentations (Seq Length: {test_dataset.sequence_length})", fontsize=14)

            # Plot 1: Processed Signal with Labels
            ax[0].plot(time_axis, signal_single, color='black', linewidth=1.0, label='Enhanced Augmented Signal')
            for t in range(signal_single.shape[0]):
                color = TEST_CLASS_COLORS.get(labels_single[t], 'magenta')
                ax[0].scatter(t, signal_single[t], color=color, s=15, zorder=3)

            ax[0].set_ylabel("Amplitude (Normalized)")
            ax[0].set_title("Enhanced Augmented Signal with Ground Truth Labels")
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
        print(f"ImportError: {e}. Ensure required libraries are installed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()