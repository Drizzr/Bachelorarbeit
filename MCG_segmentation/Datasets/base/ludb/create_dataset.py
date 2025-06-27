#Datasets/base/ludb/create_dataset.py


import os
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import resample
import logging
import random
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

class LUDBProcessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.label_dict = {'na': 0, 'p': 1, 'n': 2, 'a': 2, 't': 3}  # lowercase for consistent mapping
        self.fs_original = 500
        self.fs_target = 250
        self.channel_suffixes = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        self.gap_tolerance = 0.1  # 100ms gap tolerance
        
        # Initialize instance variables
        self.record_name = None
        self.annotations = None
        self.labels = None
        self.intervals = None
        self.waveforms = None
        self.fs = None
        self.num_channels = None

    def process_patient(self, patient_id):
        """Process a single patient record."""
        record_path = os.path.join(self.dataset_path, patient_id)
        self.record_name = patient_id

        try:
            # Load the record
            record = wfdb.rdrecord(record_path)
            LOGGER.info(f"Loaded record {patient_id} with shape {record.p_signal.shape}")
        except Exception as e:
            LOGGER.error(f"Failed to load record {patient_id}: {e}")
            return

        # Store and resample waveform data
        waveform = record.p_signal
        self.num_channels = waveform.shape[1]
        resampled_length = int(waveform.shape[0] * self.fs_target / self.fs_original)
        self.waveforms = resample(waveform, resampled_length, axis=0)
        self.fs = self.fs_target
        
        # Process each channel
        for i, suffix in enumerate(self.channel_suffixes):
            if i >= self.num_channels:
                LOGGER.warning(f"Channel {suffix} (index {i}) not available in record {patient_id}")
                continue
                
            signal = self.waveforms[:, i]

            try:
                # Load annotations for this channel
                ann = wfdb.rdann(record_path, suffix)
                self.annotations = {'sample': ann.sample, 'symbol': [s.lower() for s in ann.symbol]}
            except Exception as e:
                LOGGER.warning(f"Missing or unreadable annotation for {patient_id}, channel {suffix}: {e}")
                continue

            # Get labels for this channel
            self.labels = self._get_labels()
            if self.labels is None:
                LOGGER.warning(f"No valid labels found for {patient_id}, channel {suffix}")
                continue
                
            # Get intervals
            self.intervals = self._get_intervals()
            if not self.intervals:
                LOGGER.warning(f"No valid intervals found for {patient_id}, channel {suffix}")
                continue
            
            # Create labels array for the signal, restricted to first and last labeled indices
            labels_array, start_idx, end_idx = self._create_labels_array(len(signal))
            if labels_array is None:
                LOGGER.warning(f"Failed to create labels array for {patient_id}, channel {suffix}")
                continue
            
            # Slice signal and create time array for the labeled range
            signal = signal[start_idx:end_idx]
            time = np.linspace(start_idx / self.fs_target, (end_idx - 1) / self.fs_target, end_idx - start_idx)
            
            # Create DataFrame
            df = pd.DataFrame({
                "time": time,
                "index": np.arange(start_idx, end_idx),
                "label": [self._symbol_from_numeric(lbl) for lbl in labels_array],
                "train_label": labels_array,
                "wave_form": signal
            })

            # Save to CSV
            output_file = os.path.join(self.output_path, f"{patient_id}_{suffix}.csv")
            try:
                df.to_csv(output_file, index=False)
                LOGGER.info(f"Saved: {output_file}")
            except Exception as e:
                LOGGER.error(f"Failed to save CSV for {patient_id}, channel {suffix}: {e}")

    def _get_labels(self):
        """Return p, n, a, t labels with 'na' gaps inserted."""
        if not self.annotations:
            return None
            
        try:
            target_symbols = {'p', 'n', 'a', 't'}
            ann_samples = self.annotations['sample']
            ann_symbols = self.annotations['symbol']
            num_annotations = len(ann_samples)

            labels = []
            for ann_idx in range(1, num_annotations - 1):
                symbol = ann_symbols[ann_idx]
                if symbol in target_symbols:
                    start_sample = ann_samples[ann_idx - 1]
                    end_sample = ann_samples[ann_idx + 1]
                    peak_sample = ann_samples[ann_idx]
                    if start_sample < peak_sample <= end_sample:
                        # Resample indices to target fs
                        start_sample = int(start_sample * self.fs_target / self.fs_original)
                        end_sample = int(end_sample * self.fs_target / self.fs_original)
                        peak_sample = int(peak_sample * self.fs_target / self.fs_original)
                        labels.append({
                            'peak': peak_sample,
                            'start': start_sample,
                            'end': end_sample,
                            'label': symbol
                        })
                        
            labels.sort(key=lambda x: x['start'])
            non_overlapping_labels = [labels[0]] if labels else []
            for i in range(1, len(labels)):
                if labels[i]['start'] >= non_overlapping_labels[-1]['end']:
                    non_overlapping_labels.append(labels[i])

            return self._add_gap_labels(non_overlapping_labels)
        except Exception as e:
            LOGGER.error(f"Error processing labels for {self.record_name}: {e}")
            return None
    
    def _add_gap_labels(self, labels):
        """Add 'na' labels in gaps between valid beat intervals."""
        if not labels:
            return []

        processed_labels = [labels[0]]
        for i in range(len(labels) - 1):
            prev_label = labels[i]
            next_label = labels[i + 1]
            gap_start = prev_label['end']
            gap_end = next_label['start']
            gap_duration_samples = gap_end - gap_start

            if gap_duration_samples > 0:
                gap_duration_sec = gap_duration_samples / self.fs_target
                if gap_duration_sec > self.gap_tolerance:
                    processed_labels.append('break')
                else:
                    processed_labels.append({
                        'peak': None,
                        'start': gap_start,
                        'end': gap_end,
                        'label': 'na'
                    })
            elif gap_duration_samples < 0:
                LOGGER.warning(f"{self.record_name}: Overlap detected between label ending at {gap_start} and label starting at {gap_end}")
            processed_labels.append(next_label)
        return processed_labels

    def _symbol_from_numeric(self, label_num):
        """Convert numeric label back to symbol."""
        reverse_dict = {v: k for k, v in self.label_dict.items()}
        return reverse_dict.get(label_num, 'na')

    def _get_intervals(self):
        """Split the label list into contiguous intervals separated by 'break'."""
        if self.labels is None:
            return []

        intervals = []
        current_interval = []
        for label_item in self.labels:
            if label_item == 'break':
                if current_interval:
                    intervals.append(current_interval)
                current_interval = []
            elif isinstance(label_item, dict):
                current_interval.append(label_item)
        if current_interval:
            intervals.append(current_interval)
        return intervals

    def _create_labels_array(self, signal_length):
        """Create an array of numeric labels for the signal, restricted to first and last labeled indices."""
        if not self.intervals or self.waveforms is None or self.fs is None:
            return None, None, None

        # Find first and last labeled indices (excluding 'na' labels)
        labeled_indices = []
        for interval in self.intervals:
            for label_dict in interval:
                if label_dict['label'] != 'na':
                    labeled_indices.append(label_dict['start'])
                    labeled_indices.append(label_dict['end'])

        if not labeled_indices:
            LOGGER.warning(f"No non-'na' labels found for {self.record_name}")
            return None, None, None

        start_idx = max(0, min(labeled_indices))
        end_idx = min(signal_length, max(labeled_indices))
        if end_idx <= start_idx:
            LOGGER.warning(f"Invalid label range for {self.record_name}: start={start_idx}, end={end_idx}")
            return None, None, None

        # Create labels array for the restricted range
        labels_array = np.zeros(end_idx - start_idx, dtype=np.int32)  # Default to 'na' (0)
        
        try:
            for interval in self.intervals:
                for label_dict in interval:
                    label_str = label_dict['label']
                    label_numeric = self.label_dict.get(label_str, 0)
                    # Adjust indices relative to start_idx
                    rel_start = max(0, label_dict['start'] - start_idx)
                    rel_end = min(end_idx - start_idx, label_dict['end'] - start_idx)
                    if rel_end > rel_start:
                        labels_array[rel_start:rel_end] = label_numeric
            return labels_array, start_idx, end_idx
        except Exception as e:
            LOGGER.error(f"Error creating labels array for {self.record_name}: {e}")
            return None, None, None

    def process_all(self):
        """Process all patients in the dataset."""
        try:
            patient_ids = sorted(set(f.split('.')[0] for f in os.listdir(self.dataset_path) 
                                   if f.endswith('.dat')))
            if not patient_ids:
                LOGGER.error(f"No .dat files found in {self.dataset_path}")
                return
            LOGGER.info(f"Found {len(patient_ids)} patients to process")
            for pid in patient_ids:
                LOGGER.info(f"Processing {pid}...")
                self.process_patient(pid)
        except Exception as e:
            LOGGER.error(f"Error in process_all: {e}")

def plot_random_waveform(path):
    # Optional: define label-to-color mapping
    SEGMENT_COLORS = {
        'p': "lightblue",
        'qrs': "lightcoral",
        't': "lightgreen",
        'na': "silver"  # Optional: you may skip plotting 'na'
    }

    # Get all CSV files in the processed directory
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv') and '_' in f]
    if not csv_files:
        print(f"No CSV files found in {path}")
        return

    # Select a random CSV
    random.seed()  # Ensure fresh randomness
    csv_file = random.choice(csv_files)
    csv_path = os.path.join(path, csv_file)

    # Extract patient_id and lead from filename
    patient_id, lead = csv_file.rsplit('_', 1)
    lead = lead.replace('.csv', '')
    print(f"Selected CSV for plotting: {csv_file}")

    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load CSV {csv_file}: {e}")
        return

    # Verify required columns
    required_columns = ['time', 'index', 'label', 'train_label', 'wave_form']
    if not all(col in df.columns for col in required_columns):
        print(f"CSV {csv_file} missing required columns: {required_columns}")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.suptitle(f"Record: {patient_id}, Lead: {lead} (Fs=250 Hz)")

    # Plot waveform
    ax.plot(df['time'], df['wave_form'], color='black', lw=0.8, label='Signal')

    # Highlight segments using axvspan
    t = df['time']
    labels = df['label']
    current_label = labels.iloc[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels.iloc[i] != current_label:
            if current_label != 'na':
                label_name = current_label
                color = SEGMENT_COLORS.get(current_label, 'gray')
                ax.axvspan(t.iloc[start_idx], t.iloc[i], color=color, alpha=0.3, label=f'{label_name}')
            start_idx = i
            current_label = labels.iloc[i]
    # Final segment
    if current_label != 'na':
        color = SEGMENT_COLORS.get(current_label, 'gray')
        ax.axvspan(t.iloc[start_idx], t.iloc[-1], color=color, alpha=0.3, label=f'{current_label}')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=10, ncol=2)

    ax.set_ylabel(f'Lead {lead} Amplitude', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim([df['time'].min() - 1, df['time'].max() + 1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    dataset_path = "MCG_segmentation/Datasets/base/ludb/raw"
    output_path = "MCG_segmentation/Datasets/base/ludb/processed"
    os.makedirs(output_path, exist_ok=True)
    processor = LUDBProcessor(dataset_path, output_path)
    processor.process_all()

    # Optionally plot a random waveform
    plot_random_waveform(output_path)
    plt.show()