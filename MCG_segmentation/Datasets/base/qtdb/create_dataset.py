# 3rd party imports
# https://physionet.org/physiobank/database/qtdb/doc/node4.html
import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import logging

DATA_DIR = "MCG_segmentation/Datasets/qtdb/base"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class QTDB(object):
    def __init__(self, data_dir=DATA_DIR, random_seed=42):
        self.db_name = 'qtdb'
        self.raw_path = os.path.join(data_dir, 'raw')
        self.processed_path = os.path.join(data_dir, 'processed')
        self.record_names = None
        self.random_seed = random_seed
        self._create_folders()

    def _create_folders(self):
        _create_directory(directory=self.raw_path)
        _create_directory(directory=self.processed_path)


    def generate_db(self):
        """Generate raw and processed databases with train/val split."""
        self.generate_raw_db()
        self.generate_processed_db() # This will now handle the split

    def generate_raw_db(self):
        """Generate the raw version of the QT database in the 'raw' folder."""
        # Check if raw data exists and has content
        if os.path.exists(self.raw_path) and any(f.endswith('.dat') for f in os.listdir(self.raw_path)):
            LOGGER.info(f"Raw database seems to exist in {self.raw_path}, skipping download.")
            self.record_names = self._get_record_names()
            return # Don't download again

        LOGGER.info('Generating Raw QT Database...\nSave path: {}'.format(self.raw_path))
        try:
            wfdb.dl_database(self.db_name, self.raw_path)
            self.record_names = self._get_record_names()
            if not self.record_names:
                raise RuntimeError("Downloaded database but still could not find records.")
            LOGGER.info('Raw DB download complete!')
        except Exception as e:
            LOGGER.error(f"Failed to download raw database: {e}", exc_info=True)
            self.record_names = [] # Ensure it's empty list on failure


    # Modified to handle train/val split
    def generate_processed_db(self):
        if not self.record_names:
            LOGGER.error("No record names available. Cannot generate processed DB.")
            return

        LOGGER.info(f"Processing all {len(self.record_names)} records...")
        processed_count = 0

        for i, record_name in enumerate(self.record_names):
            try:
                record = Record(record_name=record_name, data_dir=DATA_DIR)
                if record.intervals_df is not None and not record.intervals_df.empty:
                    record.save_csv()  # No subset
                    processed_count += 1
                else:
                    LOGGER.warning(f"Record {record_name}: No valid intervals or processed DataFrame is empty.")
                if (i + 1) % 10 == 0 or (i + 1) == len(self.record_names):
                    LOGGER.info(f'Processed: {i+1}/{len(self.record_names)} Record: {record_name}')
            except Exception as e:
                LOGGER.error(f"Error processing record {record_name}: {e}", exc_info=True)

        LOGGER.info(f"Processed DB generation complete. Total saved: {processed_count}")


    def _get_record_names(self):
        """Return list of records in 'raw' path."""
        try:
            # Filter out hidden files like .DS_Store if present
            return [f.split('.')[0] for f in os.listdir(self.raw_path)
                    if f.endswith('.dat') and not f.startswith('.')]
        except FileNotFoundError:
            LOGGER.error(f"Raw data directory not found: {self.raw_path}")
            return []


def _create_directory(directory):
    """Creates directory and adds a .gitignore file."""
    os.makedirs(directory, exist_ok=True)
    gitignore_path = os.path.join(directory, '.gitignore')
    if not os.path.exists(gitignore_path): # Avoid overwriting if it exists
        try:
            with open(gitignore_path, 'w') as f:
                f.write('*\n!.gitignore\n')
        except IOError as e:
            LOGGER.warning(f"Could not create .gitignore in {directory}: {e}")


class Record(object):

    # Added data_dir to init to avoid global dependency
    def __init__(self, record_name, data_dir, gap_tolerance=1):

        # Set parameters
        self.record_name = record_name
        self.load_path = os.path.join(data_dir, 'raw')
        # self.save_path is now handled in save_csv based on subset
        self.processed_base_path = os.path.join(data_dir, 'processed') # Base path
        self.gap_tolerance = gap_tolerance

        # Set attributes
        self.label_dict = {'na': 0, 'p': 1, 'n': 2, 'a': 2, 't': 3}
        self.files = self._get_files()
        self.extensions = self._get_extensions()
        self.annotator = self._get_annotator(annotator=None, auto_return=True)
        self.waveforms = None; self.fs = None; self.num_channels = None
        self.annotations = None; self.labels = None; self.intervals = None
        self.num_intervals = 0; self.intervals_df = None

        # --- Load Data ---
        try:
            record_filepath = os.path.join(self.load_path, self.record_name)
            # Check if essential .dat and .hea files exist
            if not os.path.exists(record_filepath + ".dat") or not os.path.exists(record_filepath + ".hea"):
                raise FileNotFoundError(f"Essential record files (.dat, .hea) not found for {self.record_name}")

            record_obj = wfdb.rdrecord(record_filepath)
            self.waveforms = record_obj.p_signal
            self.fs = record_obj.fs
            self.num_channels = record_obj.n_sig
            if self.annotator:
                try:
                    self.annotations = wfdb.rdann(record_filepath, self.annotator).__dict__
                except FileNotFoundError:
                    LOGGER.warning(f"Annotation file '{self.annotator}' not found for {self.record_name}. Trying alternatives...")
                    # Simple fallback logic if preferred annotator missing
                    self.annotator = self._get_annotator(annotator=None, auto_return=True) # Re-check with auto_return maybe?
                    if self.annotator:
                        self.annotations = wfdb.rdann(record_filepath, self.annotator).__dict__
                    else:
                        LOGGER.error(f"No annotation file could be loaded for {self.record_name}")
                        self.annotations = None

            else:
                LOGGER.warning(f"No suitable annotator found initially for record {self.record_name}")

            if self.waveforms is None or self.fs is None:
                raise ValueError("Failed to load essential waveform data or sampling frequency.")

            # --- Process Labels and Intervals (only if annotations loaded) ---
            if self.annotations:
                self.labels = self._get_labels()
                if self.labels:
                    self.intervals = self._get_intervals()
                    self.num_intervals = len(self.intervals) if self.intervals else 0
                    if self.num_intervals > 0:
                        self.intervals_df = self._get_intervals_df() # Calls modified version
                        if self.intervals_df.empty:
                            LOGGER.warning(f"Processed DataFrame is empty for record {self.record_name}")
                    else:
                        LOGGER.warning(f"No valid intervals created for record {self.record_name}")
                else:
                    LOGGER.warning(f"Could not derive labels for record {self.record_name}")
            else:
                LOGGER.warning(f"No annotations available to process labels for {self.record_name}")


        except FileNotFoundError as e:
            LOGGER.error(f"Record file(s) not found for {self.record_name} in {self.load_path}: {e}")
        except ValueError as e: # Catch specific errors like checksum mismatch from rdrecord
            LOGGER.error(f"Value error processing record {self.record_name} (potential WFDB issue): {e}")
        except Exception as e:
            LOGGER.error(f"Error initializing record {self.record_name}: {e}", exc_info=True)
            # Ensure attributes remain None if initialization fails
            self.waveforms = self.fs = self.num_channels = self.annotations = self.labels = self.intervals = self.intervals_df = None
            self.num_intervals = 0

    # --- MODIFIED save_csv ---
    def save_csv(self):
        """Save each channel's data to separate CSV files in the processed directory."""
        if self.intervals_df is None or self.intervals_df.empty:
            LOGGER.debug(f"Not saving CSV for {self.record_name} due to empty DataFrame.")
            return

        _create_directory(self.processed_base_path)

        # Iterate over each channel
        for ch in range(self.num_channels):
            # Define the save path for this channel's CSV
            save_filepath = os.path.join(self.processed_base_path, f'{self.record_name}_ch{ch+1}.csv')
            
            # Create a DataFrame with the required columns
            df_channel = pd.DataFrame({
                'time': self.intervals_df['time'],
                'index': self.intervals_df['index'],
                'label': self.intervals_df['label'],
                'train_label': self.intervals_df['train_label'],
                'wave_form': self.intervals_df[f'ch{ch+1}']
            })

            try:
                df_channel.to_csv(save_filepath, index=False)
                LOGGER.info(f"Saved CSV for channel {ch+1} of {self.record_name} to {save_filepath}")
            except Exception as e:
                LOGGER.error(f"Failed to save CSV for channel {ch+1} of {self.record_name} to {save_filepath}: {e}")


    def _get_files(self):
        """Get list of files associated with record_name."""
        try:
            # Look for files starting with record_name followed by a dot
            return [file for file in os.listdir(self.load_path) if file.startswith(self.record_name + '.')]
        except FileNotFoundError:
            return []

    def _get_extensions(self):
        """Return a list file extensions for record_name."""
        # Handle cases where filename might not have an extension after splitting
        extensions = []
        for file in self.files:
            parts = file.split('.')
            if len(parts) > 1:
                extensions.append(parts[-1])
        return extensions

    def _get_annotator(self, annotator, auto_return):
        """Return annotator file extension."""
        if annotator is not None and annotator in self.extensions:
            return annotator
        elif auto_return is True:
            preferred_annotators = ['q1c', 'q2c', 'qt1', 'qt2', 'pu', 'pu1', 'pu0'] # Order of preference
            for pa in preferred_annotators:
                if pa in self.extensions:
                    LOGGER.debug(f"Using annotator '{pa}' for {self.record_name}")
                    return pa
            LOGGER.warning(f"No preferred annotator found for {self.record_name} from {self.extensions}")
            return None
        else:
            return None # Return None explicitly if no annotator specified and auto_return is False

    def _get_labels(self):
        """Return p, N, A, and t labels, with 'na' gaps inserted."""
        if self.annotations is None:
            LOGGER.warning(f"Cannot get labels for {self.record_name}: Annotations not loaded.")
            return None

        try:
            target_symbols = {'p', 'N', 'A', 't'}
            ann_samples = self.annotations['sample']
            ann_symbols = self.annotations['symbol']
            num_annotations = len(ann_samples)

            labels = []
            # Iterate through all annotations to find beat symbols
            for ann_idx in range(1, num_annotations - 1): # Avoid first and last annots for boundary checks
                symbol = ann_symbols[ann_idx]
                if symbol in target_symbols:
                    # Heuristic: Assume '(' is start marker, ')' is end marker *immediately* before/after
                    # This is a common but not guaranteed pattern in some PhysioNet annotations.
                    start_marker_symbol = ann_symbols[ann_idx - 1]
                    end_marker_symbol = ann_symbols[ann_idx + 1]

                    # Use samples directly if markers aren't parens
                    start_sample = ann_samples[ann_idx - 1]
                    end_sample = ann_samples[ann_idx + 1]

                    # More robust check might be needed if markers aren't guaranteed
                    # For now, proceed if start < peak < end
                    peak_sample = ann_samples[ann_idx]
                    if start_sample < peak_sample < end_sample:
                        labels.append({
                            'peak': peak_sample,
                            'start': start_sample,
                            'end': end_sample,
                            'label': symbol.lower()
                        })
                    else:
                        LOGGER.debug(f"Skipping beat {symbol} at sample {peak_sample} for {self.record_name}: "
                                    f"Invalid start/peak/end sample order ({start_sample}, {peak_sample}, {end_sample})")


            if not labels:
                LOGGER.warning(f"Could not extract any valid label intervals for {self.record_name} using marker logic.")
                return None

            # Sort labels by start time
            labels.sort(key=lambda x: x['start'])

            # Remove overlapping intervals (preferring the earlier one)
            non_overlapping_labels = []
            if labels:
                non_overlapping_labels.append(labels[0])
                for i in range(1, len(labels)):
                    # If current label starts after or exactly where previous ended
                    if labels[i]['start'] >= non_overlapping_labels[-1]['end']:
                        non_overlapping_labels.append(labels[i])
                    else:
                        LOGGER.debug(f"Removing overlapping label {labels[i]} starting at {labels[i]['start']} "
                                    f"because previous ended at {non_overlapping_labels[-1]['end']}")


            if not non_overlapping_labels:
                LOGGER.warning(f"No non-overlapping labels found for {self.record_name}")
                return None

            # Add gap labels
            labels_with_gaps = self._add_gap_labels(labels=non_overlapping_labels)

            return labels_with_gaps

        except Exception as e:
            LOGGER.error(f"Error processing annotations for {self.record_name}: {e}", exc_info=True)
            return None

    def _add_gap_labels(self, labels):
        """Add 'na' labels in gaps between valid beat intervals."""
        if not labels: return []

        processed_labels = [labels[0]] # Start with the first label

        for i in range(len(labels) - 1):
            prev_label = labels[i]
            next_label = labels[i+1]

            gap_start = prev_label['end']
            gap_end = next_label['start']
            gap_duration_samples = gap_end - gap_start

            if gap_duration_samples > 0: # If there is a positive gap
                gap_duration_sec = gap_duration_samples / self.fs
                if gap_duration_sec > self.gap_tolerance:
                    processed_labels.append('break')
                else: # Insert 'na' for short gaps
                    processed_labels.append({
                        'peak': None, 'start': gap_start,
                        'end': gap_end, 'label': 'na'
                    })
            elif gap_duration_samples < 0:
                LOGGER.warning(f"{self.record_name}: Overlap detected between label ending at {gap_start} and label starting at {gap_end}. Skipping gap.")
                # Implicitly skips gap if overlap

            processed_labels.append(next_label) # Add the next actual beat label

        return processed_labels

    def _get_intervals(self):
        """Split the label list into contiguous intervals separated by 'break'."""
        if self.labels is None: return []

        intervals = []
        current_interval = []
        for label_item in self.labels:
            if label_item == 'break':
                if current_interval: intervals.append(current_interval)
                current_interval = []
            elif isinstance(label_item, dict): # Only add valid label dicts
                current_interval.append(label_item)

        if current_interval: intervals.append(current_interval) # Add the last interval
        return intervals

    def _get_intervals_df(self):
        """Return contiguous intervals of labels as DataFrame, enforcing minimum gaps."""
        if not self.intervals or self.waveforms is None or self.fs is None:
            return pd.DataFrame() # Return empty DataFrame if no intervals or data

        all_indices, all_times, all_waveforms = [], [], []
        all_str_labels, all_train_labels, all_interval_ids = [], [], []

        for interval_idx, interval in enumerate(self.intervals):
            prev_label_dict = None

            for label_dict in interval:
                label_str = label_dict['label']
                label_numeric = self.label_dict.get(label_str, 0)
                start_sample = label_dict['start']
                end_sample = label_dict['end']
                num_samples = end_sample - start_sample

                if num_samples <= 0: continue

                num_relabel_as_na = 0

                waveform_segment = self.waveforms[start_sample:end_sample, :]
                index_segment = np.arange(start_sample, end_sample, dtype=np.int64)
                time_segment = index_segment / self.fs

                if num_relabel_as_na > 0:
                    str_label_segment = ['na'] * num_relabel_as_na + [label_str] * (num_samples - num_relabel_as_na)
                    train_label_segment = [self.label_dict['na']] * num_relabel_as_na + [label_numeric] * (num_samples - num_relabel_as_na)
                else:
                    str_label_segment = [label_str] * num_samples
                    train_label_segment = [label_numeric] * num_samples

                all_indices.append(index_segment)
                all_times.append(time_segment)
                all_waveforms.append(waveform_segment)
                all_str_labels.extend(str_label_segment)
                all_train_labels.extend(train_label_segment)
                all_interval_ids.extend([interval_idx] * num_samples)

                if label_str != 'na': prev_label_dict = label_dict

        if not all_indices:
            LOGGER.warning(f"No data collected for DataFrame creation in record {self.record_name}")
            return pd.DataFrame()

        try:
            final_indices = np.concatenate(all_indices, axis=0)
            final_times = np.concatenate(all_times, axis=0)
            final_waveforms = np.concatenate(all_waveforms, axis=0)

            if not (len(final_indices) == len(final_times) == final_waveforms.shape[0] == \
                    len(all_str_labels) == len(all_train_labels) == len(all_interval_ids)):
                raise ValueError(f"Length mismatch! idx:{len(final_indices)}, time:{len(final_times)}, "
                                f"wf:{final_waveforms.shape[0]}, strL:{len(all_str_labels)}, "
                                f"trainL:{len(all_train_labels)}, intID:{len(all_interval_ids)}")

            df_data = {'time': final_times, 'index': final_indices, 'label': all_str_labels,
                    'train_label': all_train_labels, 'interval': all_interval_ids}
            for ch in range(self.num_channels): df_data[f'ch{ch+1}'] = final_waveforms[:, ch]

            return pd.DataFrame(df_data)

        except ValueError as e:
            LOGGER.error(f"Error concatenating arrays for {self.record_name}: {e}", exc_info=True)
            return pd.DataFrame()

    def plot_waveform(self):
        """Plot waveform with labels."""
        if self.intervals_df is None or self.intervals_df.empty or self.waveforms is None:
            LOGGER.warning(f"Cannot plot waveform for {self.record_name}: Missing data.")
            return

        fig, axes = plt.subplots(self.num_channels, 1, figsize=(15, 5 * self.num_channels), sharex=True)
        if self.num_channels == 1: axes = [axes]
        fig.suptitle(f"Record: {self.record_name} (Fs={self.fs} Hz)")
        fig.subplots_adjust(hspace=0.1)
        full_time = np.arange(self.waveforms.shape[0]) / self.fs

        for i, ax in enumerate(axes):
            channel_name = f'ch{i+1}'
            ax.plot(full_time, self.waveforms[:, i], '-', color=[0.8, 0.8, 0.8], lw=1, label='_nolegend_')
            for interval_id in self.intervals_df['interval'].unique():
                interval_data = self.intervals_df[self.intervals_df['interval'] == interval_id]
                ax.plot(interval_data['time'], interval_data[channel_name], '-', color='k', lw=1.5, label=f'Interval {interval_id}' if i == 0 else '_nolegend_')

            ax.set_ylabel(f'Channel {i+1} Amplitude', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.tick_params(axis='y', labelsize=10)
            if i < self.num_channels - 1: ax.tick_params(labelbottom=False)
            else: ax.set_xlabel('Time (s)', fontsize=12); ax.tick_params(axis='x', labelsize=10)

        if not self.intervals_df.empty:
            min_time = self.intervals_df['time'].min() - 1; max_time = self.intervals_df['time'].max() + 1
            axes[-1].set_xlim([max(0, min_time), max_time])
        else: axes[-1].set_xlim([full_time.min(), full_time.max()])

        handles, labels = axes[0].get_legend_handles_labels()
        if handles: fig.legend(handles, labels, loc='upper right', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path = os.path.join(self.processed_base_path, f'{self.record_name}_plot.png') # Save plot in base processed dir
        try:
            plt.savefig(plot_save_path); LOGGER.info(f"Saved waveform plot to {plot_save_path}")
            plt.close(fig)
        except Exception as e: LOGGER.error(f"Failed to save plot for {self.record_name}: {e}")


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create QTDB object (specify split ratio, seed)
    qtdb = QTDB(data_dir=DATA_DIR, random_seed=123)

    # Generate raw and processed databases
    qtdb.generate_db() # This will handle download and split processing