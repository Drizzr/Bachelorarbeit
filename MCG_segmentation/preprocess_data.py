# 3rd party imports
import os
# import copy # Not needed now
import wfdb
import numpy as np
import pandas as pd
import logging
import math
import random
import sys

# Assuming DATA_DIR and LOGGER are defined as before
DATA_DIR = "MCG_segmentation/qtdb"
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# --- QTDB class remains the same as the previous version ---
# It handles downloading, finding records, splitting train/val
class QTDB(object):
    def __init__(self, data_dir=DATA_DIR, train_split_ratio=0.8, random_seed=42):
        self.db_name = 'qtdb'; self.raw_path = os.path.join(data_dir, 'raw')
        self.processed_path = os.path.join(data_dir, 'processed')
        self.train_path = os.path.join(self.processed_path, 'train')
        self.val_path = os.path.join(self.processed_path, 'val')
        self.record_names = None; self.train_split_ratio = train_split_ratio
        self.random_seed = random_seed; self._create_folders()
    
    def _create_folders(self):
        _create_directory(directory=self.raw_path)
        _create_directory(directory=self.processed_path)
        _create_directory(directory=self.train_path)
        _create_directory(directory=self.val_path)
    
    def generate_db(self): 
        self.generate_raw_db()
        self.generate_processed_db()

    def generate_raw_db(self):
        if os.path.exists(self.raw_path) and any(f.endswith('.dat') for f in os.listdir(self.raw_path)):
             LOGGER.info(f"Raw DB exists: {self.raw_path}. Skip download."); self.record_names = self._get_record_names(); return
        LOGGER.info(f'Generating Raw QT DB: {self.raw_path}');
        try: wfdb.dl_database(self.db_name, self.raw_path); self.record_names = self._get_record_names()
        except Exception as e: LOGGER.error(f"Failed raw DB download: {e}", exc_info=True); self.record_names = []
    
    def generate_processed_db(self):
        if not self.record_names: LOGGER.error("No records. Cannot process."); return
        LOGGER.info(f"Shuffling {len(self.record_names)} records (seed {self.random_seed}).")
        shuffled_records = list(self.record_names); random.seed(self.random_seed); random.shuffle(shuffled_records)
        split_index = math.floor(len(shuffled_records) * self.train_split_ratio)
        train_records, val_records = shuffled_records[:split_index], shuffled_records[split_index:]
        LOGGER.info(f"Split: {len(train_records)} train, {len(val_records)} val.")
        for subset, records in [('train', train_records), ('val', val_records)]:
            LOGGER.info(f"Processing {subset.upper()} Set..."); processed_count = 0
            for i, record_name in enumerate(records):
                try:
                    # Pass only record_name and data_dir to Record init now
                    record = Record(record_name=record_name, data_dir=DATA_DIR)
                    if record.consensus_df is not None and not record.consensus_df.empty:
                        record.save_csv(subset=subset) # Save the consensus df
                        processed_count += 1
                    else: LOGGER.warning(f"{subset.capitalize()} Record {record_name}: Consensus DataFrame empty/None.")
                    if (i + 1) % 10 == 0 or (i + 1) == len(records): LOGGER.info(f' Processed {subset}: {i+1}/{len(records)}')
                except Exception as e: LOGGER.error(f"Error processing {subset} record {record_name}: {e}", exc_info=True)
            LOGGER.info(f"Finished {subset.upper()} Set. Saved {processed_count} files.")
        LOGGER.info('Processed DB generation complete!')
    
    
    def _get_record_names(self):
        try: 
            return [f.split('.')[0] for f in os.listdir(self.raw_path) if f.endswith('.dat') and not f.startswith('.')]
        except FileNotFoundError: 
            LOGGER.error(f"Raw dir not found: {self.raw_path}")
            return []

# --- _create_directory remains the same ---
def _create_directory(directory):
    os.makedirs(directory, exist_ok=True); gitignore_path = os.path.join(directory, '.gitignore')
    if not os.path.exists(gitignore_path):
        try:
            with open(gitignore_path, 'w') as f: f.write('*\n!.gitignore\n')
        except IOError as e: LOGGER.warning(f"Could not create .gitignore in {directory}: {e}")



class Record(object):
    # Simplified init
    def __init__(self, record_name, data_dir):
        self.record_name = record_name
        self.load_path = os.path.join(data_dir, 'raw')
        self.processed_base_path = os.path.join(data_dir, 'processed')
        self.label_dict = {'na': 0, 'p': 1, 'N': 2, 'A': 2, 't': 3}
        self.target_symbols_map = {'p': 1, 'N': 2, 'A': 2, 't': 3}
        self.files = self._get_files()
        self.extensions = self._get_extensions()
        self.waveforms = None; self.fs = None; self.num_channels = None; self.sig_len = 0
        self.raw_annotations = {} # Store raw wfdb annotation objects
        self.consensus_df = None # Initialize

        # --- Load Record Header and Waveforms ---
        try:
            record_filepath = os.path.join(self.load_path, self.record_name)
            if not os.path.exists(record_filepath+".dat") or not os.path.exists(record_filepath+".hea"):
                raise FileNotFoundError(f"Files missing for {self.record_name}")
            record_obj = wfdb.rdrecord(record_filepath)
            self.waveforms = record_obj.p_signal; self.fs = record_obj.fs;
            self.num_channels = record_obj.n_sig; self.sig_len = record_obj.sig_len
            if self.fs is None or self.sig_len <= 0: raise ValueError("fs or sig_len invalid.")
            LOGGER.debug(f"{record_name}: Sig len {self.sig_len}, fs={self.fs}")
        except Exception as e: LOGGER.error(f"Error loading {self.record_name}: {e}", exc_info=True); return

        # --- Load Annotations (q1c and q2c) ---
        for ann_name in ['q1c', 'q2c']:
            if ann_name in self.extensions:
                try:
                    self.raw_annotations[ann_name] = wfdb.rdann(record_filepath, ann_name)
                    LOGGER.debug(f"Loaded '{ann_name}' annots for {self.record_name}")
                except Exception as e: LOGGER.warning(f"Could not load '{ann_name}' for {self.record_name}: {e}")

        # --- Generate Consensus DataFrame ---
        self.consensus_df = self._create_consensus_dataframe()

    def _get_files(self): 
        try: 
            return [f for f in os.listdir(self.load_path) if f.startswith(self.record_name + '.')] 
        except FileNotFoundError: 
            return []
        
    def _get_extensions(self):
        extensions = []
        parts = []
        for file in self.files: 
            parts = file.split('.')
            if len(parts) > 1: 
                extensions.append(parts[-1])

        return extensions



    # --- MODIFIED: _fill_label_array (Simpler logic) ---
    def _fill_label_array(self, annotator_name):
        """Creates a full-length array with labels filled based on annotator's intervals."""
        label_array = np.zeros(self.sig_len, dtype=np.int64) # Default to 0 ('na')
        if annotator_name not in self.raw_annotations:
            return None # Annotator doesn't exist

        ann_obj = self.raw_annotations[annotator_name]
        ann_samples = ann_obj.sample
        ann_symbols = ann_obj.symbol
        num_annotations = len(ann_samples)

        current_label = 0 # Start with 'na' label


        # Iterate through annotation points which define boundaries
        for i in range(num_annotations):
            sample_idx = ann_samples[i]
            symbol = ann_symbols[i]

            # Determine the segment end point
            segment_end = ann_samples[i+1] if i + 1 < num_annotations else self.sig_len

            # Ensure indices are within bounds
            start = max(0, sample_idx)
            end = min(self.sig_len, segment_end)

            # Fill the segment with the *previous* state's label if it was a boundary marker
            if symbol == '(':
                # Start of a new beat - the interval BEFORE this belonged to the previous state
                if start < end: # Make sure there's actually a segment to fill
                    label_array[start:end] = current_label
                 # The type of the *next* symbol determines the new state
                if i + 1 < num_annotations:
                    next_symbol = ann_symbols[i+1]
                    current_label = self.target_symbols_map.get(next_symbol, 0) # Get numeric label or 0
                    current_label_symbol = next_symbol if next_symbol in self.target_symbols_map else 'na'
                else:
                    current_label = 0 # End of annotations, revert to 'na'

            elif symbol == ')':
                # End of a beat - the interval BEFORE this belonged to the beat type
                if start < end:
                    label_array[start:end] = current_label
                # After the end marker, we revert to 'na' (unless immediately followed by another '(')
                current_label = 0


            elif symbol in self.target_symbols_map:
                # This is the beat type symbol itself, ignore boundary logic for this point
                # The label state was set by the preceding '('
                if start < end:
                    label_array[start:end] = current_label

            else: # Non-boundary, non-beat symbol - continue previous state
                if start < end:
                    label_array[start:end] = current_label

        # Fill any remaining part after the last annotation
        # last_ann_sample = ann_samples[-1]
        # if last_ann_sample < self.sig_len:
        #      label_array[last_ann_sample:] = 0 # Assume 'na' after last annotation point

        LOGGER.debug(f"{self.record_name}: Filled label array for '{annotator_name}' "
                    f"Unique labels: {np.unique(label_array)}")
        return label_array

    # --- MODIFIED: _create_consensus_dataframe (Handles single annotator) ---
    def _create_consensus_dataframe(self):
        """Generates the final DataFrame based on sample-wise label agreement
        or uses a single annotator if only one is available."""
        if self.waveforms is None or self.sig_len == 0:
            LOGGER.error(f"Cannot create DF for {self.record_name}: Waveforms missing.")
            return pd.DataFrame()

        labels_q1c = self._fill_label_array('q1c')
        labels_q2c = self._fill_label_array('q2c')


        has_q1c = labels_q1c is not None
        has_q2c = labels_q2c is not None
        consensus_labels = np.zeros(self.sig_len, dtype=np.int64) # Default to 0

        if has_q1c and has_q2c:
            LOGGER.info(f"Generating consensus labels for {self.record_name} from q1c and q2c.")
            for i in range(self.sig_len):
                # Agree AND the agreed label is NOT 'na' (0)
                if labels_q1c[i] == labels_q2c[i] and labels_q1c[i] != 0:
                    consensus_labels[i] = labels_q1c[i]
                # Otherwise, label remains 0 ('na')
        elif has_q1c:
            # Only q1c exists - use its labels directly
            LOGGER.info(f"Only q1c annotator found for {self.record_name}. Using its labels.")
            consensus_labels = labels_q1c
        elif has_q2c:
            # Only q2c exists - use its labels directly
            LOGGER.info(f"Only q2c annotator found for {self.record_name}. Using its labels.")
            consensus_labels = labels_q2c
        else:
            # Neither q1c nor q2c could be processed
            LOGGER.error(f"Could not process q1c or q2c annotations for {self.record_name}. Cannot create DataFrame.")
            return pd.DataFrame()

        # Create DataFrame
        try:
            df_data = {}
            df_data['index'] = np.arange(self.sig_len)
            df_data['time'] = df_data['index'] / self.fs
            df_data['train_label'] = consensus_labels
            idx_to_name = {v: k for k, v in self.label_dict.items()}
            df_data['label'] = [idx_to_name.get(l, 'na') for l in consensus_labels]
            df_data['interval'] = 0 # Single interval for the whole processed record

            for ch in range(self.num_channels):
                df_data[f'ch{ch+1}'] = self.waveforms[:, ch]

            return pd.DataFrame(df_data)

        except Exception as e:
            LOGGER.error(f"Error creating final DataFrame for {self.record_name}: {e}", exc_info=True)
            return pd.DataFrame()


    # --- save_csv remains the same ---
    def save_csv(self, subset):
        """Save consensus DataFrame to CSV."""
        if self.consensus_df is None or self.consensus_df.empty: 
            LOGGER.debug(f"Skip save {self.record_name} ({subset}). DF empty.")
            return
        
        if subset not in ['train', 'val']: 
            LOGGER.error(f"Invalid subset '{subset}' for {self.record_name}.")
            return
        
        subset_save_dir = os.path.join(self.processed_base_path, subset); _create_directory(subset_save_dir)
        save_filepath = os.path.join(subset_save_dir, f'{self.record_name}.csv')
        
        try: 
            self.consensus_df.to_csv(save_filepath, index=False)
        except Exception as e: 
            LOGGER.error(f"Fail save CSV {self.record_name} to {save_filepath}: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    qtdb = QTDB(data_dir=DATA_DIR, train_split_ratio=0.8, random_seed=123)
    qtdb.generate_db() # Handles download and split processing