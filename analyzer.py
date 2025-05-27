import os
import ast
import warnings
import logging
import numpy as np
import torch
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from scipy import signal
from scipy.interpolate import griddata
from scipy.signal import correlate, savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FastICA
from matplotlib.widgets import Slider
import copy

# Attempt to import local MCG_segmentation package
try:
    from MCG_segmentation.model.model import ECGSegmenter, UNet1D
except ImportError:
    logging.warning("Could not import ECGSegmenter. Segmentation features will be unavailable.")
    ECGSegmenter = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Analyzer:
    """Analyzer class for processing and analyzing MCG (Magnetocardiography) data from TDMS files.

    This class assumes all internal processing operates on data sampled at 250 Hz.
    The `prepare_data` method handles resampling from the original source rate to 250 Hz.
    It provides methods for loading, filtering, segmenting, and visualizing MCG data,
    including heart vector projections, QRS complex detection, and heatmap animations.
    """

    # Class constants
    CLASS_NAMES_MAP = {0: "No Wave", 1: "P Wave", 2: "QRS", 3: "T Wave"}
    SEGMENT_COLORS = {0: "silver", 1: "lightblue", 2: "lightcoral", 3: "lightgreen"}
    LABEL_TO_IDX = {v: k for k, v in CLASS_NAMES_MAP.items()}
    INTERNAL_SAMPLING_RATE = 250 # All internal processing assumes this rate (Hz)
    DEFAULT_SOURCE_SAMPLING_RATE = 1000 # Default assumed rate of raw TDMS files
    DEFAULT_SCALING = 2.7 / 1000
    DEFAULT_NUM_CHANNELS = 48
    DEFAULT_MODEL_CHECKPOINT_DIR = "MCG_segmentation/checkpoints/best"

    # Determine device for PyTorch computations
    try:
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
        else:
            DEVICE = torch.device("cpu")
        logging.info(f"Using device: {DEVICE}")
    except Exception as e:
        logging.warning(f"Failed to determine device, defaulting to CPU. Error: {e}")
        DEVICE = torch.device("cpu")

    def __init__(
        self,
        filename="",
        add_filename="",
        log_file_path="",
        sensor_channels_to_exclude=None,
        scaling=DEFAULT_SCALING,
        num_ch=DEFAULT_NUM_CHANNELS,
        model_checkpoint_dir=DEFAULT_MODEL_CHECKPOINT_DIR
    ):
        """Initialize the Analyzer with data and model configurations.

        Args:
            filename (str): Path to the primary TDMS data file.
            add_filename (str): Path to the additional TDMS data file.
            log_file_path (str): Path to the QZFM log file with sensor mappings.
            sensor_channels_to_exclude (dict, optional): Channels to exclude per run key.
            scaling (float): Scaling factor for TDMS data during initial load.
            num_ch (int): Number of channels expected/for plotting.
            model_checkpoint_dir (str): Directory with trained segmentation model.
        """
        # Input validation
        if not isinstance(scaling, (int, float)) or scaling <= 0:
            raise ValueError("Scaling must be a positive number")
        if not isinstance(num_ch, int) or num_ch <= 0:
            raise ValueError("Number of channels must be a positive integer")

        # Initialize attributes
        self.filename = filename
        self.add_filename = add_filename
        self.log_file_path = log_file_path
        self.scaling = scaling
        # Removed self.sampling_rate and self.target_sampling_rate
        self.num_ch = num_ch
        self.sensor_channels_to_exclude = sensor_channels_to_exclude or {}
        self.data = {}
        self.add_data = {}
        self.key_list = []
        self.quspin_gen_dict = {}
        self.quspin_channel_dict = {}
        self.quspin_position_list = []

        # Load data and sensor mappings
        self._load_tdms_files()
        self._load_sensor_log_file()

        # Adjust channel mappings for compatibility
        for key, value in self.quspin_channel_dict.items():
            if abs(value) >= 100:
                sign = -1 if value < 0 else 1
                last_two_digits = abs(value) % 100
                new_value = 31 + last_two_digits
                self.quspin_channel_dict[key] = sign * new_value

        if self.data:
            self.key_list = list(self.data.keys())
            logging.info(f"Available runs: {', '.join(self.key_list)}")

        # Configure plotting
        plt.rcParams['agg.path.chunksize'] = 100000000
        self.cmap = plt.get_cmap('nipy_spectral')
        self.cmaplist = [self.cmap(x) for x in np.linspace(0, 1, num=self.num_ch)]

        # Load segmentation model
        self.model = self._load_segmentation_model(model_checkpoint_dir)

    @staticmethod
    def bandstop_filter(data, center_frequency, bandwidth, sampling_rate, order=4):
        """Apply a bandstop filter to the data.

        Args:
            data (np.ndarray): Input signal.
            center_frequency (float): Center frequency to filter out.
            bandwidth (float): Bandwidth of the filter.
            sampling_rate (float): Sampling rate of the signal.
            order (int): Filter order.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyquist = 0.5 * sampling_rate
        low_cutoff = (center_frequency - bandwidth / 2) / nyquist
        high_cutoff = (center_frequency + bandwidth / 2) / nyquist
        b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandstop')
        return signal.filtfilt(b, a, data)

    @staticmethod
    def bandpass_filter(data, fs, lowcut, highcut, order=4):
        """Apply a bandpass filter to the data.

        Args:
            data (np.ndarray): Input signal.
            fs (float): Sampling frequency.
            lowcut (float): Low cutoff frequency.
            highcut (float): High cutoff frequency.
            order (int): Filter order.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    @staticmethod
    def apply_lowpass_filter(data, sampling_rate, cutoff_frequency, order=2):
        """Apply a lowpass filter to the data.

        Args:
            data (np.ndarray): Input signal.
            sampling_rate (float): Sampling rate.
            cutoff_frequency (float): Cutoff frequency.
            order (int): Filter order.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data, axis=1)

    @staticmethod
    def apply_highpass_filter(data, sampling_rate, cutoff_frequency, order=2):
        """Apply a highpass filter to the data.

        Args:
            data (np.ndarray): Input signal.
            sampling_rate (float): Sampling rate.
            cutoff_frequency (float): Cutoff frequency.
            order (int): Filter order.

        Returns:
            np.ndarray: Filtered signal.
        """
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high')
        return signal.filtfilt(b, a, data, axis=1)

    @staticmethod
    def remove_drift_and_offset(signal, time):
        """Remove drift and offset from the signal using polynomial detrending.

        Args:
            signal (np.ndarray): Input signal.
            time (np.ndarray): Time vector.

        Returns:
            np.ndarray: Detrended signal.
        """
        detrended = signal - np.polynomial.polynomial.polyval(
            time, np.polynomial.polynomial.polyfit(time, signal, 3)
        )
        baseline = np.median(detrended)
        return detrended - baseline

    @staticmethod
    def plot_sensor_matrix(data, time, name, path=None, save=False):
        """
        Plot a grid of time series data (e.g., from a sensor array) as subplots.

        Each subplot shows the time-domain signal from one sensor channel in a structured 
        nrows x ncols grid, typically representing the physical layout of sensors.

        Parameters:
        ----------
        data : np.ndarray
            3D array of shape (nrows, ncols, samples), representing sensor signals.
        time : np.ndarray
            1D time vector corresponding to the third dimension of `data`.
        name : str
            Title for the figure and base filename (if saving).
        path : str, optional
            Directory where the plot will be saved (if `save` is True).
        save : bool, optional
            Whether to save the plot as a PNG image (default is False).
        """
        if len(data.shape) != 3:
            raise ValueError("Data must be a 3D array of shape (nrows, ncols, samples)")
        
        nrows, ncols = data.shape[:2]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True,
                                figsize=(35/2.54, 25/2.54), dpi=100)
        fig.suptitle(name, fontsize=14, fontweight='bold', y=0.98)

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row, col] if nrows > 1 else axes[col]
                signal = data[row, col]
                if len(signal) > 0 and not np.all(signal == 0):
                    ax.plot(time, signal, '-', linewidth=0.8)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                            fontsize=10, color='red', alpha=0.5)
                ax.grid(True, linestyle='dotted', alpha=0.7)

        # Global axis labels
        fig.text(0.5, 0.06, 'Time [s]', ha='center', fontsize=12)
        fig.text(0.06, 0.5, 'Magnetic Field [pT]', va='center', rotation='vertical', fontsize=12)

        plt.subplots_adjust(wspace=0.1, hspace=0.2, left=0.12, bottom=0.1, right=0.98, top=0.93)

        if save and path:
            os.makedirs(path, exist_ok=True)
            plt.savefig(os.path.join(path, f'{name}_sensor_matrix.png'), bbox_inches='tight')
        plt.show()

    @staticmethod
    def _detect_qrs_segments(labels_ch, qrs_label):
        """
        Detects continuous segments where the label matches qrs_label.

        Args:
            labels_ch (np.ndarray): 1D array of labels for a single channel.
            qrs_label (int): The integer label representing the QRS complex.

        Returns:
            List[Tuple[int, int]]: A list of tuples, where each tuple is (start_idx, end_idx)
                                of a QRS segment (end index is exclusive).
        """
        # Get QRS mask
        is_qrs = (labels_ch == qrs_label).astype(np.int8)
        
        # Find continuous segments using run-length encoding approach
        # This handles incomplete segments at boundaries better
        where_changes = np.where(np.diff(is_qrs, prepend=0, append=0) != 0)[0]
        
        # No changes means either all QRS or no QRS
        if len(where_changes) <= 1:
            if is_qrs[0] == 1:  # All signal is QRS
                return [(0, len(is_qrs))]
            else:  # No QRS found
                return []
        
        # Process change points to get valid segments
        segments = []
        
        # Handle changes in pairs
        for i in range(0, len(where_changes) - 1, 2):
            start_idx = where_changes[i]
            # Make sure we don't go out of bounds
            if i + 1 < len(where_changes):
                end_idx = where_changes[i + 1]
                segments.append((start_idx, end_idx))
        
        # Handle odd number of change points (last segment extends to the end)
        if len(where_changes) % 2 == 1:
            segments.append((where_changes[-1], len(is_qrs)))
        
        return segments

    @staticmethod
    def _import_tdms(filename, scaling):
        """Import data from a TDMS file.

        Args:
            filename (str): Path to the TDMS file.
            scaling (float): Scaling factor for the data.

        Returns:
            dict: Dictionary with group names as keys and data arrays as values.
        """
        data = {}
        try:
            with TdmsFile.read(filename) as tdms_file:
                for group in tdms_file.groups():
                    channels = group.channels()
                    if not channels:
                        continue
                    no_of_samples = len(channels[0].read_data(scaled=True))
                    data_array = np.zeros((no_of_samples, len(channels) + 1))
                    data_array[:, 0] = np.linspace(0, no_of_samples - 1, no_of_samples)
                    for i, channel in enumerate(channels):
                        data_array[:, i + 1] = channel.read_data(scaled=True) / scaling
                    data[group.name] = data_array
        except Exception as e:
            logging.error(f"Error importing TDMS file {filename}: {e}")
            return {}
        return data

    def _load_segmentation_model(self, checkpoint_dir):
        """Load the trained ECGSegmenter model from the checkpoint directory.

        Args:
            checkpoint_dir (str): Path to the directory containing model.pth and params.json.

        Returns:
            ECGSegmenter: Loaded model instance, or None if loading fails.
        """
        if ECGSegmenter is None:
            logging.warning("ECGSegmenter not available. Returning None.")
            return None

        best_model_path = os.path.join(checkpoint_dir, "model.pth")
        best_params_path = os.path.join(checkpoint_dir, "params.json")

        if not os.path.exists(best_model_path) or not os.path.exists(best_params_path):
            raise FileNotFoundError(f"Model files not found in {checkpoint_dir}")

        model = UNet1D()
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=self.DEVICE))
            model.to(self.DEVICE)
            model.eval()
            logging.info(f"Model loaded from {best_model_path}")
        except RuntimeError as e:
            logging.error(f"Failed to load model state_dict: {e}")
            raise

        return model

    def _load_tdms_files(self):
        """Load primary and additional TDMS files into data dictionaries."""
        try:
            if self.filename:
                self.data = self._import_tdms(self.filename, self.scaling)
                logging.info(f"Loaded primary TDMS file: {self.filename}")
            if self.add_filename:
                self.add_data = self._import_tdms(self.add_filename, self.scaling)
                logging.info(f"Loaded additional TDMS file: {self.add_filename}")
        except FileNotFoundError as e:
            logging.error(f"TDMS file not found: {e}")
            self.data = {}
            self.add_data = {}
        except Exception as e:
            logging.error(f"Error loading TDMS files: {e}")
            self.data = {}
            self.add_data = {}

    def _load_sensor_log_file(self):
        """Load sensor mapping from the QZFM log file."""
        if not self.log_file_path:
            logging.warning("No log file path provided. Sensor mapping unavailable.")
            return

        try:
            with open(self.log_file_path, 'r') as file:
                log_data = ast.literal_eval(file.read())
            self.quspin_gen_dict = log_data.get('quspin_gen_dict', {})
            self.quspin_channel_dict = log_data.get('quspin_channel_dict', {})
            self.quspin_position_list = log_data.get('quspin_position_list', [])
            logging.info(f"Loaded sensor log file: {self.log_file_path}")
        except FileNotFoundError:
            logging.error(f"Sensor log file not found: {self.log_file_path}")
        except (SyntaxError, ValueError) as e:
            logging.error(f"Invalid format in sensor log file: {e}")
        except Exception as e:
            logging.error(f"Error loading sensor log file: {e}")

    def _heart_beat_score(self, confidence: np.ndarray, labels: np.ndarray, confidence_weight: float = 0.80, plausibility_weight: float = 0.2, zero_input_mask: np.ndarray = None,):
        """
        Calculate the heart beat score based on confidence and plausibility.
        Args:
            confidence: numpy array of shape (num_channels, num_samples). Confidence scores.
            labels: numpy array of shape (num_channels, num_samples). Segmentation labels.
            confidence_weight: Weight for the confidence score.
            plausibility_weight: Weight for the plausibility score.
        Returns:
            numpy array of shape (num_channels,). Final scores for each channel.
        """

        num_channels, _ = confidence.shape
        
        mean_confidence = np.mean(confidence, axis=-1) # Shape: (num_channels,)


        # Calculate segment percentages
        segment_percentages = np.zeros((num_channels, 4))
        total_samples_per_channel = labels.shape[-1]

        if total_samples_per_channel > 0:
            for segment_type in range(4): # 0, 1, 2, 3
                segment_counts = np.sum(labels == segment_type, axis=1) # Sum over time axis
                segment_percentages[:, segment_type] = (segment_counts / total_samples_per_channel) * 100
        # Else: percentages remain zero

        # Define ideal ranges (consider making these class attributes or constants)
        p_wave_range = (8, 15)      # Relaxed P-wave: 8-15%
        qrs_range = (8, 15)         # Relaxed QRS: 8-15%
        t_wave_range = (15, 30)     # Relaxed T-wave: 15-30%

        # Calculate plausibility scores (vectorized)
        p_percent = segment_percentages[:, self.LABEL_TO_IDX["P Wave"]]
        qrs_percent = segment_percentages[:, self.LABEL_TO_IDX["QRS"]]
        t_percent = segment_percentages[:, self.LABEL_TO_IDX["T Wave"]]

        # Deviations (calculate difference from range boundaries)
        p_dev = np.maximum(0, p_wave_range[0] - p_percent) + np.maximum(0, p_percent - p_wave_range[1])
        qrs_dev = np.maximum(0, qrs_range[0] - qrs_percent) + np.maximum(0, qrs_percent - qrs_range[1])
        t_dev = np.maximum(0, t_wave_range[0] - t_percent) + np.maximum(0, t_percent - t_wave_range[1])

        total_deviation = p_dev + qrs_dev + t_dev

        # Normalize deviation - use a max possible deviation or just invert
        # Simple inversion: Higher score for lower deviation. Add epsilon for stability.
        # Scale factor can be adjusted based on expected deviation range.
        plausibility_scores = 1.0 / (1.0 + total_deviation * 0.1) # Smaller multiplier = less sensitive

        # Combine scores
        # Normalize confidence to avoid scale issues if confidence varies widely

        final_scores = (confidence_weight * mean_confidence) + (plausibility_weight * plausibility_scores)

        if zero_input_mask is not None:
            final_scores[zero_input_mask] = 0.0

        return final_scores, mean_confidence, segment_percentages, plausibility_scores
    
    def _segment_heart_beat_intervall(self, data: torch.Tensor, min_duration_sec: int = 0.040):
        """
        Segment the entire run (potentially T > 2000) using a sliding window.
        Assumes input data is already at the INTERNAL_SAMPLING_RATE (250 Hz).
        Args:
            data: numpy array of shape (b, T). Assumes data is pre-filtered
                and at 250 Hz. Normalization happens per window.
            window_size: Size of the sliding window for inference.
            overlap: Fraction of overlap between consecutive windows (0.0 to < 1.0).
        Returns:
            Tuple of (final segmentation labels (b, T),
                    final confidence scores (b, T))
        """
        if data.numel() == 0:
            warnings.warn("No data to segment.")
            batch_size = data.shape[0]
            return np.empty((batch_size, 0), dtype=int), np.empty((batch_size, 0), dtype=float)
        
        # Constants
        max_len = 2000
        
        # Clip long sequences
        if data.shape[-1] > max_len:
            warnings.warn(f"Data length ({data.shape[-1]}) exceeds maximum ({max_len}). Truncating.")
            data = data[..., :max_len]
        
        data = data.to(self.DEVICE)
        
        with torch.no_grad():
            logits = self.model(data)
            probabilities = torch.softmax(logits, dim=-1)
            confidence_scores_pt, predicted_indices_pt = torch.max(probabilities, dim=-1)
            predicted_indices = predicted_indices_pt.cpu().numpy()
            confidence_scores = confidence_scores_pt.cpu().numpy()
        
        # --- Postprocessing to fix short artifacts using paper's two-rule approach ---
        batch_size, time_steps = predicted_indices.shape
        
        for b in range(batch_size):
            labels_arr = predicted_indices[b]
            i = 0
            
            while i < time_steps:
                current_label = labels_arr[i]
                start = i
                
                # Find the end of current segment
                while i < time_steps and labels_arr[i] == current_label:
                    i += 1
                end = i  # exclusive
                
                segment_length = end - start
                
                # Check if segment is too short
                if segment_length < min_duration_sec * self.INTERNAL_SAMPLING_RATE:
                    # Get neighboring labels
                    left_label = labels_arr[start - 1] if start > 0 else None
                    right_label = labels_arr[end] if end < time_steps else None
                    
                    # Apply paper's two-rule approach
                    if left_label is not None and right_label is not None:
                        # Both neighbors exist
                        if left_label == right_label:
                            # Rule 1: Same labels -> merge by extending the common label
                            new_label = left_label
                        else:
                            # Rule 2: Different labels -> assign as "none"
                            new_label = 0
                    elif left_label is not None:
                        # Only left neighbor exists
                        new_label = left_label
                    elif right_label is not None:
                        # Only right neighbor exists  
                        new_label = right_label
                    else:
                        # No neighbors exist (shouldn't happen in practice)
                        new_label = 0
                    
                    # Reassign short segment
                    labels_arr[start:end] = new_label
            
            predicted_indices[b] = labels_arr
        
        return predicted_indices, confidence_scores

    def _change_to_consistent_coordinate_system(self, data):
        """Adjust data to a consistent coordinate system based on sensor mappings.

        Args:
            data (np.ndarray): Input data, shape (num_channels, num_samples).

        Returns:
            np.ndarray: Adjusted data.
        """
        for quspin, ch in self.quspin_channel_dict.items():
            sign = -1 if str(ch).startswith('-') else 1
            ch_idx = abs(int(ch))
            data[ch_idx] *= sign
            if 'y' in quspin or self.quspin_gen_dict.get(quspin[:-2], 0) == 2:
                data[ch_idx] *= -1
        return data

    def _apply_ICA(self, data, max_iter=1000):
        """
        Apply Independent Component Analysis (ICA) to the data. Helper function.

        Args:
            data (np.ndarray): Input data, shape (channels, samples).
            max_iter (int): Maximum iterations for ICA.
            n_components (int, optional): Number of components to extract. If None, defaults to number of non-zero channels.


        Returns:
            tuple: (ICA components, mixing matrix, mean, non_zero_indices, original_channel_count) or
                (None, None, None, None, None) if ICA fails.
        """
        # Store original shape for reconstruction
        original_shape = data.shape
        
        # Reshape and identify non-zero channels
        data_reshaped = data.reshape(-1, data.shape[-1])
        non_zero_mask = ~np.all(data_reshaped == 0, axis=1)
        non_zero_indices = np.where(non_zero_mask)[0]
        arr_cleaned = data_reshaped[non_zero_mask]
        
        if arr_cleaned.shape[0] == 0:
            logging.warning("No non-zero channels for ICA")
            return None, None, None, None, None
        
        try:
            # Apply ICA - transpose for sklearn's expected format
            ica = FastICA(n_components=arr_cleaned.shape[0], random_state=0, max_iter=max_iter)
            components = ica.fit_transform(arr_cleaned.T).T
            
            # Store mixing matrix and mean for reconstruction
            mixing_matrix = ica.mixing_
            mean = ica.mean_
            
            return components, mixing_matrix, mean, non_zero_indices, original_shape
        except Exception as e:
            logging.error(f"ICA failed: {e}")
            return None, None, None, None, None

    def plot_lsd_multichannel(self, data, noise_theos, channels, labels=None, path="", name = "", save = False):

        """
        Plots the Linear Spectral Density (LSD) of multichannel time-series data using Welch's method.

        This method computes and visualizes the LSD for specified channels, optionally overlaying theoretical noise 
        levels and saving the plot to disk. The secondary Y-axis shows the corresponding linear amplitude scale.

        Parameters:
        ----------
        data : np.ndarray
            2D array where each row corresponds to a signal from one channel (shape: channels x time).
        noise_theos : list or np.ndarray
            Theoretical noise floor values for each channel, used for reference lines.
        name : str
            A string used in the plot title and filename (if saved).
        labels : list of str
            List of labels for each channel to display in the legend.
        channels : list of int
            Indices of the channels to plot.
        path : str
            Directory path where the plot will be saved (if `save` is True).
        save : bool, optional
            If True, saves the plot as a PNG file in the given `path` (default is False).

        Notes:
        -----
        - Uses a Nuttall window and NENBW of 1.9761 for spectral estimation.
        - Adds a secondary y-axis to convert LSD to linear amplitude scale.
        - The function displays the plot regardless of the `save` option.

        Raises:
        ------
        ValueError
            If the input arrays/lists are inconsistent in length or incorrectly formatted.
        """

        nenbw=1.9761
        fig, elem= plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(10,5),dpi=100)  

        for ind,ch in enumerate(channels):
            f_bins,Pxx=signal.welch(data[ch], fs=self.INTERNAL_SAMPLING_RATE, nperseg=int(self.INTERNAL_SAMPLING_RATE)*100, window='nuttall', return_onesided=True, scaling='density')
            
            Lxx=np.sqrt(Pxx)
            enbw=f_bins[1]*nenbw #calculated from acutal f_res
            if labels is None:
                elem.loglog(f_bins,Lxx,color=self.cmaplist[ind*int(len(self.cmaplist)/len(channels))], alpha = 0.5)
            else:
                elem.loglog(f_bins,Lxx,label=labels[ind],color=self.cmaplist[ind*int(len(self.cmaplist)/len(channels))], alpha = 0.5)
                
            if noise_theos[0] !=np.mean(noise_theos) and (ind<len(channels)):
                
                elem.plot(f_bins,np.full(len(f_bins),noise_theos[ind]),
                        label='Sensor theo.',color=self.cmaplist[ind],linestyle='-.')

        def forward(x):
            return x*np.sqrt(enbw)*np.sqrt(2)
        def inverse(x):
            return x/(np.sqrt(enbw)*np.sqrt(2))

        elem.plot(f_bins,np.full(len(f_bins),noise_theos[ind]),
                label='Sensor theo.',color=self.cmaplist[ind],linestyle='-.')
        secax = elem.secondary_yaxis('right', functions=(forward, inverse))
        secax.set_ylabel("LS (linear amplitude) [$ pT$]")
        elem.set_xlabel('Frequency [Hz]')
        elem.set_ylabel('LSD [$ pT$/$\sqrt{Hz}$]')
        fig.suptitle('Magnetic flux linear spectral density '
                    '[$ pT$/$\sqrt{Hz}$], NENBW=1.9761 bins\n '
                    '$f_s$=%d Hz, $f_{res}$=%.3f Hz \n' %(self.INTERNAL_SAMPLING_RATE,f_bins[1])+name,size='small',
                    y=1.0)
        elem.set_xlim(0.1)
        elem.legend(loc='lower center',bbox_to_anchor=(0.5,-0.25),ncol=4)
        elem.grid(alpha=0.7)
        elem.minorticks_on()
        elem.grid(True,which='minor',linestyle='dotted',alpha=0.7)
        # elem.set_yticks([10,100,500, 1000, 5000, 1e4,5e4,1e5,5e5])
        plt.subplots_adjust(top = 0.9, bottom = 0.2)
        if save:
            plt.savefig(path+f'{name}_LSD.png')
        plt.show()

    def plot_heart_vector_projection(self, component1, component2, proj_name, title_suffix="", ax=None, show=True):
        """Plot a 2D projection of the heart vector with metrics and filled area."""


        standalone_plot = ax is None
        if standalone_plot:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=100)

        # Plot styling
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f5f5f5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='both', which='major', labelsize=10, colors='gray')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

        # Choose plot color
        color_map = {"xy-Projection": 'dodgerblue', "xz-Projection": 'orange', "yz-Projection": 'forestgreen'}
        plot_color = color_map.get(proj_name, 'purple')

        # Plot line
        ax.plot(component1, component2, label=proj_name, color=plot_color, linewidth=2.0)

        # Fill enclosed area
        ax.fill(component1, component2, color=plot_color, alpha=0.3, label=f'{proj_name} Area')

        # Add directional arrows
        arrow_stride = max(1, len(component1) // 20)
        for i in range(0, len(component1) - arrow_stride, arrow_stride):
            next_idx = min(i + 1, len(component1) - 1)
            vec = np.array([component1[next_idx], component2[next_idx]]) - np.array([component1[i], component2[i]])
            if np.linalg.norm(vec) > 1e-6:
                ax.annotate('', xy=(component1[next_idx], component2[next_idx]), xytext=(component1[i], component2[i]),
                            arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', linewidth=1.5))

        # Calculate metrics
        metrics_text = "Metrics N/A"
        if len(component1) > 2:
            try:
                # Shoelace formula for area
                area = 0.5 * np.abs(np.dot(component1, np.roll(component2, 1)) - np.dot(component2, np.roll(component1, 1)))
                magnitudes = component1**2 + component2**2
                t_max_idx = np.argmax(magnitudes)

                t_distance = np.linalg.norm([component1[t_max_idx] - component1[0], component2[t_max_idx] - component2[0]])
                # Extra metrics: Compactness (Area / Perimeter²)
                perimeter = np.sum(np.linalg.norm(np.diff(np.stack([component1, component2], axis=1), axis=0), axis=1))
                compactness = (4 * np.pi * area) / (perimeter**2 + 1e-9)

                # Average direction (in radians and degrees)
                mean_dx = np.mean(component1)
                mean_dy = np.mean(component2)
                avg_angle_rad = np.arctan2(mean_dy, mean_dx)
                avg_angle_deg = np.degrees(avg_angle_rad)


                metrics_text = (f'\n Area: {area:.3f}\n'
                                f'T-Dist: {t_distance:.2f}\n'
                                f'Compact: {compactness:.4f}\n'
                                f'Angle: {avg_angle_deg:.1f}°')
                
            except Exception as e:
                logging.warning(f"Error calculating metrics for {proj_name}: {e}")
                metrics_text = "Metrics Error"

        # Add metrics text
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=9, fontweight='bold', color='black',
                verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.3'))

        # Set axes limits
        max_limit = max(np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_ylim()))) * 1.1
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)

        # Set labels
        labels = {
            "xy-Projection": ('$B_x$ [pT]', '$B_y$ [pT]'),
            "xz-Projection": ('$B_x$ [pT]', '$B_z$ [pT]'),
            "yz-Projection": ('$B_y$ [pT]', '$B_z$ [pT]')
        }
        ax.set_xlabel(labels.get(proj_name, ('Component 1', 'Component 2'))[0], fontsize=12)
        ax.set_ylabel(labels.get(proj_name, ('Component 1', 'Component 2'))[1], fontsize=12)

        # Format ticks
        locs_x, locs_y = ax.get_xticks(), ax.get_yticks()
        xlabels = [f"{x:.0f}" if abs(x) > 1e-3 * max_limit else "" for x in locs_x]
        ylabels = [f"{y:.0f}" if abs(y) > 1e-3 * max_limit else "" for y in locs_y]
        ax.set_xticks(locs_x)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(locs_y)
        ax.set_yticklabels(ylabels)

        ax.legend(fontsize=10, loc='lower right', frameon=True, facecolor='white', edgecolor='gray', framealpha=0.8)

        if standalone_plot:
            plt.suptitle(f"Heart Vector Projection: {proj_name}{' - ' + title_suffix if title_suffix else ''}",
                        fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if show: plt.show()

        # print out the metrics
        logging.info(f"Metrics for {proj_name}: {metrics_text}")

        return ax, {"Area": area, "T-Dist": t_distance, "Compact": compactness, "Angle": avg_angle_deg} 

    def plot_all_heart_vector_projections(self, heart_vector_components, title_suffix="", save_path=None):
        """Plot XY, XZ, and YZ projections of the heart vector.

        Args:
            heart_vector_components (np.ndarray): Shape (3, num_samples) with Bx, By, Bz.
            title_suffix (str): Suffix for the plot title.
            save_path (str, optional): Path to save the plot.
        """
        if heart_vector_components.shape[0] != 3 or heart_vector_components.ndim != 2:
            logging.error("heart_vector_components must have shape (3, num_samples)")
            return

        bx, by, bz = heart_vector_components
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        fig.suptitle(f"Mean Heart Vector Projections{': ' + title_suffix if title_suffix else ''}", fontsize=16, fontweight='bold', y=0.98)

        for ax, (comp1, comp2, name) in zip(axes, [(bx, by, 'xy-Projection'), (bx, bz, 'xz-Projection'), (by, bz, 'yz-Projection')]):
            self.plot_heart_vector_projection(comp1, comp2, name, ax=ax)
            ax.set_title("")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            try:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                logging.info(f"Saved projections plot to: {save_path}")
            except Exception as e:
                logging.error(f"Failed to save projections plot: {e}")
        plt.show()

    def plot_segmented_signal(self, signal, pred, axs=None):
        """
        Segments the heart beat interval using the loaded model. Assumes 250Hz input.
        Classifies each data point as: 0) No Wave, 1) P-Wave, 2) QRS, 3) T-Wave.

        Args:
            data: torch Tensor of shape (b, 1, T) containing normalized, 250Hz data.
            min_duration_samples: Minimum duration of segments to be considered valid.
        Returns:
            Tuple of (numpy array of segment classifications (b, T), numpy array of confidence scores (b, T))
        """
        signal = signal.squeeze()
        pred = pred.squeeze()
        t = np.arange(len(signal))

        if axs is None:
            fig, axs = plt.subplots(figsize=(15, 8))
            standalone = True
        else:
            standalone = False

        axs.plot(t, signal, color='black', linewidth=0.8, label='Signal (Processed)')
        axs.grid(True, linestyle=':', alpha=0.7)

        current_start_idx = 0
        legend_handles_map = {}
        line_signal, = axs.plot([], [], color='black', linewidth=0.8, label='Signal')
        legend_handles_map['Signal'] = line_signal

        for k in range(1, len(pred)):
            if pred[k] != pred[current_start_idx]:
                label_idx = pred[current_start_idx]
                label_name = self.CLASS_NAMES_MAP.get(label_idx, f"Class {label_idx}")
                color = self.SEGMENT_COLORS.get(label_idx, 'gray')
                h = axs.axvspan(t[current_start_idx] - 0.5, t[k] - 0.5, color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
                if f'Pred: {label_name}' not in legend_handles_map:
                    legend_handles_map[f'Pred: {label_name}'] = h
                current_start_idx = k

        label_idx = pred[current_start_idx]
        label_name = self.CLASS_NAMES_MAP.get(label_idx, f"Class {label_idx}")
        color = self.SEGMENT_COLORS.get(label_idx, 'gray')
        h = axs.axvspan(t[current_start_idx] - 0.5, t[-1] + 0.5, color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
        if f'Pred: {label_name}' not in legend_handles_map:
            legend_handles_map[f'Pred: {label_name}'] = h

        combined_handles = [legend_handles_map['Signal']]
        combined_labels = ['Signal']
        for i in sorted(self.CLASS_NAMES_MAP.keys()):
            label_name_pred = f'Pred: {self.CLASS_NAMES_MAP[i]}'
            if label_name_pred in legend_handles_map:
                patch = plt.Rectangle((0, 0), 1, 1, fc=self.SEGMENT_COLORS.get(i, 'gray'), alpha=0.3)
                combined_handles.append(patch)
                combined_labels.append(label_name_pred)

        axs.legend(combined_handles, combined_labels, loc='upper right', fontsize='x-small', ncol=2)
        if standalone:
            plt.show()

    def butterfly_plot(self, data, time, num_ch, name, path=None, save=False):
        """
        Plot time series data for multiple channels as a 'butterfly plot'.

        Parameters:
        ----------
        data : np.ndarray
            2D array of shape (num_channels, num_samples).
        time : np.ndarray
            1D array of time points.
        num_ch : int
            Number of channels to plot.
        name : str
            Plot title and filename base (if saving).
        path : str, optional
            Directory where the plot will be saved if `save` is True.
        save : bool
            If True, saves the plot as a PNG image.
        """
        fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
        fig.suptitle(name, fontsize=14, fontweight='bold', y=1.02)

        for i in range(num_ch):
            ax.plot(time, data[i], alpha=0.5, linewidth=1.0,
                    color=self.cmaplist[i % len(self.cmaplist)],
                    label=f"Ch {i+1}" if num_ch <= 10 else None)

        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Magnetic Field B [pT]', fontsize=12)
        ax.grid(True, which='major', linestyle='-', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        ax.minorticks_on()

        # Add legend if channel count is manageable
        if num_ch <= 10:
            ax.legend(loc='upper right', fontsize=9, frameon=False, ncol=2)

        # Optional annotations for first and last channel
        ax.annotate("Ch 1", xy=(time[-1], data[0][-1]), xytext=(-60, 5),
                    textcoords='offset points', fontsize=8, color=self.cmaplist[0])
        ax.annotate(f"Ch {num_ch}", xy=(time[-1], data[num_ch-1][-1]), xytext=(-60, -10),
                    textcoords='offset points', fontsize=8, color=self.cmaplist[num_ch-1])

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save and path:
            plt.savefig(os.path.join(path, f'{name}_butterfly_plot.png'), bbox_inches='tight')
        plt.show()

    def segment_entire_run(self, data: np.ndarray, window_size: int = 2000, overlap: float = 0.5):
        """
        Segment the entire run (potentially T > 2000) using a sliding window.

        Args:
            data: numpy array of shape (b, T). Assumes data is pre-filtered.
                Normalization happens per window. Sampling rate is handled.
            window_size: Size of the sliding window for inference.
            overlap: Fraction of overlap between consecutive windows (0.0 to < 1.0).

        Returns:
            Tuple of (final segmentation labels (b, T_resampled),
                    resampled data (b, T_resampled),
                    final confidence scores (b, T_resampled))
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D (b, T)")

        data = data.reshape(data.shape[0], 1, data.shape[1]) # Ensure data is 3D

        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 and < 1.0")


        data = savgol_filter(data, window_length=15, polyorder=2, axis=-1)
        # --- Model Constraints ---
        max_model_len = 2000
        if window_size > max_model_len:
            warnings.warn(f"Window size {window_size} exceeds max sequence length {max_model_len}. Clamping to {max_model_len}.")
            window_size = max_model_len
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")

        # --- Sliding Window ---
        batch_size, _, T = data.shape
        step_size = max(1, int(window_size * (1 - overlap))) # Ensure step_size is at least 1

        # Prepare storage for results - Initialize with low confidence
        final_labels = np.full((batch_size, T), -1, dtype=int) # Use -1 as initial label
        final_confidences = np.full((batch_size, T), -1.0, dtype=float) # Use -1.0 as initial confidence

        starts = range(0, T - window_size + 1, step_size)
        # Ensure the last part is processed if it doesn't align perfectly
        if T > 0 and (T - window_size) % step_size != 0:
            starts = list(starts) + [T - window_size]
            # Remove duplicate if the last calculated start is the same
            if len(starts) > 1 and starts[-1] == starts[-2]:
                starts.pop()


        for start in starts:
            end = start + window_size
            segment_np = data[:, :, start:end]

            # Convert to Tensor for model
            segment = torch.from_numpy(segment_np.astype(np.float32)).to(self.DEVICE)

            # --- Preprocessing (Apply per segment as in original) ---
            # Use torch ops for potentially faster computation if on GPU
            signal_mean = segment.mean(dim=-1, keepdim=True)
            # Check for NaN/Inf before subtraction
            valid_mean = ~torch.isnan(signal_mean) & ~torch.isinf(signal_mean)
            segment = torch.where(valid_mean, segment - signal_mean, segment)

            # Normalize by abs max
            max_vals = segment.abs().max(dim=-1, keepdim=True)[0] # [0] to get values
            # Add small epsilon to avoid division by zero
            segment = torch.where(max_vals > 1e-6, segment / (max_vals + 1e-9), segment)
            # --- Get Segment Labels ---
            labels, confidences = self._segment_heart_beat_intervall(segment)

            # --- Combine Results (Optimized) ---
            # Update final arrays where current confidence is higher
            current_slice = slice(start, end)
            mask = confidences > final_confidences[:, current_slice]
            final_labels[:, current_slice][mask] = labels[mask]
            final_confidences[:, current_slice][mask] = confidences[mask]

        # Replace any remaining -1 labels (where no window covered or confidence was always low) with 'No Wave'
        no_wave_label = self.LABEL_TO_IDX["No Wave"]
        mask_uncovered = final_labels == -1
        final_labels[mask_uncovered] = no_wave_label
        # Assign default confidence (e.g., 0.5) to these uncovered/low-conf areas
        final_confidences[mask_uncovered] = 0.5

        # Squeeze the channel dimension out of the returned data
        return final_labels, final_confidences
    
    def find_cleanest_channel(self, data: np.ndarray, window_size: int = 2000, overlap: float = 0.5, print_results: bool = True, confidence_weight: float = 0.80, plausibility_weight: float = 0.2):
        """
        Find the channel with the clearest signal based on segmentation confidence
        and physiological plausibility. Assumes input data is at 250 Hz.

        Args:
            data: numpy array of shape (num_channels, num_samples). Data at 250 Hz.
            window_size: Window size for segmentation.
            overlap: Overlap between windows.
            print_results (bool): Whether to print detailed scores.
            confidence_weight (float): Weight for mean confidence in scoring.
            plausibility_weight (float): Weight for plausibility in scoring.


        Returns:
            Tuple[int, np.ndarray,  np.ndarray, np.ndarray]:
                (Index of the best channel, labels, confidence, final scores for all channels)
                Returns index 0 and empty arrays if input is invalid or segmentation fails.

        """
        if data.ndim != 2:
            raise ValueError(f"Input data must be 2D (num_channels, num_samples), got shape {data.shape}")
        num_channels, num_samples = data.shape
        if num_channels == 0 or num_samples == 0:
            warnings.warn("Input data for find_cleanest_channel is empty.")
            # Return dummy values
            return 0, (np.empty((0,0), dtype=int), np.empty((0,0)), np.empty((0,0), dtype=float))


        # Segment all channels - segment_entire_run handles batch (channel) dimension
        # Note: Pass short_segment_threshold here if needed, otherwise use default
        labels, confidence = self.segment_entire_run(data, window_size, overlap)

        # Identify channels with all-zero input data
        zero_input_mask = np.all(data == 0, axis=1)  # Shape: (num_channels,)

        # --- Scoring ---
        if labels.size == 0: # Handle case where segmentation returned empty
            warnings.warn("Segmentation returned empty results in find_cleanest_channel.")
            return 0, (labels, confidence)

        # Calculate mean confidence (axis=1 operates over time dimension T)
        final_scores, mean_confidence, segment_percentages, plausibility_scores = self._heart_beat_score(confidence, labels, confidence_weight, plausibility_weight, zero_input_mask)

        # Find best channel
        best_channel = np.argmax(final_scores) if final_scores.size > 0 else 0

        if print_results:
            # --- Optional: Print results (keep original formatting) ---
            print("\nChannel Selection Results:")
            print(f"{'Channel':<10}{'Conf':<12}{'P-Wave %':<12}{'QRS %':<12}{'T-Wave %':<12}{'Plausibility':<15}{'Final Score':<12}")
            print("-" * 85)
            for channel in range(num_channels):
                print(f"{channel+1:<10}" 
                    f"{mean_confidence[channel]:<12.4f}"
                    f"{segment_percentages[channel, self.LABEL_TO_IDX['P Wave']]:<12.2f}"
                    f"{segment_percentages[channel, self.LABEL_TO_IDX['QRS']]:<12.2f}"
                    f"{segment_percentages[channel, self.LABEL_TO_IDX['T Wave']]:<12.2f}"
                    f"{plausibility_scores[channel]:<15.4f}"
                    f"{final_scores[channel]:<12.4f}")
            print("\nBest Channel Summary:")
            print(f"{'Channel':<10}: {best_channel+1}") # 0-based index
            if num_channels > 0:
                print(f"{'Conf':<10}: {mean_confidence[best_channel]:.4f}")
                print(f"{'Plausibility':<10}: {plausibility_scores[best_channel]:.4f}")
                print(f"{'Final Score':<10}: {final_scores[best_channel]:.4f}")
                print("Segment Distribution:")
                print(f"  P-Wave % : {segment_percentages[best_channel, self.LABEL_TO_IDX['P Wave']]:.2f}%")
                print(f"  QRS %    : {segment_percentages[best_channel, self.LABEL_TO_IDX['QRS']]:.2f}%")
                print(f"  T-Wave % : {segment_percentages[best_channel, self.LABEL_TO_IDX['T Wave']]:.2f}%")
            # --- End Print ---

        # Return 0-based index and the full results
        return best_channel, labels, confidence, final_scores

    def detect_qrs_complex_peaks_cleanest_channel(self, data: np.ndarray, confidence_threshold: float = 0.7, min_qrs_length_sec: float = 0.08, min_distance_sec: float = 0.3, print_heart_rate: bool = False):
        """
        Detects QRS complex peaks on the *cleanest* channel. Assumes input data is at 250 Hz.

        Args:
            data: numpy array of shape (num_channels, num_samples). Data at 250 Hz.
            confidence_threshold: Min avg confidence for a QRS segment to be considered.
            min_qrs_length_sec: Minimum duration (in seconds) for a QRS segment.
            min_distance_sec: Minimum distance between detected peaks in seconds.
            print_heart_rate: If True, calculates and prints HR and HRV for the cleanest channel.

        Returns:
            Tuple[List[int], int, np.ndarray, Optional[float], Optional[float]]:
                - List of peak indices in the 250Hz signal for the cleanest channel.
                - Index of the cleanest channel used for peak detection (0-based).
                - Segmentation labels for all channels (num_channels, T).
                - Average heart rate for the cleanest channel (bpm), or None.
                - HRV (SDNN in ms) for the cleanest channel, or None.
        """            
        # 1. Find cleanest channel and get segmentations
        best_channel_idx, labels, confidence, _ = self.find_cleanest_channel(data, print_results=False) 
        
        if labels.size == 0:
            warnings.warn("Segmentation data is empty, cannot detect peaks.")
            return [], best_channel_idx, 0.0, 0.0

        # Select data for the best channel
        labels_ch = labels[best_channel_idx]
        data_ch = data[best_channel_idx]
        confidence_ch = confidence[best_channel_idx]

        # Convert min length from seconds to samples
        min_qrs_length_samples = int(min_qrs_length_sec * self.INTERNAL_SAMPLING_RATE)
        min_distance_samples = int(min_distance_sec * self.INTERNAL_SAMPLING_RATE)  # Convert min distance to samples

        # 2. Find QRS segments using improved method
        qrs_label = self.LABEL_TO_IDX["QRS"]
        qrs_segments = self._detect_qrs_segments(labels_ch, qrs_label)
  
        peak_positions = []

        # 3. Process each QRS segment
        for start, end in qrs_segments:
            length = end - start
            if length < min_qrs_length_samples:
                continue

            # Check average confidence for the segment
            segment_confidence = confidence_ch[start:end]
            if np.mean(segment_confidence) < confidence_threshold:
                continue

            # Extract QRS waveform segment
            qrs_segment = data_ch[start:end]

            # Find peak within the segment (robust max absolute value)
            peak_relative_idx = np.argmax(np.abs(qrs_segment))
            peak_absolute_idx = start + peak_relative_idx
            peak_positions.append(peak_absolute_idx)

        # 4. Post-process to ensure minimum distance between peaks
        peak_positions = sorted(list(set(peak_positions)))  # Ensure unique and sorted
        filtered_peaks = []

        for peak in peak_positions:
            if not filtered_peaks or peak - filtered_peaks[-1] >= min_distance_samples:
                filtered_peaks.append(peak)

        peak_positions = filtered_peaks

        # 5. Calculate HR/HRV (optional)
        heart_rate = None
        hrv_sdnn_ms = None

        if print_heart_rate:
            if len(peak_positions) > 1:
                rr_intervals_samples = np.diff(peak_positions)
                rr_intervals_sec = rr_intervals_samples / self.INTERNAL_SAMPLING_RATE
                rr_intervals_ms = rr_intervals_sec * 1000

                # Basic outlier removal for RR intervals
                plausible_rr_ms = rr_intervals_ms[(rr_intervals_ms > 200) & (rr_intervals_ms < 2000)]

                if len(plausible_rr_ms) > 1:
                    mean_rr_sec = np.mean(plausible_rr_ms) / 1000
                    heart_rate = 60 / mean_rr_sec
                    hrv_sdnn_ms = np.std(plausible_rr_ms)  # SDNN in ms

                    logging.info(f"Heart Rate: {heart_rate:.2f} bpm")
                    logging.info(f"Heart Rate Variability (SDNN): {hrv_sdnn_ms:.2f} ms")
                    logging.info(f"Number of detected peaks: {len(peak_positions)}")
                    logging.info(f"Number of plausible RR intervals used: {len(plausible_rr_ms)}")
                else:
                    logging.warning("Not enough plausible RR intervals detected to calculate stable HR/HRV.")
                    logging.warning(f"Number of detected peaks: {len(peak_positions)}")

            elif len(peak_positions) == 1:
                logging.warning("Only one peak detected. Cannot calculate HR/HRV.")
            else:
                logging.warning("No peaks detected.")

        return peak_positions, best_channel_idx, labels, heart_rate, hrv_sdnn_ms

    def detect_qrs_complex_peaks_all_channels(self, data: np.ndarray, confidence_threshold: float = 0.7, min_qrs_length_sec: float = 0.08, min_distance_sec: float = 0.3, print_heart_rate: bool = False):
        """
        Detects QRS complex peaks for *all* channels independently. Assumes input data is at 250 Hz.

        Args:
            data: numpy array of shape (num_channels, num_samples). Data at 250 Hz.
            confidence_threshold: Min avg confidence for a QRS segment to be considered.
            min_qrs_length_sec: Minimum duration (in seconds) for a QRS segment.
            min_distance_sec: Minimum distance between detected peaks in seconds.
            print_heart_rate: If True, calculates and prints average HR and HRV across channels.

        Returns:
            Tuple[Dict[int, List[int]], int, np.ndarray, Optional[float], Optional[float]]:
                - Dictionary {channel_idx: [peak_indices]}
                - Index of the cleanest channel (determined internally).
                - Segmentation labels for all channels (num_channels, T).
                - Average heart rate across *valid* channels (bpm), or None.
                - Average HRV (SDNN in ms) across *valid* channels, or None.
        """
        cleanest_channel, labels, confidence, _ = self.find_cleanest_channel(data, print_results=False)
        
        if labels.size == 0:
            warnings.warn("Segmentation data is empty, cannot detect peaks.")
            return {}, cleanest_channel, 0.0, 0.0

        num_channels = labels.shape[0]
        min_qrs_length_samples = int(min_qrs_length_sec * self.INTERNAL_SAMPLING_RATE)
        min_distance_samples = int(min_distance_sec * self.INTERNAL_SAMPLING_RATE)  # Convert min distance to samples
        qrs_label = self.LABEL_TO_IDX["QRS"]
        peak_positions_all_channels = {}

        all_heart_rates = []
        all_hrv_sdnn = []

        # Process each channel independently
        for ch_idx in range(num_channels):
            # Use the improved QRS segment detection
            qrs_segments = self._detect_qrs_segments(labels[ch_idx], qrs_label)
            peak_positions = []

            for start, end in qrs_segments:
                length = end - start
                if length < min_qrs_length_samples:
                    continue

                segment_confidence = confidence[ch_idx][start:end]
                if np.mean(segment_confidence) < confidence_threshold:
                    continue

                qrs_segment = data[ch_idx][start:end]
                peak_relative_idx = np.argmax(np.abs(qrs_segment))
                peak_absolute_idx = start + peak_relative_idx
                peak_positions.append(peak_absolute_idx)

            # Post-process to ensure minimum distance between peaks
            peak_positions = sorted(list(set(peak_positions)))  # Ensure unique and sorted
            filtered_peaks = []

            for peak in peak_positions:
                if not filtered_peaks or peak - filtered_peaks[-1] >= min_distance_samples:
                    filtered_peaks.append(peak)

            peak_positions = filtered_peaks
            peak_positions_all_channels[ch_idx] = peak_positions

            # Calculate HR and HRV
            if len(peak_positions) > 1:
                rr_intervals_samples = np.diff(peak_positions)
                rr_intervals_sec = rr_intervals_samples / self.INTERNAL_SAMPLING_RATE
                rr_intervals_ms = rr_intervals_sec * 1000
                plausible_rr_ms = rr_intervals_ms[(rr_intervals_ms > 200) & (rr_intervals_ms < 2000)]

                if len(plausible_rr_ms) > 1:
                    mean_rr_sec = np.mean(plausible_rr_ms) / 1000
                    heart_rate = 60 / mean_rr_sec
                    hrv_sdnn_ms = np.std(plausible_rr_ms)
                    all_heart_rates.append(heart_rate)
                    all_hrv_sdnn.append(hrv_sdnn_ms)

        avg_heart_rate = np.mean(all_heart_rates) if all_heart_rates else 0.0
        std_deviation_hr = np.std(all_heart_rates) if all_heart_rates else 0.0
        avg_hrv_sdnn = np.mean(all_hrv_sdnn) if all_hrv_sdnn else 0.0
        std_deviation_hrv = np.std(all_hrv_sdnn) if all_hrv_sdnn else 0.0

        if print_heart_rate:
            if len(all_heart_rates) > 0:
                logging.info(f"Average Heart Rate: {avg_heart_rate:.2f} +/- {std_deviation_hr:.2f} bpm")
                logging.info(f"Average Heart Rate Variability (SDNN): {avg_hrv_sdnn:.2f} +/- {std_deviation_hrv:.2f} ms")
            else:
                logging.warning("No valid heart rates detected across channels.")

        return peak_positions_all_channels, cleanest_channel, labels, avg_heart_rate, avg_hrv_sdnn
    
    def avg_window(self, data, peak_positions, window_left=0.3, window_right=0.4, heart_beat_score_threshold=0.7, sigma=1):
        """Compute average windowed QRS waveform around peaks.

        Args:
            data (np.ndarray): Data array, shape (num_channels, num_samples).
            peak_positions (list or dict): Peak indices or dict of peak indices per channel.
            window_left (float): Seconds to include left of the peak.
            window_right (float): Seconds to include right of the peak.
            heart_beat_score_threshold (float): Threshold for heart beat score.
            sigma (float): Standard deviation for Gaussian filter.

        Returns:
            tuple: (avg_channels (np.ndarray), time_window (np.ndarray)).
        """
        if peak_positions is None or len(peak_positions) == 0:
            raise ValueError("No peak positions provided.")
        
        samples_left = int(window_left * self.INTERNAL_SAMPLING_RATE)
        samples_right = int(window_right * self.INTERNAL_SAMPLING_RATE)
        window_length = samples_left + samples_right
        num_channels = data.shape[0]
        avg_channels = []
        is_multichannel = isinstance(peak_positions, dict)

        for ch in range(num_channels):
            peaks = peak_positions.get(ch, []) if is_multichannel else peak_positions
            windows = []
            for pos in peaks:
                if pos - samples_left < 0 or pos + samples_right > data.shape[1]:
                    continue
                window = data[ch, pos - samples_left:pos + samples_right]
                time = np.linspace(0, len(window) / self.INTERNAL_SAMPLING_RATE, num=len(window), endpoint=False)
                windows.append(self.remove_drift_and_offset(window, time))
            
            if len(windows) > 0:
                windows = np.array(windows)
                final_labels, final_confidences = self.segment_entire_run(windows)

                scores, _, _, _ = self._heart_beat_score(final_confidences, final_labels)
                mask = scores > heart_beat_score_threshold
                windows = windows[mask]

                if len(windows) > 0:
                    avg_channels.append(np.mean(windows, axis=0))
                else:
                    avg_channels.append(np.zeros(window_length))
            else:
                avg_channels.append(np.zeros(window_length))

        time_window = np.linspace(0, window_length / self.INTERNAL_SAMPLING_RATE, num=window_length, endpoint=False)

        avg_channels = gaussian_filter1d(avg_channels, sigma=sigma, axis=-1, mode='nearest')
        return avg_channels, time_window

    def default_filter_combination(self, data, bandstop_freq=50, lowpass_freq=95, highpass_freq=1, savgol_window=61, savgol_polyorder=2, sampling_rate=1000):
        """Apply a default combination of filters to the data.

        Args:
            data (np.ndarray): Input data, shape (num_channels, num_samples).
            bandstop_freq (float): Center frequency for bandstop filter.
            lowpass_freq (float): Cutoff frequency for lowpass filter.
            highpass_freq (float): Cutoff frequency for highpass filter.
            savgol_window (int): Window length for Savitzky-Golay filter.
            savgol_polyorder (int): Polynomial order for Savitzky-Golay filter.
            sampling_rate (int): Sampling rate of the data.

        Returns:
            np.ndarray: Filtered data.
        """
        filtered_data = data
        for bandwidth, order in [(2, 2), (3, 3), (3, 3)]:
            filtered_data = self.bandstop_filter(filtered_data, bandstop_freq * (2 if bandwidth == 3 else 1), bandwidth, sampling_rate, order)
        filtered_data = self.apply_lowpass_filter(filtered_data, sampling_rate, lowpass_freq, order=3)
        filtered_data = self.apply_highpass_filter(filtered_data, sampling_rate, highpass_freq, order=2)
        filtered_data = signal.savgol_filter(filtered_data, window_length=savgol_window, polyorder=savgol_polyorder, axis=1)
        return filtered_data

    def get_field_directions(self, data, key):
        """Extract x, y, z field data for QuSpin sensors.

        Args:
            data (np.ndarray): Input data, shape (num_channels, num_samples).
            key (str): Run key for sensor exclusion.

        Returns:
            tuple: (x_data, y_data, z_data), each of shape (rows, cols, samples).
        """
        num_rows = len(self.quspin_position_list)
        x_data = np.zeros((num_rows, num_rows, data.shape[-1]))
        y_data = np.zeros((num_rows, num_rows, data.shape[-1]))
        z_data = np.zeros((num_rows, num_rows, data.shape[-1]))

        for row_idx, row in enumerate(self.quspin_position_list):
            for col_idx, quspin_id in enumerate(row):
                for suffix, target in zip(['_x', '_y', '_z'], [x_data, y_data, z_data]):
                    channel_name = quspin_id + suffix
                    channel_index = self.quspin_channel_dict.get(channel_name)
                    if channel_index is None or (self.sensor_channels_to_exclude.get(key) and channel_name in self.sensor_channels_to_exclude.get(key, [])) or \
                    (self.sensor_channels_to_exclude.get(key) and f"*{suffix}" in self.sensor_channels_to_exclude.get(key, [])):
                        continue
                    channel_index = abs(int(channel_index))
                    target[row_idx, col_idx, :] = data[channel_index]

        return x_data, y_data, z_data

    def invert_field_directions(self, x_data, y_data, z_data, key, num_channels=None):
        """
        Reconstruct the original data array from x, y, z field data. (inverse of get_field_directions)
        Args:
            x_data, y_data, z_data (np.ndarray): Field data arrays of shape (rows, cols, samples).
            key (str): Run key for sensor exclusion.
            num_channels (int, optional): Total number of channels in the original data.
        Returns:
            np.ndarray: Reconstructed data array with shape (num_channels, num_samples).
        """
        num_samples = x_data.shape[-1]
        
        if num_channels is None:
            # Use all channel indices in the dict, including potentially unused ones
            channel_indices = [abs(int(idx)) for idx in self.quspin_channel_dict.values()]
            num_channels = max(channel_indices) + 1
        
        data = np.zeros((num_channels, num_samples))

        
        for row_idx, row in enumerate(self.quspin_position_list):
            for col_idx, quspin_id in enumerate(row):
                for suffix, source in zip(['_x', '_y', '_z'], [x_data, y_data, z_data]):
                    channel_name = quspin_id + suffix
                    channel_index = self.quspin_channel_dict.get(channel_name)

                    # Skip channels that are excluded or don't exist - use exactly the same condition as in get_field_directions
                    if channel_index is None or (self.sensor_channels_to_exclude.get(key) and channel_name in self.sensor_channels_to_exclude.get(key, [])) or \
                    (self.sensor_channels_to_exclude.get(key) and f"*{suffix}" in self.sensor_channels_to_exclude.get(key, [])):
                        continue
                    
                    channel_index = abs(int(channel_index))
                    data[channel_index, :] = source[row_idx, col_idx, :]
        
        return data

    def align_multi_channel_signal(self, signal1, signal2, lag_cutoff=2000, plot=True):
        """
        Align two multi-channel signals using cross-correlation based on the mean across channels.
        In the beginning of the run a distinct signal is required to estimate the lag. F.e. a rectangular wave

        Args:
            signal1 (np.ndarray): First signal, shape (num_channels, num_samples).
            signal2 (np.ndarray): Second signal, shape (num_channels, num_samples).
            lag_cutoff (int): Maximum number of samples to consider for lag estimation.
            plot (bool): Whether to plot before/after alignment.

        Returns:
            tuple: (aligned_signal2 (np.ndarray), lag (int))
        """

        # Ensure correct shape (channels, samples)
        signal1 = signal1 if signal1.ndim == 2 else signal1[:, np.newaxis]
        signal2 = signal2 if signal2.ndim == 2 else signal2[:, np.newaxis]

        # Check signal lengths
        min_len = min(signal1.shape[1], signal2.shape[1])
        if min_len < lag_cutoff:
            raise ValueError(f"lag_cutoff ({lag_cutoff}) exceeds signal length ({min_len})")

        # Compute mean across channels
        mean_signal1 = np.mean(signal1[:, :lag_cutoff], axis=0)
        mean_signal2 = np.mean(signal2[:, :lag_cutoff], axis=0)

        # Cross-correlate the two mean signals
        correlation = correlate(mean_signal1, mean_signal2, mode='full')
        lag = np.argmax(correlation) - (lag_cutoff - 1)

        # Shift signal2 according to lag
        if lag > 0:
            aligned_signal2 = np.pad(signal2, ((0, 0), (lag, 0)), mode='constant')[:, :signal1.shape[1]]
        else:
            aligned_signal2 = signal2[:, -lag:]
            if aligned_signal2.shape[1] < signal1.shape[1]:
                aligned_signal2 = np.pad(aligned_signal2, ((0, 0), (0, signal1.shape[1] - aligned_signal2.shape[1])), mode='constant')

        # Plot before and after alignment
        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            for i in range(signal1.shape[0]):
                axs[0].plot(signal1[i, :lag_cutoff], color='blue', alpha=0.5)
            for i in range(signal2.shape[0]):
                axs[0].plot(signal2[i, :lag_cutoff], color='orange', alpha=0.5)
            axs[0].set_title('Before Alignment')
            
            for i in range(signal1.shape[0]):
                axs[1].plot(signal1[i, :lag_cutoff], color='blue', alpha=0.5)
            for i in range(aligned_signal2.shape[0]):
                axs[1].plot(aligned_signal2[i, :lag_cutoff], color='orange', alpha=0.5)
            axs[1].set_title('After Alignment')

            for ax in axs:
                ax.grid(True)
            plt.tight_layout()
            plt.show()

        return aligned_signal2, lag

    def prepare_data(self, key, apply_default_filter=False, intervall_low_sec=5, intervall_high_sec=-5, plot_alignment=False, alignment_cutoff_sec=2, input_sampling_rate=1000):
        """Prepare data for a given run key for analysis.

        Steps:
        1. Load raw data for the key from `self.data` and `self.add_data`.
        2. Align the two (resampled) datasets using cross-correlation.
        3. Concatenate aligned data.
        4. Select the time interval specified by `intervall_low_sec` and `intervall_high_sec`.
        5. Apply default filtering (optional).
        6. Apply coordinate system transformation based on log file.
        7. Resample data from `source_sampling_rate` to `self.INTERNAL_SAMPLING_RATE` (250 Hz).
        8. Extract x, y, z components into 3D grids.

        Args:
            key (str): Run key (must exist in `self.data` and potentially `self.add_data`).
            input_sampling_rate (int): The original sampling rate (in Hz) of the TDMS data.
            apply_default_filter (bool): Whether to apply default filters after resampling and alignment.
            intervall_low_sec (float): Start time in seconds relative to the beginning of the *aligned* data.
            intervall_high_sec (float): End time in seconds relative to the *end* of the *aligned* data (use negative value).
            plot_alignment (bool): Whether to plot alignment results.
            alignment_cutoff_sec (float): Duration in seconds from the start used for alignment correlation.

        Returns:
            tuple: ((x_data, y_data, z_data), time_vector, combined_run_data)
                - x/y/z_data: np.ndarray shape (rows, cols, samples) at 250Hz, filtered and oriented.
                - time_vector: np.ndarray 1D time in seconds for the final interval.
                - combined_run_data: np.ndarray shape (num_channels, samples) at 250Hz, combined, interval-selected, filtered, oriented.
                Returns (None, None, None) if data preparation fails at any critical step.
        """

        intervall_low_samples = int(intervall_low_sec * input_sampling_rate)
        intervall_high_samples = int(intervall_high_sec * input_sampling_rate)

        data1 = np.transpose(self.data[key])[1:, :]
        data2 = np.transpose(self.add_data[key])[1:, :]

        if apply_default_filter:
            data1 = self.default_filter_combination(data1, sampling_rate=input_sampling_rate)   
            data2 = self.default_filter_combination(data2, sampling_rate=input_sampling_rate)

        aligned_data2, _ = self.align_multi_channel_signal(data1, data2, plot=plot_alignment, lag_cutoff=alignment_cutoff_sec * input_sampling_rate)
        min_length = min(data1.shape[1], aligned_data2.shape[1])
        single_run = np.concatenate((data1[:, :min_length], aligned_data2[:, :min_length]), axis=0)[:, intervall_low_samples:intervall_high_samples]
        

        flipped_data = self._change_to_consistent_coordinate_system(single_run)

        if input_sampling_rate != self.INTERNAL_SAMPLING_RATE:
            num_samples_target = int(flipped_data.shape[-1] * (self.INTERNAL_SAMPLING_RATE / input_sampling_rate))

            logging.info(f"Resampling data from {input_sampling_rate}Hz to {self.INTERNAL_SAMPLING_RATE}Hz. Target length: {num_samples_target}, Original length: {flipped_data.shape[-1]} samples.")

            if num_samples_target == 0:
                warnings.warn("Resampled length is zero. Cannot proceed.")
                return np.empty((flipped_data.shape[0], 0), dtype=int), np.empty((flipped_data.shape[0], 0)), np.empty((flipped_data.shape[0], 0), dtype=float)

            resampled_data_list = []
            for i in range(flipped_data.shape[0]):
                resampled_sample = signal.resample(flipped_data[i, :], num_samples_target)
                resampled_data_list.append(resampled_sample)

            resampled_data = np.stack(resampled_data_list, axis=0)
        else:
            resampled_data = flipped_data # Use copy if no resampling needed

        time = np.linspace(0, len(resampled_data[0]) / self.INTERNAL_SAMPLING_RATE, num=len(resampled_data[0]))

        x_data, y_data, z_data = self.get_field_directions(resampled_data, key)
        return (x_data, y_data, z_data), time, resampled_data

    def ICA_filter(self, data, heart_beat_score_threshold=0.85, max_iter=5000,
            confidence_weight=0.8, plausibility_weight=0.2, print_result=False,
            plot_result=False):
        
        """
        Applies Independent Component Analysis (ICA) to filter heartbeat-related artifacts from multichannel signal data.

        This method uses ICA to decompose the input signal into components, scores them based on heartbeat-related features,
        and reconstructs the signal while removing components below a specified heartbeat score threshold.

        Parameters:
        ----------
        data : np.ndarray
            The input data array. Can be either 2D (channels x time) or 3D (grid_x x grid_y x time).
        heart_beat_score_threshold : float, optional
            Threshold for heartbeat component score above which components are retained (default is 0.85).
        max_iter : int, optional
            Maximum number of iterations for the ICA algorithm (default is 5000).
        confidence_weight : float, optional
            Weight of the confidence score in determining heartbeat relevance (default is 0.8).
        plausibility_weight : float, optional
            Weight of the plausibility score in determining heartbeat relevance (default is 0.2).
        print_result : bool, optional
            If True, prints the result of the component evaluation (default is False).
        plot_result : bool, optional
            If True and input is 3D (4x4 grid), displays an interactive plot for visual threshold adjustment (default is False).

        Returns:
        -------
        result : np.ndarray
            The filtered signal with heartbeat-related components removed or suppressed.
        ica_components : np.ndarray
            The independent components extracted from the original signal.
        best_channel_idx : int
            Index of the component with the highest combined score (best representing heartbeat activity).
        score_mask : np.ndarray
            Boolean array indicating which components were retained after thresholding.

        Raises:
        ------
        ValueError
            If input data is not 2D or 3D.
        """
    
        # Get the shape of the data
        data_shape = data.shape
        if len(data_shape) not in [2, 3]:
            raise ValueError(f"Input data must be 2D or 3D, got shape {data_shape}")

        # Check if the data is 3D or 2D
        is_3d = len(data_shape) == 3
        if is_3d:
            data_2d = data.reshape(data_shape[0] * data_shape[1], data_shape[2])
        else:
            data_2d = data

        # Apply ICA to the data
        ica_components, mixing_matrix, mean, non_zero_indices, original_shape = self._apply_ICA(
            data_2d, max_iter=max_iter
        )
        if ica_components is None:
            logging.warning("ICA failed, returning None.")
            return None, None, None, None

        # Perform heartbeat scoring
        best_channel_idx, labels, confidence, heart_beat_scores = self.find_cleanest_channel(
            ica_components,
            print_results=print_result,
            confidence_weight=confidence_weight,
            plausibility_weight=plausibility_weight
        )

        # Function to apply filter based on the threshold
        def apply_filter(threshold):
            score_mask = heart_beat_scores > threshold
            filtered_components = ica_components.copy()
            filtered_components[~score_mask, :] = 0
            reconstructed = np.dot(mixing_matrix, filtered_components) + mean.reshape(-1, 1)
            result = np.zeros((original_shape[0], original_shape[1]))
            result[non_zero_indices] = reconstructed
            if is_3d:
                result = result.reshape(data_shape)
            return result, score_mask

        # Get the filtered result based on the initial threshold
        result, score_mask = apply_filter(heart_beat_score_threshold)

        if is_3d and plot_result:
            if data_shape[0] * data_shape[1] != 16:
                print("Plotting supported only for 4x4 grid.")
            else:
                # Create a 4x4 grid plot
                fig, axes = plt.subplots(
                    nrows=data_shape[0], ncols=data_shape[1],
                    sharex=True,
                    figsize=(33/2.54, 22/2.54), dpi=100
                )

                fig.text(0.5, 0.06, 'Time [s]', ha='center', va='center', fontsize=12)
                fig.text(0.06, 0.5, 'Magnetic Field [pT]', ha='center', va='center', rotation='vertical', fontsize=12)

                length = min(data_shape[-1], 500)
                time_axis = np.arange(length) / self.INTERNAL_SAMPLING_RATE

                def plot_grid(res, unfiltered_data):
                    for i in range(data_shape[0]):
                        for j in range(data_shape[1]):
                            ax = axes[i, j]
                            ax.clear()

                            # Plot filtered signal
                            ax.plot(time_axis, res[i, j, :length], label="Filtered", color='blue', linewidth=0.8)

                            # Plot original signal
                            ax.plot(time_axis, unfiltered_data[i, j, :length], 'r:', label="Unfiltered", linewidth=0.8)

                            ax.set_title(f"Ch {i * data_shape[1] + j + 1}", fontsize=9)
                            ax.grid(True, linestyle='dotted', alpha=0.6)
                            ax.tick_params(axis='both', labelsize=8)

                    fig.canvas.draw_idle()

                plot_grid(result, data)

                # Slider for dynamic threshold adjustment
                ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
                slider = Slider(ax_slider, 'Threshold', 0.0, 1.0,
                                valinit=heart_beat_score_threshold, valstep=0.01)

                def update(val):
                    threshold = slider.val
                    result, _ = apply_filter(threshold)
                    plot_grid(result, data)

                slider.on_changed(update)
                plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, hspace=0.4, wspace=0.3)
                plt.show()

        return result, ica_components, best_channel_idx, score_mask

    def create_heat_map_animation(self, data, cleanest_i, cleanest_j, output_file='animation.gif', interval=100, resolution=500, stride=1, direction='x', key="Brustlage", dynamic_scale=True):
        """Create an animated heatmap of the data.

        Args:
            data (np.ndarray): Data array, shape (rows, cols, samples).
            cleanest_i (int): Row index of the cleanest channel.
            cleanest_j (int): Column index of the cleanest channel.
            output_file (str): Output file name for the animation.
            interval (int): Frame interval in milliseconds.
            resolution (int): Heatmap resolution.
            stride (int): Frame stride.
            direction (str): Field direction (x, y, z).
            key (str): Run key.
            dynamic_scale (bool): Whether to use dynamic color scaling.

        Returns:
            tuple: (animation (FuncAnimation), figure (plt.Figure)).
        """
        colors = ["black", "purple", "red", "yellow"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        T = data.shape[-1]
        fig, (ax_main, ax_trace) = plt.subplots(1, 2, figsize=(14, 6))

        x_highres = np.linspace(0, 3, resolution)
        y_highres = np.linspace(0, 3, resolution)
        xx_highres, yy_highres = np.meshgrid(x_highres, y_highres)

        heatmap = ax_main.imshow(np.zeros((resolution, resolution)), extent=[0, 3, 3, 0], origin='upper', cmap=cmap, interpolation='bilinear')
        cbar = fig.colorbar(heatmap, ax=ax_main)
        cbar.set_label(f'B-{direction} Field Strength {key}', fontsize=14)

        sensor_points, = ax_main.plot([], [], 'ko', markersize=6, markeredgewidth=1)
        cleanest_marker, = ax_main.plot([cleanest_j], [3 - cleanest_i], 'go', markersize=12, markerfacecolor='none', markeredgewidth=3)
        time_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes, fontsize=14)
        ax_main.set_xlim(-0.1, 3.1)
        ax_main.set_ylim(-0.1, 3.1)

        time_series = data[cleanest_i, cleanest_j, :]
        trace_plot, = ax_trace.plot(np.array(range(T)) * 1000 / self.INTERNAL_SAMPLING_RATE, time_series, 'b-', label=f'Channel ({cleanest_i}, {cleanest_j})')
        moving_point, = ax_trace.plot([], [], 'go', markersize=8)
        ax_trace.set_xlim(0, T * 1000 / self.INTERNAL_SAMPLING_RATE)
        ax_trace.set_ylim(time_series.min(), time_series.max())
        ax_trace.set_title('Time Series of Cleanest Channel', fontsize=16)
        ax_trace.set_xlabel('Time [ms]', fontsize=14)
        ax_trace.set_ylabel(f'B-{direction} Field Strength {key}', fontsize=14)
        ax_trace.legend()
        ax_trace.grid(True)

        def update(frame):
            current_data = data[:, :, frame]
            if dynamic_scale:
                valid_values = current_data[np.isfinite(current_data)]
                if valid_values.size > 0:
                    heatmap.set_clim(valid_values.min(), valid_values.max())

            points, values, sensor_x, sensor_y = [], [], [], []
            for i in range(4):
                for j in range(4):
                    if np.isfinite(current_data[i, j]) and not np.all(current_data[i, j] == 0):
                        points.append([j, 3 - i])
                        values.append(current_data[i, j])
                        sensor_x.append(j)
                        sensor_y.append(3 - i)

            if points:
                interpolated_data = griddata(np.array(points), np.array(values), (xx_highres, yy_highres),
                                            method='cubic' if len(points) > 3 else 'linear', fill_value=np.nan)
                heatmap.set_array(interpolated_data)

            sensor_points.set_data(sensor_x, sensor_y)
            time_text.set_text(f'Time: {frame * 1000 / self.INTERNAL_SAMPLING_RATE:.1f} ms')
            moving_point.set_data([frame * 1000 / self.INTERNAL_SAMPLING_RATE], [time_series[frame]])
            return heatmap, time_text, cleanest_marker, sensor_points, moving_point

        animation_T = range(0, T, stride)
        ani = FuncAnimation(fig, update, frames=animation_T, interval=interval, blit=True)
        ani.save(output_file, writer='ffmpeg', dpi=200)
        plt.tight_layout(rect=[0.15, 0.08, 0.98, 0.93])
        plt.show()
        return ani, fig
    

    def plot_segments_with_editing(self, signal, pred):
        """
        Main function to display signal with segmentation and editing capabilities
        
        Args:
            signal: The ECG signal (1D array)
            pred: Initial model predictions (1D array of class labels)
        
        Returns:
            The (possibly edited) predictions
        """
        
        # Ensure inputs are numpy arrays
        signal = np.asarray(signal).squeeze()
        pred = np.asarray(pred).squeeze()
        
        # Make a copy of the predictions that will be modified
        edited_pred = copy.deepcopy(pred)
        
        # Call the interactive segmentation function
        try:
            edited_pred = self._plot_interactive_segmentation(signal, edited_pred)
        except Exception as e:
            print(f"Warning: Interactive editing failed with error: {e}")
            print("Returning original predictions")
        
        # Make sure to return a numpy array with the correct shape
        return np.asarray(edited_pred)


    def _plot_interactive_segmentation(self, signal, pred, title="Interactive Segmentation"):
        """
        Creates an interactive plot for modifying wave segment boundaries.
        
        Args:
            signal: The ECG signal (1D array)
            pred: Initial model predictions (1D array of class labels)
            title: Title for the plot
            
        Returns:
            Modified predictions array
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        import numpy as np
        import copy
        
        # Ensure inputs are numpy arrays with correct dimensions
        signal = np.asarray(signal).squeeze()
        pred = np.asarray(pred).squeeze()
        
        # Create a deep copy of predictions that we can modify
        editable_pred = copy.deepcopy(pred)
        
        # Find segment boundaries
        boundaries = [0]  # Start of signal
        for i in range(1, len(pred)):
            if pred[i] != pred[i-1]:
                boundaries.append(i)
        boundaries.append(len(pred))  # End of signal
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(bottom=0.2)  # Make room for buttons
        
        # For storing the currently active boundary
        active_boundary = [None]
        dragging = [False]
        selected_boundary_idx = [None]
        
        # Status text for user feedback
        status_text = ["Click on a red boundary line to select, then drag to move"]
        
        def update_plot():
            """Redraw the plot with current segment boundaries"""
            ax.clear()
            ax.plot(t, signal, color='black', linewidth=0.8)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            # Plot segments
            current_start_idx = 0
            for k in range(1, len(boundaries)):
                end_idx = boundaries[k]
                label_idx = editable_pred[current_start_idx]
                label_name = self.CLASS_NAMES_MAP.get(label_idx, f"Class {label_idx}")
                color = self.SEGMENT_COLORS.get(label_idx, 'gray')
                ax.axvspan(t[current_start_idx], t[end_idx-1], 
                        color=color, alpha=0.3, ec=None)
                
                # Add label in the middle of segment
                mid_point = (current_start_idx + end_idx) // 2
                ax.text(t[mid_point], ax.get_ylim()[1] * 0.9, label_name, 
                        ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
                
                current_start_idx = end_idx
            
            # Plot boundaries for adjustment
            for i, b in enumerate(boundaries[1:-1], 1):  # Skip first and last boundaries
                if active_boundary[0] == b:
                    # Active boundary with different style
                    ax.axvline(x=t[b], color='blue', linestyle='-', linewidth=2)
                else:
                    ax.axvline(x=t[b], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Add status text
            ax.set_title(title)
            ax.text(0.5, 0.01, status_text[0], transform=ax.transAxes, 
                    ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))
            fig.canvas.draw_idle()
        
        # Create time array
        t = np.arange(len(signal))
        
        def on_click(event):
            if event.inaxes != ax:
                return
            
            if dragging[0]:  # Don't select a new boundary if we're dragging
                return
                
            # Find nearest boundary
            selected_idx = None
            min_distance = float('inf')
            
            for i, b in enumerate(boundaries[1:-1], 1):  # Skip first and last boundaries
                distance = abs(t[b] - event.xdata)
                if distance < min_distance and distance < len(t) * 0.02:  # Within 2% of signal length
                    min_distance = distance
                    selected_idx = i
            
            if selected_idx is not None:
                active_boundary[0] = boundaries[selected_idx]
                selected_boundary_idx[0] = selected_idx
                status_text[0] = "Boundary selected - drag to move or click buttons below to change segment class"
                update_plot()
            else:
                # If clicked away from boundaries, deselect
                if active_boundary[0] is not None:
                    active_boundary[0] = None
                    selected_boundary_idx[0] = None
                    status_text[0] = "Click on a red boundary line to select, then drag to move"
                    update_plot()
        
        # Set up the boundary dragging
        def on_mouse_move(event):
            if event.inaxes != ax or active_boundary[0] is None or not dragging[0]:
                return
                
            # Get the nearest time point
            idx = min(max(0, int(round(event.xdata))), len(t)-1)
            
            # Store the original segments' classes before moving boundary
            left_segment_class = editable_pred[boundaries[selected_boundary_idx[0]-1]]
            right_segment_class = editable_pred[min(boundaries[selected_boundary_idx[0]]+1, len(editable_pred)-1)]
            
            # Don't allow overlapping with adjacent boundaries
            idx = max(boundaries[selected_boundary_idx[0]-1] + 1, idx)
            idx = min(boundaries[selected_boundary_idx[0]+1] - 1, idx)
            
            # Update boundary position
            old_boundary = boundaries[selected_boundary_idx[0]]
            boundaries[selected_boundary_idx[0]] = idx
            active_boundary[0] = idx
            
            # Update predictions - preserve the classes on both sides
            for i in range(boundaries[selected_boundary_idx[0]-1], idx):
                editable_pred[i] = left_segment_class
            for i in range(idx, boundaries[selected_boundary_idx[0]+1]):
                editable_pred[i] = right_segment_class
            
            update_plot()
        
        def on_mouse_down(event):
            if event.inaxes != ax or active_boundary[0] is None:
                return
            dragging[0] = True
            status_text[0] = "Dragging boundary - release mouse to finish"
        
        def on_mouse_up(event):
            dragging[0] = False
            if active_boundary[0] is not None:
                status_text[0] = "Boundary selected - drag to move or click buttons below to change segment class"
            else:
                status_text[0] = "Click on a red boundary line to select, then drag to move"
            update_plot()
        
        # Create button axes
        class_buttons_ax = plt.axes([0.15, 0.05, 0.7, 0.1])
        class_buttons_ax.axis('off')
        
        # Add label selection buttons
        class_buttons = []
        button_width = 0.15
        spacing = 0.05
        start_pos = 0.1
        
        for i, class_name in self.CLASS_NAMES_MAP.items():
            color = self.SEGMENT_COLORS.get(i, 'gray')
            button_ax = plt.axes([start_pos + i * (button_width + spacing), 0.05, button_width, 0.05])
            button = Button(button_ax, class_name, color=color, hovercolor='0.9')
            
            # Create closure with current class index
            def make_click_handler(class_idx):
                def click_handler(event):
                    if active_boundary[0] is not None and selected_boundary_idx[0] is not None:
                        try:
                            # Change right segment class (simpler approach to avoid segfault)
                            start_idx = boundaries[selected_boundary_idx[0]]
                            end_idx = boundaries[selected_boundary_idx[0]+1]
                            editable_pred[start_idx:end_idx] = class_idx
                            status_text[0] = f"Changed segment to {self.CLASS_NAMES_MAP.get(class_idx, f'Class {class_idx}')}"
                            update_plot()
                        except Exception as e:
                            print(f"Error changing segment class: {e}")
                    else:
                        status_text[0] = "Select a boundary first before changing segment class"
                        update_plot()
                return click_handler
            
            button.on_clicked(make_click_handler(i))
            class_buttons.append(button)
        
        # Add a save button
        save_ax = plt.axes([0.85, 0.05, 0.1, 0.05])
        save_button = Button(save_ax, 'Save', color='green', hovercolor='lightgreen')
        
        # Global variable to store result
        result = [editable_pred]
        
        def on_save(event):
            try:
                plt.close(fig)
            except:
                pass
        
        save_button.on_clicked(on_save)
        
        # Connect events
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', on_mouse_down)
        fig.canvas.mpl_connect('button_release_event', on_mouse_up)
        
        update_plot()
        plt.show()
        
        # Return the modified predictions
        return editable_pred