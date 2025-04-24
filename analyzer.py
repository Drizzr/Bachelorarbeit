import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import signal
import ast
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy.signal import correlate
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import os
import pandas as pd
import re
import json
from scipy.interpolate import interp1d
from MCG_segmentation.model.model import ECGSegmenter
import torch
import warnings  # Use warnings instead of print for less intrusive messages
import numba





class Analyzer:

    # --- Constants (Match Training/Evaluation) ---
    CLASS_NAMES_MAP = {0: "No Wave", 1: "P Wave", 2: "QRS", 3: "T Wave"}
    SEGMENT_COLORS = {0: "silver", 1: "lightblue", 2: "lightcoral", 3: "lightgreen"}
    LABEL_TO_IDX = {v: k for k, v in CLASS_NAMES_MAP.items()}
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mps is not supported in this version of numpy
    
    print(f"Using device: {DEVICE}")

    def __init__(self, filename = "", add_filename = "", log_file_path = "", 
                sensor_channels_to_excllude = {"Brustlage": ["NL_x"], "Rueckenlage": ["NL_x"]}, 
                scaling = 2.7/1000, sampling = 1000, 
                num_ch = 48, model_check_point_dir = "MCG_segmentation/MCGSegmentator_s/checkpoints/best"):
        
        self.filename = filename
        self.add_filename =   add_filename
        self.log_file_path  = log_file_path

        self.scaling = scaling
        self.sampling_rate = sampling
        self.num_ch = num_ch

        try:
            self.data = self.import_tdms(self.filename, self.scaling)
            self.add_data = self.import_tdms(self.add_filename, self.scaling)
            self.sensor_channels_to_exclude = sensor_channels_to_excllude # Dictionary with sensors to exclude f.e. {"Brust": ["F1_x", "OX_y"]}


            self.quspin_gen_dict, self.quspin_channel_dict, self.quspin_position_list =  self.load_data_from_file(self.log_file_path)

            for key, value in self.quspin_channel_dict.items():
                if abs(value) >= 100:  # Überprüfen, ob der absolute Wert drei Ziffern hat
                    sign = -1 if value < 0 else 1
                    last_two_digits = abs(value) % 100  # Die letzten zwei Ziffern des Werts extrahieren
                    new_value = 31 + last_two_digits  # Neuen Wert berechnen
                    self.quspin_channel_dict[key] = sign * new_value  # Wert im Dictionary aktualisieren
            
            self.key_list = [key for key in self.data]

            print(f"Runs in this Session: {', '.join(self.key_list)} \n")
        except FileNotFoundError:
            print(f"Emtpy Initialization, no TDMS file found. \n")

            self.data = {}
            self.add_data = {}
            self.key_list = []

        plt.rcParams['agg.path.chunksize'] = 100000000
        self.cmap = plt.get_cmap('nipy_spectral')
        self.cmaplist = [self.cmap(x) for x in np.linspace(0,1,num=48)]


        try:
            self.model = self.load_trained_model(model_check_point_dir, self.DEVICE)
        except Exception as e:
            print(f"FATAL: Could not load model. Error: {e}")
            self.model = None

    @staticmethod
    def load_trained_model(checkpoint_dir, device):
        """Loads the 'best' trained ECGSegmentator model."""
        print(f"Loading model from: {checkpoint_dir}")
        best_model_path = os.path.join(checkpoint_dir, "model.pth")
        best_params_path = os.path.join(checkpoint_dir, "params.json")
        if not os.path.exists(best_model_path) or not os.path.exists(best_params_path):
            raise FileNotFoundError(f"model.pth or params.json not found in {checkpoint_dir}")
        with open(best_params_path, "r") as f:
            params = json.load(f); checkpoint_args = params.get("args", {})
            num_classes=checkpoint_args.get("num_classes", 4); num_heads=checkpoint_args.get("num_heads", 4)
            dropout_rate=checkpoint_args.get("dropout_rate", 0.3)
            # Load other args if needed by model init
            print(f"Loaded model args: num_classes={num_classes}, num_heads={num_heads}, dropout={dropout_rate}")
        model = ECGSegmenter() # Add other args if needed
        try: 
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        except RuntimeError as e: 
            print("\n*** Error loading state_dict: Model architecture mismatch? ***"); raise e
        
        model.to(device); model.eval(); print("Model loaded successfully.")
        
        return model
    
    @staticmethod
    def import_tdms(filename, scaling):
        data = {}
        with TdmsFile.read(filename) as tdms_file:
            for group in tdms_file.groups():
                no_of_channels = len(group.channels())
                no_of_samples = len(group.channels()[0].read_data(scaled=True))
                data_array = np.zeros((no_of_samples, no_of_channels+1))
            
                for i, channel in enumerate(group.channels()):
                    # Access numpy array of data for channel:
                    data_array[:, i+1] = channel.read_data(scaled=True)/scaling        
                data_array[:, 0] = np.linspace(0, (no_of_samples-1), no_of_samples)
                                            #* properties['wf_increment'], no_of_samples)
                data[group.name] = data_array
        return data

    @staticmethod
    def load_data_from_file(file_path):
        with open(file_path, 'r') as file:
            data = ast.literal_eval(file.read())
            quspin_gen_dict = data.get('quspin_gen_dict', {})
            quspin_channel_dict = data.get('quspin_channel_dict', {})
            quspin_position_list =data.get('quspin_position_list', [])
        return quspin_gen_dict, quspin_channel_dict, quspin_position_list
    
    @staticmethod
    def bandstop_filter(data, center_frequency, bandwidth, sampling_rate, order = 4):
        nyquist = 0.5 * sampling_rate
        low_cutoff = (center_frequency - bandwidth / 2) / nyquist
        high_cutoff = (center_frequency + bandwidth / 2) / nyquist
        b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='bandstop', analog=False)
        filtered_signal = signal.filtfilt(b, a, data)
        return filtered_signal
    
    @staticmethod
    def bandpass_filter(data, fs, lowcut, highcut, order=4):
        """
        Einfacher Bandpass-Filter zur Rauschreduktion.
        """
        from scipy.signal import butter, filtfilt
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def apply_lowpass_filter(data, sampling_rate, cutoff_frequency, order=2):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        return filtered_data
    
    @staticmethod
    def apply_highpass_filter(data, sampling_rate, cutoff_frequency, order=2):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        return filtered_data
    
    @staticmethod
    def remove_drift_and_offset(signal, time):
        """
        Verbesserte Methode zur Entfernung von Drift und Offset.
        Verwendet robuste polynomiale Anpassung niedriger Ordnung.
        """
        # Polynom 3. Grades für Trendentfernung
        detrended = signal - np.polynomial.polynomial.polyval(time, 
                                np.polynomial.polynomial.polyfit(time, signal, 3))
        
        # Median als robuste Baseline-Schätzung
        baseline = np.median(detrended)
        
        return detrended - baseline
    
    @staticmethod
    def plot4x4(data, time, name,path=None, save = False):
        nrows = len(data)
        ncols = len(data[0])
        
        fig, elem= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, figsize=(33/2.54,22/2.54),dpi=100)  
        fig.suptitle(f"{name}", y=0.95)
        row = 0
        col = 0

        for row in range(nrows):
            for col in range(ncols):
                d = data[row][col]          
                if len(d)>0 and not np.all(d==0):
                    elem[row,col].plot(time, d, '-')
                    #elem[row,col].axvline(window_left, color = "red", alpha= 0.3)
                    elem[row,col].grid(True,linestyle='dotted',alpha=0.7)
                    # elem[row,col].set_yscale("symlog")
                else:
                    elem[row,col].text(0.4, 0.5,'No Data', fontsize=12, color='red', alpha=0.5, ha='center', va='center')
        fig.text(0.5, 0.08, 'time [s]', ha='center', va='center', fontsize=12)
        fig.text(0.08, 0.5, 'magnetic Field [pT]', ha='center', va='center', rotation='vertical', fontsize=12)
        if save:
            plt.savefig(path+f'{name}_4x4plot.png')
        plt.show()
    
    @staticmethod
    def plot_zoomable_mulit_channel(data, time, title, channels = [1]):
        # -> needs single run type data
        # data shape (c, T)
        # time shape (T,)

        fig = go.Figure()
        for ch in channels:
            fig.add_trace(go.Scatter(x=time, y=data[ch], mode='lines', name=f'Channel {ch}'))

        fig.update_layout(
            title=title,
            title_x=0.5,  # center the title
            width=1000,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=14),
            dragmode='pan',
            uirevision=True,
            xaxis_title='Time [s]',
            yaxis_title='Field Strength [nT]',
            xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
            yaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82, 82, 82)')),
        )
        fig.show(config={'scrollZoom': True})
    
    @staticmethod
    def heart_vector_projection(data, name, label):
        component1 = np.array(data[0])
        component2 = np.array(data[1])
        
        fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
        fig.suptitle(f"Projections of 'MHV', {name}", fontsize=16, fontweight='bold', y=1.05)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f5f5f5')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.tick_params(axis='both', which='major', labelsize=10, colors='gray')
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.plot(component1, component2, label=label, color='dodgerblue', linewidth=2.5, linestyle='-')
        ax.fill(component1, component2, alpha=0.2, color='dodgerblue')
        
        for i in range(0, len(component1)-1, 10):
            arrow_start = [component1[i], component2[i]]
            arrow_end = [component1[i + 1], component2[i + 1]]
            ax.annotate('', xy=arrow_end, xytext=arrow_start,
                        arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', linewidth=2))
        
        # Calculate enclosed area using Shoelace Theorem
        area = 0.5 * np.abs(np.dot(component1, np.roll(component2, 1)) - np.dot(component2, np.roll(component1, 1)))
        
        try:
            # Calculate T-distance (T-beg to T-max)
            t_beg = np.array([component1[0], component2[0]])
            t_max_idx = np.argmax(component2**2 + component1**2)  # Assuming T-max is the max point in component2
            t_max = np.array([component1[t_max_idx], component2[t_max_idx]])
            t_distance = np.linalg.norm(t_beg - t_max)
            
            # Calculate Start-End Ratio
            t_end = np.array([component1[-1], component2[-1]])
            vector_t_beg_to_max = t_max - t_beg
            vector_t_end_to_max = t_max - t_end
            start_end_ratio = np.linalg.norm(vector_t_beg_to_max) / np.linalg.norm(vector_t_end_to_max)
            
            # Display calculated values
            ax.text(0.05, 0.95, f'Enclosed Area: {area:.2f}\nT-Distance: {t_distance:.2f}\nStart-End Ratio: {start_end_ratio:.2f}',
                    transform=ax.transAxes, fontsize=12, fontweight='bold', color='black', verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        except Exception as e:
            print(f"Error calculating metrics: {e}")
        
        # Set symmetric axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        max_abs = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)
        
        ax.set_xlabel('$B_1$ [pT]', fontsize=12)
        ax.set_ylabel('$B_1$ [pT]', fontsize=12)
        ax.xaxis.set_label_coords(0.95, 0.5)
        ax.yaxis.set_label_coords(0.5, 0.95)
        
        locs_x = ax.get_xticks()
        locs_y = ax.get_yticks()
        
        xlabels = [f"{x:.0f}" if abs(x) > 0.001 else "" for x in locs_x]
        ylabels = [f"{y:.0f}" if abs(y) > 0.001 else "" for y in locs_y]
        
        ax.set_xticks(locs_x)
        ax.set_xticklabels(xlabels)
        ax.set_yticks(locs_y)
        ax.set_yticklabels(ylabels)
        
        if label:
            ax.legend(fontsize=10, loc='best', frameon=True, facecolor='white', edgecolor='gray')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def heart_vector_projections(heart_vector_data, name, save=False, path=None):
        x = heart_vector_data[0]
        y = heart_vector_data[1]
        z = heart_vector_data[2]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
        fig.suptitle(f"Projections of 'MHV', {name}", fontsize=16, fontweight='bold', y=1.05)
        
        projections = [(x, y, 'xy-Projection', 'dodgerblue'), 
                    (x, z, 'xz-Projection', 'orange'), 
                    (y, z, 'yz-Projection', 'forestgreen')]
        
        for ax, (comp1, comp2, label, color) in zip(axes, projections):
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f5f5f5')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
            ax.tick_params(axis='both', which='major', labelsize=10, colors='gray')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            
            ax.plot(comp1, comp2, label=label, color=color, linewidth=2.5, linestyle='-')
            ax.fill(comp1, comp2, alpha=0.2, color=color)
            
            for i in range(0, len(comp1)-1, 10):
                arrow_start = [comp1[i], comp2[i]]
                arrow_end = [comp1[i + 1], comp2[i + 1]]
                ax.annotate('', xy=arrow_end, xytext=arrow_start,
                            arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', linewidth=2))
            
            try:
                area = 0.5 * np.abs(np.dot(comp1, np.roll(comp2, 1)) - np.dot(comp2, np.roll(comp1, 1)))
                t_beg = np.array([comp1[0], comp2[0]])
                t_max_idx = np.argmax(comp1**2 + comp1**2)
                t_max = np.array([comp1[t_max_idx], comp2[t_max_idx]])
                t_distance = np.linalg.norm(t_beg - t_max)
                t_end = np.array([comp1[-1], comp2[-1]])
                start_end_ratio = np.linalg.norm(t_max - t_beg) / np.linalg.norm(t_max - t_end)
                
                ax.text(0.05, 0.95, f'Area: {area:.2f}\nT-Distance: {t_distance:.2f}\nS/E Ratio: {start_end_ratio:.2f}',
                        transform=ax.transAxes, fontsize=10, fontweight='bold', color='black', verticalalignment='top',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
            except Exception as e:
                print(f"Error calculating metrics: {e}")
            
            ax.legend(fontsize=10, loc='best', frameon=True, facecolor='white', edgecolor='gray')
            
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            max_abs = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
            ax.set_xlim(-max_abs, max_abs)
            ax.set_ylim(-max_abs, max_abs)
        
        axes[0].set_xlabel('Bx [pT]', fontsize=12)
        axes[0].set_ylabel('By [pT]', fontsize=12)
        axes[1].set_xlabel('Bx [pT]', fontsize=12)
        axes[1].set_ylabel('Bz [pT]', fontsize=12)
        axes[2].set_xlabel('By [pT]', fontsize=12)
        axes[2].set_ylabel('Bz [pT]', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        if save and path is not None:
            plt.savefig(path + f'{name}_heart_vector_proj.png', bbox_inches='tight', dpi=200)



    def segment_heart_beat_intervall(self, data: torch.Tensor):
        """
        Segments the heart beat interval. Classifies each data point as:
        0) No Wave, 1) P-Wave, 2) QRS, 3) T-Wave.

        Args:
            data: torch Tensor of shape (b, 1, T) where b is the batch size and T is the number of time steps.
        Returns:
            Tuple of (numpy array of segment classifications (b, T), numpy array of confidence scores (b, T))
        """
        if data.numel() == 0:
            warnings.warn("No data to segment.")
            # Return empty arrays with correct batch dimension if possible
            batch_size = data.shape[0]
            return np.empty((batch_size, 0), dtype=int), np.empty((batch_size, 0), dtype=float)

        # Check max length *before* inference
        max_len = 2000 # Define as constant or class attribute
        if data.shape[-1] > max_len:
            warnings.warn(f"Data length ({data.shape[-1]}) exceeds maximum sequence length of {max_len}. Truncating data.")
            data = data[..., :max_len]

        # Ensure data is on the correct device
        data = data.to(self.DEVICE)

        with torch.no_grad():
            logits = self.model(data)
            probabilities = torch.softmax(logits, dim=-1) # Use torch.softmax
            confidence_scores_pt, predicted_indices_pt = torch.max(probabilities, dim=-1) # Get both max value and index

            # Move to CPU and convert to NumPy *once*
            predicted_indices = predicted_indices_pt.cpu().numpy()
            confidence_scores = confidence_scores_pt.cpu().numpy()

        return predicted_indices, confidence_scores


    def segment_entire_run(self, data: np.ndarray, window_size: int = 2000, overlap: float = 0.5):
        """
        Segment the entire run (potentially T > 2000) using a sliding window.

        Args:
            data: numpy array of shape (b, 1, T). Assumes data is pre-filtered.
                Normalization happens per window. Sampling rate is handled.
            window_size: Size of the sliding window for inference.
            overlap: Fraction of overlap between consecutive windows (0.0 to < 1.0).

        Returns:
            Tuple of (final segmentation labels (b, T_resampled),
                    resampled data (b, T_resampled),
                    final confidence scores (b, T_resampled))
        """

        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0.0 and < 1.0")

        # --- Resampling (Keep original logic, seems reasonable) ---
        if self.sampling_rate != 250:
            num_samples_target = int(data.shape[-1] * (250 / self.sampling_rate))

            if num_samples_target == 0:
                warnings.warn("Resampled length is zero. Cannot proceed.")
                return np.empty((data.shape[0], 0), dtype=int), np.empty((data.shape[0], 0)), np.empty((data.shape[0], 0), dtype=float)


            if num_samples_target < window_size:
                warnings.warn(f"Resampled length {num_samples_target} is less than window size {window_size}. "
                            f"Adjusting window size to {num_samples_target}.")
                window_size = num_samples_target

            if window_size < 200: # Lowered threshold slightly, but still warn
                warnings.warn(f"Warning: Window size {window_size} is small, might lead to suboptimal results.")

            resampled_data_list = []
            for i in range(data.shape[0]):
                resampled_sample = signal.resample(data[i, 0, :], num_samples_target)
                resampled_data_list.append(resampled_sample)

            resampled_data_np = np.stack(resampled_data_list, axis=0)
            resampled_data_np = np.expand_dims(resampled_data_np, axis=1) # Add channel dim: (b, 1, T_resampled)
        else:
            resampled_data_np = data.copy() # Use copy if no resampling needed

        # --- Model Constraints ---
        max_model_len = 2000
        if window_size > max_model_len:
            warnings.warn(f"Window size {window_size} exceeds max sequence length {max_model_len}. Clamping to {max_model_len}.")
            window_size = max_model_len
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")

        # --- Sliding Window ---
        batch_size, _, T_resampled = resampled_data_np.shape
        step_size = max(1, int(window_size * (1 - overlap))) # Ensure step_size is at least 1

        # Prepare storage for results - Initialize with low confidence
        final_labels = np.full((batch_size, T_resampled), -1, dtype=int) # Use -1 as initial label
        final_confidences = np.full((batch_size, T_resampled), -1.0, dtype=float) # Use -1.0 as initial confidence

        starts = range(0, T_resampled - window_size + 1, step_size)
        # Ensure the last part is processed if it doesn't align perfectly
        if T_resampled > 0 and (T_resampled - window_size) % step_size != 0:
            starts = list(starts) + [T_resampled - window_size]
            # Remove duplicate if the last calculated start is the same
            if len(starts) > 1 and starts[-1] == starts[-2]:
                starts.pop()


        for start in starts:
            end = start + window_size
            segment_np = resampled_data_np[:, :, start:end]

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
            labels, confidences = self.segment_heart_beat_intervall(segment)

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
        return final_labels, resampled_data_np.squeeze(axis=1), final_confidences


    def find_cleanest_channel(self, data: np.ndarray, window_size: int = 2000, overlap: float = 0.5, print_results: bool = True):
        """
        Find the channel with the clearest signal based on segmentation confidence
        and physiological plausibility.

        Args:
            data: numpy array of shape (num_channels, num_samples). Raw data.
            window_size: Window size for segmentation.
            overlap: Overlap between windows.

        Returns:
            Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
                (Index of the best channel, (labels, resampled_data, confidence) for all channels)
        """
        if data.ndim != 2:
            raise ValueError(f"Input data must be 2D (num_channels, num_samples), got shape {data.shape}")
        num_channels, num_samples = data.shape
        if num_channels == 0 or num_samples == 0:
            warnings.warn("Input data for find_cleanest_channel is empty.")
            # Return dummy values
            return 0, (np.empty((0,0), dtype=int), np.empty((0,0)), np.empty((0,0), dtype=float))


        # Reshape data to (num_channels, 1, num_samples) for segmentation function
        data_reshaped = data.reshape(num_channels, 1, -1)

        # Segment all channels - segment_entire_run handles batch (channel) dimension
        # Note: Pass short_segment_threshold here if needed, otherwise use default
        labels, resampled_data, confidence = self.segment_entire_run(data_reshaped, window_size, overlap)

        # --- Scoring ---
        if labels.size == 0: # Handle case where segmentation returned empty
            warnings.warn("Segmentation returned empty results in find_cleanest_channel.")
            return 0, (labels, resampled_data, confidence)

        # Calculate mean confidence (axis=1 operates over time dimension T)
        mean_confidence = np.mean(confidence, axis=1) # Shape: (num_channels,)

        # Calculate segment percentages
        segment_percentages = np.zeros((num_channels, 4))
        total_samples_per_channel = labels.shape[1]

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
        max_conf = np.max(mean_confidence)
        normalized_confidence = mean_confidence / (max_conf + 1e-9) if max_conf > 0 else mean_confidence

        confidence_weight = 0.6
        plausibility_weight = 0.4
        final_scores = (confidence_weight * normalized_confidence) + (plausibility_weight * plausibility_scores)

        # Find best channel
        best_channel = np.argmax(final_scores) if final_scores.size > 0 else 0

        if print_results:
            # --- Optional: Print results (keep original formatting) ---
            print("\nChannel Selection Results:")
            print(f"{'Channel':<10}{'Mean Conf':<12}{'P-Wave %':<12}{'QRS %':<12}{'T-Wave %':<12}{'Plausibility':<15}{'Final Score':<12}")
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
                print(f"{'Mean Conf':<10}: {mean_confidence[best_channel]:.4f}")
                print(f"{'Plausibility':<10}: {plausibility_scores[best_channel]:.4f}")
                print(f"{'Final Score':<10}: {final_scores[best_channel]:.4f}")
                print("Segment Distribution:")
                print(f"  P-Wave % : {segment_percentages[best_channel, self.LABEL_TO_IDX['P Wave']]:.2f}%")
                print(f"  QRS %    : {segment_percentages[best_channel, self.LABEL_TO_IDX['QRS']]:.2f}%")
                print(f"  T-Wave % : {segment_percentages[best_channel, self.LABEL_TO_IDX['T Wave']]:.2f}%")
            # --- End Print ---

        # Return 0-based index and the full results
        return best_channel, (labels, resampled_data, confidence)


    def detect_qrs_complex_peaks_cleanest_channel(self, data: np.ndarray, confidence_threshold: float = 0.7, min_qrs_length_sec: float = 0.08, min_distance_sec: float = 0.3, print_heart_rate: bool = False):
        """
        Detects QRS complex peaks on the cleanest channel.

        Args:
            data: numpy array of shape (num_channels, num_samples). Raw data.
            confidence_threshold: Min avg confidence for a QRS segment to be considered.
            min_qrs_length_sec: Minimum duration (in seconds) for a QRS segment.
            min_distance_sec: Minimum distance between detected peaks in seconds.
            print_heart_rate: If True, calculates and prints HR and HRV.

        Returns:
            Tuple[List[int], np.ndarray, int, float, float]:
                - List of peak indices in the resampled signal
                - The resampled data array for all channels (num_channels, T_resampled)
                - Index of the channel used for peak detection
                - Average heart rate across channels (bpm)
                - Average HRV (SDNN in ms) across channels
        """             
        # 1. Find cleanest channel and get segmentations
        best_channel_idx, (labels, resampled_data, confidence) = self.find_cleanest_channel(data, print_results=False)  

        if labels.size == 0:
            warnings.warn("Segmentation data is empty, cannot detect peaks.")
            return [], resampled_data, best_channel_idx, 0.0, 0.0

        # Select data for the best channel
        labels_ch = labels[best_channel_idx]
        resampled_data_ch = resampled_data[best_channel_idx]
        confidence_ch = confidence[best_channel_idx]
        T_resampled = labels_ch.shape[0]
        sampling_rate_resampled = 250  # Hardcoded based on previous logic

        # Convert min length from seconds to samples
        min_qrs_length_samples = int(min_qrs_length_sec * sampling_rate_resampled)
        min_distance_samples = int(min_distance_sec * sampling_rate_resampled)  # Convert min distance to samples

        # 2. Find QRS intervals efficiently using np.diff
        qrs_label = self.LABEL_TO_IDX["QRS"]
        is_qrs = (labels == qrs_label).astype(np.int8)
        diff_qrs = np.diff(is_qrs, prepend=0, append=0)  # Pad to catch start/end

        qrs_starts = np.where(diff_qrs == 1)[0]
        qrs_ends = np.where(diff_qrs == -1)[0]  # End index is exclusive

        if len(qrs_starts) != len(qrs_ends) or np.any(qrs_starts >= qrs_ends):
            warnings.warn("Mismatch in QRS start/end markers. Peak detection might be incomplete.")
            # Attempt to fix simple cases (e.g., starts/ends at array boundaries)
            if len(qrs_starts) > len(qrs_ends) and qrs_starts[-1] < T_resampled:
                qrs_ends = np.append(qrs_ends, T_resampled)
            if len(qrs_ends) > len(qrs_starts) and qrs_ends[0] > 0:
                qrs_starts = np.insert(qrs_starts, 0, 0)
            min_len = min(len(qrs_starts), len(qrs_ends))
            qrs_starts = qrs_starts[:min_len]
            qrs_ends = qrs_ends[:min_len]
            valid = qrs_starts < qrs_ends
            qrs_starts = qrs_starts[valid]
            qrs_ends = qrs_ends[valid]

        peak_positions = []

        # 3. Iterate through found QRS intervals
        for start, end in zip(qrs_starts, qrs_ends):
            length = end - start
            if length < min_qrs_length_samples:
                continue

            # Check average confidence for the segment
            segment_confidence = confidence_ch[start:end]
            if np.mean(segment_confidence) < confidence_threshold:
                continue

            # Extract QRS waveform segment
            qrs_segment = resampled_data_ch[start:end]

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
                rr_intervals_sec = rr_intervals_samples / sampling_rate_resampled
                rr_intervals_ms = rr_intervals_sec * 1000

                # Basic outlier removal for RR intervals (e.g., remove physiologically implausible values)
                plausible_rr_ms = rr_intervals_ms[(rr_intervals_ms > 200) & (rr_intervals_ms < 2000)]

                if len(plausible_rr_ms) > 1:
                    mean_rr_sec = np.mean(plausible_rr_ms) / 1000
                    heart_rate = 60 / mean_rr_sec
                    hrv_sdnn_ms = np.std(plausible_rr_ms)  # SDNN in ms

                    print(f"Heart Rate: {heart_rate:.2f} bpm")
                    print(f"Heart Rate Variability (SDNN): {hrv_sdnn_ms:.2f} ms")
                    print(f"Number of detected peaks: {len(peak_positions)}")
                    print(f"Number of plausible RR intervals used: {len(plausible_rr_ms)}")
                else:
                    print("Not enough plausible RR intervals detected to calculate stable HR/HRV.")
                    print(f"Number of detected peaks: {len(peak_positions)}")

            elif len(peak_positions) == 1:
                print("Only one peak detected. Cannot calculate HR/HRV.")
            else:
                print("No peaks detected.")

        return peak_positions, resampled_data, best_channel_idx, heart_rate, hrv_sdnn_ms


    def detect_qrs_complex_peaks_all_channels(self, data: np.ndarray, confidence_threshold: float = 0.7, min_qrs_length_sec: float = 0.08, min_distance_sec: float = 0.3, print_heart_rate: bool = False):
        """
        Detects QRS complex peaks for all channels and calculates HR and HRV.

        Args:
            data: numpy array of shape (num_channels, num_samples). Raw data.
            confidence_threshold: Min avg confidence for a QRS segment to be considered.
            min_qrs_length_sec: Minimum duration (in seconds) for a QRS segment.
            min_distance_sec: Minimum distance between detected peaks in seconds.
            print_heart_rate: If True, calculates and prints HR and HRV for each channel.

        Returns:
            Tuple[Dict[int, List[int]], np.ndarray, int, float, float]:
                - Dictionary of peak indices per channel
                - Resampled data (num_channels, T_resampled)
                - Index of cleanest channel
                - Average heart rate across channels (bpm)
                - Average HRV (SDNN in ms) across channels
        """
        cleanest_channel, (labels, resampled_data, confidence) = self.find_cleanest_channel(data, print_results=False)
        
        if labels.size == 0:
            warnings.warn("Segmentation data is empty, cannot detect peaks.")
            return {}, resampled_data, cleanest_channel, 0.0, 0.0

        num_channels = labels.shape[0]
        T_resampled = labels.shape[1]
        sampling_rate_resampled = 250  # Hardcoded based on previous logic
        min_qrs_length_samples = int(min_qrs_length_sec * sampling_rate_resampled)
        min_distance_samples = int(min_distance_sec * sampling_rate_resampled)  # Convert min distance to samples
        qrs_label = self.LABEL_TO_IDX["QRS"]
        peak_positions_all_channels = {}

        all_heart_rates = []
        all_hrv_sdnn = []

        # Step 2: Process each channel independently
        for ch_idx in range(num_channels):
            is_qrs = (labels[ch_idx] == qrs_label).astype(np.int8)
            diff_qrs = np.diff(is_qrs, prepend=0, append=0)

            qrs_starts = np.where(diff_qrs == 1)[0]
            qrs_ends = np.where(diff_qrs == -1)[0]

            if len(qrs_starts) != len(qrs_ends) or np.any(qrs_starts >= qrs_ends):
                warnings.warn(f"Channel {ch_idx}: Mismatch in QRS start/end markers. Peak detection might be incomplete.")
                if len(qrs_starts) > len(qrs_ends) and qrs_starts[-1] < T_resampled:
                    qrs_ends = np.append(qrs_ends, T_resampled)
                if len(qrs_ends) > len(qrs_starts) and qrs_ends[0] > 0:
                    qrs_starts = np.insert(qrs_starts, 0, 0)
                min_len = min(len(qrs_starts), len(qrs_ends))
                qrs_starts = qrs_starts[:min_len]
                qrs_ends = qrs_ends[:min_len]
                valid = qrs_starts < qrs_ends
                qrs_starts = qrs_starts[valid]
                qrs_ends = qrs_ends[valid]

            peak_positions = []

            for start, end in zip(qrs_starts, qrs_ends):
                length = end - start
                if length < min_qrs_length_samples:
                    continue

                segment_confidence = confidence[ch_idx][start:end]
                if np.mean(segment_confidence) < confidence_threshold:
                    continue

                qrs_segment = resampled_data[ch_idx][start:end]
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
            peak_positions_all_channels[ch_idx] = peak_positions

            # Step 3: Calculate HR and HRV (optional)
            if len(peak_positions) > 1:
                rr_intervals_samples = np.diff(peak_positions)
                rr_intervals_sec = rr_intervals_samples / sampling_rate_resampled
                rr_intervals_ms = rr_intervals_sec * 1000
                plausible_rr_ms = rr_intervals_ms[(rr_intervals_ms > 200) & (rr_intervals_ms < 2000)]

                if len(plausible_rr_ms) > 1:
                    mean_rr_sec = np.mean(plausible_rr_ms) / 1000
                    heart_rate = 60 / mean_rr_sec
                    hrv_sdnn_ms = np.std(plausible_rr_ms)
                    all_heart_rates.append(heart_rate)
                    all_hrv_sdnn.append(hrv_sdnn_ms)


        avg_heart_rate = np.mean(all_heart_rates) if all_heart_rates else 0.0
        avg_hrv_sdnn = np.mean(all_hrv_sdnn) if all_hrv_sdnn else 0.0

        if print_heart_rate:
            if len(all_heart_rates) > 0:
                print(f"Average Heart Rate: {avg_heart_rate:.2f} bpm")
                print(f"Average Heart Rate Variability (SDNN): {avg_hrv_sdnn:.2f} ms")
            else:
                print("No valid heart rates detected across channels.")

        return peak_positions_all_channels, resampled_data, cleanest_channel, avg_heart_rate, avg_hrv_sdnn

        
    def plotting_time_series(self, data, time, num_ch, name, path = None, save = False):
        fig, elem= plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(12, 4))  
        fig.suptitle(name,size='small', y=0.99)
        print(data.shape)
        if num_ch>1:
            # linienstile = ['-', '--', '-.', ':']
            for i in range(num_ch):
                elem.plot(time, data[i], alpha=0.4,color=self.cmaplist[i], label = f"Ch {i+1}")
            elem.legend(loc='lower center',bbox_to_anchor=(0.5,-0.35),ncol=7)
            elem.grid(alpha=0.7)
            elem.minorticks_on()
            elem.grid(True,which='minor',linestyle='dotted',alpha=0.7)
            elem.set_xlabel('Time [s]')
            elem.set_ylabel('B [pT]')#("V")#
            fig.subplots_adjust(top= 0.95, bottom=0.25)
        else:
            elem.plot(time, data,"-", label = "Single Channel")
            elem.legend(loc='lower center',bbox_to_anchor=(0.5,-0.2),ncol=5)
            elem.grid(alpha=0.7)
            elem.minorticks_on()
            elem.grid(True,which='minor',linestyle='dotted',alpha=0.7)
            elem.set_xlabel('Time [s]')
            elem.set_ylabel('B [pT]')#("V")#
            fig.subplots_adjust(bottom=0.2)
        if save:
            plt.savefig(path+f'{name}_B_vs_time.png')
        plt.show()


    def avg_window(self, data, peak_positions, window_left=0.3, window_right=0.4):
        """
        Computes the average windowed QRS waveform per channel around each peak.

        Args:
            data: numpy array of shape (num_channels, num_samples).
            peak_positions: Either a list of peak indices (for a single channel)
                            or a dict[channel_idx] = list of peak indices.
            window_left: Seconds to include to the left of the peak.
            window_right: Seconds to include to the right of the peak.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                avg_channels: (num_channels, window_length) average waveform per channel.
                time_window: (window_length,) corresponding time vector.
        """
        samples_left = int(window_left * 250)
        samples_right = int(window_right * 250)
        window_length = samples_left + samples_right

        avg_channels = []

        is_multichannel = isinstance(peak_positions, dict)

        num_channels = data.shape[0]

        for ch in range(num_channels):
            if is_multichannel:
                peaks = peak_positions.get(ch, [])
            else:
                peaks = peak_positions  # Apply same peaks to all channels

            windows = []
            for pos in peaks:
                if pos - samples_left < 0 or pos + samples_right > data.shape[1]:
                    continue  # Skip if window would be incomplete
                window = data[ch, pos - samples_left : pos + samples_right]
                time = np.linspace(0, len(window) / self.sampling_rate, num=len(window), endpoint=False)
                window_detrended = self.remove_drift_and_offset(window, time)
                windows.append(window_detrended)

            if windows:
                mean_window = np.mean(windows, axis=0)
            else:
                mean_window = np.zeros(window_length)  # Fallback if no valid windows

            avg_channels.append(mean_window)

        time_window = np.linspace(0, window_length / 250, num=window_length, endpoint=False)

        return np.array(avg_channels), time_window

    def default_filter_combination(self, data, bandstop_freq=50, lowpass_freq=95, highpass_freq=1, savgol_window=61, savgol_polyorder=2):
        filtered_data =  self.bandstop_filter(data, bandstop_freq, 2, self.sampling_rate, order = 2)
        filtered_data =  self.bandstop_filter(filtered_data, bandstop_freq, 3, self.sampling_rate, order = 3)
        filtered_data =  self.bandstop_filter(filtered_data, bandstop_freq*2, 3, self.sampling_rate, order = 3)
        filtered_data = self.apply_lowpass_filter(filtered_data, self.sampling_rate, lowpass_freq, order=3)

        filtered_data = self.apply_highpass_filter(filtered_data, self.sampling_rate, highpass_freq , order=2)
        filtered_data = signal.savgol_filter(
            filtered_data,
            window_length=savgol_window,
            polyorder=savgol_polyorder,
            axis=1  # Entlang der Zeitachse
        )
        #filtered_data = filtered_data[:,sampling*2:sampling*-2]
        return filtered_data
    
    def change_to_consistent_coordinate_system(self, data):
        uniform_cosy_data = []
        for quspin, ch in self.quspin_channel_dict.items(): 
            if "-" in str(ch):
                sign = (-1)
            else:
                sign = 1
            # sign = np.sign(ch)
            ch = abs(int(ch))
            data[ch] *= sign
            
            if "y" in quspin:
                data[ch] *=(-1)
            elif self.quspin_gen_dict[quspin[:-2]]==2:
                data[ch] *=(-1)
        uniform_cosy_data = data
        
        
        return uniform_cosy_data
    
    def get_field_directions(self, data, key):
        """
        Extracts x, y, z directional field data for each QuSpin sensor.
        """
        num_rows = len(self.quspin_position_list)
        num_cols = num_rows  # assuming fixed 4 sensors per row

        # Initialize empty containers for the x, y, z data
        x_data = np.zeros((num_rows, num_cols, data.shape[-1]))
        y_data = np.zeros((num_rows, num_cols, data.shape[-1]))
        z_data = np.zeros((num_rows, num_cols, data.shape[-1]))

        for row_idx, row in enumerate(self.quspin_position_list):
            for col_idx, quspin_id in enumerate(row):
    
                for suffix, target_list in zip(['_x', '_y', '_z'], [x_data, y_data, z_data]):
                    channel_name = quspin_id + suffix
                    channel_index = self.quspin_channel_dict.get(channel_name, None)

                    # Check if the channel is in the exclusion list
                    if channel_index is None or (self.sensor_channels_to_exclude and self.sensor_channels_to_exclude.get(key, None) and channel_name in self.sensor_channels_to_exclude.get(key,[])):
                        continue

                    elif self.sensor_channels_to_exclude and "*"+suffix in self.sensor_channels_to_exclude.get(key, []):
                        # *_x skips every channel with _x
                        continue

                    else:
                        # Ensure the channel index is a valid integer
                        channel_index = abs(int(channel_index))
                        # Extract the data for the channel
                        channel_series = data[channel_index]
                        target_list[row_idx, col_idx, :] = channel_series
        
        return x_data, y_data, z_data

    def align_multi_channel_signal(self, signal1, signal2, lag_cutoff=int(2000), plot=True):
        ox_y_channel = self.quspin_channel_dict.get("OX_y", None)
        f1_y_channel = self.quspin_channel_dict.get("F1_y", None)

        # Select alignment channels
        f1_y_data = signal1[abs(int(f1_y_channel))] * np.sign(f1_y_channel)
        ox_y_data = signal2[31 - abs(int(ox_y_channel))] * np.sign(ox_y_channel)

        # Ensure lag_cutoff is within bounds
        if len(f1_y_data) < lag_cutoff or len(ox_y_data) < lag_cutoff:
            raise ValueError("lag_cutoff exceeds the length of the signals")

        # Compute cross-correlation for alignment
        correlation = correlate(f1_y_data[:lag_cutoff], ox_y_data[:lag_cutoff], mode='full')
        lag = np.argmax(correlation) - (lag_cutoff - 1)

        # Ensure signals are two-dimensional
        if signal1.ndim == 1:
            signal1 = signal1[:, np.newaxis]
        if signal2.ndim == 1:
            signal2 = signal2[:, np.newaxis]

        # Align signal2 to signal1 using lag
        if lag > 0:
            aligned_signal2 = np.pad(
                signal2,
                pad_width=((0, 0), (lag, 0)),
                mode='constant'
            )[:, :signal1.shape[1]]  # Crop back to match signal1
        else:
            aligned_signal2 = signal2[:, -lag:]  # shift left
            if aligned_signal2.shape[1] < signal1.shape[1]:
                pad_width = signal1.shape[1] - aligned_signal2.shape[1]
                aligned_signal2 = np.pad(
                    aligned_signal2,
                    pad_width=((0, 0), (0, pad_width)),
                    mode='constant'
                )
        if plot:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            for i in range(signal1.shape[0]):
                ax[0].plot(signal1[i, :lag_cutoff], color='blue')
                ax[1].plot(signal1[i, :lag_cutoff], color='blue')
            for i in range(signal2.shape[0]):
                ax[0].plot(signal2[i, :lag_cutoff], color='orange')
            
            for i in range(aligned_signal2.shape[0]):
                ax[1].plot(aligned_signal2[i, :lag_cutoff], color='orange')

            ax[0].set_title("Signal 1 (blue) and Signal 2 (orange) before alignment")
            ax[1].set_title("Signal 1 (blue) and Signal 2 (orange) after alignment")
            plt.tight_layout()
            plt.show()


        return aligned_signal2, lag

    def prepare_data(self, key, apply_default_filter=False, intervall_low=5, intervall_high=-5, plot_alignment=False, alignment_cutoff=2000):
        intervall_low = intervall_low*self.sampling_rate
        intervall_high = intervall_high*self.sampling_rate

        data1 = np.transpose(self.data[key])[1:, :]
        data2 = np.transpose(self.add_data[key])[1:, :]

        if apply_default_filter:
            data1 = self.default_filter_combination(data1)
            data2 = self.default_filter_combination(data2)
        
        
        aligned_data2, _ = self.align_multi_channel_signal(data1, data2, plot=plot_alignment, lag_cutoff=alignment_cutoff)

        min_length = min(data1.shape[1], aligned_data2.shape[1])
        data1 = data1[:,:min_length]
        aligned_data2 = aligned_data2[:,:min_length] 

        single_run = np.concatenate((data1, aligned_data2), axis=0)[:,intervall_low:intervall_high]
        time = np.linspace(0,len(single_run[0])/self.sampling_rate,num=len(single_run[0]))

        # aling data to the same coordinate system
        flipped_data = self.change_to_consistent_coordinate_system(single_run)
        x_data, y_data, z_data = self.get_field_directions(flipped_data, key)

        return (x_data, y_data, z_data), time, single_run
    
    def apply_ICA(self, data, max_iter=1000):
        # data is inputtet as square matrix with possible enmpty channels -> data of type x_data, y_data, z_data
        # Loop over each cell in the matrix and filter out empty channels
        data = np.array(data).reshape(data.shape[0]*data.shape[1], -1)

        # drop all zero channels
        arr_cleaned = data[~np.all(data == 0, axis=1)]
        try:
            ica = FastICA(n_components=arr_cleaned.shape[0], random_state=0, max_iter=max_iter)
            S_ica = ica.fit_transform(arr_cleaned.T).T
        except:
            print("All channels are empty")
            return None

        # detect heart beat signal from the ICA components

        return S_ica  #, heart_beat_signal_index

    def detect_qrst_with_t_bounds(self, signal, time):
        """
        Detect Q, R, S, T points and T wave boundaries in an ECG/MCG-like signal of a single heartbeat window.
        Specifically designed to handle both positive and negative R peaks.

        Parameters:
        -----------
        signal : array-like
            The ECG/MCG signal.
        time : array-like
            Time points corresponding to the signal.

        Returns:
        --------
        dict
            Dictionary containing indices and time values for Q, R, S, T points
            and T wave boundaries (start, end).
        """
        
        def lorentzian(x, A, x0, gamma):
            """Lorentzian function for curve fitting."""
            return A / (1 + ((x - x0) / gamma) ** 2)

        # Convert inputs to numpy arrays if they aren't already
        signal_array = np.array(signal)
        time_array = np.array(time)
        
        # Get absolute amplitude for threshold calculations
        abs_signal = np.abs(signal_array)
        signal_mean = np.mean(abs_signal)
        signal_std = np.std(abs_signal)

        # Find peaks in BOTH positive and negative directions
        # For positive peaks (R peaks)
        pos_peaks, _ = find_peaks(signal_array, height=signal_mean + 0.5 * signal_std, distance=len(signal_array) // 4)
        
        # For negative peaks (S peaks)
        neg_peaks, _ = find_peaks(-signal_array, height=signal_mean + 0.5 * signal_std, distance=len(signal_array) // 4)
        
        # Determine if R peak is positive or negative
        is_r_negative = False
        r_idx = None

        # Handle cases where both positive and negative peaks are detected
        if len(pos_peaks) > 0 and len(neg_peaks) > 0:
            max_pos_peak = np.max(signal_array[pos_peaks])
            max_neg_peak = np.max(-signal_array[neg_peaks])
            
            if max_neg_peak > max_pos_peak:
                is_r_negative = True
                r_idx = neg_peaks[np.argmax(-signal_array[neg_peaks])]
            else:
                r_idx = pos_peaks[np.argmax(signal_array[pos_peaks])]
        elif len(pos_peaks) > 0:
            r_idx = pos_peaks[np.argmax(signal_array[pos_peaks])]
        elif len(neg_peaks) > 0:
            is_r_negative = True
            r_idx = neg_peaks[np.argmax(-signal_array[neg_peaks])]
        else:
            raise ValueError("No R peaks detected. Signal may be too noisy or too flat.")
        
        # Find Q and S points based on R peak polarity
        q_search_window = max(0, r_idx - int(len(signal_array) * 0.07))
        q_segment = signal_array[q_search_window:r_idx]
        
        if len(q_segment) > 0:
            if is_r_negative:
                q_idx = q_search_window + np.argmax(q_segment)
            else:
                q_idx = q_search_window + np.argmin(q_segment)
        else:
            q_idx = r_idx  # Fallback if window is empty

        s_search_end = min(len(signal_array), r_idx + int(len(signal_array) * 0.07))
        s_segment = signal_array[r_idx:s_search_end]
        
        if len(s_segment) > 0:
            if is_r_negative:
                s_idx = r_idx + np.argmax(s_segment)
            else:
                s_idx = r_idx + np.argmin(s_segment)
        else:
            s_idx = r_idx  # Fallback if window is empty
        
        # Find P wave (preceding Q)
        p_search_start = max(0, q_idx - int(len(signal_array) * 0.25))
        p_search_end = q_idx - int(len(signal_array) * 0.1)
        p_segment = signal_array[p_search_start:p_search_end]
        
        p_pos_peaks, _ = find_peaks(p_segment, height=0)
        p_neg_peaks, _ = find_peaks(-p_segment, height=0)
        
        if len(p_pos_peaks) > 0 and len(p_neg_peaks) > 0:
            max_p_pos = np.max(p_segment[p_pos_peaks])
            max_p_neg = np.max(-p_segment[p_neg_peaks])
            
            if max_p_pos > max_p_neg:
                p_relative_idx = p_pos_peaks[np.argmax(p_segment[p_pos_peaks])]
            else:
                p_relative_idx = p_neg_peaks[np.argmax(-p_segment[p_neg_peaks])]
        elif len(p_pos_peaks) > 0:
            p_relative_idx = p_pos_peaks[np.argmax(p_segment[p_pos_peaks])]
        elif len(p_neg_peaks) > 0:
            p_relative_idx = p_neg_peaks[np.argmax(-p_segment[p_neg_peaks])]
        else:
            p_relative_idx = np.argmax(np.abs(p_segment))
        
        p_idx = p_search_start + p_relative_idx

        # Find T wave (peak after S)
        t_search_start = s_idx + 80  # Skip a few samples to avoid S
        t_search_end = min(len(signal_array), s_idx + int(len(signal_array) * 0.4))
        
        t_idx = None
        t_segment = None  # Initialize t_segment here to avoid UnboundLocalError
        t_relative_idx = None  # Initialize t_relative_idx 
        
        if t_search_end <= t_search_start:
            t_idx = s_idx  # Fallback if window is invalid
            t_start_idx = t_idx
            t_end_idx = t_idx
        else:
            t_segment = signal_array[t_search_start:t_search_end]
            
            # T wave typically has the same polarity as the R wave in standard leads
            # but can be opposite in MCG signals sometimes
            # We'll look for both positive and negative T peaks and pick the most prominent
            t_pos_peaks, _ = find_peaks(t_segment, height=0)
            t_neg_peaks, _ = find_peaks(-t_segment, height=0)
            
            if len(t_pos_peaks) > 0 and len(t_neg_peaks) > 0:
                max_t_pos = np.max(t_segment[t_pos_peaks]) if len(t_pos_peaks) > 0 else 0
                max_t_neg = np.max(-t_segment[t_neg_peaks]) if len(t_neg_peaks) > 0 else 0
                
                if max_t_pos > max_t_neg:
                    t_relative_idx = t_pos_peaks[np.argmax(t_segment[t_pos_peaks])]
                    t_is_positive = True
                else:
                    t_relative_idx = t_neg_peaks[np.argmax(-t_segment[t_neg_peaks])]
                    t_is_positive = False
            elif len(t_pos_peaks) > 0:
                t_relative_idx = t_pos_peaks[np.argmax(t_segment[t_pos_peaks])]
                t_is_positive = True
            elif len(t_neg_peaks) > 0:
                t_relative_idx = t_neg_peaks[np.argmax(-t_segment[t_neg_peaks])]
                t_is_positive = False
            else:
                # If no clear peak, use the maximum absolute deviation from baseline
                t_relative_idx = np.argmax(np.abs(t_segment))
                t_is_positive = t_segment[t_relative_idx] > 0
                
            t_idx = t_search_start + t_relative_idx

        # Default values in case the try block fails
        t_start_idx = t_idx if t_idx is not None else s_idx
        t_end_idx = t_idx if t_idx is not None else s_idx

        # Only attempt curve fitting if we have valid t_segment and t_relative_idx
        if t_segment is not None and t_relative_idx is not None and len(t_segment) > 0:
            try:
                # Create a symmetrical window around the T wave for fitting
                # First, determine a reasonable window size for analysis
                window_half_width = min(50, len(t_segment) // 4)  # Use smaller of 50 samples or 1/4 of T segment
                
                # Create symmetrical indices for fitting, centered on T peak
                fit_start_idx = max(0, t_relative_idx - window_half_width)
                fit_end_idx = min(len(t_segment), t_relative_idx + window_half_width)
                
                # Extract the segment for fitting
                x_data = np.arange(fit_start_idx, fit_end_idx)
                y_data = t_segment[fit_start_idx:fit_end_idx]
                
                # Initial guess for Lorentzian fitting: [Amplitude, Peak position (t_relative_idx), Width]
                # Adjust amplitude based on T wave polarity
                peak_amplitude = t_segment[t_relative_idx]
                initial_guess = [peak_amplitude, t_relative_idx, window_half_width / 2.5]
                
                # Perform the Lorentzian fitting
                popt, _ = curve_fit(lorentzian, x_data, y_data, p0=initial_guess)
                A, x0, gamma = popt  # Unpack the fitted parameters

                # Compute the Full Width at Half Maximum (FWHM)
                fwhm = 3 * gamma  # Use 3.0 times gamma for a standard definition of T wave boundaries
                
                # Define the start and end indices based on FWHM (in a symmetric window around the T peak)
                # Ensure symmetry around the fitted peak (x0)
                start_fwhm_idx = max(0, int(x0 - fwhm))
                end_fwhm_idx = min(len(t_segment) - 1, int(x0 + fwhm))
                
                # If boundaries are asymmetric due to signal limits, adjust to maintain symmetry
                width_to_start = x0 - start_fwhm_idx
                width_to_end = end_fwhm_idx - x0
                symmetric_width = min(width_to_start, width_to_end)
                
                # Redefine boundaries to be perfectly symmetrical
                start_fwhm_idx = max(0, int(x0 - symmetric_width))
                end_fwhm_idx = min(len(t_segment) - 1, int(x0 + symmetric_width))
                
                # Adjust these indices to the overall signal range (taking t_search_start into account)
                t_start_idx = start_fwhm_idx + t_search_start
                t_end_idx = end_fwhm_idx + t_search_start
                
                # Final bounds check
                t_start_idx = max(0, min(len(signal_array) - 1, t_start_idx))
                t_end_idx = max(0, min(len(signal_array) - 1, t_end_idx))

            except (RuntimeError, ValueError) as e:
                # Fallback if fitting fails: use a fixed symmetrical window around detected T peak
                print(f"T wave fitting failed: {e}")
                fallback_window = min(30, len(t_segment) // 6)  # Conservative window
                t_start_idx = max(0, t_idx - fallback_window)
                t_end_idx = min(len(signal_array) - 1, t_idx + fallback_window)

        # Compile results
        results = {
            'p_idx': p_idx,
            'q_idx': q_idx,
            'r_idx': r_idx,
            's_idx': s_idx,
            't_idx': t_idx if t_idx is not None else s_idx,  # T peak position (manually detected)
            't_start_idx': t_start_idx,  # T wave start
            't_end_idx': t_end_idx,  # T wave end
            'p_time': time_array[p_idx],
            'q_time': time_array[q_idx],
            'r_time': time_array[r_idx],
            's_time': time_array[s_idx],
            't_time': time_array[t_idx if t_idx is not None else s_idx],
            't_start_time': time_array[t_start_idx],
            't_end_time': time_array[t_end_idx],
            'is_r_negative': is_r_negative
        }
        
        return results

    def create_heat_map_animation(self, data, cleanest_i, cleanest_j, output_file='animation.mp4', interval=100, resolution=500, stride=1, direction='x', key="Brustlage", dynamic_scale=True):
        
        """
        data: 4x4xT array
        cleanest_i, cleanest_j: indices of the cleanest channel
        output_file: name of the output file
        interval: time interval between frames in milliseconds
        resolution: resolution of the heatmap
        stride: number of frames to skip for animation

        returns: animation object and figure
        """

        # Custom colormap mit stärkerem Blau-Gelb-Gradienten
        colors = ["black", "purple", "red", "yellow"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
        T = data.shape[-1]  # Anzahl der Zeitschritte

        fig, (ax_main, ax_trace) = plt.subplots(1, 2, figsize=(14, 6))
        
        x_highres = np.linspace(0, 3, resolution)
        y_highres = np.linspace(0, 3, resolution)
        xx_highres, yy_highres = np.meshgrid(x_highres, y_highres)
        
        heatmap = ax_main.imshow(np.zeros((resolution, resolution)), extent=[0, 3, 3, 0],
                                origin='upper', cmap=cmap, interpolation='bilinear')
        cbar = fig.colorbar(heatmap, ax=ax_main)
        cbar.set_label(f'B-{direction} Feldstärke {key}', fontsize=14)
        
        sensor_points, = ax_main.plot([], [], 'ko', markersize=6, markeredgewidth=1)
        cleanest_marker, = ax_main.plot([], [], 'go', markersize=12, markerfacecolor='none', markeredgewidth=3)
        time_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes, fontsize=14)
        
        ax_main.set_xlim(-0.1, 3.1)
        ax_main.set_ylim(-0.1, 3.1)
        
        time_series = data[cleanest_i, cleanest_j, :]
        trace_plot, = ax_trace.plot(np.array(range(T)) * 1000 /self.sampling_rate, time_series, 'b-', label=f'Kanal ({cleanest_i}, {cleanest_j})')
        moving_point, = ax_trace.plot([], [], 'go', markersize=8)
        ax_trace.set_xlim(0, T)
        ax_trace.set_ylim(time_series.min(), time_series.max())
        ax_trace.set_title('Zeitverlauf des festen saubersten Kanals', fontsize=16)
        ax_trace.set_xlabel('Zeit [ms]', fontsize=14)
        ax_trace.set_ylabel(f'B-{direction} Feldstärke {key}', fontsize=14)
        ax_trace.legend()
        ax_trace.grid(True)
        
        def update(frame):
            current_data = data[:, :, frame]
            
            if dynamic_scale:
                valid_values = current_data[np.isfinite(current_data)]
                if valid_values.size > 0:
                    vmin, vmax = valid_values.min(), valid_values.max()
                    heatmap.set_clim(vmin, vmax)

            points, values, sensor_x, sensor_y = [], [], [], []
            for i in range(4):
                for j in range(4):
                    if np.isfinite(current_data[i, j]):
                        if not np.all(current_data[i, j] == 0):
                            points.append([j, 3-i])
                            values.append(current_data[i, j])
                            sensor_x.append(j)
                            sensor_y.append(3-i)
            
            method = 'cubic' if len(points) > 3 else 'linear'
            if points:
                points, values = np.array(points), np.array(values)
                interpolated_data = griddata(points, values, (xx_highres, yy_highres), method=method, fill_value=np.nan)
                heatmap.set_array(interpolated_data)
            
            sensor_points.set_data(sensor_x, sensor_y)
            time_text.set_text(f'Zeit: {frame*1000/self.sampling_rate} ms')
            
            moving_point.set_data([frame], [time_series[frame]])
            
            return heatmap, time_text, cleanest_marker, sensor_points, moving_point
        
        cleanest_marker.set_data([cleanest_j], [3-cleanest_i])

        animation_T = [i for i in range(0, T, stride)]
        ani = FuncAnimation(fig, update, frames=animation_T, interval=interval, blit=True)
        plt.tight_layout()
        ani.save(output_file, writer='ffmpeg', dpi=200)
        plt.show()
        
        return ani, fig
    
    def loop_over_dataset(self, function):
        # Load the overview Excel file
        overview = pd.read_excel("Data/overview.xlsx")

        # Iterate through the directories in the dataset folder
        for dir in sorted(os.listdir("Data/")):
            dir_path = os.path.join("Data/", dir)
            print(f"Processing directory: {dir}")

            if dir.startswith("P") and os.path.isdir(dir_path):  # Check for valid directory
                # Extract patient number from directory name
                number = re.findall(r'\d+', dir)
                patient_number = int(number[0]) if number else None

                if overview.loc[overview["Porbanten Nr."] == patient_number, "runs"].values[0].strip() == "-":
                    print(f"No runs found for patient {patient_number}. Skipping...")
                    continue

                # Find the corresponding runs for this patient number
                self.key_list = [key for key in self.data]#overview.loc[overview["Porbanten Nr."] == patient_number, "runs"].values[0].strip().split(", ")

                # Find the relevant .tdms files (up to 2 files)
                count = 0
                for file in os.listdir(dir_path):
                    if file.endswith(".tdms"):
                        count += 1
                        if count == 1:
                            self.filename = os.path.join(dir_path, file)
                        elif count == 2:
                            self.add_filename = os.path.join(dir_path, file)
                        elif count > 2:
                            break

                # If the required .tdms files are found, proceed
                if count >= 2:
                    # Load the log file and data
                    self.log_file_path = os.path.join(dir_path, "QZFM_log_file.txt")
                    self.data = self.import_tdms(self.filename, self.scaling)
                    self.key_list = overview.loc[overview["Porbanten Nr."] == patient_number, "runs"].values[0].strip().split(", ")

                    self.add_data = self.import_tdms(self.add_filename, self.scaling)
                    self.sensor_channels_to_exclude = json.loads(overview.loc[overview["Porbanten Nr."] == patient_number, "Sensors to exclude"].values[0])

                    # Load data for quspin channels
                    self.quspin_gen_dict, self.quspin_channel_dict, self.quspin_position_list = self.load_data_from_file(self.log_file_path)

                    # Update the quspin channel dictionary based on value conditions
                    for key, value in self.quspin_channel_dict.items():
                        if abs(value) >= 100:  # Check if the absolute value has three digits
                            sign = -1 if value < 0 else 1
                            last_two_digits = abs(value) % 100  # Extract the last two digits
                            new_value = 31 + last_two_digits  # Calculate the new value
                            self.quspin_channel_dict[key] = sign * new_value  # Update the value

                    # Prepare and process the data for each key in the key list
                    for key in self.key_list:
                        data = self.prepare_data(key, apply_default_filter=True)
                        # Pass the data to the function to process further
                        function(self, data, key, dir)
                else:
                    print(f"Not enough .tdms files in directory {dir}. Skipping...")

            else:
                print(f"Skipping non-patient directory: {dir}")




if __name__ == "__main__":
    # Example usage
    analyzer = Analyzer()
