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
from functools import wraps
import pandas as pd
import re
import json
from scipy.interpolate import interp1d




class Analyzer:

    def __init__(self, filename = "", add_filename = "", log_file_path = "", sensor_channels_to_excllude = {"Brustlage": ["NL_x"], "Rueckenlage": ["NL_x"]}, scaling = 2.7/1000, sampling = 1000, num_ch = 48):
        self.filename = filename
        self.add_filename =   add_filename
        self.log_file_path  = log_file_path

        self.scaling = scaling
        self.sampling = sampling
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
        self.cmaplist = [self.cmap(x) for x in np.linspace(0,1,num=40)]

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

    def find_cleanest_channel(self, data):
        """
        Find the channel with the clearest peaks in the data.
        This function detects both positive and negative peaks and returns the channel index
        with the highest mean peak height.
        """
        best_mean = 0
        argmax = 0
        best_peak_type = 'positive'
        for ch in range(len(data)):
            meanvalue = np.median(data[ch])
            # Detect positive peaks
            positive_peaks, positive_prop = find_peaks(data[ch], height=meanvalue * 5, distance=500, width=(15, 50))
            positive_heights = positive_prop["peak_heights"]
            mean_positive = np.mean(positive_heights) if len(positive_heights) > 0 else 0
            # Detect negative peaks
            negative_peaks, negative_prop = find_peaks(-data[ch], height=meanvalue * 5, distance=500, width=(15, 50))
            negative_heights = negative_prop["peak_heights"]  # These are -data[ch] at minima, positive values
            mean_negative = np.mean(negative_heights) if len(negative_heights) > 0 else 0
            # Determine which peak type is clearer for this channel
            if mean_positive > mean_negative:
                mean = mean_positive
                peak_type = 'positive'
            else:
                mean = mean_negative
                peak_type = 'negative'
            # Update best channel if this mean is higher
            if ch == 0 or mean > best_mean:
                best_mean = mean
                argmax = ch
                best_peak_type = peak_type
        return argmax, best_peak_type

    def find_peak_positions_fixed_ch(self, data, time, ch, peak_type='positive', z_threshold=2.5):
        meanvalue = np.median(data[ch])   
        
        if peak_type == 'positive':
            peak_positions, peak_prop = find_peaks(data[ch], height=meanvalue*5, distance=500)
            peak_heights = peak_prop["peak_heights"]
        elif peak_type == 'negative':
            peak_positions, peak_prop = find_peaks(-data[ch], height=meanvalue*5, distance=500)
            peak_heights = peak_prop["peak_heights"]  # These are -data[ch] at minima, positive values
        else:
            raise ValueError("peak_type must be 'positive' or 'negative'")

        # Filter peaks using z-score
        if len(peak_positions) > 0:
            z_scores = np.abs((peak_heights - np.median(peak_heights)) / np.std(peak_heights))
            peak_positions = peak_positions[z_scores < z_threshold]
        
        # Compute intervals in milliseconds
        if len(peak_positions) > 1:
            # Assume time is in seconds; convert to milliseconds
            peak_times = time[peak_positions] * 1000 / (self.sampling) # ms
            intervals = np.diff(peak_times)  # ms
            
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            hrv = std_interval  # HRV is typically defined as std of RR intervals
            
            print(f"{len(peak_positions)} beats detected; average bpm: {round(len(peak_positions) / time[-1] * 60, 2)}")
            print(f"Mean inter-beat interval: {round(mean_interval, 2)} ms")
            print(f"STD of inter-beat intervals: {round(std_interval, 2)} ms")
        return peak_positions
        
    def plotting_time_series(self, data, time, num_ch, name, path = None, save = False):
        fig, elem= plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(3.94, 3.54),dpi=100)  
        fig.suptitle(name,size='small', y=0.99)
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
    
    def detect_cardiac_cycles(self, data, reference_channel, peak_type='positive'):
        """
        Erkennt Herzzyklen mit verbesserter Robustheit gegenüber Arrhythmien.
        """
        # Referenzkanal filtern
        filtered_data = self.bandpass_filter(data[reference_channel], self.sampling, 0.5, 40)
        
        baseline = np.median(filtered_data)
        mad = np.median(np.abs(filtered_data - baseline))
        
        # Adaptive Schwellenwerte
        threshold = baseline + 3.5 * mad if peak_type == 'positive' else baseline - 3.5 * mad
        data_to_analyze = filtered_data if peak_type == 'positive' else -filtered_data
        
        # Peaks erkennen
        peaks, _ = find_peaks(data_to_analyze, 
                             height=threshold, 
                             distance=int(0.3 * self.sampling),
                             width=(int(0.02 * self.sampling), int(0.12 * self.sampling)))
        
        # RR-Intervalle prüfen, um Arrhythmien zu identifizieren
        rr_intervals = np.diff(peaks)
        median_rr = np.median(rr_intervals)
        
        # Ungewöhnliche RR-Intervalle identifizieren (mögliche Arrhythmien)
        anomalous_intervals = np.where(np.abs(rr_intervals - median_rr) > 0.2 * median_rr)[0]
        
        # Anomale Peaks markieren (für spätere spezielle Behandlung)
        anomalous_peaks = []
        for idx in anomalous_intervals:
            # Der zweite Peak des anomalen Intervalls wird markiert
            anomalous_peaks.append(peaks[idx + 1])
        
        return peaks, anomalous_peaks
    
    def phase_aligned_averaging(self, data, time):
        """
        Performs phase-aligned averaging to compensate for heart rate variations.
        
        Args:
            data: Multi-channel signal data (n_channels x n_samples)
            time: Time vector corresponding to the data
            
        Returns:
            tuple: (averaged_cycles, time_window) where averaged_cycles is the phase-aligned
                average across channels and time_window is the normalized time vector
        """
        try:
            # Find optimal reference channel for peak detection
            reference_channel, peak_type = self.find_cleanest_channel(data)
            
            # Detect cardiac cycle peaks
            peaks = self.find_peak_positions_fixed_ch(data, time, reference_channel, peak_type)
            
            # Check if sufficient peaks are detected
            if len(peaks) < 3:
                return None, None
                
            # Calculate RR intervals and median
            rr_intervals = np.diff(peaks)
            median_rr = np.median(rr_intervals)
            
            # Define parameters
            REFERENCE_LENGTH = 1000  # Normalized phase length
            PRE_PEAK_FRACTION = 0.2  # Fraction of RR before peak
            POST_PEAK_FRACTION = 0.1  # Fraction of RR after peak
            
            # Create normalized time base
            normalized_time = np.linspace(0, 1, REFERENCE_LENGTH)
            
            # Process each channel
            n_channels = data.shape[0]
            avg_channels = np.zeros((n_channels, REFERENCE_LENGTH))
            
            for ch in range(n_channels):
                cycles = []
                for i in range(len(peaks) - 1):
                    # Calculate cycle boundaries
                    start_idx = max(0, peaks[i] - int(PRE_PEAK_FRACTION * median_rr))
                    end_idx = min(data.shape[1], peaks[i + 1] - int(POST_PEAK_FRACTION * median_rr))
                    
                    # Skip invalid cycles
                    if end_idx <= start_idx:
                        continue
                        
                    # Extract cycle
                    cycle = data[ch, start_idx:end_idx]
                    cycle_time = np.linspace(0, 1, len(cycle))
                    
                    # Interpolate to reference length
                    try:
                        interpolator = interp1d(cycle_time, cycle, 
                                            kind='cubic', 
                                            bounds_error=False, 
                                            fill_value='extrapolate')
                        resampled_cycle = interpolator(normalized_time)
                        cycles.append(resampled_cycle)
                    except ValueError:
                        continue
                
                # Compute average for channel
                avg_channels[ch] = np.mean(cycles, axis=0) if cycles else np.zeros(REFERENCE_LENGTH)
            
            # Scale time window to seconds
            time_window = normalized_time * (median_rr / self.sampling)
            
            return avg_channels, time_window
            
        except Exception as e:
            print(f"Error in phase_aligned_averaging: {str(e)}")
            return None, None

    def avg_window(self, data, peak_positions, num_ch, window_left = 0.3, window_right =0.4):
        samples_left = int(window_left * self.sampling)
        samples_right = int(window_right * self.sampling)
        window_length = samples_left + samples_right

        avg_channels = []

        for ch in range(num_ch):
            windows = []
            for pos in peak_positions:
                # Sicherstellen, dass das Fenster nicht aus dem Signal rausfällt
                if pos - samples_left < 0 or pos + samples_right > data.shape[1]:
                    continue  # Fenster unvollständig → überspringen
                window = data[ch, pos - samples_left : pos + samples_right]
                time = np.linspace(0, len(window) / self.sampling, num=len(window), endpoint=False)
                # Drift entfernen
                window_detrended = self.remove_drift_and_offset(window, time)
                windows.append(window_detrended)
            if windows:
                mean_window = np.mean(windows, axis=0)
            else:
                mean_window = np.zeros(window_length)  # Fallback falls kein Fenster gültig war

            avg_channels.append(mean_window)
        time_window = np.linspace(0, window_length / self.sampling, num=window_length, endpoint=False)

        return np.array(avg_channels), time_window

    def default_filter_combination(self, data, bandstop_freq=50, lowpass_freq=95, highpass_freq=1, savgol_window=61, savgol_polyorder=2):
        filtered_data =  self.bandstop_filter(data, bandstop_freq, 2, self.sampling, order = 2)
        filtered_data =  self.bandstop_filter(filtered_data, bandstop_freq, 3, self.sampling, order = 3)
        filtered_data =  self.bandstop_filter(filtered_data, bandstop_freq*2, 3, self.sampling, order = 3)
        filtered_data = self.apply_lowpass_filter(filtered_data, self.sampling, lowpass_freq, order=3)

        filtered_data = self.apply_highpass_filter(filtered_data, self.sampling, highpass_freq , order=2)
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
        intervall_low = intervall_low*self.sampling
        intervall_high = intervall_high*self.sampling

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
        time = np.linspace(0,len(single_run[0])/self.sampling,num=len(single_run[0]))

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
        trace_plot, = ax_trace.plot(np.array(range(T)) * 1000 /self.sampling, time_series, 'b-', label=f'Kanal ({cleanest_i}, {cleanest_j})')
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
            time_text.set_text(f'Zeit: {frame*1000/self.sampling} ms')
            
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
                        function(data, key, dir)
                else:
                    print(f"Not enough .tdms files in directory {dir}. Skipping...")

            else:
                print(f"Skipping non-patient directory: {dir}")


