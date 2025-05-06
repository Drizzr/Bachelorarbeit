# MCG Data Analyzer

## Overview

The `Analyzer` class is a comprehensive Python tool for processing, analyzing, and visualizing Magnetocardiography (MCG) data stored in TDMS (`.tdms`) files. It is designed to handle multi-channel MCG recordings, offering functionalities from raw data loading and filtering to advanced signal processing techniques like Independent Component Analysis (ICA), cardiac cycle segmentation, QRS complex detection, and various visualizations.

A key feature of this class is its standardized internal processing at **250 Hz**. The `prepare_data` method handles resampling from the original source sampling rate to this internal rate, ensuring consistency across all analysis steps.

<!-- Placeholder for a general overview image or diagram of the MCG analysis workflow -->

## Features

*   **TDMS Data Loading**: Imports data from primary and additional TDMS files.
*   **Sensor Log Integration**: Parses QZFM log files for sensor mapping and orientation.
*   **Data Preparation**:
    *   Alignment of data from two separate TDMS files (e.g., two halves of a recording).
    *   Time interval selection.
    *   Resampling to an internal standard of 250 Hz.
    *   Coordinate system transformation based on sensor logs.
*   **Signal Filtering**:
    *   Bandstop, bandpass, lowpass, and highpass filters.
    *   Savitzky-Golay smoothing.
    *   Polynomial detrending and baseline removal.
    *   A `default_filter_combination` for common noise removal.
*   **Cardiac Segmentation**:
    *   Utilizes a pre-trained PyTorch model (`ECGSegmenter` from the `MCG_segmentation` package) to classify heartbeats into P-wave, QRS complex, T-wave, and no-wave segments.
    *   Sliding window approach for segmenting long recordings.
    *   Post-processing to remove short artifactual segments.
*   **Channel Quality Assessment**:
    *   `find_cleanest_channel` method to identify the channel with the most physiologically plausible and confident heartbeat segmentation.
    *   Scoring based on segmentation confidence and segment duration plausibility (P, QRS, T percentages).
*   **QRS Complex Detection**:
    *   Detects QRS peaks on the cleanest channel or all channels.
    *   Calculates heart rate (HR) and heart rate variability (HRV - SDNN).
*   **Waveform Averaging**: Averages QRS complexes or other segments around detected peaks.
*   **Independent Component Analysis (ICA)**:
    *   Applies FastICA to decompose signals.
    *   Filters components based on their "heart beat score" to potentially remove artifacts or isolate cardiac activity.
    *   Interactive plotting for threshold adjustment (for 3D grid data).
*   **Visualization**:
    *   **Sensor Matrix Plot**: Grid plot of time-series data from sensor arrays.
    *   **Butterfly Plot**: Overlaid plot of all channel signals.
    *   **Segmented Signal Plot**: Time-series plot with highlighted P, QRS, T segments.
    *   **Heart Vector Projections**: 2D projections (XY, XZ, YZ) of the derived heart vector, with calculated metrics.
    *   **Linear Spectral Density (LSD) Plot**: `plot_lsd_multichannel` for frequency domain analysis.
    *   **Heatmap Animation**: Animated spatial heatmap of field strength over time.
*   **Configurable**: Device selection for PyTorch (CUDA/CPU), sampling rates, scaling factors, model paths.
*   **Logging**: Integrated `logging` for monitoring processing steps and potential issues.

## Dependencies

The class relies on several Python libraries:

*   `os`
*   `ast`
*   `warnings`
*   `logging`
*   `numpy`
*   `torch` (for segmentation model)
*   `nptdms` (for reading TDMS files)
*   `matplotlib` (for plotting)
*   `scipy` (for signal processing: filters, interpolation, correlation)
*   `sklearn` (for FastICA)

Optional but recommended for full functionality:
*   `MCG_segmentation` package (local package for `ECGSegmenter`). If not found, segmentation features will be unavailable.

## Setup

### 1. Python Environment

It's recommended to use a virtual environment. Install the required packages:

```bash
pip install numpy torch nptdms matplotlib scipy scikit-learn
Use code with caution.
Markdown
Ensure you have a PyTorch version compatible with your system (CPU or CUDA). Visit pytorch.org for installation instructions.

2. MCG_segmentation Package
The Analyzer class attempts to import ECGSegmenter from a local package named MCG_segmentation.
For segmentation features to work, this package must be structured correctly and be importable (e.g., in your PYTHONPATH or in the same directory). The expected structure for the model within this package is:

MCG_segmentation/
├── model/
│   └── model.py  (containing ECGSegmenter class)
└── MCGSegmentator_s/
    └── checkpoints/
        └── best/
            ├── model.pth
            └── params.json
Use code with caution.
If MCG_segmentation.model.model.ECGSegmenter cannot be imported, a warning will be logged, and segmentation-dependent methods will not function correctly.

3. Model Checkpoints
The default path for the segmentation model checkpoints is MCG_segmentation/MCGSegmentator_s/checkpoints/best. If your model is located elsewhere, you can specify the path during Analyzer initialization using the model_checkpoint_dir parameter.

The Analyzer Class
Core Concept: Internal 250 Hz Sampling Rate
All internal processing, especially segmentation and QRS detection, assumes that the data is sampled at 250 Hz. The prepare_data method is responsible for resampling the input data from its original input_sampling_rate (defaulted from DEFAULT_SOURCE_SAMPLING_RATE if not specified) to this INTERNAL_SAMPLING_RATE. This ensures consistency and compatibility with the pre-trained segmentation model.

Initialization
from your_module import Analyzer # Assuming Analyzer is in your_module.py

analyzer = Analyzer(
    filename="path/to/primary_data.tdms",
    add_filename="path/to/additional_data.tdms", # Optional
    log_file_path="path/to/sensor_log.txt",
    sensor_channels_to_exclude={'run_key1': ['SensorA_x', 'SensorB_y']}, # Optional
    scaling=2.7 / 1000, # Default scaling factor
    num_ch=48, # Expected number of channels after combining primary and additional data
    model_checkpoint_dir="path/to/your/segmentation_model_checkpoints" # Optional
)
Use code with caution.
Python
Parameters:

filename (str): Path to the primary TDMS data file.
add_filename (str, optional): Path to an additional TDMS data file. Data from this file will be aligned and concatenated with the primary file.
log_file_path (str): Path to the QZFM log file. This file contains sensor mappings (e.g., quspin_channel_dict, quspin_position_list) used for interpreting sensor orientations and positions.
sensor_channels_to_exclude (dict, optional): A dictionary where keys are run names (from TDMS group names) and values are lists of channel names (e.g., Q01_x) to exclude from processing for that specific run. Wildcards like *_x can be used to exclude all x-components.
scaling (float): Scaling factor applied to TDMS data during initial loading. Default: 2.7 / 1000.
num_ch (int): The total number of channels expected after combining data from filename and add_filename (if provided). Used for plotting and array initialization. Default: 48.
model_checkpoint_dir (str): Directory containing the model.pth and params.json for the ECGSegmenter model. Default: "MCG_segmentation/MCGSegmentator_s/checkpoints/best".
Upon initialization, the Analyzer loads TDMS data, sensor log information, and the segmentation model (if available). It also sets up the computation device (CUDA or CPU) for PyTorch.

Key Attributes
data (dict): Data loaded from filename. Keys are run names, values are NumPy arrays.
add_data (dict): Data loaded from add_filename.
key_list (list): List of run keys (group names) found in the primary TDMS file.
quspin_gen_dict, quspin_channel_dict, quspin_position_list: Loaded from the sensor log file.
model (ECGSegmenter instance or None): The loaded segmentation model.
INTERNAL_SAMPLING_RATE (int): Fixed at 250 Hz.
DEVICE (torch.device): The device (CPU or CUDA) used for model computations.
Main Methods and Workflow
A typical workflow involves:

Initializing the Analyzer.
Preparing data for a specific run using prepare_data(). This handles alignment, interval selection, filtering (optional), coordinate transformation, and resampling to 250 Hz.
Performing analysis such as segmentation, QRS detection, or ICA.
Visualizing results.
Data Loading and Preparation
__init__(...): Loads raw data from TDMS files and sensor log.
prepare_data(key, apply_default_filter=False, intervall_low_sec=5, intervall_high_sec=-5, plot_alignment=False, alignment_cutoff_sec=2, input_sampling_rate=1000):
Selects data for the given key (run name).
Aligns data from self.data and self.add_data if both exist for the key.
Optionally applies default_filter_combination.
Selects a time interval from the combined data.
Applies sensor coordinate system adjustments.
Resamples data to self.INTERNAL_SAMPLING_RATE (250 Hz).
Returns (x_data, y_data, z_data), time_vector, combined_run_data_250Hz.
The x/y/z_data are 3D grids (rows, cols, samples) based on quspin_position_list.
combined_run_data_250Hz is the (channels, samples) array at 250 Hz.
Signal Filtering
default_filter_combination(data, bandstop_freq=50, lowpass_freq=95, highpass_freq=1, savgol_window=61, savgol_polyorder=2, sampling_rate=1000): Applies a pre-defined sequence of bandstop, lowpass, highpass, and Savitzky-Golay filters. Note: sampling_rate here refers to the rate of the input data to this method.
Static filter methods: bandstop_filter, bandpass_filter, apply_lowpass_filter, apply_highpass_filter, remove_drift_and_offset.
Segmentation and QRS Detection (Requires MCG_segmentation package and model)
find_cleanest_channel(data_250Hz, ...): Segments all channels in data_250Hz (must be at 250 Hz) and scores them based on segmentation confidence and physiological plausibility of P/QRS/T wave percentages. Returns the index of the best channel, along with segmentation labels and confidence scores for all channels.
detect_qrs_complex_peaks_cleanest_channel(data_250Hz, ...): Runs find_cleanest_channel and then detects QRS peaks on this cleanest channel. Returns peak indices, cleanest channel index, all labels, HR, and HRV.
detect_qrs_complex_peaks_all_channels(data_250Hz, ...): Detects QRS peaks independently for all channels. Returns a dictionary of peak indices per channel, cleanest channel index, all labels, average HR, and average HRV.
avg_window(data_250Hz, peak_positions, ...): Averages waveforms around the provided peak_positions. Can filter windows based on a heartbeat score threshold.
Independent Component Analysis (ICA)
ICA_filter(data_250Hz, heart_beat_score_threshold=0.85, ...):
Applies FastICA to data_250Hz (must be at 250 Hz).
Scores each independent component using a "heart beat score" derived from segmentation.
Reconstructs the signal, nullifying components below heart_beat_score_threshold.
Can optionally plot results with an interactive threshold slider for 3D grid data.
Returns filtered_result, ica_components, best_component_idx, score_mask.
Visualization
plot_sensor_matrix(data, time, name, ...): Plots sensor data in a grid layout.
butterfly_plot(data, time, num_ch, name, ...): Overlays all channel signals.
plot_segmented_signal(signal_250Hz, prediction_labels, ...): Plots a single channel's signal with colored regions indicating P/QRS/T segments.
plot_all_heart_vector_projections(heart_vector_components_250Hz, ...): Plots XY, XZ, YZ projections of the heart vector. heart_vector_components should be a (3, num_samples) array (Bx, By, Bz).
plot_lsd_multichannel(data, noise_theos, freqs, name, ...): Plots Linear Spectral Density.
create_heat_map_animation(data_grid_250Hz, cleanest_i, cleanest_j, ...): Creates an MP4 animation of the spatial field distribution over time. data_grid_250Hz should be (rows, cols, samples).
Static Utility Methods
The class also includes several static methods for common signal processing tasks (e.g., bandstop_filter, apply_lowpass_filter, remove_drift_and_offset) and plotting helpers.

Basic Usage Example
This example demonstrates a common workflow: loading data, preparing it, finding the cleanest channel, detecting QRS peaks, and plotting the segmented signal of the cleanest channel.

import numpy as np
import matplotlib.pyplot as plt
# from your_module import Analyzer # Assuming Analyzer is in your_module.py

# --- Dummy Data and File Creation (for a runnable example) ---
# In a real scenario, you would have existing TDMS and log files.
def create_dummy_tdms(filepath, group_name="Run1", num_channels=24, num_samples=20000, fs=1000):
    from nptdms import TdmsWriter, ChannelObject
    with TdmsWriter(filepath) as tdms_writer:
        channels = []
        time_data = np.linspace(0, num_samples / fs, num_samples)
        for i in range(num_channels):
            # Simple sine wave with noise
            signal_data = np.sin(2 * np.pi * 1 * time_data + i * np.pi/4) + \
                          0.2 * np.sin(2 * np.pi * 50 * time_data) + \
                          0.1 * np.random.randn(num_samples)
            channels.append(ChannelObject(group_name, f'Channel{i+1}', signal_data * 1000 * (2.7/1000) )) # Scaled
        tdms_writer.write_segment(channels)

def create_dummy_log_file(log_path, num_main_sensors=8): # Assumes 3 axes per sensor for 24 channels
    sensor_positions = [
        ['Q01', 'Q02'],
        ['Q03', 'Q04'],
        ['Q05', 'Q06'],
        ['Q07', 'Q08']
    ] # Simplified 4x2 layout for 8 QuSpins
    
    quspin_channel_dict = {}
    ch_idx = 0
    for r_idx, row in enumerate(sensor_positions):
        for c_idx, q_id in enumerate(row):
            quspin_channel_dict[f'{q_id}_x'] = ch_idx
            ch_idx += 1
            quspin_channel_dict[f'{q_id}_y'] = ch_idx
            ch_idx += 1
            quspin_channel_dict[f'{q_id}_z'] = ch_idx
            ch_idx += 1

    log_content = {
        'quspin_gen_dict': {f'Q{i:02d}': 1 for i in range(1, num_main_sensors + 1)},
        'quspin_channel_dict': quspin_channel_dict,
        'quspin_position_list': sensor_positions
    }
    import ast, json
    with open(log_path, 'w') as f:
        # Log file uses a string representation of a dict
        f.write(str(log_content))

# Create dummy files
primary_tdms_path = "dummy_primary_data.tdms"
# additional_tdms_path = "dummy_additional_data.tdms" # For this example, we'll use only primary
log_file_path = "dummy_sensor_log.txt"
run_key = "Run1"
num_primary_channels = 24 # 8 sensors * 3 axes
# num_additional_channels = 24
source_sampling_rate = 1000

create_dummy_tdms(primary_tdms_path, run_key, num_primary_channels, fs=source_sampling_rate)
# create_dummy_tdms(additional_tdms_path, run_key, num_additional_channels, fs=source_sampling_rate)
create_dummy_log_file(log_file_path, num_main_sensors=8)
# --- End of Dummy Data Creation ---

try:
    # 1. Initialize Analyzer
    # For this example, we'll only use one TDMS file, so num_ch = num_primary_channels
    # If using add_filename, num_ch would be num_primary_channels + num_additional_channels
    analyzer = Analyzer(
        filename=primary_tdms_path,
        # add_filename=additional_tdms_path, # Uncomment if using a second TDMS
        log_file_path=log_file_path,
        num_ch=num_primary_channels # Adjust if using add_filename
        # model_checkpoint_dir can be specified if not default
    )

    if not analyzer.key_list:
        print(f"No runs found in TDMS file: {primary_tdms_path}")
    else:
        selected_run_key = analyzer.key_list[0] # e.g., "Run1"
        print(f"Processing run: {selected_run_key}")

        # 2. Prepare data for the selected run
        # This resamples data to analyzer.INTERNAL_SAMPLING_RATE (250 Hz)
        # For dummy data, source_sampling_rate is 1000 Hz.
        (grid_x, grid_y, grid_z), time_250Hz, combined_data_250Hz = analyzer.prepare_data(
            key=selected_run_key,
            apply_default_filter=True, # Apply standard filters
            intervall_low_sec=1,    # Start 1 sec into the recording
            intervall_high_sec=-1,  # End 1 sec before the end
            input_sampling_rate=source_sampling_rate # Specify original sampling rate
        )

        if combined_data_250Hz is not None and combined_data_250Hz.size > 0:
            print(f"Prepared data shape (channels, samples) @ 250Hz: {combined_data_250Hz.shape}")
            print(f"Time vector length for 250Hz data: {time_250Hz.shape[0]}")

            # 3. Find the cleanest channel (optional, but good for focusing analysis)
            # This requires the segmentation model to be working
            if analyzer.model is not None:
                best_channel_idx, all_labels_250Hz, all_confidence_250Hz, _ = analyzer.find_cleanest_channel(
                    combined_data_250Hz,
                    print_results=True
                )
                print(f"Cleanest channel index: {best_channel_idx}")

                # 4. Detect QRS complexes on the cleanest channel
                qrs_peaks_250Hz, _, _, hr, hrv = analyzer.detect_qrs_complex_peaks_cleanest_channel(
                    combined_data_250Hz,
                    print_heart_rate=True
                )
                print(f"Detected {len(qrs_peaks_250Hz)} QRS peaks on channel {best_channel_idx}.")
                if hr is not None:
                    print(f"Heart Rate: {hr:.2f} bpm, HRV (SDNN): {hrv:.2f} ms")

                # 5. Plot the segmented signal for the cleanest channel
                cleanest_signal_250Hz = combined_data_250Hz[best_channel_idx, :]
                cleanest_labels_250Hz = all_labels_250Hz[best_channel_idx, :]

                fig, ax = plt.subplots(figsize=(15, 5))
                analyzer.plot_segmented_signal(
                    signal=cleanest_signal_250Hz.reshape(1, -1), # Expects (b, T)
                    pred=cleanest_labels_250Hz.reshape(1, -1),   # Expects (b, T)
                    axs=ax
                )
                ax.set_title(f"Segmented Signal for Cleanest Channel ({best_channel_idx}) - Run {selected_run_key}")
                plt.show()
                
                # <!-- Placeholder for Segmented Signal Plot Image -->

                # Example: Averaging QRS complexes
                if qrs_peaks_250Hz:
                    avg_qrs_waveforms, time_window_avg = analyzer.avg_window(
                        combined_data_250Hz,
                        peak_positions=qrs_peaks_250Hz, # Use peaks from cleanest channel for all
                        window_left=0.15, # seconds before peak
                        window_right=0.25 # seconds after peak
                    )
                    plt.figure(figsize=(10,6))
                    plt.plot(time_window_avg, avg_qrs_waveforms[best_channel_idx])
                    plt.title(f"Average QRS Waveform - Channel {best_channel_idx}")
                    plt.xlabel("Time (s) relative to peak")
                    plt.ylabel("Amplitude (pT)")
                    plt.grid(True)
                    plt.show()
                     # <!-- Placeholder for Average QRS Waveform Image -->

            else:
                print("Segmentation model not available. Skipping segmentation-dependent analysis.")

            # Example: Plotting sensor matrix for Z-component (if grid data is valid)
            if grid_z is not None and grid_z.shape[0] > 0 and grid_z.shape[1] > 0 :
                analyzer.plot_sensor_matrix(grid_z[:,:,:int(5*analyzer.INTERNAL_SAMPLING_RATE)], time_250Hz[:int(5*analyzer.INTERNAL_SAMPLING_RATE)], f"Sensor Matrix Z - {selected_run_key} (First 5s @ 250Hz)", save=False)
                # <!-- Placeholder for Sensor Matrix Plot Image -->

        else:
            print(f"Failed to prepare data for run {selected_run_key} or data is empty.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure dummy files are created or paths are correct.")
except ImportError as e:
    print(f"Import error: {e}. Make sure all dependencies are installed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

# Cleanup dummy files (optional)
# import os
# os.remove(primary_tdms_path)
# os.remove(log_file_path)
# if os.path.exists(additional_tdms_path): os.remove(additional_tdms_path)
Use code with caution.
Python
<!-- Placeholder for Basic Usage Example Output Image -->
Detailed Method Explanations
(Refer to the class docstrings for detailed parameter descriptions. Key aspects are highlighted here.)

prepare_data(...)
This is a crucial first step for any analysis.

It takes a key (run name) and parameters for interval selection and filtering.
Crucially, input_sampling_rate must be provided if it's different from the default 1000 Hz, as this is used for correct initial time interval selection and resampling.
Handles alignment of primary and additional TDMS files if add_filename was provided.
Applies sensor orientation corrections via _change_to_consistent_coordinate_system.
Resamples the data to self.INTERNAL_SAMPLING_RATE (250 Hz).
Organizes channel data into 3D grids (x_data, y_data, z_data) based on quspin_position_list from the sensor log.
Output: (x_data, y_data, z_data), time_vector_250Hz, combined_run_data_250Hz. All returned data is at 250 Hz.
ICA_filter(data_250Hz, ...)
Input data_250Hz must be at 250 Hz. It can be 2D (channels, samples) or 3D (grid_x, grid_y, samples).
Decomposes the signal into independent components.
If ECGSegmenter is available, it segments each component to assess its "heartbeat-likeness."
Components with a score below heart_beat_score_threshold are zeroed out before reconstructing the signal.
If plot_result=True and input is 3D, an interactive plot allows dynamic adjustment of the threshold.
# Assuming 'combined_data_250Hz' is available from prepare_data()
if analyzer.model is not None:
    filtered_data_250Hz, ica_components, best_comp_idx, score_mask = analyzer.ICA_filter(
        combined_data_250Hz,
        heart_beat_score_threshold=0.8,
        plot_result=False # Set to True for interactive plot with 3D data
    )
    if filtered_data_250Hz is not None:
        print(f"ICA filtered data shape: {filtered_data_250Hz.shape}")
        # Further analysis or plotting on filtered_data_250Hz
        analyzer.butterfly_plot(filtered_data_250Hz, time_250Hz, filtered_data_250Hz.shape[0], "ICA Filtered Butterfly Plot")
        # <!-- Placeholder for ICA Filtered Butterfly Plot Image -->
else:
    print("ICA_filter with heartbeat scoring requires the segmentation model.")
Use code with caution.
Python
Plotting Methods
All plotting methods take data (usually at 250 Hz if processed) and relevant parameters. Examples:

plot_all_heart_vector_projections(heart_vector_components_250Hz, title_suffix, save_path=None)
heart_vector_components_250Hz should be a NumPy array of shape (3, num_samples) representing Bx, By, Bz components of the heart vector, typically derived from averaged beats or specific ICA components.
# Example: Assuming avg_qrs_waveforms_xyz are (3, N_samples) from some processing
# For instance, if grid_x, grid_y, grid_z are averaged waveforms from avg_window
# and then processed to get dominant X, Y, Z components.
# This is a conceptual example of preparing input for heart vector plots.

# Placeholder for actual heart vector derivation logic
# If avg_qrs_waveforms is (num_channels, window_length)
# And we can map these channels to X, Y, Z components
# For simplicity, let's assume we have derived mean Bx, By, Bz signals
# from the averaged beats across relevant sensors.

# Example: if avg_qrs_waveforms contains averaged data for all channels
# and we assume specific channels correspond to global X, Y, Z.
# This is highly dependent on the sensor setup and how Bx,By,Bz are defined.
# Let's make up some dummy data for plotting:

if 'avg_qrs_waveforms' in locals() and avg_qrs_waveforms.shape[1] > 0:
    t_win = time_window_avg
    # Create synthetic Bx, By, Bz components for demonstration
    # In reality, these would be derived from the actual MCG data,
    # e.g., by averaging specific sensor orientations or from ICA.
    example_bx = np.sin(2 * np.pi * 1 * t_win) * np.exp(-t_win * 2)
    example_by = np.cos(2 * np.pi * 1 * t_win + np.pi/4) * np.exp(-t_win * 2)
    example_bz = np.sin(2 * np.pi * 1 * t_win + np.pi/2) * np.exp(-t_win * 2.5)
    
    heart_vector_example = np.vstack([example_bx, example_by, example_bz])

    analyzer.plot_all_heart_vector_projections(
        heart_vector_example,
        title_suffix=f"Avg Beat - Run {selected_run_key}",
        save_path=f"heart_vector_projections_{selected_run_key}.png"
    )
    # <!-- Placeholder for Heart Vector Projections Image -->
Use code with caution.
Python
create_heat_map_animation(data_grid_250Hz, cleanest_i, cleanest_j, ...)
data_grid_250Hz needs to be a 3D array (rows, cols, samples_250Hz).
cleanest_i, cleanest_j are row/col indices for a reference trace in the animation.
# Assuming grid_x_250Hz is (rows, cols, samples) from prepare_data()
# And we have determined cleanest_i, cleanest_j (e.g., from sensor layout)
# This can be computationally intensive and requires ffmpeg.
# For this example, we'll use a small segment of data.
# Note: The dummy sensor log creates a 4x2 layout (8 QuSpins).
# The get_field_directions method pads this to a square matrix if quspin_position_list is not square.
# Check grid_x.shape to ensure it's suitable.

# Example: If grid_x is (4,4,N), and we pick a channel, e.g. (0,0)
# The actual indices for cleanest_i, cleanest_j would depend on your specific sensor array logic
# or could be arbitrary for a general heatmap.
if grid_x is not None and grid_x.ndim == 3 and grid_x.shape[2] > 100:
    print(f"Creating heatmap for X-component (shape: {grid_x.shape})")
    # analyzer.create_heat_map_animation(
    #     grid_x[:, :, :int(2 * analyzer.INTERNAL_SAMPLING_RATE)], # Animate first 2 seconds
    #     cleanest_i=0, # Example row index
    #     cleanest_j=0, # Example col index
    #     output_file=f'heatmap_x_{selected_run_key}.mp4',
    #     direction='x',
    #     key=selected_run_key
    # )
    print("Heatmap animation creation commented out for brevity in example.")
    # <!-- Placeholder for Heatmap Animation Still Frame or GIF -->
Use code with caution.
Python
Input File Formats
TDMS Files (.tdms)
Standard National Instruments TDMS files.
The class expects data to be organized in groups (runs), with each group containing multiple channels.
If using add_filename, it's assumed that group names (run keys) correspond between the primary and additional files if they are to be processed together for a given run.
Sensor Log File (.txt)
A text file containing a Python dictionary representation. This dictionary should include:
quspin_gen_dict: Maps QuSpin sensor IDs (e.g., 'Q01') to generation or type.
quspin_channel_dict: Maps full sensor channel names (e.g., 'Q01_x', 'Q01_y', 'Q01_z') to their numerical channel indices in the TDMS data. Negative values indicate inverted polarity.
Note: The __init__ method includes logic to adjust channel indices val where abs(val) >= 100 to sign * (31 + abs(val) % 100). This is a specific transformation that might need adjustment based on your channel mapping conventions.
quspin_position_list: A list of lists representing the grid layout of QuSpin sensors, e.g., [['Q01', 'Q02'], ['Q03', 'Q04']]. Used by get_field_directions.
Example sensor_log.txt content:

{
    'quspin_gen_dict': {'Q01': 1, 'Q02': 1, ...},
    'quspin_channel_dict': {'Q01_x': 0, 'Q01_y': 1, 'Q01_z': 2, 'Q02_x': 3, ...},
    'quspin_position_list': [['Q01', 'Q02', 'Q03', 'Q04'],
                             ['Q05', 'Q06', 'Q07', 'Q08'],
                             ... ]
}
Use code with caution.
Python
Logging
The class uses the Python logging module. By default, INFO level messages and above are printed to the console, including timestamps and log levels. This helps track the progress and diagnose issues.

Troubleshooting / Notes
Segmentation Model: If ECGSegmenter or its model files are not found, segmentation-related features (find_cleanest_channel, QRS detection, ICA filtering with heartbeat score) will be impaired or unavailable. Warnings will be logged.
Memory Usage: Processing long recordings, especially with ICA or animations, can be memory-intensive. Consider processing data in chunks if memory becomes an issue, though the current class structure processes entire runs loaded into memory.
Sampling Rates: Pay close attention to input_sampling_rate in prepare_data and the sampling_rate parameter in default_filter_combination to ensure filters are designed correctly for the data at that stage. All data fed into segmentation, QRS detection, and ICA methods (unless otherwise specified by the method's parameters) is expected to be at INTERNAL_SAMPLING_RATE (250 Hz).
ffmpeg: The create_heat_map_animation method requires ffmpeg to be installed and in your system's PATH to save animations.
Coordinate System: The _change_to_consistent_coordinate_system method applies specific sign changes based on sensor names (e.g., if 'y' in quspin_id or quspin_gen_dict indicates type 2). This logic might need customization based on your specific sensor setup and desired coordinate conventions.
Use code with caution.
