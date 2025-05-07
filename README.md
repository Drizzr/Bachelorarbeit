

# MCG Data Analyzer

## Overview

The `Analyzer` class is a comprehensive Python tool for processing, analyzing, and visualizing Magnetocardiography (MCG) data stored in TDMS (`.tdms`) files. It is designed to handle multi-channel MCG recordings, offering functionalities from raw data loading and filtering to advanced signal processing techniques like Independent Component Analysis (ICA), cardiac cycle segmentation, QRS complex detection, and various visualizations.

A key feature of this class is its standardized internal processing at **250 Hz**. The `prepare_data` method handles resampling from the original source sampling rate to this internal rate, ensuring consistency across all analysis steps.

## Features

- **TDMS Data Loading**: Imports data from primary and additional TDMS files.
- **Sensor Log Integration**: Parses QZFM log files for sensor mapping and orientation.
- **Data Preparation**:
  - Alignment of data from two separate TDMS files (e.g., two halves of a recording).
  - Time interval selection.
  - Resampling to an internal standard of 250 Hz.
  - Coordinate system transformation based on sensor logs.
- **Signal Filtering**:
  - Bandstop, bandpass, lowpass, and highpass filters.
  - Savitzky-Golay smoothing.
  - Polynomial detrending and baseline removal.
  - A `default_filter_combination` for common noise removal.
- **Cardiac Segmentation**:
  - Utilizes a pre-trained PyTorch model (`ECGSegmenter` from the `MCG_segmentation` package) to classify heartbeats into P-wave, QRS complex, T-wave, and no-wave segments.
  - Sliding window approach for segmenting long recordings.
  - Post-processing to remove short artifactual segments.
- **Channel Quality Assessment**:
  - `find_cleanest_channel` method to identify the channel with the most physiologically plausible and confident heartbeat segmentation.
  - Scoring based on segmentation confidence and segment duration plausibility (P, QRS, T percentages).
- **QRS Complex Detection**:
  - Detects QRS peaks on the cleanest channel or all channels.
  - Calculates heart rate (HR) and heart rate variability (HRV - SDNN).
- **Waveform Averaging**: Averages QRS complexes or other segments around detected peaks.
- **Independent Component Analysis (ICA)**:
  - Applies FastICA to decompose signals.
  - Filters components based on their "heart beat score" to potentially remove artifacts or isolate cardiac activity.
  - Interactive plotting for threshold adjustment (for 3D grid data).
- **Visualization**:
  - **Sensor Matrix Plot**: Grid plot of time-series data from sensor arrays.
  - **Butterfly Plot**: Overlaid plot of all channel signals.
  - **Segmented Signal Plot**: Time-series plot with highlighted P, QRS, T segments.
  - **Heart Vector Projections**: 2D projections (XY, XZ, YZ) of the derived heart vector, with calculated metrics.
  - **Linear Spectral Density (LSD) Plot**: `plot_lsd_multichannel` for frequency domain analysis.
  - **Heatmap Animation**: Animated spatial heatmap of field strength over time.
- **Configurable**: Device selection for PyTorch (CUDA/CPU), sampling rates, scaling factors, model paths.
- **Logging**: Integrated `logging` for monitoring processing steps and potential issues.

## Dependencies

The class relies on several Python libraries to install them run:

```bash
pip install -r reuqirements.txt
```

in the root directory


## Setup

### 1. Dependencies

The class relies on several Python libraries to install them run:

```bash
pip install -r reuqirements.txt
```

in the root directory

### 2. MCG_segmentation Package

The `Analyzer` class attempts to import `ECGSegmenter` from a local package named `MCG_segmentation`. For segmentation features to work, this package must be structured correctly and importable (e.g., in your `PYTHONPATH` or the same directory). The expected structure is:

```
MCG_segmentation/
├── model/
│   └── model.py  (containing ECGSegmenter class)
└── MCGSegmentator_s/
    └── checkpoints/
        └── best/
            ├── model.pth
            └── params.json
```

If `MCG_segmentation.model.model.ECGSegmenter` cannot be imported, a warning will be logged, and segmentation-dependent methods will not function correctly.

### 3. Model Checkpoints

The default path for the segmentation model checkpoints is `MCG_segmentation/MCGSegmentator_s/checkpoints/best`. If your model is located elsewhere, specify the path during `Analyzer` initialization using the `model_checkpoint_dir` parameter.

## The Analyzer Class

### Core Concept: Internal 250 Hz Sampling Rate

All internal processing, especially segmentation and QRS detection, assumes the data is sampled at 250 Hz. The `prepare_data` method resamples input data from its original `input_sampling_rate` (defaulted from `DEFAULT_SOURCE_SAMPLING_RATE` if not specified) to this `INTERNAL_SAMPLING_RATE`, ensuring consistency and compatibility with the pre-trained segmentation model.

### Initialization

```python
from your_module import Analyzer  # Assuming Analyzer is in your_module.py

analyzer = Analyzer(
    filename="path/to/primary_data.tdms",
    add_filename="path/to/additional_data.tdms",  # Optional
    log_file_path="path/to/sensor_log.txt",
    sensor_channels_to_exclude={'run_key1': ['SensorA_x', 'SensorB_y']},  # Optional
    scaling=2.7 / 1000,  # Default scaling factor
    num_ch=48,  # Expected number of channels after combining primary and additional data
    model_checkpoint_dir="path/to/your/segmentation_model_checkpoints"  # Optional
)
```

**Parameters**:

- `filename` (str): Path to the primary TDMS data file.
- `add_filename` (str, optional): Path to an additional TDMS data file. Data from this file will be aligned and concatenated with the primary file.
- `log_file_path` (str): Path to the QZFM log file containing sensor mappings (e.g., `quspin_channel_dict`, `quspin_position_list`).
- `sensor_channels_to_exclude` (dict, optional): Dictionary where keys are run names (from TDMS group names) and values are lists of channel names (e.g., `Q01_x`) to exclude. Wildcards like `*_x`, which excludes the entire x-Channels are supported.
- `scaling` (float): Scaling factor applied to TDMS data during loading. Default: `2.7 / 1000`.
- `num_ch` (int): Total number of channels expected after combining data. Default: 48.
- `model_checkpoint_dir` (str): Directory containing `model.pth` and `params.json` for the `ECGSegmenter` model. Default: `MCG_segmentation/MCGSegmentator_s/checkpoints/best`.

Upon initialization, the `Analyzer` loads TDMS data, sensor log information, and the segmentation model (if available). It sets up the computation device (CUDA/CPU/mps) for PyTorch. (the fastest option is selected automatically)

### Key Attributes

- `data` (dict): Data loaded from `filename`. Keys are run names, values are NumPy arrays.
- `add_data` (dict): Data loaded from `add_filename`.
- `key_list` (list): List of run keys (group names) in the primary TDMS file.
- `quspin_gen_dict`, `quspin_channel_dict`, `quspin_position_list`: Loaded from the sensor log file.
- `model` (ECGSegmenter instance or None): The loaded segmentation model.
- `INTERNAL_SAMPLING_RATE` (int): Fixed at 250 Hz.
- `DEVICE` (torch.device): Device (CPU or CUDA) for model computations.

### Main Methods

A typical workflow involves:

1. Initializing the `Analyzer`.
2. Preparing data for a specific run using `prepare_data()`. This handles alignment, interval selection, filtering (optional), coordinate transformation, and resampling to 250 Hz.
3. Performing analysis such as segmentation, QRS detection, or ICA.
4. Visualizing results.

### Data Preparation and Alignment

#### `prepare_data`
Prepares time-aligned, spatially-oriented data from multiple sensor recordings, handling alignment, filtering, resampling, and coordinate transformations in one streamlined step.

**Workflow**:

Load raw sensor data from internal sources.
Align datasets using cross-correlation (using `align_multi_channel_signal`)
Concatenate and crop to the specified time interval.
Optionally apply default filters. (using `default_filter_combination`)
Convert to a consistent coordinate system.
Resample to internal rate (default 250 Hz).
Extract x, y, z field components in a grid layout. (using 
**Usage**:
```python
(x_data, y_data, z_data), time, combined = processor.prepare_data("run_01", apply_default_filter=True)
```
**Parameters**:

| **Parameter**          | **Type** | **Default** | **Description**                            |
| ---------------------- | -------- | ----------- | ------------------------------------------ |
| `key`                  | `str`    | —           | Dataset key (must exist in internal data). |
| `apply_default_filter` | `bool`   | `False`     | Apply filtering after alignment.           |
| `intervall_low_sec`    | `float`  | `5`         | Start time (sec) from aligned start.       |
| `intervall_high_sec`   | `float`  | `-5`        | End time (sec) from aligned end.           |
| `plot_alignment`       | `bool`   | `False`     | Plot alignment visualization.              |
| `alignment_cutoff_sec` | `float`  | `2`         | Max duration for alignment (in seconds).   |
| `input_sampling_rate`  | `int`    | `1000`      | Sampling rate of original data.            |


**Returns**:

(x_data, y_data, z_data) (np.ndarray): 3D spatial signals (rows × cols × time) at 250 Hz.
time (np.ndarray): 1D time vector in seconds.
combined_run_data (np.ndarray): Combined and preprocessed raw signal (channels × time).

**Example alignment plot**:
![image](https://github.com/user-attachments/assets/80065fda-0be1-4fd9-807e-5163a04950e8)


#### `align_multi_channel_signal`
Aligns two multichannel signals using cross-correlation of the averaged signal over channels, typically used for synchronizing measurements from multiple acquisition systems.

**Usage**:
```python
aligned, lag = processor.align_multi_channel_signal(signal1, signal2, lag_cutoff=2000)
```

**Parameters:**

| **Parameter** | **Type**     | **Default** | **Description**                      |
| ------------- | ------------ | ----------- | ------------------------------------ |
| `signal1`     | `np.ndarray` | —           | First signal `(channels × samples)`. |
| `signal2`     | `np.ndarray` | —           | Second signal to align.              |
| `lag_cutoff`  | `int`        | `2000`      | Max samples to consider for lag.     |
| `plot`        | `bool`       | `True`      | Plot before/after alignment.         |


**Returns:**

aligned_signal2 (np.ndarray): Shifted version of signal2, aligned to signal1.
lag (int): Estimated lag in samples.
Raises:

ValueError — If lag_cutoff exceeds signal length.

#### `get_field_directions`
Converts multichannel flat data into spatial 3D field representations in x, y, z directions, using known sensor layout metadata, based on the log file specified in `log_file_path`.

**Usage**:

```python
x_data, y_data, z_data = processor.get_field_directions(data, key="run_01")
```

**Parameters**:

| **Parameter** | **Type**     | **Description**                   |
| ------------- | ------------ | --------------------------------- |
| `data`        | `np.ndarray` | 2D array `(channels × samples)`.  |
| `key`         | `str`        | Dataset key for sensor exclusion. |

**Returns**:

x_data, y_data, z_data (np.ndarray): Arrays of shape (rows, cols, samples) for each field direction.


#### `invert_field_directions`
Performs the inverse of get_field_directions, reconstructing the original channel-wise signal from 3D x/y/z field representations, based on the log file specified in `log_file_path`.

**Usage**:
```python
reconstructed = processor.invert_field_directions(x_data, y_data, z_data, key="run_01")
```

**Parameters**:

| **Parameter**                | **Type**     | **Default** | **Description**                                       |
| ---------------------------- | ------------ | ----------- | ----------------------------------------------------- |
| `x_data`, `y_data`, `z_data` | `np.ndarray` | —           | Grid arrays for field directions.                     |
| `key`                        | `str`        | —           | Dataset key for sensor exclusion.                     |
| `num_channels`               | `int`        | `None`      | Total number of output channels (inferred if `None`). |

**Returns:**

data (np.ndarray) — Reconstructed signal array (channels × samples).


### Signal Filtering

#### `default_filter_combination`

Applies a predefined sequence of filters to multichannel signal data to remove noise and prepare it for further analysis. The pipeline includes:

* **Bandstop filter**: Attenuates narrow frequency bands (e.g., powerline interference).
* **Lowpass filter**: Removes high-frequency noise.
* **Highpass filter**: Eliminates baseline drift or slow signal trends.
* **Savitzky-Golay filter**: Smooths the signal while preserving features.

**Usage**:

```python
filtered = processor.default_filter_combination(data)
```

**Parameters**:

| Name               | Type         | Default | Description                                 |
| ------------------ | ------------ | ------- | ------------------------------------------- |
| `data`             | `np.ndarray` | —       | Input array of shape `(channels, samples)`. |
| `bandstop_freq`    | `float`      | `50`    | Center frequency for bandstop filter (Hz).  |
| `lowpass_freq`     | `float`      | `95`    | Lowpass cutoff frequency (Hz).              |
| `highpass_freq`    | `float`      | `1`     | Highpass cutoff frequency (Hz).             |
| `savgol_window`    | `int`        | `61`    | Window size for Savitzky-Golay smoothing.   |
| `savgol_polyorder` | `int`        | `2`     | Polynomial order for Savitzky-Golay filter. |
| `sampling_rate`    | `int`        | `1000`  | Data sampling rate (Hz).                    |

**Returns**:
`np.ndarray` — Filtered data array with the same shape as the input.

---

#### `ICA_filter`

Uses **Independent Component Analysis (ICA)** to isolate and remove heartbeat-related artifacts from multichannel data. Supports both 2D and 3D input (e.g., 4×4 sensor grids).

**Features**:

* Scores ICA components based on physiological plausibility.
* Retains only components with a high likelihood of containing real neural signals.
* Optional plotting tool for interactive threshold adjustment in 3D datasets.

**Usage**:

```python
result, components, best_idx, mask = processor.ICA_filter(data, plot_result=True)
```

If `print_result, plot_result = True, True` this returns:

![Figure_1](https://github.com/user-attachments/assets/85a9f730-a2a9-4f71-9c9f-c671ef00f704)
(where the red dotted line indicates the original signal)

```
Channel Selection Results:
Channel   Conf        P-Wave %    QRS %       T-Wave %    Plausibility   Final Score 
-------------------------------------------------------------------------------------
1         0.8761      10.60       11.08       21.08       1.0000         0.9008      
2         0.8017      1.28        43.56       23.92       0.2208         0.6855      
3         0.8290      7.84        9.32        19.36       0.9843         0.8600      
4         0.8182      2.08        37.60       25.44       0.2596         0.7064      
5         0.8030      0.84        54.60       11.84       0.1669         0.6758      
6         0.8421      12.80       13.24       20.04       1.0000         0.8737      
7         0.8117      6.24        37.12       20.24       0.2952         0.7084      
8         0.8811      13.60       10.96       24.92       1.0000         0.9049      
9         0.8165      8.40        28.20       28.92       0.4310         0.7394      
10        0.7869      9.64        20.40       35.92       0.4690         0.7233      
11        0.7874      6.72        14.68       34.16       0.6477         0.7594      
12        0.8606      12.44       12.80       21.04       1.0000         0.8885      
13        0.8529      11.84       9.40        22.36       1.0000         0.8823      
14        0.8069      11.28       12.48       34.52       0.6887         0.7832      
15        0.8120      6.84        23.72       24.32       0.5030         0.7502      

Best Channel Summary:
Channel   : 8
Conf      : 0.8811
Plausibility: 1.0000
Final Score: 0.9049
Segment Distribution:
  P-Wave % : 13.60%
  QRS %    : 10.96%
  T-Wave % : 24.92%
```

**Parameters**:

| Name                         | Type         | Default | Description                                                          |
| ---------------------------- | ------------ | ------- | -------------------------------------------------------------------- |
| `data`                       | `np.ndarray` | —       | Input data, shape `(channels × time)` or `(grid_x × grid_y × time)`. |
| `heart_beat_score_threshold` | `float`      | `0.85`  | Score threshold for keeping ICA components.                          |
| `max_iter`                   | `int`        | `5000`  | Max iterations for ICA convergence.                                  |
| `confidence_weight`          | `float`      | `0.8`   | Weight for confidence score in scoring.                              |
| `plausibility_weight`        | `float`      | `0.2`   | Weight for plausibility score.                                       |
| `print_result`               | `bool`       | `False` | If `True`, prints evaluation mectrics for the channels.              |
| `plot_result`                | `bool`       | `False` | If `True` and data is 4×4 grid, displays interactive plot.           |

**Returns**:

* `result` (`np.ndarray`) — Reconstructed signal with heartbeat components removed.
* `ica_components` (`np.ndarray`) — Extracted ICA components.
* `best_channel_idx` (`int`) — Index of component most strongly related to heartbeat.
* `score_mask` (`np.ndarray`) — Boolean mask of retained components.

**Raises**:

* `ValueError` — If the input data is not 2D or 3D.




#### Segmentation and QRS Detection (Requires MCG_segmentation package and model)

- `find_cleanest_channel(data_250Hz, ...)`: Segments all channels in `data_250Hz` (must be at 250 Hz) and scores them based on segmentation confidence and physiological plausibility of P/QRS/T wave percentages. Returns the best channel index, segmentation labels, and confidence scores.
- `detect_qrs_complex_peaks_cleanest_channel(data_250Hz, ...)`: Runs `find_cleanest_channel` and detects QRS peaks on the cleanest channel. Returns peak indices, cleanest channel index, all labels, HR, and HRV.
- `detect_qrs_complex_peaks_all_channels(data_250Hz, ...)`: Detects QRS peaks independently for all channels. Returns a dictionary of peak indices per channel, cleanest channel index, all labels, average HR, and average HRV.
- `avg_window(data_250Hz, peak_positions, ...)`: Averages waveforms around provided `peak_positions`. Can filter windows based on a heartbeat score threshold.

#### Independent Component Analysis (ICA)

- `ICA_filter(data_250Hz, heart_beat_score_threshold=0.85, ...)`:
  - Applies FastICA to `data_250Hz` (must be at 250 Hz).
  - Scores each independent component using a "heart beat score" derived from segmentation.
  - Reconstructs the signal, nullifying components below `heart_beat_score_threshold`.
  - Optionally plots results with an interactive threshold slider for 3D grid data.
  - Returns `filtered_result`, `ica_components`, `best_component_idx`, `score_mask`.

#### Visualization

- `plot_sensor_matrix(data, time, name, ...)`: Plots sensor data in a grid layout.
- `butterfly_plot(data, time, num_ch, name, ...)`: Overlays all channel signals.
- `plot_segmented_signal(signal_250Hz, prediction_labels, ...)`: Plots a single channel's signal with colored P/QRS/T segments.
- `plot_all_heart_vector_projections(heart_vector_components_250Hz, ...)`: Plots XY, XZ, YZ projections of the heart vector. `heart_vector_components` should be a (3, num_samples) array (Bx, By, Bz).
- `plot_lsd_multichannel(data, noise_theos, freqs, name, ...)`: Plots Linear Spectral Density.
- `create_heat_map_animation(data_grid_250Hz, cleanest_i, cleanest_j, ...)`: Creates an MP4 animation of spatial field distribution. `data_grid_250Hz` should be (rows, cols, samples).

#### Static Utility Methods

The class includes static methods for signal processing (e.g., `bandstop_filter`, `apply_lowpass_filter`, `remove_drift_and_offset`) and plotting helpers.

### Basic Usage Example

This example demonstrates loading data, preparing it, finding the cleanest channel, detecting QRS peaks, and plotting the segmented signal.

```python
import numpy as np
import matplotlib.pyplot as plt
# from your_module import Analyzer  # Assuming Analyzer is in your_module.py

# --- Dummy Data and File Creation ---
def create_dummy_tdms(filepath, group_name="Run1", num_channels=24, num_samples=20000, fs=1000):
    from nptdms import TdmsWriter, ChannelObject
    with TdmsWriter(filepath) as tdms_writer:
        channels = []
        time_data = np.linspace(0, num_samples / fs, num_samples)
        for i in range(num_channels):
            signal_data = (np.sin(2 * np.pi * 1 * time_data + i * np.pi/4) +
                           0.2 * np.sin(2 * np.pi * 50 * time_data) +
                           0.1 * np.random.randn(num_samples))
            channels.append(ChannelObject(group_name, f'Channel{i+1}', signal_data * 1000 * (2.7/1000)))
        tdms_writer.write_segment(channels)

def create_dummy_log_file(log_path, num_main_sensors=8):
    sensor_positions = [['Q01', 'Q02'], ['Q03', 'Q04'], ['Q05', 'Q06'], ['Q07', 'Q08']]
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
    with open(log_path, 'w') as f:
        f.write(str(log_content))

# Create dummy files
primary_tdms_path = "dummy_primary_data.tdms"
log_file_path = "dummy_sensor_log.txt"
run_key = "Run1"
num_primary_channels = 24
source_sampling_rate = 1000

create_dummy_tdms(primary_tdms_path, run_key, num_primary_channels, fs=source_sampling_rate)
create_dummy_log_file(log_file_path, num_main_sensors=8)

# --- Workflow ---
try:
    # 1. Initialize Analyzer
    analyzer = Analyzer(
        filename=primary_tdms_path,
        log_file_path=log_file_path,
        num_ch=num_primary_channels
    )

    if not analyzer.key_list:
        print(f"No runs found in TDMS file: {primary_tdms_path}")
    else:
        selected_run_key = analyzer.key_list[0]
        print(f"Processing run: {selected_run_key}")

        # 2. Prepare data
        (grid_x, grid_y, grid_z), time_250Hz, combined_data_250Hz = analyzer.prepare_data(
            key=selected_run_key,
            apply_default_filter=True,
            intervall_low_sec=1,
            intervall_high_sec=-1,
            input_sampling_rate=source_sampling_rate
        )

        if combined_data_250Hz is not None and combined_data_250Hz.size > 0:
            print(f"Prepared data shape (channels, samples) @ 250Hz: {combined_data_250Hz.shape}")
            print(f"Time vector length for 250Hz data: {time_250Hz.shape[0]}")

            # 3. Find cleanest channel
            if analyzer.model is not None:
                best_channel_idx, all_labels_250Hz, all_confidence_250Hz, _ = analyzer.find_cleanest_channel(
                    combined_data_250Hz,
                    print_results=True
                )
                print(f"Cleanest channel index: {best_channel_idx}")

                # 4. Detect QRS complexes
                qrs_peaks_250Hz, _, _, hr, hrv = analyzer.detect_qrs_complex_peaks_cleanest_channel(
                    combined_data_250Hz,
                    print_heart_rate=True
                )
                print(f"Detected {len(qrs_peaks_250Hz)} QRS peaks on channel {best_channel_idx}.")
                if hr is not None:
                    print(f"Heart Rate: {hr:.2f} bpm, HRV (SDNN): {hrv:.2f} ms")

                # 5. Plot segmented signal
                cleanest_signal_250Hz = combined_data_250Hz[best_channel_idx, :]
                cleanest_labels_250Hz = all_labels_250Hz[best_channel_idx, :]

                fig, ax = plt.subplots(figsize=(15, 5))
                analyzer.plot_segmented_signal(
                    signal=cleanest_signal_250Hz.reshape(1, -1),
                    pred=cleanest_labels_250Hz.reshape(1, -1),
                    axs=ax
                )
                ax.set_title(f"Segmented Signal for Cleanest Channel ({best_channel_idx}) - Run {selected_run_key}")
                plt.show()

                # Example: Averaging QRS complexes
                if qrs_peaks_250Hz:
                    avg_qrs_waveforms, time_window_avg = analyzer.avg_window(
                        combined_data_250Hz,
                        peak_positions=qrs_peaks_250Hz,
                        window_left=0.15,
                        window_right=0.25
                    )
                    plt.figure(figsize=(10, 6))
                    plt.plot(time_window_avg, avg_qrs_waveforms[best_channel_idx])
                    plt.title(f"Average QRS Waveform - Channel {best_channel_idx}")
                    plt.xlabel("Time (s) relative to peak")
                    plt.ylabel("Amplitude (pT)")
                    plt.grid(True)
                    plt.show()

            else:
                print("Segmentation model not available. Skipping segmentation-dependent analysis.")

            # Example: Plotting sensor matrix
            if grid_z is not None and grid_z.shape[0] > 0 and grid_z.shape[1] > 0:
                analyzer.plot_sensor_matrix(
                    grid_z[:, :, :int(5 * analyzer.INTERNAL_SAMPLING_RATE)],
                    time_250Hz[:int(5 * analyzer.INTERNAL_SAMPLING_RATE)],
                    f"Sensor Matrix Z - {selected_run_key} (First 5s @ 250Hz)",
                    save=False
                )

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
```

## Detailed Method Explanations

(Refer to the class docstrings for detailed parameter descriptions. Key aspects are highlighted here.)

### `prepare_data(...)`

This is a crucial first step for any analysis.

- Takes a `key` (run name) and parameters for interval selection and filtering.
- Requires `input_sampling_rate` if different from the default 1000 Hz for correct time interval selection and resampling.
- Handles alignment of primary and additional TDMS files if `add_filename` is provided.
- Applies sensor orientation corrections via `_change_to_consistent_coordinate_system`.
- Resamples data to `self.INTERNAL_SAMPLING_RATE` (250 Hz).
- Organizes channel data into 3D grids (`x_data`, `y_data`, `z_data`) based on `quspin_position_list`.
- **Output**: `(x_data, y_data, z_data)`, `time_vector_250Hz`, `combined_run_data_250Hz`. All returned data is at 250 Hz.

### `ICA_filter(data_250Hz, ...)`

- Input `data_250Hz` must be at 250 Hz, either 2D (channels, samples) or 3D (grid_x, grid_y, samples).
- Decomposes the signal into independent components.
- If `ECGSegmenter` is available, segments each component to assess its "heartbeat-likeness."
- Components below `heart_beat_score_threshold` are zeroed out before signal reconstruction.
- If `plot_result=True` and input is 3D, an interactive plot allows dynamic threshold adjustment.

```python
if analyzer.model is not None:
    filtered_data_250Hz, ica_components, best_comp_idx, score_mask = analyzer.ICA_filter(
        combined_data_250Hz,
        heart_beat_score_threshold=0.8,
        plot_result=False
    )
    if filtered_data_250Hz is not None:
        print(f"ICA filtered data shape: {filtered_data_250Hz.shape}")
        analyzer.butterfly_plot(
            filtered_data_250Hz,
            time_250Hz,
            filtered_data_250Hz.shape[0],
            "ICA Filtered Butterfly Plot"
        )
else:
    print("ICA_filter with heartbeat scoring requires the segmentation model.")
```

### Plotting Methods

All plotting methods take data (usually at 250 Hz if processed) and relevant parameters.

#### `plot_all_heart_vector_projections(heart_vector_components_250Hz, title_suffix, save_path=None)`

- `heart_vector_components_250Hz` should be a (3, num_samples) NumPy array (Bx, By, Bz), typically derived from averaged beats or ICA components.

```python
if 'avg_qrs_waveforms' in locals() and avg_qrs_waveforms.shape[1] > 0:
    t_win = time_window_avg
    example_bx = np.sin(2 * np.pi * 1 * t_win) * np.exp(-t_win * 2)
    example_by = np.cos(2 * np.pi * 1 * t_win + np.pi/4) * np.exp(-t_win * 2)
    example_bz = np.sin(2 * np.pi * 1 * t_win + np.pi/2) * np.exp(-t_win * 2.5)
    heart_vector_example = np.vstack([example_bx, example_by, example_bz])

    analyzer.plot_all_heart_vector_projections(
        heart_vector_example,
        title_suffix=f"Avg Beat - Run {selected_run_key}",
        save_path=f"heart_vector_projections_{selected_run_key}.png"
    )
```

#### `create_heat_map_animation(data_grid_250Hz, cleanest_i, cleanest_j, ...)`

- `data_grid_250Hz` must be a 3D array (rows, cols, samples_250Hz).
- `cleanest_i`, `cleanest_j` are row/col indices for a reference trace.

```python
if grid_x is not None and grid_x.ndim == 3 and grid_x.shape[2] > 100:
    print(f"Creating heatmap for X-component (shape: {grid_x.shape})")
    # analyzer.create_heat_map_animation(
    #     grid_x[:, :, :int(2 * analyzer.INTERNAL_SAMPLING_RATE)],
    #     cleanest_i=0,
    #     cleanest_j=0,
    #     output_file=f'heatmap_x_{selected_run_key}.mp4',
    #     direction='x',
    #     key=selected_run_key
    # )
    print("Heatmap animation creation commented out for brevity.")
```

## Input File Formats

### TDMS Files (.tdms)

- Standard National Instruments TDMS files.
- Data organized in groups (runs), each containing multiple channels.
- If using `add_filename`, group names (run keys) should correspond between primary and additional files for combined processing.

### Sensor Log File (.txt)

- A text file containing a Python dictionary representation with:
  - `quspin_gen_dict`: Maps QuSpin sensor IDs (e.g., `Q01`) to generation or type.
  - `quspin_channel_dict`: Maps channel names (e.g., `Q01_x`, `Q01_y`, `Q01_z`) to numerical channel indices in TDMS data. Negative values indicate inverted polarity.
  - `quspin_position_list`: A list of lists representing the grid layout of QuSpin sensors, e.g., `[['Q01', 'Q02'], ['Q03', 'Q04']]`.

Example `sensor_log.txt` content:

```python
{
    'quspin_gen_dict': {'Q01': 1, 'Q02': 1, ...},
    'quspin_channel_dict': {'Q01_x': 0, 'Q01_y': 1, 'Q01_z': 2, 'Q02_x': 3, ...},
    'quspin_position_list': [['Q01', 'Q02', 'Q03', 'Q04'],
                             ['Q05', 'Q06', 'Q07', 'Q08'],
                             ... ]
}
```

**Note**: The `__init__` method adjusts channel indices where `abs(val) >= 100` to `sign * (31 + abs(val) % 100)`. Adjust this logic based on your channel mapping conventions.

## Logging

The class uses Python's `logging` module. By default, `INFO` level messages and above are printed to the console with timestamps and log levels, aiding progress tracking and issue diagnosis.

## Troubleshooting / Notes

- **Segmentation Model**: If `ECGSegmenter` or its model files are missing, segmentation features (`find_cleanest_channel`, QRS detection, ICA filtering with heartbeat score) will be impaired. Warnings will be logged.
- **Memory Usage**: Processing long recordings, especially with ICA or animations, can be memory-intensive. Consider chunked processing if memory is a concern.
- **Sampling Rates**: Ensure `input_sampling_rate` in `prepare_data` and `sampling_rate` in `default_filter_combination` match the data's rate at each stage. Data for segmentation, QRS detection, and ICA must be at `INTERNAL_SAMPLING_RATE` (250 Hz) unless specified otherwise.
- **ffmpeg**: The `create_heat_map_animation` method requires `ffmpeg` in the system's PATH.
- **Coordinate System**: The `_change_to_consistent_coordinate_system` method applies sign changes based on sensor names (e.g., if `'y' in quspin_id` or `quspin_gen_dict` indicates type 2). Customize this logic for your sensor setup and coordinate conventions.

