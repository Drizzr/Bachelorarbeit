# MCG Data Analyzer Documentation
- [Overview](#overview)
  - [Key Features](#key-features)
- [Installation and Setup](#installation-and-setup)
  - [Dependencies](#dependencies)
  - [MCG Segmentation Package](#mcg-segmentation-package)
  - [Model Checkpoints](#model-checkpoints)
  - [FFmpeg](#ffmpeg)
- [Analyzer Class](#analyzer-class)
  - [Core Concept: 250 Hz Internal Sampling Rate](#core-concept-250-hz-internal-sampling-rate)
  - [Initialization](#initialization)
  - [Key Attributes](#key-attributes)
- [Input File Formats](#input-file-formats)
  - [TDMS Files](#tdms-files)
  - [Sensor Log File](#sensor-log-file)
- [Methods](#methods)
  - [Data Preparation](#data-preparation)
    - [prepare_data](#prepare_data)
    - [align_multi_channel_signal](#align_multi_channel_signal)
    - [get_field_directions](#get_field_directions)
    - [invert_field_directions](#invert_field_directions)
  - [Signal Filtering](#signal-filtering)
    - [default_filter_combination](#default_filter_combination)
    - [ICA_filter](#ica_filter)
  - [Cardiac Segmentation and QRS Detection](#cardiac-segmentation-and-qrs-detection)
    - [segment_entire_run](#segment_entire_run)
    - [find_cleanest_channel](#find_cleanest_channel)
    - [detect_qrs_complex_peaks_cleanest_channel](#detect_qrs_complex_peaks_cleanest_channel)
    - [detect_qrs_complex_peaks_all_channels](#detect_qrs_complex_peaks_all_channels)
    - [avg_window](#avg_window)
  - [Visualization](#visualization)
    - [plot_sensor_matrix](#plot_sensor_matrix)
    - [plot_lsd_multichannel](#plot_lsd_multichannel)
    - [plot_heart_vector_projection](#plot_heart_vector_projection)
    - [plot_all_heart_vector_projections](#plot_all_heart_vector_projections)
    - [plot_segmented_signal](#plot_segmented_signal)
    - [butterfly_plot](#butterfly_plot)
    - [create_heat_map_animation](#create_heat_map_animation)
- [Example Workflow](#example-workflow)
  - [Workflow Steps](#workflow-steps)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)
  
## Overview

The `Analyzer` class is a robust Python tool designed for processing, analyzing, and visualizing **Magnetocardiography (MCG)** data stored in **TDMS** (`.tdms`) files. It supports multi-channel MCG recordings with a standardized internal sampling rate of **250 Hz**, ensuring consistent analysis across various processing stages. The class provides a comprehensive suite of functionalities, including data loading, signal filtering, cardiac cycle segmentation, QRS complex detection, Independent Component Analysis (ICA), and advanced visualizations.

### Key Features
- **Data Handling**: Loads and aligns data from primary and additional TDMS files, with support for sensor log integration.
- **Signal Processing**: Applies filters (bandstop, bandpass, lowpass, highpass, Savitzky-Golay) and ICA for noise reduction.
- **Cardiac Analysis**: Segments heartbeats into P-wave, QRS complex, T-wave, and no-wave using a pre-trained PyTorch model (`ECGSegmenter`).
- **Channel Selection**: Identifies the cleanest channel based on segmentation confidence and physiological plausibility.
- **QRS Detection**: Detects QRS peaks with heart rate (HR) and heart rate variability (HRV) calculations.
- **Visualization**: Generates sensor matrix plots, butterfly plots, segmented signal plots, heart vector projections, linear spectral density (LSD) plots, and animated heatmaps.
- **Configurability**: Supports customizable sampling rates, scaling factors, and PyTorch device selection (CPU/CUDA).
- **Logging**: Uses Python’s `logging` module for detailed process tracking and debugging.

## Installation and Setup

### Dependencies
The `Analyzer` class requires several Python libraries. Install them by running:

```bash
pip install -r requirements.txt
```

Required libraries include:
- `numpy`, `scipy`, `matplotlib`, `torch`, `sklearn`, `nptdms`, `logging`, `ffmpeg-python` (for animations).

### MCG Segmentation Package
The `ECGSegmenter` model, used for cardiac segmentation, must be available in a local package named `MCG_segmentation`. The expected directory structure is:

```
MCG_segmentation/
├── model/
│   └── model.py  # Contains ECGSegmenter class
└── MCGSegmentator_s/
    └── checkpoints/
        └── best/
            ├── model.pth
            └── params.json
```

Ensure `MCG_segmentation` is in your `PYTHONPATH` or the working directory. If unavailable, segmentation-related methods (`segment_entire_run`, `find_cleanest_channel`, `detect_qrs_complex_peaks_*`, `ICA_filter` with heartbeat scoring) will log warnings and may not function.

### Model Checkpoints
The default model checkpoint path is `MCG_segmentation/MCGSegmentator_s/checkpoints/best`. Specify a custom path using the `model_checkpoint_dir` parameter during `Analyzer` initialization.

### FFmpeg
For `create_heat_map_animation`, ensure `ffmpeg` is installed and accessible in the system’s PATH.

## Analyzer Class

### Core Concept: 250 Hz Internal Sampling Rate
All processing, including segmentation and QRS detection, operates at a fixed **250 Hz** internal sampling rate (`INTERNAL_SAMPLING_RATE`). The `prepare_data` method resamples input data from its original sampling rate to 250 Hz, ensuring compatibility with the pre-trained `ECGSegmenter` model.

### Initialization
```python
from analyzer import Analyzer

analyzer = Analyzer(
    filename="path/to/primary_data.tdms",
    add_filename="path/to/additional_data.tdms",  # Optional
    log_file_path="path/to/sensor_log.txt",
    sensor_channels_to_exclude={'run_key1': ['SensorA_x', 'SensorB_y']},
    scaling=2.7 / 1000,
    num_ch=48,
    model_checkpoint_dir="path/to/MCG_segmentation/MCGSegmentator_s/checkpoints/best"
)
```

#### Parameters
| Parameter                  | Type   | Default                  | Description                                                                 |
|----------------------------|--------|--------------------------|-----------------------------------------------------------------------------|
| `filename`                 | `str`  | —                        | Path to primary TDMS file.                                                  |
| `add_filename`             | `str`  | `None`                   | Path to additional TDMS file for alignment and concatenation.               |
| `log_file_path`            | `str`  | —                        | Path to QZFM sensor log file (contains sensor mappings and orientations).   |
| `sensor_channels_to_exclude` | `dict` | `None`                  | Dict of run keys to lists of channel names to exclude (supports wildcards). |
| `scaling`                  | `float`| `2.7 / 1000`             | Scaling factor for TDMS data.                                               |
| `num_ch`                   | `int`  | `48`                     | Expected number of channels after combining data.                          |
| `model_checkpoint_dir`     | `str`  | See above                | Path to `ECGSegmenter` model checkpoints.                                   |

#### Key Attributes
- `data`/`add_data`: Dictionaries storing TDMS data (keys: run names, values: NumPy arrays).
- `key_list`: List of run keys from the primary TDMS file.
- `quspin_gen_dict`/`quspin_channel_dict`/`quspin_position_list`: Sensor metadata from the log file.
- `model`: Loaded `ECGSegmenter` instance (or `None` if unavailable).
- `DEVICE`: PyTorch device (auto-selects CUDA/CPU/mps for optimal performance).
- `INTERNAL_SAMPLING_RATE`: Fixed at 250 Hz.

## Input File Formats

### TDMS Files
- Format: National Instruments TDMS (`.tdms`).
- Structure: Organized in groups (run keys), each containing multiple channels.
- Note: If using `add_filename`, ensure group names align between primary and additional files.

### Sensor Log File
- Format: Text file with a Python dictionary containing:
  - `quspin_gen_dict`: Maps sensor IDs (e.g., `Q01`) to generation/type.
  - `quspin_channel_dict`: Maps channel names (e.g., `Q01_x`) to TDMS indices (negative for inverted polarity).
  - `quspin_position_list`: Grid layout of sensors (e.g., `[['Q01', 'Q02'], ['Q03', 'Q04']]`).
- Example:
  ```python
  {
      'quspin_gen_dict': {'Q01': 1, 'Q02': 1},
      'quspin_channel_dict': {'Q01_x': 0, 'Q01_y': 1, 'Q01_z': 2},
      'quspin_position_list': [['Q01', 'Q02'], ['Q03', 'Q04']]
  }
  ```

## Methods

### Data Preparation

#### `prepare_data`
Prepares and aligns multi-channel MCG data, applying resampling, filtering, and coordinate transformations.

**Steps**:
1. Loads raw sensor data from TDMS files.
2. Aligns primary and additional datasets using cross-correlation.
3. Crops to the specified time interval.
4. Applies optional default filters.
5. Converts to a consistent coordinate system based on sensor logs.
6. Resamples to 250 Hz.
7. Organizes into x, y, z field components in a grid layout.

**Usage**:
```python
(x_data, y_data, z_data), time, combined = analyzer.prepare_data(
    key="run_01", apply_default_filter=True, plot_alignment=True
)
```

**Parameters**:
| Parameter             | Type   | Default | Description                                       |
|-----------------------|--------|---------|---------------------------------------------------|
| `key`                 | `str`  | —       | Run key from TDMS file.                          |
| `apply_default_filter`| `bool` | `False` | Apply default filter combination.                |
| `intervall_low_sec`   | `float`| `5`     | Start time (seconds) from aligned start.         |
| `intervall_high_sec`  | `float`| `-5`    | End time (seconds) from aligned end.             |
| `plot_alignment`      | `bool` | `False` | Plot alignment visualization.                    |
| `alignment_cutoff_sec`| `float`| `2`     | Max duration for alignment (seconds).            |
| `input_sampling_rate` | `int`  | `1000`  | Original data sampling rate (Hz).                |

**Returns**:
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`: `(x_data, y_data, z_data)`, 3D arrays `(rows, cols, samples)` at 250 Hz.
- `np.ndarray`: Time vector (seconds) at 250 Hz.
- `np.ndarray`: Combined raw signal `(channels, samples)` at 250 Hz.

**Example Output**:
![Alignment Plot](https://github.com/user-attachments/assets/80065fda-0be1-4fd9-807e-5163a04950e8)

#### `align_multi_channel_signal`
Aligns two multi-channel signals using cross-correlation of their averaged signals.

**Usage**:
```python
aligned_signal, lag = analyzer.align_multi_channel_signal(signal1, signal2, lag_cutoff=2000)
```

**Parameters**:
| Parameter    | Type         | Default | Description                              |
|--------------|--------------|---------|------------------------------------------|
| `signal1`    | `np.ndarray` | —       | First signal `(channels, samples)`.      |
| `signal2`    | `np.ndarray` | —       | Second signal to align.                  |
| `lag_cutoff` | `int`        | `2000`  | Max samples for lag estimation.          |
| `plot`       | `bool`       | `True`  | Plot alignment before/after.             |

**Returns**:
- `np.ndarray`: Aligned `signal2`.
- `int`: Estimated lag (samples).

**Raises**:
- `ValueError`: If `lag_cutoff` exceeds signal length.

#### `get_field_directions`
Converts flat multi-channel data into 3D spatial field representations (x, y, z) using sensor metadata.

**Usage**:
```python
x_data, y_data, z_data = analyzer.get_field_directions(data, key="run_01")
```

**Parameters**:
| Parameter | Type         | Default | Description                              |
|-----------|--------------|---------|------------------------------------------|
| `data`    | `np.ndarray` | —       | Input `(channels, samples)`.             |
| `key`     | `str`        | —       | Run key for sensor exclusion.            |

**Returns**:
- `np.ndarray`: `(x_data, y_data, z_data)`, each `(rows, cols, samples)`.

#### `invert_field_directions`
Reconstructs flat channel-wise signals from 3D x, y, z field representations.

**Usage**:
```python
combined = analyzer.invert_field_directions(x_data, y_data, z_data, key="run_01", num_channels=48)
```

**Parameters**:
| Parameter      | Type         | Default | Description                              |
|----------------|--------------|---------|------------------------------------------|
| `x_data`, `y_data`, `z_data` | `np.ndarray` | — | 3D field arrays `(rows, cols, samples)`. |
| `key`          | `str`        | —       | Run key for sensor exclusion.            |
| `num_channels` | `int`        | `None`  | Total output channels (inferred if `None`). |

**Returns**:
- `np.ndarray`: Reconstructed signal `(channels, samples)`.

### Signal Filtering

#### `default_filter_combination`
Applies a sequence of filters to remove noise (bandstop, lowpass, highpass, Savitzky-Golay).

**Usage**:
```python
filtered = analyzer.default_filter_combination(data)
```

**Parameters**:
| Parameter          | Type         | Default | Description                                 |
|--------------------|--------------|---------|---------------------------------------------|
| `data`             | `np.ndarray` | —       | Input `(channels, samples)`.               |
| `bandstop_freq`    | `float`      | `50`    | Bandstop center frequency (Hz).            |
| `lowpass_freq`     | `float`      | `95`    | Lowpass cutoff frequency (Hz).             |
| `highpass_freq`    | `float`      | `1`     | Highpass cutoff frequency (Hz).            |
| `savgol_window`    | `int`        | `61`    | Savitzky-Golay window size.                |
| `savgol_polyorder` | `int`        | `2`     | Savitzky-Golay polynomial order.           |
| `sampling_rate`    | `int`        | `1000`  | Input sampling rate (Hz).                  |

**Returns**:
- `np.ndarray`: Filtered data, same shape as input.

#### `ICA_filter`
Applies FastICA to decompose signals, filtering components based on heartbeat plausibility.

**Steps**:
1. Decomposes input into independent components.
2. Segments components using `ECGSegmenter` (if available).
3. Scores components for heartbeat likelihood.
4. Reconstructs signal, excluding low-scoring components.
5. Optionally plots interactive threshold adjustment for 3D data.

**Usage**:
```python
filtered, components, best_idx, mask = analyzer.ICA_filter(data, heart_beat_score_threshold=0.8, plot_result=True)
```

**Parameters**:
| Parameter                   | Type         | Default | Description                                 |
|-----------------------------|--------------|---------|---------------------------------------------|
| `data`                      | `np.ndarray` | —       | Input `(channels, samples)` or `(grid_x, grid_y, samples)`. |
| `heart_beat_score_threshold`| `float`      | `0.85`  | Threshold for retaining components.         |
| `max_iter`                  | `int`        | `5000`  | Max ICA iterations.                         |
| `confidence_weight`         | `float`      | `0.8`   | Weight for confidence in scoring.           |
| `plausibility_weight`       | `float`      | `0.2`   | Weight for plausibility in scoring.         |
| `print_result`              | `bool`       | `False` | Print channel evaluation metrics.           |
| `plot_result`               | `bool`       | `False` | Show interactive plot for 3D data.          |

**Returns**:
- `np.ndarray`: Reconstructed signal.
- `np.ndarray`: ICA components.
- `int`: Index of the most heartbeat-related component.
- `np.ndarray`: Boolean mask of retained components.

**Raises**:
- `ValueError`: If input is not 2D or 3D.

**Example Output**:
![ICA Filter Plot](https://github.com/user-attachments/assets/85a9f730-a2a9-4f71-9c9f-c671ef00f704)

```
Channel Selection Results:
Channel   Conf        P-Wave %    QRS %       T-Wave %    Plausibility   Final Score
-------------------------------------------------------------------------------------
1         0.8761      10.60       11.08       21.08       1.0000         0.9008
...
8         0.8811      13.60       10.96       24.92       1.0000         0.9049
...
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

### Cardiac Segmentation and QRS Detection

#### `segment_entire_run`
Segments long MCG signals into P-wave, QRS complex, T-wave, and no-wave using a sliding window approach.

**Steps**:
1. Validates input and overlap.
2. Applies Savitzky-Golay smoothing.
3. Segments data into overlapping windows.
4. Normalizes each window (mean subtraction, max scaling).
5. Predicts labels and confidences using `ECGSegmenter`.
6. Aggregates results, prioritizing high-confidence predictions.

**Usage**:
```python
labels, confidences = analyzer.segment_entire_run(data, window_size=2000, overlap=0.5)
```

**Parameters**:
| Parameter     | Type         | Default | Description                                 |
|---------------|--------------|---------|---------------------------------------------|
| `data`        | `np.ndarray` | —       | Input `(batch, samples)` at 250 Hz.         |
| `window_size` | `int`        | `2000`  | Sliding window size.                        |
| `overlap`     | `float`      | `0.5`   | Overlap fraction (0 to <1).                 |

**Returns**:
- `np.ndarray`: Labels `(batch, samples)`.
- `np.ndarray`: Confidence scores `(batch, samples)`.

**Raises**:
- `ValueError`: If input is not 2D, overlap is invalid, or window size is non-positive.
- `Warning`: If window size exceeds model max (2000), clamps to 2000.

**Example Output**:
![Segmented Signal](https://github.com/user-attachments/assets/9e5616e0-f852-4573-b0b9-dbcb8bd8fd66)

#### `find_cleanest_channel`
Identifies the channel with the clearest signal based on segmentation confidence and physiological plausibility.

**Steps**:
1. Segments all channels using `segment_entire_run`.
2. Scores channels by combining confidence and plausibility.
3. Selects the highest-scoring channel.
4. Optionally prints scores and segment distributions.

**Usage**:
```python
best_channel, labels, confidence, scores = analyzer.find_cleanest_channel(data, print_results=True)
```

**Parameters**:
| Parameter            | Type         | Default | Description                                 |
|----------------------|--------------|---------|---------------------------------------------|
| `data`               | `np.ndarray` | —       | Input `(channels, samples)` at 250 Hz.      |
| `window_size`        | `int`        | `2000`  | Window size for segmentation.               |
| `overlap`            | `float`      | `0.5`   | Overlap fraction.                           |
| `print_results`      | `bool`       | `True`  | Print scores and distributions.             |
| `confidence_weight`  | `float`      | `0.8`   | Confidence weight in scoring.               |
| `plausibility_weight`| `float`      | `0.2`   | Plausibility weight in scoring.             |

**Returns**:
- `int`: Index of the best channel (0-based).
- `np.ndarray`: Labels `(channels, samples)`.
- `np.ndarray`: Confidences `(channels, samples)`.
- `np.ndarray`: Scores for all channels.

**Raises**:
- `ValueError`: If input is not 2D.
- `Warning`: If input is empty or segmentation fails.

#### `detect_qrs_complex_peaks_cleanest_channel`
Detects QRS peaks in the cleanest channel.

**Steps**:
1. Identifies cleanest channel using `find_cleanest_channel`.
2. Detects QRS segments based on labels and confidence.
3. Finds peaks within valid QRS segments.
4. Filters peaks to ensure minimum distance.
5. Optionally computes and prints HR and HRV.

**Usage**:
```python
peaks, best_channel, labels, hr, hrv = analyzer.detect_qrs_complex_peaks_cleanest_channel(
    data, print_heart_rate=True
)
```

**Parameters**:
| Parameter            | Type         | Default | Description                                 |
|----------------------|--------------|---------|---------------------------------------------|
| `data`               | `np.ndarray` | —       | Input `(channels, samples)` at 250 Hz.      |
| `confidence_threshold`| `float`     | `0.7`   | Min confidence for QRS segments.            |
| `min_qrs_length_sec` | `float`      | `0.08`  | Min QRS duration (seconds).                 |
| `min_distance_sec`   | `float`      | `0.3`   | Min distance between peaks (seconds).       |
| `print_heart_rate`   | `bool`       | `False` | Print HR and HRV.                           |

**Returns**:
- `List[int]`: Peak indices for the cleanest channel.
- `int`: Cleanest channel index.
- `np.ndarray`: Labels `(channels, samples)`.
- `Optional[float]`: HR (bpm) or `None`.
- `Optional[float]`: HRV (SDNN, ms) or `None`.

**Raises**:
- `Warning`: If segmentation data is empty.

**Example Output**:
![QRS Detection](https://github.com/user-attachments/assets/724c40f4-38d5-4458-822f-3a02c9da62e4)

#### `detect_qrs_complex_peaks_all_channels`
Detects QRS peaks independently for all channels.

**Steps**:
1. Identifies cleanest channel.
2. Detects QRS segments per channel.
3. Finds and filters peaks per channel.
4. Optionally computes and prints average HR and HRV across channels.

**Usage**:
```python
peaks_dict, cleanest_channel, labels, avg_hr, avg_hrv = analyzer.detect_qrs_complex_peaks_all_channels(
    data, print_heart_rate=True
)
```

**Parameters**:
| Parameter            | Type         | Default | Description                                 |
|--------------------|--------------|---------|---------------------------------------------|
| `data`             | `np.ndarray` | —       | Input `(channels, samples)` at 250 Hz.      |
| `confidence_threshold`| `float`     | `0.7`   | Min confidence for QRS segments.            |
| `min_qrs_length_sec` | `float`     | `0.08`  | Min QRS duration (seconds).                 |
| `min_distance_sec`   | `float`      | `0.3`   | Min distance between peaks (seconds).       |
| `print_heart_rate`   | `bool`       | `False` | Print average HR and HRV.                   |

**Returns**:
- `Dict[int, List[int]]`: Channel indices to peak indices.
- `int`: Cleanest channel index.
- `np.ndarray`: Labels `(channels, samples)`.
- `Optional[float]`: Average HR (bpm) or `None`.
- `Optional[float]`: Average HRV (SDNN, ms) or `None`.

**Raises**:
- `Warning`: If segmentation data is empty.

#### `avg_window`
Computes average waveforms around detected peaks (e.g., QRS complexes).

**Steps**:
1. Extracts windows around peaks.
2. Removes drift and offset from windows.
3. Segments windows to evaluate quality.
4. Filters windows by heartbeat score.
5. Averages valid windows per channel.
6. Applies Gaussian smoothing.

**Usage**:
```python
avg_waveforms, time_window = analyzer.avg_window(data, peak_positions, window_left=0.3, window_right=0.5)
```

**Parameters**:
| Parameter                   | Type             | Default | Description                                 |
|-----------------------------|------------------|---------|---------------------------------------------|
| `data`                      | `np.ndarray`     | —       | Input `(channels, samples)` at 250 Hz.      |
| `peak_positions`            | `list` or `dict` | —       | Peak indices or per-channel indices.        |
| `window_left`               | `float`          | `0.3`   | Seconds left of peak.                       |
| `window_right`              | `float`          | `0.5`   | Seconds right of peak.                      |
| `heart_beat_score_threshold`| `float`          | `0.0`   | Min score for window inclusion.             |
| `sigma`                     | `float`          | `1`     | Std deviation for gaussian filter.          |

**Returns**:
- `np.ndarray`: Average waveforms `(channels, window_length)`.
- `np.ndarray`: Time array `(window_length,)`.

**Raises**:
- `ValueError`: If peak positions are empty or invalid.

### Visualization

#### `plot_sensor_matrix`
Plots a grid of time-series data from a sensor array.

**Steps**:
1. Creates a subplot grid matching sensor array layout.
2. Plots each sensor’s signal or marks “No Data” for invalid signals.
3. Adds grid lines, global labels, and title.
4. Optionally saves as PNG.

**Usage**:
```python
Analyzer.plot_sensor_matrix(data, time, name="Sensor Grid", path="./plots", save=True)
```

**Parameters**:
| Parameter | Type         | Default | Description                                 |
|-----------|--------------|---------|---------------------------------------------|
| `data`    | `np.ndarray` | —       | Input `(rows, cols, samples)`.              |
| `time`    | `np.ndarray` | —       | Time vector.                                |
| `name`    | `str`        | —       | Title and filename base.                    |
| `path`    | `str`        | `None`  | Save directory.                             |
| `save`    | `bool`       | `False` | Save as PNG.                                |

**Returns**: None.

**Example Output**:
![Sensor Matrix](https://github.com/user-attachments/assets/2015e780-456a-49ec-b4c0-3eab05510e21)

#### `plot_lsd_multichannel`
Plots Linear Spectral Density (LSD) of multi-channel data using Welch’s method.

**Steps**:
1. Computes LSD with a Nuttall window.
2. Plots on a log-log scale with optional noise floors.
3. Adds secondary y-axis for linear amplitude.
4. Includes grid, legend, and labels.
5. Optionally saves as PNG.

**Usage**:
```python
analyzer.plot_lsd_multichannel(data, noise_theos, freqs, name="LSD Plot", labels=["Ch1", "Ch2"], channels=[0, 1], path="./plots", save=True)
```

![Leermessung_Noise_Spectrum_Triax_Sensor_LSD](https://github.com/user-attachments/assets/d8217a25-4953-4f22-9028-807334be9a24)


**Parameters**:
| Parameter    | Type             | Default | Description                                 |
|--------------|------------------|---------|---------------------------------------------|
| `data`       | `np.ndarray`     | —       | Input `(channels, samples)` at 250 Hz.      |
| `noise_theos`| `list` or `np.ndarray` | — | Theoretical noise floors.                    |
| `name`       | `str`            | —       | Title and filename base.                    |
| `labels`     | `list of str`    | —       | Channel labels for legend.                   |
| `channels`   | `list of int`    | —       | Channel indices to plot.                    |
| `path`       | `str`            | —       | Save directory.                             |
| `save`       | `bool`           | `False` | Save as PNG.                                |

**Returns**: None.

**Raises**:
- `ValueError`: If input arrays are inconsistent.

#### `plot_heart_vector_projection`
Plots a 2D projection of the heart vector with metrics and filled area.

**Steps**:
1. Sets up an equal-aspect plot with styled grid.
2. Plots projection line and fills enclosed area.
3. Adds directional arrows.
4. Computes metrics (area, T-distance, compactness, angle).
5. Displays metrics and labels axes.
6. Supports standalone or subplot use.

**Usage**:
```python
analyzer.plot_heart_vector_projection(component1, component2, proj_name="xy-Projection", title_suffix="Subject1")
```

**Parameters**:
| Parameter     | Type                   | Default | Description                                 |
|---------------|------------------------|---------|---------------------------------------------|
| `component1`  | `np.ndarray`           | —       | First component (e.g., Bx).                 |
| `component2`  | `np.ndarray`           | —       | Second component (e.g., By).                |
| `proj_name`   | `str`                  | —       | Projection type (`xy-Projection`, etc.).    |
| `title_suffix`| `str`                  | `""`    | Title suffix.                               |
| `ax`          | `matplotlib.axes.Axes` | `None`  | Axis for plotting (or new figure).          |

**Returns**:
- `matplotlib.axes.Axes`: Plot axis.

**Raises**:
- `Exception`: Logs warning if metrics fail.

**Example Output**:
![Heart Vector Projection](https://github.com/user-attachments/assets/755764c0-819b-4ff1-83cb-a3491631e6f9)

#### `plot_all_heart_vector_projections`
Plots XY, XZ, and YZ heart vector projections in a single figure.

**Steps**:
1. Validates input shape `(3, samples)`.
2. Creates a 1x3 subplot grid.
3. Calls `plot_heart_vector_projection` for each projection.
4. Adds unified title and adjusts layout.
5. Optionally saves as PNG.

**Usage**:
```python
analyzer.plot_all_heart_vector_projections(heart_vector_components, title_suffix="Subject1", save_path="./plots/projections.png")
```

**Parameters**:
| Parameter                | Type         | Default | Description                                 |
|--------------------------|--------------|---------|---------------------------------------------|
| `heart_vector_components`| `np.ndarray` | —       | Input `(3, samples)` with Bx, By, Bz.       |
| `title_suffix`           | `str`        | `""`    | Title suffix.                               |
| `save_path`              | `str`        | `None`  | Save path for PNG.                          |

**Returns**: None.

**Raises**:
- `Exception`: Logs error if input shape is invalid.

#### `plot_segmented_signal`
Plots a signal with overlaid heartbeat segmentations.

**Steps**:
1. Plots the signal as a time series.
2. Overlays colored spans for segments (No Wave, P-Wave, QRS, T-Wave).
3. Builds a legend for signal and segments.
4. Adds grid and labels.
5. Supports standalone or existing axis.

**Usage**:
```python
analyzer.plot_segmented_signal(signal, pred)
```

**Parameters**:
| Parameter | Type                   | Default | Description                                 |
|-----------|------------------------|---------|---------------------------------------------|
| `signal`  | `np.ndarray`           | —       | Input signal `(samples,)` at 250 Hz.        |
| `pred`    | `np.ndarray`           | —       | Segment labels `(samples,)`.                |
| `axs`     | `matplotlib.axes.Axes` | `None`  | Axis for plotting (or new figure).          |

**Returns**: None.

**Example Output**:
![Segmented Signal](https://github.com/user-attachments/assets/b84801ec-2e0e-4c94-9d95-8b0da44f2fd3)

#### `butterfly_plot`
Plots multi-channel time-series data overlaid on a single axis.

**Steps**:
1. Sets up a single-axis figure.
2. Plots each channel with distinct colors and transparency.
3. Adds major/minor grids, time, and magnetic field labels.
4. Includes legend (for ≤10 channels) and annotates first/last channels.
5. Optionally saves as PNG.

**Usage**:
```python
analyzer.butterfly_plot(data, time, num_ch=48, name="Signal Plot", path="./plots", save=True)
```

**Parameters**:
| Parameter | Type         | Default | Description                                 |
|-----------|--------------|---------|---------------------------------------------|
| `data`    | `np.ndarray` | —       | Input `(channels, samples)` at 250 Hz.      |
| `time`    | `np.ndarray` | —       | Time points.                                |
| `num_ch`  | `int`        | —       | Number of channels to plot.                 |
| `name`    | `str`        | —       | Title and filename base.                    |
| `path`    | `str`        | `None`  | Save directory.                             |
| `save`    | `bool`       | `False` | Save as PNG.                                |

**Returns**: None.

**Example Output**:
![Butterfly Plot](https://github.com/user-attachments/assets/50478143-9f66-44d2-96f3-73c2338fedfb)

#### `create_heat_map_animation`
Creates an animated heatmap of sensor data with a time-series trace of the cleanest channel.

**Steps**:
1. Sets up a figure with heatmap and time-series subplots.
2. Initializes a high-resolution interpolated heatmap.
3. Plots sensor locations and highlights the cleanest channel.
4. Displays the cleanest channel’s time series with a moving marker.
5. Animates by updating heatmap and trace per frame.
6. Saves as GIF/video using FFmpeg.

**Usage**:
```python
ani, fig = analyzer.create_heat_map_animation(
    data, cleanest_i=1, cleanest_j=2, output_file="heatmap_animation.mp4", direction="x"
)
```

**Parameters**:
| Parameter       | Type         | Default          | Description                                 |
|-----------------|--------------|------------------|---------------------------------------------|
| `data`          | `np.ndarray` | —                | Input `(rows, cols, samples)` at 250 Hz.    |
| `cleanest_i`    | `int`        | —                | Row index of cleanest channel.              |
| `cleanest_j`    | `int`        | —                | Column index of cleanest channel.           |
| `output_file`   | `str`        | `"animation.gif"`| Output file name (GIF/video).               |
| `interval`      | `int`        | `100`            | Frame interval (ms).                        |
| `resolution`    | `int`        | `500`            | Heatmap resolution.                         |
| `stride`        | `int`        | `1`              | Frame stride.                               |
| `direction`     | `str`        | `"x"`            | Field direction (`x`, `y`, `z`).            |
| `key`           | `str`        | `"Brustlage"`    | Run key for labeling.                       |
| `dynamic_scale` | `bool`       | `True`           | Dynamically adjust heatmap color scale.     |

**Returns**:
- `matplotlib.animation.FuncAnimation`: Animation object.
- `matplotlib.figure.Figure`: Figure object.

## Example Workflow
This workflow demonstrates loading, processing, analyzing, and visualizing MCG data using the `Analyzer` class.

```python
from analyzer import Analyzer
import matplotlib.pyplot as plt
import numpy as np

# Initialize Analyzer
analyzer = Analyzer(
    filename="data/primary.tdms",
    add_filename="data/additional.tdms",
    log_file_path="data/sensor_log.txt",
    sensor_channels_to_exclude={'run_01': ['Q01_x', 'Q02_y']}
)

# Prepare data for a specific run
key = "run_01"
(x_data, y_data, z_data), time, single_run = analyzer.prepare_data(
    key, apply_default_filter=True, plot_alignment=True
)

# Select a time interval
intervall_start, intervall_end = 250, 1250
x_data_intervall = x_data[:, :, intervall_start:intervall_end]
y_data_intervall = y_data[:, :, intervall_start:intervall_end]
z_data_intervall = z_data[:, :, intervall_start:intervall_end]
time_intervall = time[intervall_start:intervall_end]
single_run_intervall = single_run[:, intervall_start:intervall_end]

# Apply ICA filtering
x_data_filtered, _, _, _ = analyzer.ICA_filter(x_data_intervall, heart_beat_score_threshold=0.74)
y_data_filtered, _, _, _ = analyzer.ICA_filter(y_data_intervall, heart_beat_score_threshold=0.73)
z_data_filtered, _, _, _ = analyzer.ICA_filter(z_data_intervall, heart_beat_score_threshold=0.77)

# Reconstruct filtered signal
single_run_filtered = analyzer.invert_field_directions(
    x_data_filtered, y_data_filtered, z_data_filtered, key, num_channels=48
)

# Visualize filtered data
analyzer.butterfly_plot(
    single_run_filtered, time_intervall, num_ch=48, name=f"Filtered {key}", path="./plots", save=True
)

# Detect QRS peaks
peak_positions, ch, labels, hr, hrv = analyzer.detect_qrs_complex_peaks_cleanest_channel(
    single_run_filtered, print_heart_rate=True, confidence_threshold=0.7
)

# Plot QRS peaks
if peak_positions and len(peak_positions) > 0:
    plt.figure(figsize=(12, 4))
    plt.plot(single_run_filtered[ch], label="Signal", linewidth=1.2)
    plt.plot(peak_positions, single_run_filtered[ch, peak_positions], "ro", markersize=6, label="R Peaks")
    plt.title(f"QRS Detection - Cleanest Channel {ch + 1}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (pT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/qrs_peaks.png")
    plt.show()
else:
    print("No R peaks detected.")

# Visualize segmented signal
analyzer.plot_segmented_signal(single_run_filtered[ch], labels[ch], path="./plots", save=True)

# Compute and visualize averaged waveforms
avg_channels, time_window = analyzer.avg_window(
    single_run_filtered, peak_positions, window_left=0.3, window_right=0.5
)
analyzer.butterfly_plot(
    avg_channels, time_window, num_ch=48, name=f"Averaged Waveforms {key}", path="./plots", save=True
)

# Create heatmap animation (example for x-data)
if x_data_filtered.shape[2] > 100:
    analyzer.create_heat_map_animation(
        x_data_filtered, cleanest_i=1, cleanest_j=2, output_file=f"./plots/heatmap_x_{key}.mp4",
        direction="x", key=key
    )
```

### Workflow Steps
1. **Initialization**: Configures `Analyzer` with TDMS files, sensor log, and optional exclusions.
2. **Data Preparation**: Loads, aligns, filters, and resamples data to 250 Hz.
3. **Interval Selection**: Extracts a focused time interval.
4. **ICA Filtering**: Removes noise from x, y, z components using ICA.
5. **Signal Reconstruction**: Combines filtered components into a channel-wise signal.
6. **Visualization**: Plots filtered signals as a butterfly plot.
7. **QRS Detection**: Identifies QRS peaks in the cleanest channel and computes HR/HRV.
8. **Peak Visualization**: Plots the cleanest channel with marked R peaks.
9. **Segmentation Visualization**: Shows segmented cardiac cycles.
10. **Waveform Averaging**: Computes and visualizes averaged QRS waveforms.
11. **Heatmap Animation**: Generates an animated spatial heatmap of field strength.

## Troubleshooting

- **Segmentation Model**: If `ECGSegmenter` or its checkpoints are missing, segmentation features will fail with logged warnings. Verify `MCG_segmentation` package and model paths.
- **Sampling Rates**: Ensure `input_sampling_rate` matches the data’s rate in `prepare_data`. All segmentation and QRS detection require 250 Hz input.
- **Memory Usage**: Long recordings or animations may require significant memory. Use interval selection or chunked processing.
- **FFmpeg**: Required for `create_heat_map_animation`. Install via `conda install ffmpeg` or system package manager.
- **Coordinate System**: The `_change_to_consistent_coordinate_system` method applies sign corrections based on sensor names and types. Customize if your sensor setup differs.
- **Channel Mapping**: The `__init__` method adjusts channel indices (`abs(val) >= 100`). Modify this logic if your TDMS channel mapping differs.

## Notes
- **Performance**: CUDA is automatically selected if available, significantly speeding up segmentation and ICA.
- **Logging**: Set `logging` level to `DEBUG` for detailed diagnostics or `ERROR` for minimal output.
- **Extensibility**: Static utility methods (e.g., `bandstop_filter`) can be used independently for custom processing.
- **Data Integrity**: Validate TDMS and log files before processing to avoid runtime errors.

This documentation provides a unified guide to using the `Analyzer` class for MCG data analysis, with consistent formatting and clear examples to facilitate robust cardiac signal processing and visualization.
