# MCG Data Analyzer Documentation
- [Overview](#overview)
  - [Key Features](#key-features)
  - [Scientific Context & Methodology](#scientific-context--methodology)
    - [Background](#background)
    - [Methods](#methods)
    - [Measurements](#measurements)
    - [Statistical Analysis](#statistical-analysis)
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
    - [visualize_heart_vector](#visualize_heart_vector)
    - [visualize_heart_vectors](#visualize_heart_vectors)
    - [plot_segmented_signal](#plot_segmented_signal)
    - [plot_segments_with_editing](#plot_segments_with_editing)
    - [butterfly_plot](#butterfly_plot)
    - [create_heat_map_animation](#create_heat_map_animation)
- [The Analysis Pipeline in Practice](#the-analysis-pipeline-in-practice)
  - [Configuration: setup.json](#configuration-setupjson)
  - [Step 1: Feature Extraction (heart_beat.py)](#step-1-feature-extraction-heart_beatpy)
  - [Step 2: Statistical Analysis (statistical_analysis.py)](#step-2-statistical-analysis-statistical_analysispy)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Overview

This repository contains a comprehensive Python pipeline for processing, analyzing, and visualizing Magnetocardiography (MCG) data. The primary goal of this project is to leverage high-precision MCG recordings from optically pumped magnetometers (OPMs) to investigate potential biomarkers for Arrhythmogenic Cardiomyopathy (ACM). The pipeline is designed to distinguish between ACM patients and healthy controls by extracting and statistically evaluating geometric features of the heart's magnetic field vector.

The project is structured around three main components:

1.  **`analyzer.py`**: A powerful and versatile Python class that serves as the core engine for all data handling. It provides functionalities for loading TDMS files, advanced signal processing (filtering, ICA), deep learning-based cardiac segmentation, and calculation of heart vector metrics with uncertainty propagation.
2.  **`heart_beat.py`**: An automated script that executes the end-to-end analysis pipeline for a given patient. It handles data loading based on a central setup file, applies preprocessing and filtering, performs window-averaging, facilitates manual segmentation refinement, calculates key vector metrics (Area, Distance, Compactness, Angle), and saves the results.
3.  **`statistical_analysis.py`**: A script designed to consume the outputs of `heart_beat.py`. It performs demographic analysis, conducts statistical comparisons (Welch's t-test, ROC analysis) between the ACM and healthy cohorts, and uses Monte Carlo simulations to incorporate measurement uncertainties into the final statistical results. It supports both individual sensor analysis and aggregated analysis over sensor grid regions.

This pipeline was developed for the study **Magnetocardiography to Screen Adults with Arrhythmogenic Cardiomyopathy: A pilot study**, providing a tool for researchers to explore non-invasive cardiac screening methods.

### Key Features
- **End-to-End Workflow**: From raw TDMS data to final statistical significance plots.
- **Advanced Signal Processing**: Includes robust filtering (bandpass, bandstop, Savitzky-Golay) and Independent Component Analysis (ICA) for artifact removal, guided by a physiological plausibility score and an interactive thresholding UI.
- **AI-Powered Segmentation**: Utilizes a pre-trained neural network (e.g., `UNet1D`, `MCGSegmenter`) to segment cardiac cycles, with an option for manual correction via an interactive UI.
- **Heart Vector Analysis**: Computes 2D projections of the magnetic heart vector and quantifies their geometry using metrics like Enclosed Area, Distance, Compactness, and Angle.
- **Uncertainty Quantification**: Implements Monte Carlo simulations using the `uncertainties` library to model timing uncertainties in segment boundaries and propagate them to the final metrics and statistical tests.
- **Comprehensive Statistical Evaluation**: Provides tools to assess the diagnostic performance of extracted features, including t-tests, ROC/AUC analysis, and confusion matrices. Analysis can be run on individual sensors or aggregated over regions of the sensor grid.
- **Highly Configurable & Extensible**: The entire pipeline is controlled via a central `setup.json` file and configuration dictionaries within the main scripts, making it easy to manage patients, runs, analysis parameters, and model selection.

## Scientific Context & Methodology
### Background
Arrhythmogenic cardiomyopathy is associated with a mechanical or electrical dysfunction of the heart. This mainly affects the right, but can also affect the left ventricle, in which heart muscle cells are replaced by fibrotic fatty tissue. This remodeling of the ventricle leads to the arrhythmogenicity. We investigated magnetocardiography to measure the magnetic field of the heart precisely and to evaluate an innovative potential screening tool.

### Methods
Magnetocardiography was used to monitor potential changes in the magnetic field of the heart. The recordings were made using an innovative optically pumped system and a person-sized magnetic shield established at German Heart Center and TUM University, Munich, Germany. MCG is a non-invasive method that detects the cardiac magnetic field generated by electrical currents in the heart. MCG data were preprocessed using a digital bandpass filter (1–95 Hz) to preserve relevant cardiac frequencies, and a bandstop filter to attenuate 50 Hz powerline interference and its harmonics. Baseline drift and DC offset were removed via third-order polynomial detrending.

A lightweight neural network segmented the MCG signal into four waveform classes: no wave, P wave, QRS complex, and T wave. Based on the predicted segment durations, a heartbeat plausibility score was computed by comparing the relative segment lengths to physiologically expected ranges. This score guided artifact removal through Independent Component Analysis (ICA). Specifically, ICA was applied separately to each magnetic field component, using the respective sensor channels as input. Output components showing high similarity to valid heartbeat morphology—based on the plausibility score—were retained, whereas components with low cardiac relevance were discarded using an interactive UI for threshold selection.

### Measurements
For QRS detection, the single cleanest channel—selected from all sensors and their two to three orthogonal components—was used. QRS complexes were identified within this channel over intervals spanning at least four heartbeats. Subsequently, average waveform segments centered around the detected QRS peaks were computed using a windowed averaging approach to enhance signal-to-noise ratio and suppress statistical fluctuations. All analyses were based on these average waveforms. Boundaries of the P wave, QRS complex, ST segment, and T wave were initially estimated by the neural network and subsequently refined manually using a dedicated graphical interface. A timing uncertainty of ±40 milliseconds was assumed for each segment boundary and modeled as Gaussian-distributed.

Three key cardiac segments were analyzed:
*   **QRS complex**: from the beginning of the QRS complex to its end.
*   **ST segment**: from the end of the QRS complex to the start of the T wave.
*   **T wave**: from the start to the end of the T wave.

For each segment and each of the 16 sensors, we computed the magnetic heart vector, defined as the two-dimensional projection of the time-varying magnetic field vector during the respective segment. To characterize the geometry and dynamics of this vector trajectory, we computed the following metrics:

*   **Enclosed Area**: Represents the spatial extent of the vector loop, calculated using the shoelace formula: `Area = ½ × |Σ (xᵢ · yᵢ₊₁ – yᵢ · xᵢ₊₁)|`.
*   **Distance**: Quantifies the displacement of the magnetic heart vector from its starting point to the point of maximum magnitude during the segment: `Distance = ||P(t_max_magnitude) - P(t_start)||`.
*   **Compactness**: A dimensionless metric that quantifies the circularity of the vector loop: `Compactness = (4 × π × Area) / Perimeter²`.
*   **Angle**: The angle of the average heart vector for the segment, calculated as `atan2(mean(By), mean(Bx))`.

To account for timing uncertainty in the segment definitions, a Monte Carlo simulation was conducted. In each iteration, segment boundaries were randomly perturbed within the assumed Gaussian distribution, and all metrics were recomputed to estimate uncertainty bounds.

### Statistical Analysis
Statistical comparisons were performed between ACM patients and healthy controls. Outliers were optionally removed using the IQR method, configurable for each feature. Group differences were assessed using Welch’s two-sample t-test (p < 0.05). Diagnostic performance was evaluated using receiver operating characteristic (ROC) curve analysis, with optimal thresholds determined by maximizing Youden’s index. Measurement uncertainty was incorporated via a Monte Carlo simulation with N=500 iterations, providing confidence intervals for p-values, AUC, and other statistical outputs.


## Installation and Setup

The Project has the following structure:

```
.
├── Data/
│   ├── setup.json             # Central configuration for patients, runs, and parameters.
│   ├── P001/                  # Example patient data directory
│   │   ├── P001_S01.tdms      # Primary TDMS data file.
│   │   └── QZFM_log_file.txt  # Sensor layout and mapping file.
│   └── ...
├── Results/
│   ├── Q01_xy/                # Example results directory for a sensor/projection
│   │   ├── result.csv         # CSV with extracted metrics for all patients.
│   │   ├── Generated_Plots_MC/ # Plots from statistical analysis.
│   │   └── ...
│   └── ...
├── MCG_segmentation/          # Package for the cardiac segmentation model.
│   ├── trained_models/        # Directory containing trained model checkpoints
│   │   └── UNet_1D_15M/       # Example model directory
│   │       ├── checkpoints/
│   │       └── config.json
│   └── ...
├── analyzer.py                # The core Analyzer class.
├── heart_beat.py              # Main script to run the analysis on a patient.
├── statistical_analysis.py    # Script to perform statistical analysis on results.
├── requirements.txt           # Python dependencies.
└── README.md                  # This documentation file.
```

### Dependencies
The `Analyzer` class requires several Python libraries. Install them by running:

```bash
pip install -r requirements.txt
```
(Note that all of this code was developed and tested using python 3.11.2 64-bit. Newer versions should work as well, although you might have to use different version of the libraries listed in the requirements.txt file)

Required libraries include:
- `numpy`, `scipy`, `matplotlib`, `torch`, `sklearn`, `nptdms`, `logging`, `ffmpeg-python` (for animations), `uncertainties`.

### MCG Segmentation Package
The cardiac segmentation model (e.g., `UNet1D`, `MCGSegmenter`) must be available in a local package named `MCG_segmentation`. Ensure it is in your `PYTHONPATH` or the working directory. If unavailable, segmentation-related methods will fail.

### Model Checkpoints
The model checkpoint path is configurable in `heart_beat.py`. A typical path is `MCG_segmentation/trained_models/UNet_1D_15M`. The `Analyzer` class expects this directory to contain a `config.json` file and a `checkpoints/best/model.pth` file.

### FFmpeg
For `create_heat_map_animation`, ensure `ffmpeg` is installed and accessible in the system’s PATH.

## Analyzer Class

### Core Concept: 250 Hz Internal Sampling Rate
All internal processing, including segmentation and QRS detection, operates at a fixed **250 Hz** sampling rate (`INTERNAL_SAMPLING_RATE`). The `prepare_data` method is responsible for resampling input data from its original sampling rate to 250 Hz.

### Initialization
```python
from analyzer import Analyzer, UNet1D

analyzer = Analyzer(
    filename="path/to/primary_data.tdms",
    add_filename="path/to/additional_data.tdms",
    log_file_path="path/to/sensor_log.txt",
    sensor_channels_to_exclude={'run_key1': ['SensorA_x']},
    model_checkpoint_dir="MCG_segmentation/trained_models/UNet_1D_15M",
    model_class=UNet1D
)
```

#### Parameters
| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | — | Path to primary TDMS file. |
| `add_filename` | `str` | `""` | Path to additional TDMS file for alignment and concatenation. |
| `log_file_path` | `str` | — | Path to QZFM sensor log file (contains mappings and orientations). |
| `sensor_channels_to_exclude` | `dict` | `None` | Dict of run keys to lists of channel names to exclude. |
| `scaling` | `float` | `2.7 / 1000` | Scaling factor for TDMS data. |
| `num_ch` | `int` | `48` | Expected number of channels. |
| `model_checkpoint_dir` | `str` | `MCG.../MCGsegmenter_s` | Path to the directory containing model checkpoints and `config.json`. |
|`model_class`|`torch.nn.Module`| `MCGSegmenter`| The Python class of the segmentation model to load (e.g., `UNet1D`, `MCGSegmenter`). |

#### Key Attributes
- `data`/`add_data`: Dictionaries storing TDMS data.
- `key_list`: List of run keys from the primary TDMS file.
- `quspin_..._dict`/`quspin_position_list`: Sensor metadata from the log file.
- `model`: Loaded segmentation model instance (or `None` if loading fails).
- `DEVICE`: PyTorch device (auto-selects CUDA/CPU/MPS).
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
<p align="center">
  <img width="900" alt="Alignment Plot" src="https://github.com/user-attachments/assets/80065fda-0be1-4fd9-807e-5163a04950e8">
</p>


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
| `confidence_weight`         | `float`      | `0.9`   | Weight for confidence in scoring.           |
| `plausibility_weight`       | `float`      | `0.1`   | Weight for plausibility in scoring.         |
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
<p align="center">
  <img width="900" alt="ICA Filter Plot" src="https://github.com/user-attachments/assets/85a9f730-a2a9-4f71-9c9f-c671ef00f704">
</p>


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
(ploted with the `plot_segmented_signal` function)
<p align="center">
  <img width="900" alt="Segmented Signal" src="https://github.com/user-attachments/assets/9e5616e0-f852-4573-b0b9-dbcb8bd8fd66">
</p>

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
| `confidence_weight`  | `float`      | `0.8`   | Confidence weight in scoring.               |
| `plausibility_weight`| `float`      | `0.2`   | Plausibility weight in scoring.             |

**Returns**:
- `List[int]`: Peak indices for the cleanest channel.
- `int`: Cleanest channel index.
- `np.ndarray`: Labels `(channels, samples)`.
- `Optional[float]`: HR (bpm) or `None`.
- `Optional[float]`: HRV (SDNN, ms) or `None`.

**Raises**:
- `Warning`: If segmentation data is empty.

**Example Output**:
<p align="center">
  <img width="900" alt="QRS Detection" src="https://github.com/user-attachments/assets/724c40f4-38d5-4458-822f-3a02c9da62e4">
</p>


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
| `window_right`              | `float`          | `0.4`   | Seconds right of peak.                      |
| `heart_beat_score_threshold`| `float`          | `0.7`   | Min score for window inclusion.             |
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

<p align="center">
  <img width="700" alt="Leermessung_Noise_Spectrum_Triax_Sensor_LSD" src="https://github.com/user-attachments/assets/2015e780-456a-49ec-b4c0-3eab05510e21">
</p>


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

<p align="center">
  <img width="700" alt="Leermessung_Noise_Spectrum_Triax_Sensor_LSD" src="https://github.com/user-attachments/assets/d8217a25-4953-4f22-9028-807334be9a24">
</p>




**Parameters**:
| Parameter    | Type             | Default | Description                                 |
|--------------|------------------|---------|---------------------------------------------|
| `data`       | `np.ndarray`     | —       | Input `(channels, samples)` at 250 Hz.      |
| `noise_theos`| `list` or `np.ndarray` | `None` | Theoretical noise floors.                    |
| `name`       | `str`            | `""`      | Title and filename base.                    |
| `labels`     | `list of str`    | `None`      | Channel labels for legend.                   |
| `channels`   | `list of int`    |   `None`     | Channel indices to plot.                    |
| `path`       | `str`            | `""`       | Save directory.                             |
| `save`       | `bool`           | `False` | Save as PNG.                                |

**Returns**: None.

**Raises**:
- `ValueError`: If input arrays are inconsistent.


#### `visualize_heart_vector`
Plots a 2D projection of the heart vector for a given segment and calculates key geometric metrics, incorporating measurement uncertainty.

**Steps**:
1.  Validates that the input data and segment indices are valid.
2.  Extracts the specified signal segment from the `original_data`.
3.  Calls `calculate_metrics_with_uncertainty` to run an internal Monte Carlo simulation. This helper function perturbs the segment boundaries according to `uncertainty_ms` over `n_realizations` to estimate the mean and standard deviation for each metric.
4.  Sets up an equal-aspect plot with a styled grid.
5.  Plots the vector projection line and fills the enclosed area.
6.  Adds directional arrows to indicate the trajectory's flow.
7.  Displays the calculated metrics with their uncertainties (e.g., `Area: 150.2 ± 5.3`) in a text box on the plot.
8.  Handles saving the plot to a file if `save_path` is provided.
9.  Supports both standalone plotting and integration into a larger subplot grid via the `ax` parameter.

**Usage**:
```python
# Assuming 'sensor_data' is a (2, N) numpy array and segment boundaries are known
ax, uncertain_metrics = analyzer.visualize_heart_vector(
    original_data=sensor_data,
    segment_start_global=100,
    segment_end_global=180,
    proj_name="xy-Projection",
    title_suffix="T-Wave Segment",
    save_path="t_wave_vector.pdf",
    uncertainty_ms=40,
    n_realizations=100
)
```

**Parameters**:
| Parameter | Type | Default | Description |
|---|---|---|---|
| `original_data` | `np.ndarray` | — | The full, averaged waveform data, shape `(n_components, n_samples)`. |
| `segment_start_global` | `int` | — | Global start index of the segment to analyze within `original_data`. |
| `segment_end_global` | `int` | — | Global end index of the segment to analyze within `original_data`. |
| `proj_name` | `str` | — | Name of the projection (e.g., `'xy-Projection'`). Used for titles and labels. |
| `title_suffix`| `str` | `""` | Additional text to append to the plot title. |
| `ax` | `matplotlib.axes.Axes` | `None` | An existing Matplotlib axis to plot on. If `None`, a new figure is created. |
| `show` | `bool` | `True` | Whether to display the plot with `plt.show()`. |
| `save_path` | `str` | `None` | File path to save the generated plot. If `None`, the plot is not saved. |
| `uncertainty_ms` | `int` | `100` | The uncertainty (in milliseconds) of segment boundaries, used for the Monte Carlo simulation. |
| `n_realizations`| `int` | `100` | The number of Monte Carlo iterations to run for uncertainty estimation. |
| `display_legend`| `bool` | `True` | Whether to display the legend with metrics. Default is True. |

**Returns**:
- `tuple`: A tuple `(ax, uncertain_metrics)` where:
  - `ax` is the `matplotlib.axes.Axes` object for the plot.
  - `uncertain_metrics` is a dictionary containing the calculated metrics as `ufloat` objects (value ± uncertainty).

**Raises**:
- `ValueError`: If `original_data` is invalid or segment indices are out of bounds.

**Example Output**:
<p align="center">
  <img width="600" alt="Heart Vector Projection" src="https://github.com/user-attachments/assets/3bc59e5a-985f-4596-96c8-b524208e5a7d">
</p>


#### `visualize_heart_vectors`
Plots the XY, XZ, and YZ heart vector projections for a specified segment in a single figure, with each projection including its own uncertainty metrics.

**Steps**:
1.  Validates that the input `heart_vector_components` has the correct shape `(3, samples)`.
2.  Creates a 1x3 subplot grid to hold the three projections.
3.  Iterates through the XY, XZ, and YZ combinations. For each one, it calls `plot_heart_vector_projection`, passing the relevant data slice (`Bx` and `By`, etc.) along with the global segment boundaries and uncertainty parameters.
4.  Each subplot independently calculates and displays its own geometric metrics with uncertainty.
5.  Adds a unified title to the entire figure.
6.  Adjusts the layout for clean presentation and saves the figure to `save_path` if provided.

**Usage**:
```python
# Assuming 'avg_heart_vector' is a (3, N) numpy array with Bx, By, Bz components
analyzer.plot_all_heart_vector_projections(
    heart_vector_components=avg_heart_vector,
    segment_start_global=100,
    segment_end_global=180,
    title_suffix="T-Wave Segment Analysis",
    save_path="./plots/all_projections_T_wave.pdf",
    uncertainty_ms=40
)
```

**Parameters**:
| Parameter | Type | Default | Description |
|---|---|---|---|
| `heart_vector_components` | `np.ndarray` | — | Input array of shape `(3, samples)` containing the Bx, By, and Bz components. |
| `segment_start_global` | `int` | — | Global start index of the segment to analyze within `heart_vector_components`. |
| `segment_end_global` | `int` | — | Global end index of the segment to analyze within `heart_vector_components`. |
| `title_suffix` | `str` | `""` | Suffix for the main figure title. |
| `save_path` | `str` | `None` | File path to save the generated plot. If `None`, the plot is not saved. |
| `uncertainty_ms` | `int` | `100` | The uncertainty (in milliseconds) of segment boundaries, passed to each subplot for metric calculation. |
| `n_realizations`| `int` | `100` | The number of Monte Carlo iterations, passed to each subplot. |

**Returns**:
-   `None`. The function generates and displays/saves a plot but does not return any objects.

**Raises**:
-   Logs an error via `logging.error` if the input `heart_vector_components` does not have the shape `(3, num_samples)`.


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
<p align="center">
  <img width="1000" alt="Segmented Signal" src="https://github.com/user-attachments/assets/b84801ec-2e0e-4c94-9d95-8b0da44f2fd3">
</p>




#### `plot_segmented_signal_with_editing`
Visualizes an ECG signal with interactive segmentation, enabling users to modify segment boundaries and reclassify segments through a graphical interface. The function displays the signal with color-coded segments and provides tools for adjusting boundaries and segment labels.

**Functionality**:
1. Creates a deep copy of the input predictions to preserve the original data.
2. Converts the input signal and predictions to NumPy arrays for processing.
3. Launches an interactive Matplotlib plot with draggable segment boundaries and buttons for segment reclassification.
4. Handles user interactions for editing segment boundaries and labels, updating the plot in real-time.
5. Returns the modified predictions as a NumPy array.

**User Interface**:
- **Plot Display**: The ECG signal is plotted as a black line on a grid, with segments shaded in colors based on their class labels (defined in `self.SEGMENT_COLORS`). Segment boundaries are marked with red dashed lines (blue when selected).
- **Segment Labels**: Each segment displays its class name (from `self.CLASS_NAMES_MAP`) at the top center of the segment for easy identification.
- **Boundary Interaction**: 
  - Click a red boundary line to select it (turns blue). The selection threshold is within 2% of the signal length.
  - Drag the selected boundary to adjust its position, constrained to avoid overlapping adjacent boundaries.
  - Click away from boundaries to deselect.
- **Class Selection Buttons**: Located at the bottom of the plot, buttons correspond to class labels (from `self.CLASS_NAMES_MAP`) with associated colors. Clicking a button reclassifies the segment to the right of the selected boundary.
- **Save Button**: A green "Save" button closes the plot and returns the edited predictions.
- **Status Text**: A text box at the bottom provides feedback (e.g., "Click on a red boundary line to select, then drag to move" or "Dragging boundary - release mouse to finish").


**Usage**:
```python
edited_predictions = analyzer.plot_segmented_signal_with_editing(signal, pred)
```

**Parameters**:
| Parameter  | Type         | Default | Description                                       |
|------------|--------------|---------|---------------------------------------------------|
| `signal`   | `np.ndarray` | —       | 1D array of the ECG signal.                      |
| `pred`     | `np.ndarray` | —       | 1D array of initial model predictions (class labels). |

**Returns**:
- `np.ndarray`: Edited predictions array with the same shape as the input `pred`.


**Example Output**:
<p align="center">
  <img width="1000" alt="Segmented Signal" src="https://github.com/user-attachments/assets/5cb5ecdd-8fd6-42a4-a17c-af545bd4eb76">
</p>



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
<p align="center">
  <img width="900" alt="Butterfly Plot" src="https://github.com/user-attachments/assets/50478143-9f66-44d2-96f3-73c2338fedfb">
</p>

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

## The Analysis Pipeline in Practice

The end-to-end workflow is managed by the `heart_beat.py` and `statistical_analysis.py` scripts, orchestrated by a central configuration file.

### Configuration: `Data/setup.json`
This file is the single source of truth for the entire pipeline. It defines all patients, their demographic data, their cohort (ACM or Healthy), and the specific parameters for each experimental run.

**Example `setup.json` structure:**
```json
{
  "P001": {
    "gender": "male",
    "height": "180",
    "age": "45",
    "runs": {
      "S01": {
        "ARVC": true,
        "interval_start": 250,
        "interval_end": 1250,
        "ICA_filter": [0.74, 0.73, 0.77],
        "sensors_to_exclude": {
          "Brustlage": ["Q01_z"]
        }
      }
    }
  },
  "P002": { ... }
}
```
- **Patient ID (`P001`)**: Top-level key for each subject.
- **`ARVC`**: `true` for ACM patients, `false` for healthy controls. This label is essential for the statistical analysis.
- **`interval_start`/`interval_end`**: The time window (in samples at the original sampling rate) of the recording to analyze.
- **`ICA_filter`**: A list of three float values representing the heartbeat score thresholds for the x, y, and z components during ICA filtering.
- **`sensors_to_exclude`**: A dictionary to exclude specific sensor channels from a run, preventing them from being processed.



### Step 1: Feature Extraction (`heart_beat.py`)
This script is the main driver for processing a single patient's data. It orchestrates calls to the `Analyzer` class to perform a full analysis chain, incorporating several stages of user interaction for quality control.

**Workflow**:
1.  **User Input & Configuration**: The script prompts the user for a Patient ID (e.g., `P001`) and an optional Run ID (e.g., `S01`). It then loads configurations from two sources:
    - A global `CONFIG` dictionary at the top of the script for file paths and model details.
    - The central `Data/setup.json` file for patient-specific parameters like analysis intervals, ICA thresholds, and excluded sensors.
    - The segmentation model (e.g., `UNet1D`) is dynamically loaded based on the configuration.

2.  **Data Preparation**: It calls `analyzer.prepare_data()`, which loads the raw TDMS files, applies a default filter combination at the original sampling rate, aligns the primary and additional data files, crops the signal to the specified interval, and finally resamples the data to the internal 250 Hz processing rate.

3.  **Interactive ICA Filtering**: This is a key interactive step for artifact removal. The script applies `analyzer.ICA_filter()` separately to the x, y, and z components of the data. For each component, it displays an **interactive plot with a slider**, allowing the user to visually inspect the effect of the heartbeat score threshold and choose the optimal value for cleaning the signal before proceeding.

4.  **QRS Detection, Averaging, and Segmentation**:
    - The script uses `analyzer.detect_qrs_complex_peaks_cleanest_channel()` to find R-peaks in the filtered signal.
    - It then plots the detected R-peaks on the cleanest channel's signal for immediate visual validation by the user.
    - Using these peaks, it calls `analyzer.avg_window()` to create a high-SNR, averaged heartbeat waveform for all channels.
    - **Crucially**, it calls `analyzer.plot_segments_with_editing()` on the averaged waveform. This opens another interactive UI where the user can manually drag segment boundaries and reclassify P, QRS, and T waves to ensure maximum accuracy.

5.  **Metric Calculation, Plotting, and Saving**: After the user saves their segmentation edits, the script executes a comprehensive loop for every sensor in the grid:
    - It identifies the valid 2D projection for the sensor (e.g., xy, xz, or yz).
    - It uses the finalized segment boundaries to calculate heart vector metrics (**Area, Distance, Compactness, and Angle**) for the **T, QRS, and ST segments** by calling `analyzer.visualize_heart_vector()`. This method internally runs a Monte Carlo simulation to estimate the uncertainty of each metric.
    - **Plot Generation**:
        - It saves a detailed, individual **PDF plot of the heart vector for each segment of each sensor** in a structured directory (e.g., `Results/Q01_xy/Patients/P001_S01/T_heart_vector/`).
        - It also generates and saves **4x4 grid plots** (one for each segment: T, QRS, ST) that provide a full overview of all sensor projections for the current patient.
    - **CSV Output**: For each sensor, it appends a single row containing all calculated metrics and their uncertainties (e.g., `t_Area`, `t_Area_unc`, `qrs_Distance`, `qrs_Distance_unc`, etc.) to a `result.csv` file located in the sensor's results directory (e.g., `Results/Q01_xy/`).



### Step 2: Statistical Analysis (`statistical_analysis.py`)
After running `heart_beat.py` for all subjects, this script performs a highly configurable, multi-level statistical analysis to compare the ACM and healthy cohorts.

**Workflow**:
1.  **Configuration and Data Loading**: The script begins by loading a global `feature_analysis_config` dictionary. This powerful feature allows for fine-grained control over the analysis of each metric, specifying parameters like the t-test hypothesis (`not_equal`, `data1_greater`), whether to remove outliers using the IQR method, and units for plotting. It then automatically finds and loads all `result.csv` files from the `Results` directory, merging them with cohort information from `Data/setup.json`.

2.  **Demographic Analysis**: It first performs a detailed demographic analysis, generating and saving separate plots of age, height, and gender distributions for the "All", "ACM", and "Healthy" groups. This provides a clear overview of the study population.

<p align="center">
  <img width="600" alt="Demographic analysis plots for age, height, and gender" src="https://github.com/user-attachments/assets/11a517e2-9d35-4622-a5e2-3c2f07a6f985">
</p>

3.  **Two-Tiered Statistical Analysis**: The script's main analysis is performed in two distinct stages to provide both localized and regional insights:
    *   **A. Individual Sensor Analysis**: The script iterates through every sensor projection (e.g., `Q01_xy`, `F2_yz`, etc.) and performs a full, independent statistical analysis on each one.
    *   **B. Aggregated Quadrant Analysis**: In this powerful new step, the script aggregates data from sensors located in four predefined quadrants of the sensor grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right) for each projection type (`xy`, `yz`). It then runs the same full statistical analysis on these aggregated regional datasets.

4.  **Comprehensive Statistical Comparisons**: For each feature (e.g., T-Wave Area) in each analysis run (both individual and aggregated), the script performs a comprehensive comparison between the two groups:
    *   **T-Test Analysis**: It runs `perform_t_test()`, which calculates Welch's t-test to get a p-value. It generates a plot of the t-distribution and a detailed boxplot that includes individual data points and Monte Carlo-derived confidence intervals for the median.
    *   **ROC Analysis**: It runs `determine_optimal_threshold()` to evaluate diagnostic performance. This function calculates the AUC, F1-score, sensitivity, and specificity, and generates an ROC curve plot and a confusion matrix.

5.  **Monte Carlo Simulation for Uncertainty**: A core strength of the script is its robust use of Monte Carlo simulations. The `_unc` columns (e.g., `t_Area_unc`) from the `result.csv` files are used to run the `perform_t_test()` and `determine_optimal_threshold()` functions hundreds of times on data perturbed according to its measurement uncertainty. This produces robust confidence intervals for all key statistical outputs (p-value, AUC, optimal threshold, etc.), providing a rigorous assessment of the findings.

6.  **Saving Outputs**: The script generates a wealth of structured output:
    *   For individual sensor analyses, all plots and a summary CSV are saved to a `Generated_Plots_MC` and `Generated_Tables_MC` subdirectory within that sensor's result folder.
    *   For aggregated analyses, plots and tables are saved in the global `Results/Overall_Generated_Plots/` and `Results/Overall_Generated_Tables/` directories.
    *   Finally, master summary CSV files are created, consolidating the results from all individual and all aggregated analyses.

A typical set of output plots for a single feature analysis includes the following:

| Box Plot with MC Uncertainty | ROC Curve Analysis | Confusion Matrix |
|:-----------------------------:|:------------------:|:----------------:|
| ![Box Plot of T-Wave Area](https://github.com/user-attachments/assets/898bc20a-c7db-4f1f-a361-baa14656cadc) | ![ROC Curve for T-Wave Area](https://github.com/user-attachments/assets/6f34e19d-17ee-4f65-a408-5d364264409d) | ![Confusion Matrix at Optimal Threshold](https://github.com/user-attachments/assets/4c8afc26-c580-44c1-9eef-b47815e44221) |

## Troubleshooting

- **Segmentation Model**: If the model (`MCG_segmentation`, `UNet-1D` etc.) or its checkpoints are missing, segmentation features will fail with logged warnings. 
- **Sampling Rates**: Ensure `input_sampling_rate` matches the data’s rate in `prepare_data`. All segmentation and QRS detection require 250 Hz input.
- **Memory Usage**: Long recordings or animations may require significant memory. Use interval selection or chunked processing.
- **FFmpeg**: Required for `create_heat_map_animation`. Install via `conda install ffmpeg` or system package manager.
- **Coordinate System**: The `_change_to_consistent_coordinate_system` method applies sign corrections based on sensor names and types. Customize if your sensor setup differs.
- **Channel Mapping**: The `__init__` method adjusts channel indices (`abs(val) >= 100`). Modify this logic if your TDMS channel mapping differs.

## Notes
- **Performance**: CUDA or MPS (Apple Silicon) is automatically selected if available, significantly speeding up segmentation and ICA.
- **Logging**: Set `logging` level to `DEBUG` for detailed diagnostics or `ERROR` for minimal output.
- **Extensibility**: Static utility methods (e.g., `bandstop_filter`) can be used independently for custom processing.
- **Data Integrity**: Validate TDMS and log files before processing to avoid runtime errors.


