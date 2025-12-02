
# MCG Data Analyzer & ACM Screening Pipeline

## Overview

This repository contains a comprehensive Python pipeline for processing, analyzing, and visualizing Magnetocardiography (MCG) data. The primary goal is to leverage high-precision recordings from optically pumped magnetometers (OPMs) to investigate biomarkers for **Arrhythmogenic Cardiomyopathy (ACM)**.

The pipeline offers an end-to-end workflow: from raw TDMS data ingestion, through AI-based segmentation and interactive artifact removal, to rigorous statistical analysis using Monte Carlo simulations for uncertainty quantification.

### Key Features
*   **End-to-End Workflow**: Raw data $\to$ Preprocessing $\to$ Feature Extraction $\to$ Statistical Significance.
*   **Interactive Cleaning**: Independent Component Analysis (ICA) with a UI for real-time artifact removal thresholds.
*   **AI Segmentation**: Integrated PyTorch models (https://github.com/Drizzr/MCG_segmentation) for automatic P-QRS-T segmentation, coupled with a **drag-and-drop UI** for manual refinement.
*   **Advanced Geometrics**: Calculates geometric features of the Heart Vector, including **Convex Hull Area**, **Solidity** (Twist), **Compactness**, and **Angle**.
*   **Uncertainty Propagation**: Implements Monte Carlo simulations to propagate timing uncertainties from segmentation into final statistical metrics.
*   **Aggregated Statistics**: Performs analysis on individual sensors and aggregates data spatially (Quadrants) to increase statistical power.

---

## Scientific Context & Methodology

### Background
Arrhythmogenic cardiomyopathy (ACM) involves mechanical or electrical dysfunction, often characterized by the replacement of heart muscle with fibrotic fatty tissue. This project uses MCG to detect subtle changes in the magnetic field vector of the heart caused by this remodeling.

### Measurements
All internal processing operates at a fixed **250 Hz**.
*   **Timing Uncertainty**: A Gaussian uncertainty ($\sigma \approx 40\text{ms}$) is applied to manual segment boundaries during metric calculation to ensure results are robust against annotation errors.
*   **Analysis Segments**: QRS Complex, ST Segment, T-Wave.

### Geometric Features
For each segment, the 2D projection of the magnetic field vector is analyzed. Key metrics include:

1.  **Enclosed Area (Net)**: Calculated via the Shoelace formula. Represents the signed area.
2.  **Area Hull (Gross)**: The area of the Convex Hull surrounding the vector loop. This measures total dispersion and is often more sensitive to ACM loop expansion.
3.  **Solidity**: Ratio of `Area Net / Area Hull`. Quantifies loop "twist" or concavity. A value of 1.0 indicates a perfect convex loop; lower values indicate twisting or jaggedness.
4.  **Distance**: Max amplitude displacement from the start point.
5.  **Compactness**: Circularity metric: $(4 \pi \cdot Area) / Perimeter^2$.
6.  **Angle**: Average trajectory angle.

---

## Installation

### Prerequisites
*   Python 3.10 or 3.11 (recommended).
*   FFmpeg (required for heatmap animations).

### 1. Clone the Repository
This project uses submodules for the segmentation model.
```bash
git clone --recurse-submodules https://github.com/Drizzr/Bachelorarbeit
cd Bachelorarbeit
```
*If you already cloned without submodules:* `git submodule update --init --recursive`

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── Data/
│   ├── setup.json             # CENTRAL CONFIGURATION (Patients, runs, filters)
│   ├── P001/                  # Patient Data Directory
│   │   ├── P001_S01.tdms      # Primary recording
│   │   └── QZFM_log_file.txt  # Sensor geometry/log
│   └── ...
├── Results/                   # Output Directory
│   ├── Overall_Generated_Plots/   # Aggregated statistical plots
│   ├── Overall_Generated_Tables/  # Summary CSVs
│   └── <Sensor_ID>_<Proj>/        # Individual sensor results (e.g., Q01_xy)
├── MCG_segmentation/          # Submodule: PyTorch Segmentation Models
├── analyzer.py                # Core API: Data processing logic
├── heart_beat.py              # Pipeline Step 1: Feature extraction script
├── statistical_analysis.py    # Pipeline Step 2: Statistics script
└── requirements.txt
```

---

## The Analysis Pipeline in Practice

The workflow consists of two main scripts driven by the `Data/setup.json` configuration.

### Configuration (`setup.json`)
Controls parameters for every patient and run.

```json
{
  "P001": {
    "gender": "male", "height": "180", "age": "45",
    "runs": {
      "S01": {
        "ACM": true,                 // True = Patient, False = Healthy Control
        "interval_start": 250,       // Start sample (at source sampling rate)
        "interval_end": 1250,        // End sample
        "ICA_filter": [0.74, 0.73, 0.77], // Thresholds for [x, y, z] components
        "sensors_to_exclude": {
          "Brustlage": ["Q01_z"]     // Exclude broken channels
        }
      }
    }
  }
}
```

### Step 1: Feature Extraction (`heart_beat.py`)
Run this script to process a specific patient.
```bash
python heart_beat.py
```
**Workflow:**
1.  **Input**: Enter Patient ID (e.g., `P001`) and Run ID.
2.  **ICA Filtering**: An interactive plot appears. Adjust the slider to find the threshold that removes noise but keeps the heartbeat. Close the window to apply.
    <p align="center"><img width="900" alt="ICA Filter Plot" src="https://github.com/user-attachments/assets/33241fc1-28f2-4d5d-b532-dc42f9c57a05"></p>
3.  **QRS Detection**: Verifies R-peak detection on the cleanest channel.
    <p align="center"><img width="900" alt="QRS Detection" src="https://github.com/user-attachments/assets/724c40f4-38d5-4458-822f-3a02c9da62e4"></p>
4.  **Segmentation Editing**: An interactive plot appears showing the averaged heartbeat.
    *   **Click** a boundary (red line) to select it (turns blue).
    *   **Drag** to adjust timing.
    *   **Click Class Buttons** to reassign segment types (P-Wave, QRS, T-Wave).
    *   **Save** to proceed.
    <p align="center"><img width="1000" alt="Segmented Signal Editing" src="https://github.com/user-attachments/assets/5cb5ecdd-8fd6-42a4-a17c-af545bd4eb76"></p>
5.  **Output**: Geometric metrics (Area, Solidity, etc.) are calculated for every sensor and saved to `Results/`. Vector plots are generated.
    <p align="center"><img width="600" alt="Heart Vector Projection" src="https://github.com/user-attachments/assets/54a0428e-67ac-40ef-8b21-edd517d095fa"></p>

### Step 2: Statistical Analysis (`statistical_analysis.py`)
Run this script to compare ACM vs. Healthy cohorts.
```bash
python statistical_analysis.py
```
**Workflow:**
1.  **Demographics**: Plots age, gender, and height distributions.
2.  **Individual Analysis**: Runs t-tests and ROC analysis for every feature on every sensor.
3.  **Aggregated Analysis**: Groups sensors into **Quadrants** (TopLeft, TopRight, etc.) and performs analysis on the pooled data to reduce noise.
4.  **Monte Carlo**: Re-runs stats 5000 times (configurable) using the measurement uncertainties (from Step 1) to generate robust p-values and confidence intervals.
5.  **Output**: Saves summary tables and plots to `Results/Overall_Generated_...`.

| Box Plot with MC Uncertainty | P-Value Distribution | F1-Score Distribution |
|:-----------------------------:|:------------------:|:----------------:|
| ![Box Plot](https://github.com/user-attachments/assets/1c438666-9b1a-4953-a179-4a2e183809c5) | ![P-Value Dist](https://github.com/user-attachments/assets/0b1cdf4c-d018-4409-82cb-4754009310ad) | ![F1 Dist](https://github.com/user-attachments/assets/4249b7f0-39ad-4e21-95b7-6c8ae2c84239) |

---

## Analyzer Class API Documentation

The `Analyzer` class (`analyzer.py`) is the core engine. It assumes an internal sampling rate of **250 Hz**.

### Initialization
```python
analyzer = Analyzer(
    filename="path/to/data.tdms",
    add_filename="path/to/extra_data.tdms",
    log_file_path="path/to/log.txt",
    model_checkpoint_dir="path/to/checkpoints",
    model_class=UNet1D,
    sensor_channels_to_exclude={'run_key': ['Q01_z']}
)
```

### Data Preparation Methods

#### `prepare_data`
Loads TDMS files, applies default filters (Bandstop 50Hz, Lowpass 95Hz, Highpass 1Hz, Savitzky-Golay), aligns primary and additional recordings via cross-correlation, crops to the requested interval, resamples to 250Hz, and organizes data into spatial grids (x, y, z).
<p align="center"><img width="900" alt="Alignment Plot" src="https://github.com/user-attachments/assets/80065fda-0be1-4fd9-807e-5163a04950e8"></p>

#### `ICA_filter`
Decomposes the signal using FastICA.
*   **Logic**: Components are scored based on "Heartbeat Plausibility" (presence of P, QRS, T waves relative to physiological norms).
*   **Interaction**: If `plot_result=True`, opens a slider UI to manually set the score threshold.
*   **Returns**: Reconstructed signal with noise components removed.

### Segmentation & QRS Methods

#### `segment_entire_run`
Applies the loaded PyTorch model (e.g., `UNet1D`) to segment the signal.
*   **Input**: Normalized 250Hz data.
*   **Output**: Labels (0=No Wave, 1=P, 2=QRS, 3=T) and confidence scores.

#### `find_cleanest_channel`
Identifies the best sensor channel by analyzing segmentation confidence and physiological plausibility scores across all channels.

#### `detect_qrs_complex_peaks_cleanest_channel`
Detects R-peaks on the cleanest channel using a minimum-distance logic (default 0.3s) and confidence thresholding. Returns peak indices and calculated Heart Rate/HRV.

#### `plot_segmented_signal_with_editing`
**Interactive UI Feature.** Plots the signal with colored segments.
*   **Editing**: Allows selecting boundaries and dragging them.
*   **Reclassification**: Buttons to change a segment's class (e.g., mistake T-wave for Noise).
*   **Returns**: The modified prediction array.

### Visualization & Analysis Methods

#### `visualize_heart_vector`
The core metric calculator.
1.  **Extracts** the segment (e.g., T-wave) from 2D projection data.
2.  **Simulates** uncertainty: Perturbs start/end points randomly ($N=100$) based on `uncertainty_ms`.
3.  **Calculates** metrics (Area, Hull, Solidity, etc.) for every iteration.
4.  **Returns**: `uncertainties.ufloat` objects (mean ± std_dev) and plots the trajectory.

#### `plot_sensor_matrix`
Plots time-series traces for all 16 sensors in a 4x4 grid layout.
<p align="center"><img width="700" alt="Sensor Matrix" src="https://github.com/user-attachments/assets/2015e780-456a-49ec-b4c0-3eab05510e21"></p>

#### `plot_lsd_multichannel`
Plots the Linear Spectral Density (Noise Spectrum) using Welch's method.
<p align="center"><img width="700" alt="LSD Plot" src="https://github.com/user-attachments/assets/d8217a25-4953-4f22-9028-807334be9a24"></p>

#### `butterfly_plot`
Overlays all channels on a single axis to visualize global signal coherence.
<p align="center"><img width="900" alt="Butterfly Plot" src="https://github.com/user-attachments/assets/50478143-9f66-44d2-96f3-73c2338fedfb"></p>

#### `create_heat_map_animation`
Generates an MP4/GIF heatmap of the magnetic field strength evolving over time, interpolated from the sensor positions.

---

## Troubleshooting

- **Missing Model**: If segmentation fails, ensure the `MCG_segmentation` submodule is initialized and `model_checkpoint_dir` in `heart_beat.py` points to a valid `.pth` file.
- **"No Data" in plots**: Check `setup.json` intervals. If `interval_start` is larger than the recording length, arrays will be empty.
- **Interactive plots freezing**: Ensure your Python environment supports GUI backends (e.g., `TkAgg` or `Qt5Agg`).
- **Memory Usage**: High-resolution animations or very long intervals may consume RAM. Reduce `resolution` in `create_heat_map_animation`.
- **FFmpeg Error**: If animation fails, install FFmpeg: `conda install ffmpeg` or via your system package manager.