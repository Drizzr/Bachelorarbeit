

# ECG Segmentation Project Documentation

## Table of Contents

- [1. Overview](#1-overview)
  - [1.1. Key Features](#11-key-features)
- [2. Project Structure](#2-project-structure)
- [3. Installation and Setup](#3-installation-and-setup)
  - [3.1. Dependencies](#31-dependencies)
  - [3.2. Dataset](#32-dataset)
  - [3.3. Data Preprocessing (`create_dataset.py` and `create_training_data.py`)](#33-data-preprocessing-create_datasetpy-and-create_training_datapy)
  - [3.4. Prepared Data Format](#34-prepared-data-format)
- [4. Data Loading (`data_loader.py`)](#4-data-loading-data_loaderpy)
  - [4.1. `ECGFullDataset` Class](#41-ecgfulldataset-class)
  - [4.2. Testing and Visualization](#42-testing-and-visualization)
- [5. Model Architecture (`model.py`)](#5-model-architecture-modelpy)
  - [5.1. Model Overview](#51-model-overview)
  - [5.2. Unet-1D-15M](#52-unet-1d-15m)
  - [5.3. Unet-1D-900k](#53-unet-1d-900k)
  - [5.4. MCG-Segmentator_s](#54-mcg-segmentator_s)
  - [5.5. MCG-Segmentator_xl](#55-mcg-segmentator_xl)
  - [5.6. DENS-Model](#56-dens-model)
- [6. Training (`trainer.py` and `train.py`)](#6-training-trainerpy-and-trainpy)
  - [6.1. `Trainer` Class](#61-trainer-class)
  - [6.2. Training Script (`train.py`)](#62-training-script-trainpy)
- [7. Evaluation (`evaluate.py`)](#7-evaluation-evaluatepy)
  - [7.1. Usage Example](#71-usage-example)
  - [7.2. Key Arguments](#72-key-arguments)
  - [7.3. Outputs](#73-outputs)
  - [7.4. Evaluation Metrics](#74-evaluation-metrics)
  - [7.5. Post-Processing](#75-post-processing)
  - [7.6. Evaluation Process](#76-evaluation-process)
- [8. Training Process and Results](#8-training-process-and-results)
  - [8.1. General Training Setup](#81-general-training-setup)
  - [8.2. Summary of Model Performance](#82-summary-of-model-performance)
- [9. Possible Applications to Magnetocardiography and ECG](#9-possible-applications-to-magnetocardiography-and-ecg)
- [10. References](#10-references)

## 1. Overview

The ECG Segmentation Project provides a robust Python-based framework for segmenting Electrocardiogram (ECG) and Magnetocardiogram (MCG) signals into their core cardiac cycle components: No Wave, P-Wave, QRS Complex, and T-Wave [1]. While the primary goal is to develop deep learning models for precise MCG segmentation, the lack of open-source MCG datasets necessitates the use of established ECG databases for training. This project addresses key challenges in this domain:
- **Short-Window Analysis**: Models are designed to be effective on very short segments (1-2 seconds), unlike many state-of-the-art solutions that require longer data periods (5-10 seconds).
- **Signal Normalization**: A critical two-step normalization process is implemented to handle the vastly different absolute scales of ECG and MCG signals.
- **Independent Channel Processing**: Models treat each channel independently, which is crucial for MCG analysis where directional components do not correspond to standard ECG leads.

Built using PyTorch, this framework processes ECG data from the QT Database (QTDB) and the Lobachevsky University Database (LUDB), supporting advanced features like data augmentation, multi-scale feature extraction with self-attention, and comprehensive evaluation tools. The ultimate goal is to enable applications like robust QRS peak detection, average waveform generation, and the computation of novel biomarkers for cardiac conditions like Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC).

(c.f. [Magnetocardiography To Screen Adults with Arrhythmogenic Cardiomyopathy: a pilot study](https://github.com/Drizzr/Bachelorarbeit))

### 1.1. Key Features

- **Data Sources**: Utilizes the QT Database (QTDB) and the Lobachevsky University Database (LUDB) from PhysioNet [2], providing diverse ECG datasets for training and testing.
- **Advanced Preprocessing**: Includes scripts to download and parse datasets, harmonize sampling frequencies, and generate artificial T-onsets for the QTDB to handle incomplete annotations. Data is split into dedicated training, validation and test sets.
- **Robust Data Augmentation**: Enhances model generalization through a pipeline of augmentations including amplitude scaling, time shifting, Gaussian noise, baseline wander, and high-frequency sinusoidal noise to simulate real-world conditions.
- **Critical Signal Normalization**: Implements a two-step zero-mean and max-absolute scaling process to ensure the model learns features independent of absolute signal amplitude.
- **State-of-the-Art Model Architectures**:
  - `Unet-1D-15M` & `Unet-1D-900k`: 1D U-Net models with an integrated **Multi-Head Self-Attention (MHSA)** mechanism [3] in the bottleneck to capture long-range dependencies in the signal, inspired by recent literature.
  - `MCG-Segmentator_s` & `_xl`: Custom models combining multi-scale convolutions, BiLSTM, and Transformer-based self-attention.
  - `DENS-Model`: An implementation based on the DENS-ECG paper [4].
- **Configurable Training Pipeline**: Features the AdamW optimizer [5], Cosine Annealing learning rate scheduling, Focal Loss [6] for handling class imbalance, and detailed logging of training metrics.
- **Standardized Evaluation**: Implements an event-based evaluation protocol adhering to AAMI standards (±150 ms tolerance window) and calculates key metrics including Sensitivity (Se), Positive Predictive Value (PPV), F1-Score, and mean error (m±σ) for wave onsets and offsets.
- **Post-processing**: Employs a simple yet effective post-processing step to remove small, erroneous segment predictions, improving final delineation accuracy.
- **Extensibility**: Modular design facilitates the integration of new models, datasets, or custom augmentation techniques.

## 2. Project Structure

The project is organized as follows:

```
project_root/
├── Datasets/
│   ├── base/
│   │   ├── qtdb/
│   │   │   ├── raw/                # Raw QTDB files (downloaded by create_dataset.py)
│   │   │   └── processed/          # Processed QTDB CSV files
│   │   │   └── create_dataset.py   # Preprocess QTDB data
│   │   ├── ludb/
│   │   │   ├── raw/                # Raw LUDB files (downloaded by create_dataset.py)
│   │   │   └── processed/          # Processed LUDB CSV files
│   │   │   └── create_dataset.py   # Preprocess LUDB data
│   ├── train/                      # Training CSV files (generated by create_training_data.py)
│   ├── val/                        # Validation CSV files (generated by create_training_data.py)
│   └── create_training_data.py     # Logic to create the train/test split
├── model/
│   ├── data_loader.py              # Dataset and DataLoader for ECG data
│   ├── model.py                    # Model architectures and loss functions
│   └── trainer.py                  # Training logic and logging
├── train.py                        # Main training script
├── evaluate.py                     # Evaluation script
├── trained_models/
│   ├── Unet-1D-15M/
│   │   ├── checkpoints/            # Saved model checkpoints for Unet-1D-15M
│   │   ├── logs/                   # Training metrics CSV for Unet-1D-15M
│   │   └── evaluation_results/     # Evaluation plots and metrics for Unet-1D-15M
│   ├── Unet-1D-900k/
│   │   └── ...
│   ├── MCG-Segmentator_s/
│   │   └── ...
│   ├── MCG-Segmentator_xl/
│   │   └── ...
│   └── DENS-Model/
│       └── ...
└── requirements.txt                # Python dependencies
```

## 3. Installation and Setup

### 3.1. Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Required libraries include:
- torch, numpy, pandas, matplotlib, scikit-learn, seaborn, tqdm, json, csv, logging, wfdb

Ensure PyTorch is compatible with your hardware (CUDA for GPU support). Check PyTorch’s official site for installation instructions.

### 3.2. Dataset

The project utilizes two key open-source ECG datasets from PhysioNet [2] for training and benchmarking: the **QT Database (QTDB)** and the **Lobachevsky University Database (LUDB)**.

#### 3.2.1. QT Database (QTDB)

The [QT Database (QTDB)](https://physionet.org/content/qtdb/1.0.0/) is a standard dataset for evaluating ECG analysis algorithms [7].
- **Composition**: 105 two-lead ECG recordings, each 15 minutes long, sampled at 250 Hz.
- **Annotations**: For each record, at least 30 individual heartbeats are annotated. While it includes P, QRS, and T-wave annotations, many recordings are missing T-wave onset information.

#### 3.2.2. Lobachevsky University Database (LUDB)

The [Lobachevsky University Database (LUDB)](https://physionet.org/content/ludb/1.0.1/) is a more recent dataset designed to address some shortcomings of older databases [8, 9].
- **Composition**: 200 12-lead ECG recordings from unique subjects, each 10 seconds long, sampled at 500 Hz.
- **Annotations**: Provides more complete and consistent annotations for P, QRS, and T-wave onsets and offsets across all leads.

#### 3.2.3. Dataset Comparison

The following table, adapted from the thesis, summarizes the key characteristics of the datasets used for model training.

| Data Source | # Recordings | Annotated Duration | Frequency | Leads | Annotations                          |
|-------------|--------------|--------------------|-----------|-------|--------------------------------------|
| QTDB        | 105          | ~30 seconds        | 250 Hz    | 2     | P on/offsets, QRS on/offsets, T offsets |
| LUDB        | 200          | 10 seconds         | 500 Hz    | 12    | P on/offsets, QRS on/offsets, T on/offsets |


### 3.3. Data Preprocessing (`create_dataset.py` and `create_training_data.py`)

The project employs a multi-step preprocessing pipeline to prepare the QTDB and LUDB datasets for training.

#### 3.3.1. Key Operations

-   **Download**: Scripts automatically fetch data from PhysioNet [2] into `Datasets/base/qtdb/raw/` and `Datasets/base/ludb/raw/`.
-   **Harmonization**: LUDB signals are downsampled from 500 Hz to 250 Hz to match QTDB's sampling frequency. Each lead from both datasets is processed independently.
-   **Annotation Parsing & Labeling**: Waveform and annotation files are parsed to identify P-wave, QRS complex, and T-wave segments. Gaps between waves are labeled as "No Wave". Integer labels are assigned: 0 (No Wave), 1 (P-Wave), 2 (QRS Complex), 3 (T-Wave).
-   **Artificial T-Wave Onset Generation (for QTDB)**: To address the frequent absence of T-wave onset annotations in QTDB, an artificial T-onset is generated for each beat by sampling from a Gaussian distribution centered around a physiologically normal ST segment length of 100 ms, with a standard deviation of 20 ms.
-   **Data Filtering**:
    -   For QTDB, records containing only QRS annotations are initially discarded. Additionally, a predefined list of records with known quality issues (e.g., `sel102`, `sel104`) are moved to an `unused` directory and excluded from the training, validation, and test sets.
    -   For LUDB, only the annotated intervals are used, as annotations are often missing in the first and last seconds of a recording.
-   **Data Splitting and Set Composition (`create_training_data.py`)**: To ensure a rigorous and unbiased evaluation, the processed data is partitioned at the **patient level**, meaning all leads from a single patient record are assigned exclusively to one set. The final sets are a mixture from both databases, composed as follows:
    -   **Training Set**: Used to train the model's parameters.
        -   *From QTDB*: 94% of the usable patient records.
        -   *From LUDB*: 55% of the patient records.
    -   **Validation Set**: Used for hyperparameter tuning and model selection (i.e., early stopping).
        -   *From QTDB*: The remaining 6% of usable patient records.
        -   *From LUDB*: 5% of the patient records.
    -   **Test Set**: A hold-out set used *only* for the final, unbiased performance evaluation.
        -   *From LUDB*: The remaining 40% of patient records.
        -   **Note**: No data from QTDB is included in the test set, ensuring compatability with performance metrics in literature.

#### 3.3.2. Running Preprocessing

1.  **For QTDB**:
    ```bash
    cd Datasets/base/qtdb
    python create_dataset.py
    ```

2.  **For LUDB**:
    ```bash
    cd Datasets/base/ludb
    python create_dataset.py
    ```

3.  **For Data Splitting**:
    ```bash
    cd Datasets
    python create_training_data.py
    ```

### 3.4. Prepared Data Format

After preprocessing, ECG data is stored in CSV format in `Datasets/train/`, `Datasets/test/` and `Datasets/val/`. Each CSV file represents a single channel/lead and contains:
- **Columns**: `time`, `index`, `label` (string), `train_label` (integer), and `wave_form` (signal amplitude).

## 4. Data Loading (`data_loader.py`)

### 4.1. `ECGFullDataset` Class

The `ECGFullDataset` class, defined in `model/data_loader.py`, is a core component responsible for loading processed ECG data, slicing it into sequences, and applying data augmentations and normalization to prepare data for model training.

**Key Features**:
- Loads processed CSV files from a specified directory.
- Slices long ECG recordings into shorter, overlapping sequences for model input.
- Applies data augmentations during training to simulate physiological and environmental noise, improving model robustness.
- Normalizes signals using a two-step process to reduce sensitivity to amplitude variations.
- Returns single-channel sequences of shape `(1, sequence_length)` with corresponding integer labels.

**Initialization**:

```python
from model.data_loader import ECGFullDataset

dataset = ECGFullDataset(
    data_dir="Datasets/train",
    label_column="train_label",
    file_extension= ".csv",
    sequence_length=512,
    overlap=256,
    sinusoidal_noise_mag=0.05,
    gaussian_noise_std=0.04,
    baseline_wander_mag=0.1,
    baseline_wander_freq_max=0.5,
    augmentation_prob= 0.5,
    amplitude_scale_range=(0.9, 1.1),
    time_shift_ratio=0.01
)
```

**Parameters**:

| Parameter                 | Type           | Default         | Description                                                                 |
|---------------------------|----------------|-----------------|-----------------------------------------------------------------------------|
| `data_dir`                | str            | —               | Directory containing processed CSV files (e.g., `Datasets/train/`).         |
| `label_column`            | str            | "train_label"   | Column name for integer labels (0: No Wave, 1: P-Wave, 2: QRS, 3: T-Wave). |
| `file_extension`          | str            | ".csv"          | File extension for data files.                                              |
| `sequence_length`         | int            | 500             | Length of each input sequence (in samples).                                 |
| `overlap`                 | int            | 125             | Overlap between consecutive sequences (in samples).                         |
| `sinusoidal_noise_mag`    | float          | 0.05            | Magnitude of powerline noise (simulating 50 Hz interference).                |
| `gaussian_noise_std`      | float          | 0.04            | Standard deviation of Gaussian noise (in mV).                               |
| `baseline_wander_mag`     | float          | 0.1             | Magnitude of baseline wander noise.                                         |
| `baseline_wander_freq_max`| float          | 0.5             | Maximum frequency for baseline wander (in Hz).
| `augmentation_prob`| float          | 0.5             | Probability of each augmentation step beeing applied                               |
| `amplitude_scale_range`   | Tuple[float, float] | (0.9, 1.1) | Range for random amplitude scaling (e.g., (0.9, 1.1) for ±10% scaling).     |
| `time_shift_ratio`        | float          | 0.01            | Ratio of sequence length for random time shifting (e.g., 0.01 for 1% shift).|


**Data Augmentation and Signal Normalization**

To enhance model generalization and robustness, a series of data augmentations are applied to each 1D signal during training. Each augmentation is applied independently with a specified probability (e.g., 80%).

-   **Amplitude Scaling**: The signal's amplitude is randomly scaled by a factor `α` drawn from a predefined uniform distribution.
    $$ x_{\text{scaled}}[t] = \alpha \cdot x[t] $$

-   **Time Shifting**: The input signal is cyclically shifted by a random integer offset `s`, ensuring no data points are lost.
    $$ x_{\text{shifted}}[t] = x[(t - s) \pmod{T}] $$

-   **Baseline Wander**: To simulate low-frequency physiological drift, a noise signal composed of 1 to 2 low-frequency sinusoids is added.
    $$ x_{\text{bw}}[t] = x[t] + \sum_{k=1}^{K} A_k \sin(2\pi f_k t + \phi_k), \quad K \in \{1, 2\} $$
    where amplitude $A_k \sim U(-A_{\max}, A_{\max})$, frequency $f_k \sim U(0, f_{\max})$, and phase $\phi_k \sim U(0, 2\pi)$.

-   **Gaussian Noise**: To model general sensor and electronic noise, zero-mean Gaussian noise is added.
    $$ x_{\text{gauss}}[t] = x[t] + \epsilon_t, \quad \text{where } \epsilon_t \sim \mathcal{N}(0, \sigma^2) $$

-   **High-Frequency Sinusoidal Noise**: To simulate electrical or muscular interference (e.g., from power lines or EMG artifacts), a high-frequency noise signal composed of 2 to 4 sinusoids is added.
    $$ x_{\text{hf}}[t] = x[t] + \sum_{k=1}^{K} A_k \sin(2\pi f_k t + \phi_k), \quad K \in \{2, 3, 4\} $$
    where amplitude $A_k \sim U(0, A_{\max})$, frequency $f_k$ is drawn from $\{1, 2, ..., 29\}$ Hz, and phase $\phi_k \sim U(0, 2\pi)$.

Following the augmentation pipeline, a two-step normalization is applied. This step is critical because the model is ultimately intended for MCG data, which has a vastly different absolute scale than ECG data.

1.  **Zero-Mean Normalization**: The augmented signal is centered by subtracting its mean.
    $$ x_{\text{zero}}[t] = x_{\text{aug}}[t] - \frac{1}{T} \sum_{i=0}^{T-1} x_{\text{aug}}[i] $$

2.  **Max-Absolute Scaling**: The zero-mean signal is scaled to lie within the range `[-1, 1]`.
    $$ x_{\text{norm}}[t] = \frac{x_{\text{zero}}[t]}{\max_i |x_{\text{zero}}[i]| + \varepsilon} $$
    where $x_{\text{aug}}$ is the augmented signal and $\varepsilon$ is a small constant (e.g., 10⁻⁶) to prevent division by zero.

<div align="center">
  <img src="https://github.com/user-attachments/assets/da91c08b-345b-404b-8ccc-1984da9f7444" alt="augmentation_result" width="100%">
  <br>
  <em>Example of the data augmentation pipeline applied to a sample from the training dataset. The original signal is transformed by adding noise, baseline wander, and other augmentations before being normalized.</em>
</div>


### 4.2. Testing and Visualization

The `data_loader.py` script includes a `__main__` block to verify dataset functionality. When run, it loads a sample, applies the full augmentation and normalization pipeline, and generates a plot showing the final signal and its corresponding ground truth labels. This is useful for visually inspecting the effect of the preprocessing steps.

## 5. Model Architecture (`model.py`)

This project implements and evaluates several deep learning architectures for ECG segmentation. The primary models, `Unet-1D-15M` and `Unet-1D-900k`, are based on a U-Net architecture with an integrated self-attention mechanism.

### 5.1. Model Overview

| Model Name         | # Parameters | Key Features                               |
|--------------------|--------------|--------------------------------------------|
| Unet-1D-15M        | ~15M         | 4-level U-Net, 8-head self-attention, dropout |
| Unet-1D-900k       | ~900k        | 3-level U-Net, 4-head self-attention, no dropout |
| MCG-Segmentator_s  | ~375k        | Multi-scale Conv, BiLSTM, Transformer      |
| MCG-Segmentator_xl | ~1.3M        | Larger version of MCG-Segmentator_s        |
| DENS-Model         | ~1.4M        | Conv layers followed by BiLSTM layers      |

### 5.2. Unet-1D-15M

This is a large-capacity model designed for high performance.
- **Architecture**: A 1D U-Net with a contracting path (encoder), a bottleneck, and an expansive path (decoder).
- **Encoder**: Four levels with channel sizes of [64, 128, 256, 512].
- **Bottleneck**: Integrates an 8-head **Multi-Head Self-Attention (MHSA)** mechanism [3] to capture global dependencies across the sequence. A dropout rate of 0.4 is applied for regularization.
- **Decoder**: Symmetrically mirrors the encoder, using transposed convolutions and skip connections to recover high-resolution details.
- **Output**: A final 1x1 convolutional layer maps features to the 4 output classes.
- **Parameters**: Approximately 15 million.

**Architecture Diagram**:
<div align="center">
  <img src="https://github.com/user-attachments/assets/6efcd9ba-54be-438d-9627-da8c27e2bcbe" alt="UNet Architecture" width="90%">
  <br>
  <em>Architecture of the U-Net-1D-15M model. The diagram illustrates the data flow and tensor shape transformations, where B is the batch size (e.g., 64) and T is the number of time steps (500, corresponding to a 2-second segment). The dimensions of the blocks represent the number of channels and the temporal length, though they are not drawn to scale.</em>
</div>


### 5.3. Unet-1D-900k

A lightweight and efficient version of the larger model.
- **Architecture**: Similar to the 15M model but with a shallower and narrower design.
- **Encoder**: Three levels with channel sizes of [32, 64, 128].
- **Bottleneck**: Employs a simpler 4-head attention mechanism and does not use dropout.
- **Decoder**: Three upsampling levels with skip connections.
- **Parameters**: Approximately 900,000.

### 5.4. MCG-Segmentator_s

The MCG-Segmentator_s is a smaller variant of the `ECGSegmenter` model, designed for efficiency in resource-constrained environments.

**Architecture**:

- **Positional Encoding**: Adds temporal context for sequences up to `max_seq_len`.
- **Initial Convolution**: Expands input to 16 channels (kernel size 7).
- **Multi-Scale Convolutions**: Kernels [3, 7, 15] for varied receptive fields.
- **Residual Blocks**: Four blocks with dilations [1, 2, 4, 8].
- **BiLSTM**: Single layer with 20 hidden units (bidirectional).
- **Transformer Encoder**: Single layer with 8 attention heads.
- **Skip Connection**: Combines early convolutional features with LSTM output.
- **Classifier**: Linear layer to `num_classes`.

**Architecture Diagram**:
<div align="center">
  <img src="https://github.com/user-attachments/assets/a6500515-d7b1-465f-85a0-fced5c35e874" alt="UNet Architecture" width="90%">
  <br>
  <em>Architecture of the MCGSegmenter_s model. The diagram illustrates the data flow and tensor shape transformations, where B is the batch size (e.g., 64) and T is the number of time steps (500, corresponding to a 2-second segment). The dimensions of the blocks represent the number of channels and the temporal length, though they are not drawn to scale.</em>
</div>

**Initialization**:

```python
from model.model import ECGSegmenter

model = ECGSegmenter(num_classes=4, input_channels=1, hidden_channels=16, lstm_hidden=20, dropout_rate=0.3, max_seq_len=2000)
```

### 5.5. MCG-Segmentator_xl

The MCG-Segmentator_xl is a larger variant of the `ECGSegmenter` model, designed for enhanced performance on complex ECG patterns.

**Architecture**:

- Similar to MCG-Segmentator_s but with increased capacity: 64 hidden channels and 64 LSTM hidden units.
- **Positional Encoding**: Same as MCG-Segmentator_s.
- **Initial Convolution**: Expands to 32 channels.
- **Multi-Scale Convolutions**: Same kernel sizes [3, 7, 15].
- **Residual Blocks**: Four blocks with dilations [1, 2, 4, 8].
- **BiLSTM**: Single layer with 64 hidden units (bidirectional).
- **Transformer Encoder**: Single layer with 8 attention heads.
- **Skip Connection**: Combines early features with LSTM output.
- **Classifier**: Linear layer to `num_classes`.

**Initialization**:

```python
from model.model import ECGSegmenter

model = ECGSegmenter(num_classes=4, input_channels=1, hidden_channels=64, lstm_hidden=64, dropout_rate=0.3, max_seq_len=2000)
```

### 5.6. DENS-Model

The DENS-Model, inspired by the DENS-ECG paper [4], combines convolutional and recurrent layers for robust ECG segmentation.

**Architecture**:

- **Convolutional Layers**: Three Conv1d layers with channels [32, 64, 128] (kernel size 3, padding 1).
- **BiLSTM Layers**: Two bidirectional LSTMs with hidden sizes 250 and 125.
- **Dropout**: Applied after BiLSTM (p=0.2).
- **Classifier**: Linear layer mapping to `num_classes`, with softmax applied during inference (implicitly handled by CrossEntropyLoss during training).

**Initialization**:

```python
from model.model import DENS_ECG_segmenter

model = DENS_ECG_segmenter(input_channels=1, num_classes=4)
```

## 6. Training (`trainer.py` and `train.py`)

This section describes the training pipeline implemented in `trainer.py` and `train.py`.

### 6.1. `Trainer` Class

The `Trainer` class, defined in `trainer.py`, manages the training and validation loops for the model, including metric logging, checkpoint saving, and detailed assessment of significant point detection in ECG signals.

#### 6.1.1. Initialization

The `Trainer` class is initialized with the following parameters:

```python
from trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    args=args,
    optimizer=optimizer,
    device=device,
    log_filepath="path/to/logs/training_metrics.csv",
    lr_scheduler=None,
    init_epoch=1
)
```

- **model**: The neural network model to be trained.
- **train_loader**: DataLoader for the training dataset.
- **val_loader**: DataLoader for the validation dataset.
- **args**: Configuration object containing hyperparameters (e.g., `num_epochs`, `clip`, `save_dir`).
- **optimizer**: The optimizer used for training (e.g., `torch.optim.Adam`).
- **device**: The device to run the model on (e.g., `cuda` or `cpu`).
- **log_filepath**: Path to the CSV file for logging metrics.
- **lr_scheduler**: Optional learning rate scheduler (defaults to `None`).
- **init_epoch**: Starting epoch number (defaults to 1).

The class initializes a log file (`training_metrics.csv`) and defines class names for ECG wave types: `No Wave` (0), `P Wave` (1), `QRS` (2), and `T Wave` (3).

#### 6.1.2. Key Methods

##### `train()`
Executes the training loop over multiple epochs. For each epoch:
- Computes loss using Focal Loss (`alpha=0.25`, `gamma=2.0`) [6].
- Updates model parameters with gradient clipping (if `args.clip > 0`).
- Tracks training loss and accuracy using a progress bar (`tqdm`).
- Calls `validate()` to evaluate the model on the validation set.
- Logs metrics and saves checkpoints (best model when validation F1-score improves, or every 5 epochs).
- Updates the learning rate if a scheduler is provided.

##### `validate()`
Evaluates the model on the validation set, computing:
- Average validation loss (using Focal Loss).
- Validation accuracy.
- Macro-averaged F1-score.
- Detailed classification report with precision, recall, and F1-score per class.
- Significant point detection metrics for ECG waves (onsets and offsets for P Wave, QRS, and T Wave) with a tolerance of ±150 ms.

The method also prints a detailed classification report and detection metrics, then returns the validation loss, accuracy, and macro F1-score.

##### `save_model(is_best=False)`
Saves the model, optimizer, and scheduler states, along with training parameters, to a checkpoint directory:
- If `is_best=True`, saves to `save_dir/best/`.
- Otherwise, saves to `save_dir/checkpoint_epoch_{epoch}/`.
- Saves `model.pth`, `optimizer.pth`, `lr_scheduler.pth` (if applicable), and `params.json`.

##### `_log_epoch_metrics(epoch_metrics: dict)`
Logs epoch metrics to the specified CSV file (`log_filepath`). The metrics are written as a row with the following headers:
- `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `val_f1_macro`, `learning_rate`

##### `_extract_significant_points(labels)`
Extracts onset and offset points for P Wave, QRS, and T Wave from a sequence of labels. Returns a dictionary with onsets and offsets for each wave type.

##### `_evaluate_detection_metrics(...)`
Computes detection metrics for significant points (onsets and offsets) of ECG waves:
- **True Positives (TP)**: Predicted points within ±150 ms of ground truth.
- **False Positives (FP)**: Predicted points without a matching ground truth point.
- **False Negatives (FN)**: Ground truth points without a matching predicted point.
- **Metrics per wave type (P Wave, QRS, T Wave)**:
  - Sensitivity (Se): `TP / (TP + FN)`.
  - Precision (PPV): `TP / (TP + FP)`.
  - F1-score: `2 * PPV * Se / (PPV + Se)`.
  - Mean and standard deviation of errors (in samples and milliseconds).
- Assumes a default sample rate of 250 Hz and a tolerance of 150 ms.

##### `focal_loss(logits, labels, alpha=0.25, gamma=2.0)`
Implements Focal Loss [6] to address class imbalance:
- **alpha**: Weighting factor for rare classes (default: 0.25).
- **gamma**: Focusing parameter to reduce the impact of easy examples (default: 2.0).
- Computes loss using softmax probabilities and adds a small epsilon (`1e-9`) to prevent `log(0)` errors.

#### 6.1.3. Logged Metrics

Metrics are logged to `logs/training_metrics.csv` with the following columns:

| Metric            | Description                                |
|-------------------|--------------------------------------------|
| `epoch`           | Current epoch number.                      |
| `train_loss`      | Average training loss for the epoch.       |
| `train_acc`       | Average training accuracy for the epoch.   |
| `val_loss`        | Average validation loss for the epoch.     |
| `val_acc`         | Average validation accuracy for the epoch. |
| `val_f1_macro`    | Macro-averaged F1-score on validation set. |
| `learning_rate`   | Current learning rate.                     |

### 6.2. Training Script (`train.py`)

The `train.py` script configures and executes the training pipeline. It handles command-line argument parsing, dataset and DataLoader setup, model initialization, and integration with the `Trainer` class. The script supports resuming training from checkpoints and includes robust error handling.

#### 6.2.1. Usage Example

```bash
python train.py \
    --data_dir_train Datasets/train \
    --data_dir_val Datasets/val \
    --save_dir trained_models/Unet-1D-900k/checkpoints \
    --metrics_file trained_models/Unet-1D-900k/logs/training_metrics.csv \
    --model_name Unet-1D-900k \
    --num_epochs 100 \
    --batch_size 64 \
    --max_lr 1e-3 \
    --sequence_length 500 \
    --overlap 400 \
    --augmentation_prob 0.8
```

#### 6.2.2. Key Arguments

| Argument                  | Type  | Default                                    | Description                                               |
|---------------------------|-------|--------------------------------------------|-----------------------------------------------------------|
| `--num_epochs`            | int   | 50                                         | Number of training epochs.                                |
| `--batch_size`            | int   | 64                                         | Training batch size.                                      |
| `--val_batch_size`        | int   | 4                                          | Validation batch size.                                    |
| `--max_lr`                | float | 1e-3                                       | Maximum learning rate for scheduler.                      |
| `--base_lr`               | float | 1e-5                                       | Minimum learning rate for cosine annealing.               |
| `--clip`                  | float | 1.0                                        | Gradient clipping value (0 to disable).                   |
| `--from_check_point`      | bool  | False                                      | Resume training from a checkpoint.                        |
| `--load_dir`              | str   | `MCG_segmentation/checkpoints`             | Directory to load checkpoint from.                        |
| `--save_dir`              | str   | `MCG_segmentation/checkpoints`             | Directory for saving new checkpoints.                     |
| `--data_dir_train`        | str   | `MCG_segmentation/Datasets/train`          | Training data directory.                                  |
| `--data_dir_val`          | str   | `MCG_segmentation/Datasets/val`            | Validation data directory.                                |
| `--sequence_length`       | int   | 500                                        | Input sequence length for the model.                      |
| `--overlap`               | int   | 400                                        | Overlap when creating sequences from files.               |
| `--num_workers`           | int   | 4                                          | Number of DataLoader workers.                             |
| `--augmentation_prob`     | float | 0.80                                       | Probability of applying augmentations during training.    |
| `--sinusoidal_noise_mag`  | float | 0.04                                       | Magnitude of sinusoidal noise for augmentation.           |
| `--gaussian_noise_std`    | float | 0.04                                       | Standard deviation of Gaussian noise for augmentation.    |
| `--baseline_wander_mag`   | float | 0.10                                       | Magnitude of baseline wander for augmentation.            |
| `--metrics_file`          | str   | `MCG_segmentation/logs/training_metrics.csv` | Path for CSV logging of metrics.                        |

#### 6.2.3. Script Overview

The script initializes datasets, DataLoaders, the model, optimizer, and scheduler. It supports resuming from checkpoints and calls the `Trainer` class to execute the main training loop, handling errors and interruptions gracefully by saving the current model state.

## 7. Evaluation (`evaluate.py`)

The `evaluate.py` script evaluates a trained ECG segmentation model on a test dataset, computing metrics and generating visualizations to assess performance according to the **AAMI standard**. It supports loading checkpoints, processing ECG data, and producing detailed event-based and sample-wise metrics.

### 7.1. Usage Example

```bash
python evaluate.py \
    --load_dir trained_models/MCGSegmenter_s \
    --data_dir_eval Datasets/test \
    --output_dir trained_models/MCGSegmenter_s/evaluation_results \
    --sequence_length 1250 \
    --eval_batch_size 16 \
    --num_workers 4
```

### 7.2. Key Arguments

| Argument            | Type  | Default (example)                                           | Description                                                                     |
|---------------------|-------|-------------------------------------------------------------|---------------------------------------------------------------------------------|
| `--load_dir`        | str   | `.../trained_models/MCGSegmenter_s`            | Directory containing the model checkpoint (e.g., `checkpoints/best/model.pth`). |
| `--data_dir_eval`   | str   | `.../Datasets/test`                             | Evaluation data directory.                                                      |
| `--output_dir`      | str   | `.../trained_models/MCGSegmenter_s/evaluation_results` | Directory for saving evaluation outputs.                                        |
| `--eval_batch_size` | int   | 16                                                          | Batch size for evaluation.                                                      |
| `--sequence_length` | int   | 1250                                                        | Sequence length for evaluation (e.g., 5 seconds at 250 Hz).                     |
| `--num_workers`     | int   | 4                                                           | Number of DataLoader workers.                                                   |

### 7.3. Outputs

Saved in `output_dir`:
- **Confusion Matrix** (`confusion_matrix.png`): A heatmap showing the sample-wise confusion matrix as percentages.
- **Sample Plot** (`ecg_segment_random_batch_X.pdf`): A plot displaying the ECG signal with predicted segments and ground truth labels.
- **Console Output**: Includes overall accuracy, a detailed classification report, and event-based delineation metrics.

### 7.4. Evaluation Metrics

The evaluation follows AAMI standards, assessing onsets and offsets of P-Wave, QRS, and T-Wave within a **150 ms tolerance window** (37.5 samples at 250 Hz).

- **Significant Point Detection**: TP, FP, and FN are counted for onsets and offsets.
- **Error Metrics**: Mean Error (m) and Standard Deviation of Error (σ) are calculated.
- **Performance Metrics**: Sensitivity (Se), Positive Predictive Value (PPV), F1-Score, and sample-wise accuracy are reported.

### 7.5. Post-Processing

Predictions are refined before evaluation:
- Segments shorter than **40 ms** (10 samples at 250 Hz) are reassigned to the label of their neighbors. This removes physiologically implausible artifacts.

### 7.6. Evaluation Process

The script loads the best model checkpoint, prepares the test dataset (with augmentations disabled), and processes the data in batches. It generates visualizations, computes sample-wise and event-based metrics, and prints the results to the console.

## 8. Training Process and Results

### 8.1. General Training Setup

- **UNet Models (`-15M`, `-900k`)**: The datasets were partitioned into training, validation, and test sets using a strict patient-level split to prevent data leakage across sets. The test set was composed exclusively of records from 40\% of the LUDB patients. The validation set, used to guide model selection, was formed from 5\% of patients from the QTDB and 5\% of the remaining LUDB patients. Consequently, the final training set consisted of all other records, comprising 95\% of suitable QTDB patients and the remaining 55\% of LUDB patients.
- **Other Models (`DENS`, `MCG-Segmentator`)**: Trained and validated on the QTDB dataset only and tested on the entire LUDB dataset. This difference in training data diversity should be considered when comparing performance.
- **Training Parameters**: Models were trained for up to 100 epochs with AdamW [5], cosine annealing LR, and Focal Loss [6]. Input sequences were 500 samples (2 seconds at 250 Hz) with an overlap of 400 samples, and data augmentation was applied with a probability of 80% during training.

<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/943e5f4a-0d0a-41c2-98b1-42d1544a8daf" alt="accuracy" width="95%">
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/dce211c5-c293-4279-ace8-9f79d56025d6" alt="lr_scheduler" width="95%">
      </td>
    </tr>
  </table>
  <br>
  <em><b>(a)</b> Training and validation accuracy curves over 100 epochs. As shown, the peak validation performance is reached around epoch 40. <b>(b)</b> The cosine annealing learning rate schedule, decaying from 10<sup>-3</sup> to 10<sup>-5</sup>.</em>
</div>

 To select the optimal model and prevent overfitting, a strategy of early stopping was employed. The selection criterion was the sample-wise F1-score on the validation set, which evaluates the classification performance for each individual time step in the signal. The model weights from the epoch yielding the highest sample-wise F1-score were then retained for the final, event-based evaluation.

### 8.2. Summary of Model Performance

The following table presents the delineation performance of the proposed UNet models against several state-of-the-art algorithms on the LUDB test set. Performance is measured by Sensitivity (Se), Positive Predictive Value (PPV), F1-Score, and the mean error with standard deviation (m ± σ) in milliseconds.

**Table: Delineation performance of the proposed models and other state-of-the-art algorithms on the LUDB test set.**
| Model | Metric | P onset | P offset | QRS onset | QRS offset | T onset | T offset |
|:--- |:--- |:---:|:---:|:---:|:---:|:---:|:---:|
| **Kalyakulina et al. [8]** | Se (%) | 98.46 | 98.46 | 99.61 | 99.61 | – | 98.03 |
| | PPV (%) | 96.41 | 96.41 | 99.87 | 99.87 | – | 98.84 |
| | F1 (%) | 97.42 | 97.42 | 99.74 | 99.74 | – | 98.43 |
| | m±σ (ms) | -2.7±10.2 | 0.4±11.4 | -8.1±7.7 | 3.8±8.8 | – | 5.7±15.5 |
| **Sereda et al. [10]** | Se (%) | 95.20 | 95.39 | 99.51 | 99.50 | 97.95 | 97.56 |
| | PPV (%) | 82.66 | 82.59 | 98.17 | 97.96 | 94.81 | 94.96 |
| | F1 (%) | 88.49 | 88.53 | 98.84 | 98.72 | 96.35 | 96.24 |
| | m±σ (ms) | 2.7±21.9 | -7.4±28.6 | 2.6±12.4 | -1.7±14.1 | 8.4±28.2 | -3.1±28.2 |
| **Moskalenko et al. [11]** | Se (%) | 97.38 | 97.36 | 99.96 | 99.96 | 99.43 | 99.48 |
| | PPV (%) | 95.53 | 95.52 | 99.84 | 99.84 | 98.88 | 98.94 |
| | F1 (%) | 96.47 | 96.43 | 99.90 | 99.90 | 99.15 | 99.21 |
| | m±σ (ms) | 0.9±14.1 | -3.5±15.7 | 2.1±9.8 | 1.6±9.8 | 1.3±20.9 | -0.3±22.9 |
| **Joung et al. [12]** | Se (%) | 98.16 | 98.20 | 99.67 | 99.97 | 99.82 | 99.63 |
| | PPV (%) | 96.39 | 96.36 | 99.29 | 99.59 | 99.66 | 99.42 |
| | F1 (%) | 97.24 | 97.27 | 99.48 | 99.78 | 99.74 | 99.52 |
| | m±σ (ms) | 7.4±14.1 | -1.8±9.9 | 6.1±10.5 | 2.0±10.7 | 3±25.2 | 4.5±24.4 |
| **DENS-Model [4]** | Se (%) | 94.41 | 94.58 | 95.62 | 96.00 | 97.65 | 97.61 |
| | PPV (%) | 87.25 | 87.41 | 99.41 | 99.80 | 97.59 | 97.55 |
| | **F1 (%)** | **90.69** | **90.86** | **97.47** | **97.86** | **97.62** | **97.58** |
| | m±σ (ms) | -14.1±28.1 | 0.7±23.9 | 1.2±10.5 | -2.9±14.3 | -25.2±38.6 | 10.4±33.6 |
| **This work (U-Net-1D-900k)** | Se (%) | 95.58 | 95.60 | 99.37 | 99.50 | 97.84 | 97.67 |
| | PPV (%) | 94.88 | 94.90 | 99.70 | 99.83 | 98.33 | 98.15 |
| | **F1 (%)** | **95.23** | **95.25** | **99.53** | **99.67** | **98.08** | **97.91** |
| | m±σ (ms) | 2.5±17.3 | 1.3±16.5 | 0.2±7.4 | -1.2±10.7 | -0.3±26.5 | -0.8±24.2 |
| **This work (U-Net-1D-15M)** | Se (%) | 96.12 | 96.12 | 99.30 | 99.43 | 98.91 | 98.72 |
| | PPV (%) | 96.10 | 96.10 | 99.55 | 99.68 | 98.98 | 98.79 |
| | **F1 (%)** | **96.11** | **96.11** | **99.42** | **99.56** | **98.95** | **98.75** |
| | m±σ (ms) | -0.2±17.2 | 2.5±16.3 | 1.0±7.8 | -1.5±10.6 | -0.5±26.1 | 3.2±23.0 |
| **This work (MCG Segmenter s)** | Se (%) | 96.17 | 96.22 | 97.08 | 97.45 | 97.46 | 97.22 |
| | PPV (%) | 89.45 | 89.50 | 99.23 | 99.62 | 97.74 | 97.49 |
| | **F1 (%)** | **92.69** | **92.74** | **98.15** | **98.52** | **97.60** | **97.36** |
| | m±σ (ms) | -8.2±25.3| 1.5±21.9 | 1.4±11.5| -3.6±14.5 | -19.05±36.86 | 11.1±30.2 |
| **This work (MCG Segmenter xl)** | Se (%) | 95.71 | 95.75 | 96.94 | 97.35 | 97.44 | 97.22 |
| | PPV (%) | 90.00 | 90.03 | 99.34 | 99.76 | 97.55 | 97.33 |
| | **F1 (%)** | **92.77** | **92.80** | **98.13** | **98.54** | **97.50** | **97.28** |
| | m±σ (ms) | -5.8±23.6 | 2.3±21.1 | 1.6±10.8 | -3.9±14.7 | -21.3±38.6 | 13.1±31.7 |

#### 8.2.1. Discussion

- **Performance**: The proposed UNet architectures achieve performance comparable to state-of-the-art solutions, even when trained on a less diverse dataset and with significantly shorter input intervals. This is likely attributable to the **Multi-Head Self-Attention** mechanism [3], which effectively captures long-range dependencies in the signal.
- **P-Wave Detection**: A notable weakness, however, is observed in P-wave delineation. Compared to state-of-the-art results, both proposed models show a lower $F_1$-Score, indicating a higher rate of false positive and false negative detections. This performance gap is likely a direct consequence of the training data composition. The inclusion of the QTDB, where nearly every annotated heartbeat contains a P-wave, appears to have induced a model bias. This leads the model to over-predict P-waves when evaluated on the more diverse LUDB test set, which includes arrhythmic recordings where P-waves may be absent. This inclusion was a deliberate trade-off. Leveraging the QTDB was found to be crucial for achieving robust performance on the primary target data (magnetocardiography signals), as a subset of the LUDB recordings exhibits high-frequency and atypically sharp artifacts, not representative of typical ECG/MCG signals.
- **Model Efficiency**: The `Unet-1D-900k` model performs only marginally worse than its 15M-parameter counterpart and other much larger models from the literature. This demonstrates its high efficiency and suitability for deployment on resource-constrained infrastructure.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e28cfee9-f8db-44f3-9681-8abfa5e51d85" alt="ECG Segmentation Performance" width="85%">
  <br>
  <em>Segmentation performance of the U-Net-1D-900k model on a LUDB sample from the test set. This figure illustrates the type of sharp, high-frequency artifacts present in some LUDB recordings, which can pose a challenge for generalizing to MCG signals.</em>
</div>
<div align="center">
  <img  width="85%" src="https://github.com/user-attachments/assets/f071c2bf-fa0d-498e-bcbe-e0b318f436e5">
  <br>
  <em>Example segmentation performance of the U-Net-1D-15M on a random sample from the test dataset. The model correctly identifies the P-wave, QRS complex, and an inverted T-wave, demonstrating its robustness to pathological signals.</em>
</div>
<div align="center">
  <img src="https://github.com/user-attachments/assets/8e15773b-631a-43b0-8128-89116f7e0a55" alt="ECG Segmentation Performance" width="85%">
  <br>
  <em>Segmentation of a noisy single-channel MCG signal using the trained U-Net-1D-15M model. Despite the rather high noise level, the model successfully delineates the P-wave, QRS complex, and T-wave.</em>
</div>





## 9. Possible Applications to Magnetocardiography and ECG

Beyond its primary function as a segmentation tool, the trained U-Net model can be leveraged for several practical downstream applications that enhance cardiac signal analysis.

**Robust Peak Detection**
The high F1-score of the model's QRS complex detection (>98%) enables a more robust peak detection methodology. By constraining the search for the signal's maximum absolute value (the R-peak) to only those intervals identified as QRS segments, the algorithm's reliability is significantly improved, preventing erroneous detection of noise spikes. This is particularly crucial for accurately timing cardiac events in noisy MCG recordings.

<div align="center">
  <img width="90%" alt="qrs_detection" src="https://github.com/user-attachments/assets/19ac67c6-d7c7-4f57-b141-4a4e920db735" />
  <br>
  <em>Illustration of robust R-peak detection on a noisy MCG signal. The search for the peak is constrained to the interval identified by the model as the QRS complex, preventing erroneous detection of noise spikes.</em>
</div>


**A Novel "Heartbeat Score" for Signal Quality Assessment**
For multi-channel recordings like MCG, it is beneficial to automatically identify the channel with the cleanest signal. To automate this, a novel "Heartbeat Score" is introduced. This metric quantifies the likelihood that a signal segment represents a plausible cardiac rhythm by assigning a score between 0 and 1, based on two criteria:

1.  **Prediction Confidence**: The model's softmax output probabilities are aggregated to gauge the certainty of its classifications. Higher average probabilities suggest a more confident prediction and thus a higher quality signal.
2.  **Physiological Plausibility**: The relative proportions of the detected P, QRS, and T segments are compared against physiologically expected ranges for a typical heartbeat (e.g., P-wave: 8-15%, QRS: 8-15%, T-wave: 15-30%). Signals with segment distributions that deviate significantly from these norms receive a lower score.

The final score combines these two components, providing a single metric to rank channels by signal quality.

**ICA Filtering for Signal Quality Improvement**
The "Heartbeat Score" can be used to improve **Independent Component Analysis (ICA)** filtering. ICA decomposes a multi-channel signal into statistically independent components, some of which may represent noise. By calculating the Heartbeat Score for each independent component, we can identify and remove components that are unlikely to originate from a human heartbeat (i.e., those with low confidence or implausible segment distributions). This enhances the signal-to-noise ratio of the reconstructed signal by isolating components that align with expected cardiac patterns.

This approach has been successfully implemented and demonstrated in a related project, available at: [https://github.com/Drizzr/Bachelorarbeit](https://github.com/Drizzr/Bachelorarbeit).

## 10. References

[1] Sattar, Y., & Chhabra, L. (2023). Electrocardiogram. In *StatPearls [Internet]*. StatPearls Publishing.

[2] Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220.

[3] Vaswani, A., et al. (2017). Attention Is All You Need. *arXiv preprint arXiv:1706.03762*.

[4] Peimankar, A., & Puthusserypady, S. (2021). DENS-ECG: A deep learning approach for ECG signal delineation. *Expert Systems with Applications*, 165, 113911.

[5] Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic Optimization. *arXiv preprint arXiv:1412.6980*.

[6] Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *arXiv preprint arXiv:1708.02002*.

[7] Laguna, P., et al. (1997). A database for evaluation of algorithms for measurement of QT and other waveform intervals in the ECG. *Computers in Cardiology*, 24, 673-676.

[8] Kalyakulina, A. I., et al. (2020). LUDB: A New Open-Access Validation Tool for Electrocardiogram Delineation Algorithms. *IEEE Access*, 8, 186181-186190.

[9] Kalyakulina, A., et al. (2021). Lobachevsky University Electrocardiography Database (version 1.0.1). *PhysioNet*.

[10] Sereda, I., et al. (2018). ECG Segmentation by Neural Networks: Errors and Correction. *arXiv preprint arXiv:1812.10386*.

[11] Moskalenko, V., et al. (2020). Deep Learning for ECG Segmentation. *arXiv preprint arXiv:2001.04689*.

[12] Joung, C., et al. (2024). Deep learning based ECG segmentation for delineation of diverse arrhythmias. *PLOS ONE*, 19(6), e0303178.

[13] Kanhe, R. K., & Hamde, S. T. (2014). Wavelet-based compression of ECG signals. *International Journal of Bioelectromagnetism*, 16(4), 297-314.
