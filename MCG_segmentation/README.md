# ECG Wave Segmentation Project

This project focuses on the segmentation of electrocardiogram (ECG) signals into distinct physiological waveforms: P wave, QRS complex, T wave, and periods of no significant wave activity (labeled as 'No Wave'). It utilizes deep learning models trained on the QT Database (QTDB) from PhysioNet.

The primary goal is to accurately classify each time point in an ECG signal into one of these categories. The project includes scripts for data preprocessing, model training, and evaluation. Two different neural network architectures are implemented and compared for this task.

**Label Mapping:**
*   0: No Wave (Baseline or Isoelectric Line)
*   1: P Wave
*   2: QRS Complex
*   3: T Wave

## Features

*   **Data Preprocessing:** Downloads and processes the QTDB dataset, extracts waveforms and labels, handles annotations, and splits data into training and validation sets (`preprocess_data.py`).
*   **Data Augmentation:** Includes various augmentation techniques during training (applied in `model/data_loader.py`):
    *   Sinusoidal Noise
    *   Gaussian Noise
    *   Baseline Wander
    *   Amplitude Scaling
    *   Time Shifting
*   **Model Architectures:** Implements two distinct deep learning models for segmentation (`model/model.py`):
    *   `ECGSegmenter`: A custom architecture involving multi-scale convolutions, residual blocks, BiLSTM, and self-attention.
    *   `DENS_ECG_segmenter`: A simpler architecture based on CNN layers followed by BiLSTM layers.
*   **Training:** Flexible training script (`train.py`) with features like:
    *   Checkpointing (saving model, optimizer, and scheduler states).
    *   Resuming training from checkpoints.
    *   Learning rate scheduling (Cosine Annealing).
    *   Gradient Clipping.
    *   Logging metrics (Loss, Accuracy, F1-score) to a CSV file (`logs/training_metrics.csv`).
    *   Saving the best model based on validation F1-score.
*   **Evaluation:** Script (`evaluate.py`) to evaluate a trained model on a dataset, providing:
    *   Overall Loss and Accuracy.
    *   Detailed Classification Report (Precision, Recall, F1-score per class).
    *   Confusion Matrix visualization.
    *   Optional plotting of individual sample predictions against ground truth.

## Dataset

This project uses the **QT Database (QTDB)** from PhysioNet. The `preprocess_data.py` script handles downloading the raw data using the `wfdb` library and processing it into a suitable format.

Preprocessing steps include:
1.  Reading raw `.dat`, `.hea`, and annotation files.
2.  Extracting signal data and annotations (P, QRS, T wave onsets, peaks, offsets).
3.  Mapping annotations to time-point labels (0, 1, 2, 3).
4.  Handling gaps and potential overlaps between annotated waves.
5.  Splitting records into training and validation sets based on a specified ratio (default 80/20).
6.  Saving processed data (signal segments and corresponding labels) as CSV files in `MCG_segmentation/qtdb/processed/train` and `MCG_segmentation/qtdb/processed/val`.

## Model Architectures

Two architectures are implemented for comparison:

### 1. `ECGSegmenter`

This is a more complex custom architecture designed to capture features at multiple scales and leverage temporal dependencies effectively. Its key components include:
*   **Positional Encoding:** Adds information about the position of each time step.
*   **Initial Convolution:** Increases the number of channels and extracts initial features.
*   **Multi-Scale Convolutions:** Uses parallel convolutional layers with different kernel sizes (3, 7, 15) to capture patterns at various resolutions.
*   **Residual Blocks with Dilation:** Employs stacked residual blocks with increasing dilation factors (1, 2, 4, 8) to expand the receptive field without losing resolution.
*   **Bidirectional LSTM (BiLSTM):** Processes the sequence in both forward and backward directions to model temporal context.
*   **Transformer Encoder Layer (Self-Attention):** Allows the model to weigh the importance of different time steps within the sequence when making predictions for a specific point.
*   **Skip Connection:** Adds features from an earlier layer to the output of the attention block, potentially helping with gradient flow and preserving lower-level details.
*   **Final Classifier:** A linear layer outputs the probability distribution over the classes for each time step.

### 2. `DENS_ECG_segmenter`

This architecture follows a more standard pattern often seen in sequence modeling tasks:
*   **1D Convolutional Layers:** A stack of 1D convolutional layers (with increasing channel depth: 32 -> 64 -> 128) acts as a feature extractor, learning local patterns in the signal. ReLU activation is used.
*   **Bidirectional LSTM (BiLSTM) Layers:** Two stacked BiLSTM layers (hidden sizes 250 -> 125, bidirectional) process the features extracted by the CNNs, capturing longer-range temporal dependencies.
*   **Dropout:** Applied after the BiLSTMs to reduce overfitting.
*   **TimeDistributed Dense Layer:** A final linear layer (classifier) is applied independently to the output of the BiLSTM at each time step to produce the class predictions. Softmax activation is applied.

### Comparison

*   **Complexity:** `ECGSegmenter` is significantly more complex, incorporating multi-scale processing, dilated convolutions, and self-attention, while `DENS_ECG_segmenter` uses a more straightforward CNN-RNN structure.
*   **Feature Extraction:** `ECGSegmenter` explicitly uses multi-scale convolutions and dilated convolutions for potentially richer feature extraction across different temporal ranges. `DENS_ECG_segmenter` relies on standard stacked CNNs.
*   **Sequence Modeling:** Both use BiLSTMs. `ECGSegmenter` adds a self-attention mechanism on top, allowing potentially more sophisticated modeling of relationships between distant time steps compared to the standard BiLSTM context.
*   **Potential Strengths:** `ECGSegmenter` might be better at handling signals with variations in wave morphology or timing due to its multi-scale and attention features. `DENS_ECG_segmenter` might be faster to train and less prone to overfitting on smaller datasets due to its relative simplicity.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch numpy pandas wfdb scikit-learn matplotlib seaborn tqdm
    ```
    *(Alternatively, if you create a `requirements.txt` file, use `pip install -r requirements.txt`)*

4.  **Data Preparation:** The first run of `preprocess_data.py` will attempt to download the QTDB data if it's not found in `MCG_segmentation/qtdb/raw`. Ensure you have an internet connection.

## Usage

### 1. Preprocess Data

Run the preprocessing script. This only needs to be done once.
```bash
python preprocess_data.py
Use code with caution.
Markdown
This will download the QTDB data (if needed) into MCG_segmentation/qtdb/raw and create processed CSV files in MCG_segmentation/qtdb/processed/train and MCG_segmentation/qtdb/processed/val.

2. Train a Model
Run the training script. Adjust arguments as needed.

python train.py \
    --num_epochs 60 \
    --batch_size 64 \
    --max_lr 1e-4 \
    --base_lr 1e-5 \
    --sequence_length 500 \
    --save_dir MCG_segmentation/checkpoints/my_experiment \
    --metrics_file MCG_segmentation/logs/my_experiment_metrics.csv \
    --data_dir_train MCG_segmentation/qtdb/processed/train \
    --data_dir_val MCG_segmentation/qtdb/processed/val
Use code with caution.
Bash
To resume training from a checkpoint:

python train.py \
    --from_check_point \
    --load_dir MCG_segmentation/checkpoints/my_experiment/checkpoint_epoch_X \
    --save_dir MCG_segmentation/checkpoints/my_experiment \
    --metrics_file MCG_segmentation/logs/my_experiment_metrics.csv
    # Add other arguments if they changed, otherwise they might be loaded from params.json
Use code with caution.
Bash
(Replace checkpoint_epoch_X with the specific checkpoint directory)

3. Evaluate a Model
Evaluate a trained model (e.g., the best saved model).

python evaluate.py \
    --load_dir MCG_segmentation/checkpoints/my_experiment/best \
    --data_dir_eval MCG_segmentation/qtdb/processed/val \
    --output_dir MCG_segmentation/evaluation_results/my_experiment_best \
    --sequence_length 500 \
    --eval_batch_size 128
Use code with caution.
Bash
To plot a specific sample from the evaluation set:

python evaluate.py \
    --load_dir MCG_segmentation/checkpoints/my_experiment/best \
    --data_dir_eval MCG_segmentation/qtdb/processed/val \
    --output_dir MCG_segmentation/evaluation_results/my_experiment_best \
    --sequence_length 500 \
    --plot_sample_index 42  # Index of the sample in the dataset
Use code with caution.
Bash
Results and Evaluation
This section presents the performance comparison between the ECGSegmenter and DENS_ECG_segmenter models.

(Note: Add your actual results, tables, and plots here)

Performance Metrics
Model	Overall Accuracy	Macro F1-Score	F1 - No Wave	F1 - P Wave	F1 - QRS	F1 - T Wave
ECGSegmenter	[Your Value]	[Your Value]	[Your Value]	[Your Value]	[Value]	[Your Value]
DENS_ECG_segmenter	[Your Value]	[YourValue]	[Your Value]	[Your Value]	[Value]	[Your Value]
Confusion Matrices
ECGSegmenter:

[Placeholder for Confusion Matrix Plot - ECGSegmenter]
Use code with caution.
(Example:
![alt text](MCG_segmentation/evaluation_results/ECGSegmenter/confusion_matrix.png)
)

DENS_ECG_segmenter:

[Placeholder for Confusion Matrix Plot - DENS_ECG_segmenter]
Use code with caution.
(Example:
![alt text](MCG_segmentation/evaluation_results/DENS_ECG/confusion_matrix.png)
)

Sample Prediction Plots
ECGSegmenter:

[Placeholder for Sample Prediction Plot - ECGSegmenter]
Use code with caution.
(Example:
![alt text](MCG_segmentation/evaluation_results/ECGSegmenter/sample_X.png)
)

DENS_ECG_segmenter:

[Placeholder for Sample Prediction Plot - DENS_ECG_segmenter]
Use code with caution.
(Example:
![alt text](MCG_segmentation/evaluation_results/DENS_ECG/sample_Y.png)
)

Learning Curves
[Placeholder for Learning Curve Plot (Loss vs Epoch)]
Use code with caution.
(Example:
![alt text](MCG_segmentation/logs/loss_curves.png)
)

[Placeholder for Learning Curve Plot (F1-Score vs Epoch)]
Use code with caution.
(Example:
![alt text](MCG_segmentation/logs/f1_curves.png)
)

License
[Specify Your License Here, e.g., MIT License]

**Remember to:**

1.  Replace `<your-repository-url>` with the actual URL if you host it online.
2.  Fill in the `[Your Value]` placeholders in the Results section with your actual metrics once you run the evaluations.
3.  Replace the `[Placeholder for ... Plot]` text with markdown image links (`![alt text](path/to/image.png)`) pointing to your generated plots once you have them. Make sure the paths are correct relative to the README file or provide full URLs if hosted elsewhere.
4.  Choose and specify a license in the License section.
Use code with caution.
