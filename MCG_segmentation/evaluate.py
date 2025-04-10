# evaluate.py

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
# --- Import the correct model ---
from model.model import Conv1D_BiLSTM_Segmenter # Import the 1D model
# --- Import dataset class (output format changed) ---
from model.data_loader import ECGFullDataset
import sys
import json
import os
import numpy as np
from tqdm import tqdm # For progress bar
import sklearn.metrics as metrics # For detailed metrics
import matplotlib.pyplot as plt
import seaborn as sns # For plotting confusion matrix

# --- Constants ---
# Class mapping MUST match training
CLASS_NAMES_MAP = {
    0: "No Wave",
    1: "P Wave",
    2: "QRS",
    3: "T Wave"
}
# Colors for plotting segments/dots
PLOT_CLASS_COLORS = { # Used for dots
    0: "silver", 1: "blue", 2: "red", 3: "green",
}
SEGMENT_COLORS = { # Used for background
    0: "whitesmoke", 1: "lightblue", 2: "lightcoral", 3: "lightgreen",
}
LABEL_TO_IDX = {v: k for k, v in CLASS_NAMES_MAP.items()}

# Model checkpoint path resolution (assuming subdir structure)
# The script expects --load_dir to point to a specific subdir like 'best' or 'checkpoint_epoch_N'

# Device setup
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
DEVICE = torch.device("cpu") # Force CPU
print(f"Using device: {DEVICE}")
# --- End Constants ---


# --- Updated Load Model Function ---
def load_model_for_evaluation(load_dir, device):
    """Loads the 'best' or specified Conv1D_BiLSTM_Segmenter model."""
    load_directory = load_dir # Expecting path like .../checkpoints_1d/best
    print(f"Attempting to load model checkpoint from: {load_directory}")

    if not load_directory or not os.path.isdir(load_directory):
        raise FileNotFoundError(f"Checkpoint directory not found or not specified: {load_directory}")

    model_path = os.path.join(load_directory, "model.pth")
    param_path = os.path.join(load_directory, "params.json")

    if not os.path.exists(model_path) or not os.path.exists(param_path):
        raise FileNotFoundError(f"model.pth or params.json not found in {load_directory}")

    # Load parameters saved during training
    print(f"Loading parameters from: {param_path}")
    try:
        with open(param_path, "r") as f:
            params = json.load(f)
            checkpoint_args = params.get("args", {})
            # Load critical architecture parameters from checkpoint args
            num_classes = checkpoint_args.get("num_classes", 4) # Default if missing
            dropout_rate = checkpoint_args.get("dropout_rate", 0.2) # Default if missing
            # --- Load other model-specific args if they were saved ---
            # Example: These defaults should match Conv1D_BiLSTM_Segmenter if not saved
            cnn_filters_loaded = checkpoint_args.get("cnn_filters", (32, 64, 128))
            lstm_units_loaded = checkpoint_args.get("lstm_units", (250, 125))
            input_channels_loaded = checkpoint_args.get("input_channels", 1) # Usually 1

            print(f"Loaded model args: num_classes={num_classes}, dropout={dropout_rate}")
            # Print others if loaded: print(f" cnn_filters={cnn_filters_loaded}, lstm_units={lstm_units_loaded}")

    except Exception as e:
         raise IOError(f"Error reading parameter file {param_path}: {e}")

    # --- Instantiate the CORRECT 1D Model ---
    model = Conv1D_BiLSTM_Segmenter(
        num_classes=num_classes,
        input_channels=input_channels_loaded,
        cnn_filters=cnn_filters_loaded,
        lstm_units=lstm_units_loaded,
        dropout_rate=dropout_rate
        # Ensure all required __init__ args are provided
    )

    # Load model state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model state loaded successfully from {model_path}")
    except FileNotFoundError: raise FileNotFoundError(f"Model state file not found: {model_path}")
    except RuntimeError as e: print("\n*** Error loading state_dict: Model definition likely mismatch! ***"); raise e
    except Exception as e: raise RuntimeError(f"Error loading model state dict: {e}")

    model.to(device)
    model.eval() # Set model to evaluation mode

    # Return the model and key parameters needed for data loading (less relevant now)
    loaded_params = {
        "num_classes": num_classes,
        "sequence_length": checkpoint_args.get("sequence_length", 250), # From saved args
        "overlap": checkpoint_args.get("overlap", 125), # From saved args
    }
    return model, loaded_params
# --- End Load Model Function ---


# --- REMOVED preprocess_signal_for_model function ---


# --- Modified Plot Function for 1D Signal + Segments ---
def plot_evaluation_sample(signal_1d, true_labels, predicted_labels, output_dir, class_names_map, sample_index=0, sequence_length=None):
    """
    Generates a 2-panel plot:
    1. Processed 1D signal with true labels as dots and predicted labels as background.
    2. True vs Predicted label sequences as step plots.
    """
    print(f"\nGenerating plot for evaluation sample {sample_index}...")

    # Ensure data is on CPU and NumPy format for plotting
    signal_np = signal_1d.cpu().numpy() if isinstance(signal_1d, torch.Tensor) else signal_1d
    true_labels_np = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels
    predicted_labels_np = predicted_labels.cpu().numpy() if isinstance(predicted_labels, torch.Tensor) else predicted_labels

    # Ensure lengths match
    if not (len(signal_np) == len(true_labels_np) == len(predicted_labels_np)):
         print(f"Warning: Length mismatch in plot data for sample {sample_index}. Sig:{len(signal_np)}, True:{len(true_labels_np)}, Pred:{len(predicted_labels_np)}. Skipping plot.")
         return

    time_axis = np.arange(len(signal_np)) # Use sample index for time

    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True) # 2 panels
    title_str = f"Evaluation Sample {sample_index}"
    if sequence_length: title_str += f" (Seq Length: {sequence_length})"
    fig.suptitle(title_str, fontsize=14)

    # --- Plot 1: Signal + True Dots + Predicted Background ---
    ax[0].plot(time_axis, signal_np, color='black', linewidth=0.8, label='Signal (Processed)')
    ax[0].grid(True, linestyle=':', alpha=0.7)

    # Overlay predicted segments as background
    current_start_idx = 0
    legend_handles_map = {}
    line_signal, = ax[0].plot([],[], color='black', linewidth=0.8, label='Signal') # Dummy for legend
    legend_handles_map['Signal'] = line_signal

    for k in range(1, len(predicted_labels_np)):
        if predicted_labels_np[k] != predicted_labels_np[current_start_idx]:
            label_idx = predicted_labels_np[current_start_idx]
            label_name = class_names_map.get(label_idx, f"Class {label_idx}")
            color = SEGMENT_COLORS.get(label_idx, 'gray')
            # Use index for x-axis limits
            h = ax[0].axvspan(time_axis[current_start_idx] - 0.5, time_axis[k] - 0.5, color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
            if f'Pred: {label_name}' not in legend_handles_map: legend_handles_map[f'Pred: {label_name}'] = h
            current_start_idx = k
    # Last segment
    label_idx = predicted_labels_np[current_start_idx]
    label_name = class_names_map.get(label_idx, f"Class {label_idx}")
    color = SEGMENT_COLORS.get(label_idx, 'gray')
    h = ax[0].axvspan(time_axis[current_start_idx] - 0.5, time_axis[-1] + 0.5, color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
    if f'Pred: {label_name}' not in legend_handles_map: legend_handles_map[f'Pred: {label_name}'] = h

    # Overlay true labels as dots
    dot_handles = {}
    for t in range(len(signal_np)):
        true_lbl_idx = true_labels_np[t]
        true_color = PLOT_CLASS_COLORS.get(true_lbl_idx, 'magenta')
        marker_size = 5 if true_labels_np[t] == predicted_labels_np[t] else 8 # Make errors slightly bigger
        zorder = 3 if true_labels_np[t] == predicted_labels_np[t] else 4
        p = ax[0].scatter(time_axis[t], signal_np[t], color=true_color, s=marker_size**2, zorder=zorder, marker='.', label=f'True: {class_names_map.get(true_lbl_idx)}')
        label_name_true = f'True: {class_names_map.get(true_lbl_idx)}'
        if label_name_true not in dot_handles: dot_handles[label_name_true] = p # Store one handle per class

    ax[0].set_ylabel("Amplitude (Normalized)")
    ax[0].set_title("Processed Signal with True Labels (dots) and Predicted Segments (background)")

    # Create legend for Panel 1 combining segments and dots
    combined_handles = [legend_handles_map['Signal']]
    combined_labels = ['Signal']
    # Add True Dot handles/labels sorted by class index
    for i in sorted(class_names_map.keys()):
        name = class_names_map[i]
        label_name_true = f'True: {name}'
        if label_name_true in dot_handles:
             combined_handles.append(dot_handles[label_name_true])
             combined_labels.append(label_name_true)
    # Add Predicted Segment handles/labels sorted by class index
    for i in sorted(class_names_map.keys()):
        name = class_names_map[i]
        label_name_pred = f'Pred: {name}'
        if label_name_pred in legend_handles_map:
            # Use a patch handle for axvspan legend
             patch = plt.Rectangle((0, 0), 1, 1, fc=SEGMENT_COLORS.get(i, 'gray'), alpha=0.3)
             combined_handles.append(patch)
             combined_labels.append(label_name_pred)

    ax[0].legend(combined_handles, combined_labels, loc='upper right', fontsize='x-small', ncol=2)


    # --- Plot 2: True vs. Predicted Label Sequence ---
    ax[1].plot(time_axis, true_labels_np, drawstyle='steps-post', label='True Labels', color='darkorange', linewidth=1.5)
    ax[1].plot(time_axis, predicted_labels_np, drawstyle='steps-post', label='Predicted Labels', color='purple', linestyle='--', linewidth=1.5)
    diff_indices = np.where(true_labels_np != predicted_labels_np)[0]
    if len(diff_indices) > 0:
        ax[1].scatter(diff_indices, predicted_labels_np[diff_indices], color='red', s=25, zorder=5, label='Misclassified Point', marker='x')
    ax[1].set_xlabel("Time Step (Sample Index)")
    ax[1].set_ylabel("Label Index")
    ax[1].set_title("Ground Truth vs. Predicted Labels")
    ax[1].grid(True, linestyle=':', alpha=0.7)
    ax[1].legend(loc='upper right', fontsize='small')
    combined_label_indices = list(np.unique(true_labels_np)) + list(np.unique(predicted_labels_np))
    ax[1].set_yticks(np.unique(combined_label_indices))
    ax[1].set_ylim(min(combined_label_indices)-0.5, max(combined_label_indices)+0.5) # Adjust y-limits


    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(output_dir, f"evaluation_sample_{sample_index}_1D_plot.png")
    try:
        plt.savefig(plot_filename); print(f"Sample plot saved to: {plot_filename}"); plt.close(fig)
    except Exception as e: print(f"Warning: Could not save sample plot: {e}")
# --- End Plot Function ---


# --- Modified Evaluate Model Function ---
def evaluate_model(model, dataloader, loss_fn, device, num_classes, output_dir, class_names_map, plot_sample_info=None): # Changed plot arg
    """Evaluates the 1D model, calculates detailed metrics, saves results, and optionally plots."""
    print("--- Starting Evaluation ---")
    total_loss = 0.0
    all_preds = []
    all_labels = []
    # No need to store raw signals/tf_maps separately for this plot type
    plotted_sample = False

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        # DataLoader yields (signal_batch, labels_batch)
        for i, (signals_batch, labels_batch) in enumerate(progress_bar):
            # signals_batch shape: (B, 1, T)
            # labels_batch shape: (B, T)
            
            signals = signals_batch.to(device, non_blocking=True)
            labels = labels_batch.to(device, non_blocking=True).long()

            # Forward pass
            logits = model(signals) # Output shape: (B, T, num_classes)

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item() * labels.numel() # Accumulate total loss

            # Get predictions
            preds = torch.argmax(logits, dim=-1) # (B, T)

            # Store flattened predictions and labels for overall metrics
            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

            # --- Optional: Plotting the specified sample ---
            # plot_sample_info is dict {'batch_idx': B, 'item_idx_in_batch': I, 'dataset_idx': D}
            if plot_sample_info and i == plot_sample_info['batch_idx'] and not plotted_sample:
                 item_idx = plot_sample_info['item_idx_in_batch']
                 dataset_idx = plot_sample_info['dataset_idx']
                 # Need the processed 1D signal *before* batching/device placement, or get from batch
                 # The DataLoader output signal_batch is the processed one
                 signal_to_plot = signals_batch[item_idx].squeeze(0) # Get (T,) tensor
                 true_labels_to_plot = labels_batch[item_idx] # Get (T,) tensor
                 pred_labels_to_plot = preds[item_idx] # Get (T,) tensor from predictions

                 plot_evaluation_sample(
                     signal_1d=signal_to_plot, # Pass the 1D signal
                     true_labels=true_labels_to_plot,
                     predicted_labels=pred_labels_to_plot,
                     output_dir=output_dir,
                     class_names_map=class_names_map,
                     sample_index=dataset_idx, # Use original dataset index for filename
                     sequence_length=signal_to_plot.shape[0] # Pass sequence length
                 )
                 plotted_sample = True
            # --- End Plotting ---

            # Update progress bar (optional)
            # current_acc = metrics.accuracy_score(all_labels, all_preds) if all_labels else 0
            # progress_bar.set_postfix({"Running Acc": f"{current_acc:.4f}"})


    # --- Calculate Overall Metrics ---
    # (Metrics calculation and saving report/CM remains the same as before)
    total_samples = len(all_labels)
    if total_samples == 0: print("Error: No samples evaluated."); return 0, 0
    avg_loss = total_loss / total_samples
    overall_accuracy = metrics.accuracy_score(all_labels, all_preds)
    print("--- Evaluation Finished ---")
    print(f"Average Loss: {avg_loss:.4f}"); print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Total Samples Evaluated: {total_samples}")
    print("\n--- Detailed Classification Report ---")
    report_target_names = [class_names_map.get(i, f"Class {i}") for i in range(num_classes)]
    present_label_indices = sorted(list(set(all_labels + all_preds)))
    report_target_names_present = [class_names_map.get(i, f"Class {i}") for i in present_label_indices]
    report_str = metrics.classification_report(all_labels, all_preds, labels=present_label_indices, target_names=report_target_names_present, zero_division=0, digits=4)
    print(report_str)
    report_filename = os.path.join(output_dir, "classification_report.txt")
    with open(report_filename, "w") as f: f.write(f"Eval Results...\n{report_str}"); print(f"Report saved: {report_filename}")
    print("\n--- Confusion Matrix ---")
    all_possible_indices = list(range(num_classes)); cm = metrics.confusion_matrix(all_labels, all_preds, labels=all_possible_indices)
    plt.figure(figsize=(max(6, num_classes*1.2), max(5, num_classes*1.0))); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=report_target_names, yticklabels=report_target_names)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix'); plt.tight_layout()
    cm_filename = os.path.join(output_dir, "confusion_matrix.png")
    try: plt.savefig(cm_filename); print(f"CM plot saved: {cm_filename}"); plt.close()
    except Exception as e: print(f"Warn: Could not save CM plot: {e}")
    # No misclassified examples saved in this version

    return avg_loss, overall_accuracy
# --- End Evaluate Model Function ---


# --- Modified Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate 1D CNN-BiLSTM ECG Segmentation Model")

    # Args
    parser.add_argument("--load_dir", type=str, default="MCG_segmentation/checkpoints/best", help="Directory of the specific checkpoint SUBDIR to load (e.g., .../checkpoints_1d/best)")
    parser.add_argument("--data_dir_eval", type=str, default="MCG_segmentation/qtdb/processed/val", help="Path to evaluation data CSV directory")
    parser.add_argument("--output_dir", type=str, default="MCG_segmentation/evaluation_results/misclassified_examples", help="Directory to save evaluation results")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_sample_index", type=int, default=None, help="0-based index of the sample in the dataset to plot")
    # Fallback args if not in params.json (though loading from params is better)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)

    global args # Make args accessible if needed globally (e.g., by plot func via evaluate_model)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    try:
        model, loaded_params = load_model_for_evaluation(args.load_dir, DEVICE)
        # Use sequence length/overlap from checkpoint if not overridden
        sequence_length = args.sequence_length if args.sequence_length is not None else loaded_params["sequence_length"]
        overlap = args.overlap if args.overlap is not None else loaded_params["overlap"]
        num_classes = loaded_params["num_classes"]
        print(f"Using Sequence Length: {sequence_length}, Overlap: {overlap}")
    except Exception as e: print(f"Error loading model: {e}"); sys.exit(1)
    # --- End Load Model ---

    # --- Dataset & DataLoader Setup ---
    # No CWT params needed for ECGFullDataset here
    try:
        print("Setting up evaluation dataset for 1D model...")
        eval_dataset = ECGFullDataset(
            data_dir=args.data_dir_eval,
            overlap=overlap, # Use loaded/arg value
            sequence_length=sequence_length, # Use loaded/arg value
            # Turn off augmentations for evaluation
            sinusoidal_noise_mag=0.0, gaussian_noise_std=0.0,
            baseline_wander_mag=0.0, amplitude_scale_range=0.0, max_time_shift=0,
        )
        if len(eval_dataset) == 0: print(f"Error: Eval dataset empty: {args.data_dir_eval}"); sys.exit(1)

        eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                               drop_last=False, num_workers=args.num_workers, pin_memory=True)
        print(f"Eval samples: {len(eval_dataset)}, Eval batches: {len(eval_loader)}")

        # Calculate info needed for plotting specific sample
        plot_info = None
        if args.plot_sample_index is not None:
            if 0 <= args.plot_sample_index < len(eval_dataset):
                batch_idx = args.plot_sample_index // args.eval_batch_size
                item_idx_in_batch = args.plot_sample_index % args.eval_batch_size
                plot_info = {
                    'batch_idx': batch_idx,
                    'item_idx_in_batch': item_idx_in_batch,
                    'dataset_idx': args.plot_sample_index
                }
                print(f"Will plot sample {args.plot_sample_index} (found in batch {batch_idx}, item {item_idx_in_batch})")
            else: print(f"Warning: --plot_sample_index ({args.plot_sample_index}) out of bounds. Disabling plot.")

    except Exception as e: print(f"Error initializing dataset: {e}"); sys.exit(1)
    # --- End Dataset & DataLoader Setup ---

    # --- Loss Function ---
    loss_fn = nn.CrossEntropyLoss()
    # --- End Loss Function ---

    # --- Run Evaluation ---
    try:
        evaluate_model(model, eval_loader, loss_fn, DEVICE, num_classes, args.output_dir, CLASS_NAMES_MAP,
                       plot_sample_info=plot_info) # Pass plot info dict
    except Exception as e: print(f"\nError during evaluation: {e}"); import traceback; traceback.print_exc(); sys.exit(1)
    # --- End Run Evaluation ---

if __name__ == "__main__":
    main()