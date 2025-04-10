# evaluate.py

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model.model import WaveletCNNClassifier
from model.data_loader import ECGFullDataset, compute_wavelet
import sys
import json
import os
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib.lines import Line2D # Import Line2D for custom legends

# --- User-Defined Class Names and Colors ---
CLASS_NAMES_MAP = {
    0: "No Wave",
    1: "P Wave",
    2: "QRS",
    3: "T Wave"
}

PLOT_CLASS_COLORS = {
    0: "silver",  # No Wave
    1: "blue",    # P Wave
    2: "red",     # QRS
    3: "green",   # T Wave
}
# --- End User-Defined Constants ---

# --- load_model_for_evaluation function ---
# (Remains IDENTICAL to the previous version - no changes needed here)
def load_model_for_evaluation(load_dir, device):
    """Load the 'best' model and its parameters from a checkpoint directory."""
    print(f"Attempting to load best model from checkpoint directory: {load_dir}")

    if not load_dir or not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Checkpoint directory not found or not specified: {load_dir}")

    # Prioritize loading the 'best' checkpoint
    best_model_path = os.path.join(load_dir, "best/model.pth")
    best_params_path = os.path.join(load_dir, "best/params.json")

    if os.path.exists(best_model_path) and os.path.exists(best_params_path):
        print(f"Loading from best checkpoint files: {best_model_path}, {best_params_path}")
        model_path = best_model_path
        param_path = best_params_path
    else:
        print(f"Warning: 'best/model.pth' or 'best/params.json' not found in {load_dir}.")
        # Fallback: try finding the highest epoch number checkpoint
        found_model = None
        found_params = None
        latest_epoch = -1
        for f in os.listdir(load_dir):
             if f.startswith("checkpoint_epoch_") and f.endswith("_model.pth"):
                 try:
                     epoch_num = int(f.split("_")[2])
                     corresponding_params = f.replace("_model.pth", "_params.json")
                     if os.path.exists(os.path.join(load_dir, corresponding_params)):
                         if epoch_num > latest_epoch:
                             latest_epoch = epoch_num
                             found_model = os.path.join(load_dir, f)
                             found_params = os.path.join(load_dir, corresponding_params)
                 except (IndexError, ValueError):
                     continue

        if found_model and found_params:
             print(f"Falling back to latest found epoch checkpoint: Epoch {latest_epoch}")
             print(f"Model: {found_model}, Params: {found_params}")
             model_path = found_model
             param_path = found_params
        else:
             raise FileNotFoundError(f"Could not find suitable model (.pth) and parameter (.json) files (neither 'best' nor epoch-specific) in: {load_dir}")


    # Load parameters
    try:
        with open(param_path, "r") as f:
            params = json.load(f)
            checkpoint_args = params.get("args", {})
            num_classes = checkpoint_args.get("num_classes")
            if num_classes is None:
                 raise KeyError("'num_classes' not found in checkpoint's saved arguments.")
            num_heads = checkpoint_args.get("num_heads", 4)
            dropout_rate = checkpoint_args.get("dropout_rate", 0.3)
            sequence_length = checkpoint_args.get("sequence_length", 250)
            overlap = checkpoint_args.get("overlap", 125)

            print(f"Loaded parameters from checkpoint: num_classes={num_classes}, num_heads={num_heads}, dropout={dropout_rate}, seq_len={sequence_length}, overlap={overlap}")

            if num_classes != len(CLASS_NAMES_MAP):
                 print(f"\n*** WARNING: Number of classes loaded from checkpoint ({num_classes}) "
                       f"does not match the number of entries in CLASS_NAMES_MAP ({len(CLASS_NAMES_MAP)}). ***")
                 print(f"*** Please ensure CLASS_NAMES_MAP in evaluate.py is correctly defined for this model. ***\n")


    except FileNotFoundError:
        raise FileNotFoundError(f"Parameter file not found: {param_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from parameter file: {param_path}")
    except KeyError as e:
         raise KeyError(f"Missing expected key {e} in loaded params.json args. Check if the correct args were saved during training in '{param_path}'.")

    # Define model structure
    model = WaveletCNNClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        num_heads=num_heads
    )

    # Load model state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model state loaded successfully from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model state file not found: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading model state dict from {model_path}: {e}")

    model.to(device)
    model.eval()

    loaded_params = {
        "num_classes": num_classes,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
        "sequence_length": sequence_length,
        "overlap": overlap,
    }

    return model, loaded_params
# --- End load_model_for_evaluation function ---


# --- Plot function matching the ORIGINAL data_loader.py layout ---
# --- but adding predicted markers where different from true ---
def plot_evaluation_sample(raw_signal, tf_map, true_labels, predicted_labels, output_dir, class_names_map, sample_index=0):
    """
    Generates a 3-panel plot similar to the data_loader test plot:
    1. CWT map.
    2. Signal with true labels as dots ('o') and predicted labels as crosses ('x')
       only where prediction differs from truth.
    3. True label sequence step plot.
    """
    print(f"\nGenerating original layout plot for evaluation sample {sample_index}...")

    # Ensure data is on CPU and NumPy format for plotting
    signal_np = raw_signal.cpu().numpy() if isinstance(raw_signal, torch.Tensor) else raw_signal
    tf_map_np = tf_map.squeeze(0).cpu().numpy() if isinstance(tf_map, torch.Tensor) else tf_map.squeeze(0)
    true_labels_np = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else true_labels
    predicted_labels_np = predicted_labels.cpu().numpy() if isinstance(predicted_labels, torch.Tensor) else predicted_labels

    time_axis = np.arange(signal_np.shape[0])

    # Use layout parameters similar to original data_loader test plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={'height_ratios': [1, 2, 1]}) # Ratios from original

    fig.suptitle(f"Evaluation Sample {sample_index} (Original Layout Style)", fontsize=14)

    # Plot 1: Wavelet Scalogram (Input to Model) - As before
    im = ax[0].imshow(tf_map_np, aspect='auto', cmap='viridis',
                      extent=[0, time_axis[-1], tf_map_np.shape[0], 1])
    ax[0].set_ylabel("Wavelet Scale Index") # Changed label slightly
    ax[0].set_title("Wavelet Transform (Input to CNN)")
    plt.colorbar(im, ax=ax[0], label='Magnitude', pad=0.02)

    # Plot 2: Original Signal with Ground Truth (dots) AND Predictions (crosses, if different)
    ax[1].plot(time_axis, signal_np, color='black', linewidth=1.0, label='Input Signal')
    # Plot ground truth dots first
    for t in range(signal_np.shape[0]):
        true_lbl_idx = true_labels_np[t]
        true_color = PLOT_CLASS_COLORS.get(true_lbl_idx, 'magenta')
        ax[1].scatter(t, signal_np[t]-0.05, color=true_color, s=20, marker='o', zorder=3) # True labels as dots

    for t in range(signal_np.shape[0]):
        pred_lbl_idx = predicted_labels_np[t]
        pred_color = PLOT_CLASS_COLORS.get(pred_lbl_idx, 'black') # Color based on predicted class
        # Use 'x' marker for predictions, potentially red color for emphasis
        ax[1].scatter(t, signal_np[t]+0.05, color=pred_color, s=35, marker='x', zorder=4, linewidths=1.5)

    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Input Signal with Ground Truth (o) and Prediction (x, if different)")
    ax[1].grid(True, linestyle=':', alpha=0.7)

    # Create legend for Panel 2
    legend_elements_panel2 = [Line2D([0], [0], color='black', lw=1, label='Signal')]
    for lbl_idx, name in class_names_map.items():
        color = PLOT_CLASS_COLORS.get(lbl_idx, 'magenta')
        legend_elements_panel2.append(Line2D([0], [0], marker='o', color='w', label=f'True: {name}',
                                       markerfacecolor=color, markersize=7))
        legend_elements_panel2.append(Line2D([0], [0], marker='x', color='w', label=f'Pred: {name}',
                                       markerfacecolor=color, markersize=7, markeredgewidth=1.5))

    ax[1].legend(handles=legend_elements_panel2, loc='upper right', fontsize='x-small', ncol=math.ceil(len(legend_elements_panel2)/2))


    # Plot 3: Ground Truth Label Sequence (Step Plot) - Only True Labels
    ax[2].plot(time_axis, true_labels_np, drawstyle='steps-post', label='True Label Sequence', color='darkorange')
    ax[2].set_xlabel("Time Step (Sample)")
    ax[2].set_ylabel("Label Index")
    ax[2].set_title("Ground Truth Label Sequence")
    ax[2].grid(True, linestyle=':', alpha=0.7)
    ax[2].legend(loc='upper right', fontsize='small')
    ax[2].set_yticks(np.unique(true_labels_np))


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    plot_filename = os.path.join(output_dir, f"evaluation_sample_{sample_index}_plot.png")
    try:
        plt.savefig(plot_filename)
        print(f"Sample plot saved to: {plot_filename}")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save sample plot: {e}")
# --- End plot_evaluation_sample ---


# --- evaluate_model function ---
# (Remains IDENTICAL to the previous version - no changes needed here,
#  it will correctly call the updated plot_evaluation_sample function)
def evaluate_model(model, dataloader, loss_fn, device, num_classes, output_dir, class_names_map, plot_sample_index=None):
    """
    Evaluates the model, calculates detailed metrics using descriptive names,
    saves results, and optionally plots a specific sample using the original 3-panel layout.
    """
    print("--- Starting Evaluation ---")
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_raw_signals = []
    all_tf_maps = []
    save_misclassified_limit = 50
    plotted_sample = False

    report_target_names = [class_names_map.get(i, f"Class {i}") for i in range(num_classes)]

    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, (raw_signal_batch, tf_map_batch, labels_batch) in enumerate(progress_bar):
            tf_maps = tf_map_batch.to(device, non_blocking=True)
            labels = labels_batch.to(device, non_blocking=True).long()
            raw_signals = raw_signal_batch

            logits = model(tf_maps)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item() * labels.numel()

            preds = torch.argmax(logits, dim=-1)

            # --- Optional: Plotting the specified sample ---
            if plot_sample_index is not None and i == plot_sample_index and not plotted_sample:
                 # Calls the *newly updated* plot_evaluation_sample function
                 plot_evaluation_sample(
                     raw_signal=raw_signals[0],
                     tf_map=tf_map_batch[0],
                     true_labels=labels[0],
                     predicted_labels=preds[0],
                     output_dir=output_dir,
                     class_names_map=class_names_map,
                     sample_index=args.plot_sample_index
                 )
                 plotted_sample = True
            # --- End Plotting ---

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

            misclassified_mask = (preds != labels).cpu()
            if misclassified_mask.any() and len(all_raw_signals) < save_misclassified_limit:
                batch_indices, time_indices = torch.where(misclassified_mask)
                for k in range(len(batch_indices)):
                    b_idx = batch_indices[k].item()
                    if len(all_raw_signals) < save_misclassified_limit:
                         all_raw_signals.append(raw_signals[b_idx].numpy())
                         all_tf_maps.append(tf_map_batch[b_idx].cpu().numpy())
                    else:
                        break

            current_acc = metrics.accuracy_score(all_labels, all_preds) if all_labels else 0
            progress_bar.set_postfix({"Running Acc": f"{current_acc:.4f}"})

    # Calculate Overall Metrics, Print/Save Reports, CM, Misclassified examples
    # (This part remains IDENTICAL to the previous version)
    total_samples = len(all_labels)
    if total_samples == 0:
        print("Error: No samples were evaluated.")
        return 0, 0

    avg_loss = total_loss / total_samples
    overall_accuracy = metrics.accuracy_score(all_labels, all_preds)

    print("--- Evaluation Finished ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Total Samples Evaluated: {total_samples}")

    print("\n--- Detailed Classification Report ---")
    present_label_indices = sorted(list(set(all_labels + all_preds)))
    report_target_names_present = [class_names_map.get(i, f"Class {i}") for i in present_label_indices]

    report_str = metrics.classification_report(
        all_labels, all_preds,
        labels=present_label_indices,
        target_names=report_target_names_present,
        zero_division=0, digits=4
    )
    report_dict = metrics.classification_report(
        all_labels, all_preds,
        labels=present_label_indices,
        target_names=report_target_names_present,
        zero_division=0, output_dict=True
    )
    print(report_str)

    report_filename = os.path.join(output_dir, "classification_report.txt")
    with open(report_filename, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write("="*20 + "\n")
        f.write(f"Class Names Mapping Used: {class_names_map}\n\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Total Samples Evaluated: {total_samples}\n\n")
        f.write("Detailed Classification Report (F1-Score is harmonic mean of Precision and Recall):\n")
        f.write(report_str)
    print(f"Classification report saved to: {report_filename}")

    print("\n--- Confusion Matrix ---")
    all_possible_indices = list(range(num_classes))
    cm = metrics.confusion_matrix(all_labels, all_preds, labels=all_possible_indices)

    plt.figure(figsize=(max(6, num_classes * 1.2), max(5, num_classes * 1.0)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=report_target_names,
                yticklabels=report_target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    cm_filename = os.path.join(output_dir, "confusion_matrix.png")
    try:
        plt.savefig(cm_filename)
        print(f"Confusion matrix plot saved to: {cm_filename}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")


    if all_raw_signals:
        print(f"\nSaving up to {save_misclassified_limit} misclassified example sequences...")
        misclassified_dir = os.path.join(output_dir, "misclassified_examples")
        os.makedirs(misclassified_dir, exist_ok=True)
        try:
            np.savez_compressed(
                os.path.join(misclassified_dir, "misclassified_data.npz"),
                raw_signals=np.array(all_raw_signals, dtype=object),
                tf_maps=np.array(all_tf_maps, dtype=object)
            )
            print(f"Saved {len(all_raw_signals)} misclassified sequences to {misclassified_dir}/misclassified_data.npz")
        except Exception as e:
            print(f"Warning: Could not save misclassified examples: {e}")


    return avg_loss, overall_accuracy
# --- End evaluate_model function ---


# --- main function ---
# (Remains IDENTICAL to the previous version - no changes needed here)
def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG Segmentation Model with Detailed Metrics and Descriptive Class Names")

    parser.add_argument("--load_dir", type=str, required=True, help="Directory of the checkpoint to load")
    parser.add_argument("--data_dir_eval", type=str, required=True, help="Path to evaluation data directory")
    parser.add_argument("--output_dir", type=str, default="Code/MCG_segmentation/evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--plot_sample_index", type=int, default=None, help="Specify the index (0-based) of the sample in the dataset to plot")
    parser.add_argument("--sequence_length", type=int, default=None, help="Length of ECG sequence segments (override loaded params if specified)")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap between segments (override loaded params if specified)")

    global args
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        model, loaded_params = load_model_for_evaluation(args.load_dir, device)
        sequence_length = args.sequence_length if args.sequence_length is not None else loaded_params["sequence_length"]
        overlap = args.overlap if args.overlap is not None else loaded_params["overlap"]
        num_classes = loaded_params["num_classes"]

        if num_classes > len(CLASS_NAMES_MAP):
             print(f"Warning: Model has {num_classes} classes, but only {len(CLASS_NAMES_MAP)} names defined in CLASS_NAMES_MAP.")
             for i in range(len(CLASS_NAMES_MAP), num_classes):
                 CLASS_NAMES_MAP[i] = f"Class {i}"
                 PLOT_CLASS_COLORS[i] = plt.cm.get_cmap("tab10")(i % 10)

        if args.sequence_length is not None or args.overlap is not None:
            print("Overriding sequence length/overlap from command line.")
        print(f"Using Sequence Length: {sequence_length}, Overlap: {overlap}")

    except (FileNotFoundError, ValueError, RuntimeError, KeyError, TypeError) as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    wavelet_scales = np.logspace(np.log2(2), np.log2(128), num=64, base=2.0)
    print(f"Using fixed Wavelet Scales: {wavelet_scales.min()} to {wavelet_scales.max()}")

    try:
        print("Setting up evaluation dataset...")
        eval_dataset = ECGFullDataset(
            data_dir=args.data_dir_eval,
            overlap=overlap,
            sequence_length=sequence_length,
            noise_mag=0.00,
            wavelet_scales=wavelet_scales
        )

        if len(eval_dataset) == 0:
             print(f"Error: Evaluation dataset is empty. Check path: {args.data_dir_eval}")
             sys.exit(1)

        print("Setting up evaluation dataloader...")
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        print(f"Evaluation samples found: {len(eval_dataset)}")
        print(f"Evaluation batches: {len(eval_loader)}")

        plot_arg_for_eval = None
        if args.plot_sample_index is not None:
            if args.plot_sample_index < 0 or args.plot_sample_index >= len(eval_dataset):
                print(f"Warning: --plot_sample_index ({args.plot_sample_index}) is out of bounds for dataset size ({len(eval_dataset)}). Disabling plotting.")
            else:
                batch_idx_to_plot = args.plot_sample_index // args.eval_batch_size
                print(f"Plotting sample at dataset index {args.plot_sample_index} (found in batch index {batch_idx_to_plot}, plotting item 0 of that batch).")
                plot_arg_for_eval = batch_idx_to_plot


    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)

    loss_fn = nn.CrossEntropyLoss()

    try:
        evaluate_model(
            model,
            eval_loader,
            loss_fn,
            device,
            num_classes,
            args.output_dir,
            CLASS_NAMES_MAP,
            plot_sample_index=plot_arg_for_eval
        )
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# --- End main function ---

if __name__ == "__main__":
    main()