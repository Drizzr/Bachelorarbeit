# evaluate.py

import os, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from torch.utils.data import DataLoader
from torch import nn

from model.model import ECGSegmenter
from model.data_loader import ECGFullDataset

# Constants
CLASS_NAMES = {0: "No Wave", 1: "P Wave", 2: "QRS", 3: "T Wave"}
PLOT_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}
SEGMENT_COLORS = {0: "whitesmoke", 1: "lightblue", 2: "lightcoral", 3: "lightgreen"}
DEVICE = torch.device("cpu")


def load_model(load_dir, device):
    """Load trained model and its parameters."""
    model_path = os.path.join(load_dir, "model.pth")
    param_path = os.path.join(load_dir, "params.json")

    if not os.path.exists(model_path) or not os.path.exists(param_path):
        raise FileNotFoundError("Model or params not found.")

    with open(param_path) as f:
        args = json.load(f).get("args", {})

    model = ECGSegmenter()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    return model, args.get("num_classes", 4)

def plot_sample(signal, true, pred, output_dir, index=0, seq_len=None):
    """Plot signal with true/predicted labels."""
    signal_np, true_labels_np, predicted_labels_np = map(lambda x: x.cpu().numpy(), [signal, true, pred])
    t = np.arange(len(signal_np))

    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f"Sample {index} (Seq Length: {seq_len})" if seq_len else f"Sample {index}")

    # --- Panel 1: Signal + True Dots + Predicted Background ---
    axs[0].plot(t, signal_np, color='black', linewidth=0.8, label='Signal (Processed)')
    axs[0].grid(True, linestyle=':', alpha=0.7)

    current_start_idx = 0
    legend_handles_map = {}
    line_signal, = axs[0].plot([],[], color='black', linewidth=0.8, label='Signal')  # Dummy for legend
    legend_handles_map['Signal'] = line_signal

    for k in range(1, len(predicted_labels_np)):
        if predicted_labels_np[k] != predicted_labels_np[current_start_idx]:
            label_idx = predicted_labels_np[current_start_idx]
            label_name = CLASS_NAMES.get(label_idx, f"Class {label_idx}")
            color = SEGMENT_COLORS.get(label_idx, 'gray')
            h = axs[0].axvspan(t[current_start_idx] - 0.5, t[k] - 0.5, color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
            if f'Pred: {label_name}' not in legend_handles_map:
                legend_handles_map[f'Pred: {label_name}'] = h
            current_start_idx = k

    # Final segment
    label_idx = predicted_labels_np[current_start_idx]
    label_name = CLASS_NAMES.get(label_idx, f"Class {label_idx}")
    color = SEGMENT_COLORS.get(label_idx, 'gray')
    h = axs[0].axvspan(t[current_start_idx] - 0.5, t[-1] + 0.5, color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
    if f'Pred: {label_name}' not in legend_handles_map:
        legend_handles_map[f'Pred: {label_name}'] = h

    # True label dots
    dot_handles = {}
    for i in range(len(t)):
        true_lbl_idx = true_labels_np[i]
        true_color = PLOT_COLORS.get(true_lbl_idx, 'magenta')
        marker_size = 5 if true_labels_np[i] == predicted_labels_np[i] else 8
        zorder = 3 if true_labels_np[i] == predicted_labels_np[i] else 4
        p = axs[0].scatter(t[i], signal_np[i], color=true_color, s=marker_size**2, zorder=zorder, marker='.', label=f'True: {CLASS_NAMES.get(true_lbl_idx)}')
        label_name_true = f'True: {CLASS_NAMES.get(true_lbl_idx)}'
        if label_name_true not in dot_handles:
            dot_handles[label_name_true] = p

    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("True (dots) vs Predicted (background)")

    # Combined legend
    combined_handles = [legend_handles_map['Signal']]
    combined_labels = ['Signal']
    for i in sorted(CLASS_NAMES.keys()):
        label_name_true = f'True: {CLASS_NAMES[i]}'
        if label_name_true in dot_handles:
            combined_handles.append(dot_handles[label_name_true])
            combined_labels.append(label_name_true)
    for i in sorted(CLASS_NAMES.keys()):
        label_name_pred = f'Pred: {CLASS_NAMES[i]}'
        if label_name_pred in legend_handles_map:
            patch = plt.Rectangle((0, 0), 1, 1, fc=SEGMENT_COLORS.get(i, 'gray'), alpha=0.3)
            combined_handles.append(patch)
            combined_labels.append(label_name_pred)

    axs[0].legend(combined_handles, combined_labels, loc='upper right', fontsize='x-small', ncol=2)

    # --- Panel 2: Label Comparison ---
    axs[1].plot(t, true_labels_np, drawstyle='steps-post', label="True", color='orange')
    axs[1].plot(t, predicted_labels_np, drawstyle='steps-post', label="Pred", color='purple', linestyle='--')
    axs[1].scatter(t[true_labels_np != predicted_labels_np], predicted_labels_np[true_labels_np != predicted_labels_np], color='red', marker='x', s=25, label='Mismatch')

    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Label Index")
    axs[1].legend()
    axs[1].grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"sample_{index}.png"))
    plt.close()

def evaluate(model, dataloader, loss_fn, device, num_classes, output_dir, sample_info=None):
    """Evaluate model on dataset, compute metrics, and optionally plot a sample."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    plotted = False

    for i, (x, y) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
        x, y = x.to(device), y.to(device).long()
        logits = model(x)
        loss = loss_fn(logits.view(-1, num_classes), y.view(-1))
        total_loss += loss.item() * y.numel()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().flatten().tolist())
        all_labels.extend(y.cpu().flatten().tolist())

        if sample_info and not plotted and i == sample_info['batch_idx']:
            idx = sample_info['item_idx_in_batch']
            plot_sample(x[idx].squeeze(0), y[idx], preds[idx], output_dir, sample_info['dataset_idx'], x.shape[-1])
            plotted = True

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\nEval Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[CLASS_NAMES.get(i, f"Class {i}") for i in range(num_classes)], digits=4))

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser("Evaluate ECG Segmenter")
    parser.add_argument("--load_dir", type=str, default="MCG_segmentation/checkpoints/best")
    parser.add_argument("--data_dir_eval", type=str, default="MCG_segmentation/qtdb/processed/val")
    parser.add_argument("--output_dir", type=str, default="MCG_segmentation/evaluation_results")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_sample_index", type=int)
    parser.add_argument("--sequence_length", type=int, default=500) 


    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, num_classes = load_model(args.load_dir, DEVICE)

    dataset = ECGFullDataset(
        data_dir=args.data_dir_eval,
        sequence_length=args.sequence_length,
        overlap=0,
        sinusoidal_noise_mag=0.05,
        gaussian_noise_std=0.02,
        baseline_wander_mag=0.03,
        amplitude_scale_range=0.1,
        max_time_shift=5,
        augmentation_prob=1.00,
    )

    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers)

    sample_info = None
    if args.plot_sample_index is not None and 0 <= args.plot_sample_index < len(dataset):
        sample_info = {
            'batch_idx': args.plot_sample_index // args.eval_batch_size,
            'item_idx_in_batch': args.plot_sample_index % args.eval_batch_size,
            'dataset_idx': args.plot_sample_index
        }

    loss_fn = nn.CrossEntropyLoss()
    evaluate(model, dataloader, loss_fn, DEVICE, num_classes, args.output_dir, sample_info)

if __name__ == "__main__":
    main()
