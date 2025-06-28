import os
import json
import argparse
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from model.model import UNet1D, DENS_ECG_segmenter, ECGSegmenter
from model.data_loader import ECGFullDataset
import random

# Constants
CLASS_NAMES = {0: "No Wave", 1: "P Wave", 2: "QRS", 3: "T Wave"}
PLOT_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}
SEGMENT_COLORS = {0: "whitesmoke", 1: "lightblue", 2: "lightcoral", 3: "lightgreen"}

# Device setup
try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
except Exception as e:
    print(f"Failed to determine device, defaulting to CPU. Error: {e}")
    DEVICE = torch.device("cpu")

def load_model(load_dir, device):
    best_model_path = os.path.join(load_dir, "checkpoints/best/model.pth")
    config_path = os.path.join(load_dir, "config.json")

    if not os.path.exists(best_model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model files not found in {load_dir}")

    # Load model parameters
    with open(config_path, "r") as f:
        model_params = json.load(f)

    model = UNet1D(**model_params)

    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        model.eval()
    except RuntimeError as e:
        raise
    
    return model


def sample_from_model(model, device, data: torch.Tensor, min_duration_sec: float = 0.04):
    if data.numel() == 0:
        warnings.warn("No data to segment.")
        batch_size = data.shape[0]
        return []

    max_len = 2000
    if data.shape[-1] > max_len:
        warnings.warn(f"Data length ({data.shape[-1]}) exceeds maximum ({max_len}). Truncating.")
        data = data[..., :max_len]

    data = data.to(device)

    with torch.no_grad():
        logits = model(data)
        probabilities = torch.softmax(logits, dim=-1)
        confidence_scores_pt, predicted_indices_pt = torch.max(probabilities, dim=-1)

    predicted_indices = predicted_indices_pt.cpu().numpy()
    confidence_scores = confidence_scores_pt.cpu().numpy()

    batch_size, time_steps = predicted_indices.shape

    for b in range(batch_size):
        labels_arr = predicted_indices[b]
        i = 0
        while i < time_steps:
            current_label = labels_arr[i]
            start = i
            while i < time_steps and labels_arr[i] == current_label:
                i += 1
            end = i

            segment_length = end - start
            if segment_length < min_duration_sec * 250:
                left_label = labels_arr[start - 1] if start > 0 else None
                right_label = labels_arr[end] if end < time_steps else None

                if left_label is not None and right_label is not None:
                    new_label = left_label if left_label == right_label else 0
                else:
                    new_label = left_label if left_label is not None else right_label or 0

                labels_arr[start:end] = new_label

        predicted_indices[b] = labels_arr

    # Return flattened list for sklearn compatibility
    return predicted_indices.flatten().tolist()

def extract_significant_points(labels):
    points = {1: {'onsets': [], 'offsets': []}, 2: {'onsets': [], 'offsets': []}, 3: {'onsets': [], 'offsets': []}}
    current = labels[0]
    start = 0 if current in points else None

    for i in range(1, len(labels)):
        if labels[i] != current:
            if current in points and start is not None:
                points[current]['onsets'].append(start)
                points[current]['offsets'].append(i - 1)
            start = i if labels[i] in points else None
            current = labels[i]

    if current in points and start is not None:
        points[current]['onsets'].append(start)
        points[current]['offsets'].append(len(labels) - 1)

    return points

def plot_segmented_signal(signal, pred, ground_truth, output_dir, sample_rate=250, filename_prefix="ecg_segment"):
    """
    Visualisiert ein EKG-Signal mit vorhergesagten Segmenten als Flächen und Ground-Truth-Labels als Scatter-Plot.
    Assumes 250Hz input. Classifies each data point as: 0) No Wave, 1) P-Wave, 2) QRS, 3) T-Wave.

    Args:
        signal (np.ndarray): EKG-Signal (Shape: [time_steps]).
        pred (np.ndarray): Vorhergesagte Labels (Shape: [time_steps]).
        ground_truth (np.ndarray): Ground-Truth-Labels (Shape: [time_steps]).
        output_dir (str): Verzeichnis zum Speichern des Plots.
        sample_rate (int): Abtastrate in Hz (Standard: 250).
        filename_prefix (str): Präfix für den Dateinamen des Plots.
    """
    signal = signal.squeeze()
    pred = pred.squeeze()
    ground_truth = ground_truth.squeeze()
    t = np.arange(len(signal)) / sample_rate  # Zeitachse in Sekunden

    fig, axs = plt.subplots(figsize=(15, 8))
    axs.plot(t, signal, color='black', linewidth=0.8, label='Signal (Processed)')
    axs.grid(True, linestyle=':', alpha=0.7)

    # Plot vorhergesagte Segmente
    current_start_idx = 0
    legend_handles_map = {}
    line_signal, = axs.plot([], [], color='black', linewidth=0.8, label='Signal')
    legend_handles_map['Signal'] = line_signal

    for k in range(1, len(pred)):
        if pred[k] != pred[current_start_idx]:
            label_idx = pred[current_start_idx]
            label_name = CLASS_NAMES.get(label_idx, f"Class {label_idx}")
            color = SEGMENT_COLORS.get(label_idx, 'gray')
            h = axs.axvspan(t[current_start_idx] - 0.5/sample_rate, t[k] - 0.5/sample_rate, 
                           color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
            if f'Pred: {label_name}' not in legend_handles_map:
                legend_handles_map[f'Pred: {label_name}'] = h
            current_start_idx = k

    label_idx = pred[current_start_idx]
    label_name = CLASS_NAMES.get(label_idx, f"Class {label_idx}")
    color = SEGMENT_COLORS.get(label_idx, 'gray')
    h = axs.axvspan(t[current_start_idx] - 0.5/sample_rate, t[-1] + 0.5/sample_rate, 
                   color=color, alpha=0.3, ec=None, label=f'Pred: {label_name}')
    if f'Pred: {label_name}' not in legend_handles_map:
        legend_handles_map[f'Pred: {label_name}'] = h

    # Plot Ground-Truth-Labels als Scatter-Plot
    for i in sorted(CLASS_NAMES.keys()):
        mask = ground_truth == i
        if np.any(mask):
            label_name = f'True: {CLASS_NAMES[i]}'
            axs.scatter(t[mask], signal[mask], color=PLOT_COLORS.get(i, 'gray'), 
                       s=10, alpha=0.8, label=label_name, marker='o')
            legend_handles_map[label_name] = axs.scatter([], [], color=PLOT_COLORS.get(i, 'gray'), 
                                                        s=10, alpha=0.8, label=label_name)

    # Legende
    combined_handles = [legend_handles_map['Signal']]
    combined_labels = ['Signal']
    for i in sorted(CLASS_NAMES.keys()):
        label_name_pred = f'Pred: {CLASS_NAMES[i]}'
        label_name_true = f'True: {CLASS_NAMES[i]}'
        if label_name_pred in legend_handles_map:
            patch = plt.Rectangle((0, 0), 1, 1, fc=SEGMENT_COLORS.get(i, 'gray'), alpha=0.3)
            combined_handles.append(patch)
            combined_labels.append(label_name_pred)
        if label_name_true in legend_handles_map:
            scatter = plt.scatter([], [], color=PLOT_COLORS.get(i, 'gray'), s=10, alpha=0.8)
            combined_handles.append(scatter)
            combined_labels.append(label_name_true)

    axs.legend(combined_handles, combined_labels, loc='upper right', fontsize='x-small', ncol=2)
    axs.set_xlabel("Zeit (s)")
    axs.set_ylabel("Amplitude")
    axs.set_title("EKG-Signal mit vorhergesagten Segmenten und Ground-Truth-Labels")

    plt.savefig(os.path.join(output_dir, f"{filename_prefix}.png"), bbox_inches='tight')
    plt.close()



def evaluate_detection_metrics(all_preds, all_labels, sequence_length, sample_rate=250, tolerance_ms=150):
    tolerance_samples = int((tolerance_ms / 1000) * sample_rate)
    stats = {wave: {f'{pt}_{metric}': 0 for pt in ['TP', 'FP', 'FN'] for metric in ['onset', 'offset']} |
             {f'{pt}_errors': [] for pt in ['onset', 'offset']} for wave in [1, 2, 3]}

    for i in range(0, len(all_labels), sequence_length):
        true_seq = all_labels[i:i + sequence_length]
        pred_seq = all_preds[i:i + sequence_length]
        true_pts = extract_significant_points(true_seq)
        pred_pts = extract_significant_points(pred_seq)

        for wave in [1, 2, 3]:
            for pt in ['onsets', 'offsets']:
                matched = [False] * len(true_pts[wave][pt])
                for p in pred_pts[wave][pt]:
                    found = False
                    for j, t in enumerate(true_pts[wave][pt]):
                        if not matched[j] and abs(p - t) <= tolerance_samples:
                            stats[wave][f'TP_{pt[:-1]}'] += 1
                            stats[wave][f'{pt[:-1]}_errors'].append(p - t)
                            matched[j] = True
                            found = True
                            break
                    if not found:
                        stats[wave][f'FP_{pt[:-1]}'] += 1
                stats[wave][f'FN_{pt[:-1]}'] += matched.count(False)

    print(f"\n=== Significant Point Detection Metrics (±{tolerance_ms} ms) ===")
    for wave in [1, 2, 3]:
        name = CLASS_NAMES[wave]
        for point in ['onset', 'offset']:
            TP = stats[wave][f'TP_{point}']
            FP = stats[wave][f'FP_{point}']
            FN = stats[wave][f'FN_{point}']
            errors = np.array(stats[wave][f'{point}_errors'])

            mean_err = errors.mean() if len(errors) else 0
            std_err = errors.std() if len(errors) else 0
            Se = TP / (TP + FN) if (TP + FN) else 0
            PPV = TP / (TP + FP) if (TP + FP) else 0
            F1 = 2 * PPV * Se / (PPV + Se) if (PPV + Se) else 0

            print(f"\n{name} {point.capitalize()}:")
            print(f"  TP={TP}, FP={FP}, FN={FN}")
            print(f"  Mean Error: {mean_err:.2f} samples ({mean_err / sample_rate * 1000:.2f} ms)")
            print(f"  Std Dev: {std_err:.2f} samples ({std_err / sample_rate * 1000:.2f} ms)")
            print(f"  Sensitivity (Se): {Se:.4f}, Precision (PPV): {PPV:.4f}, F1 Score: {F1:.4f}")


def evaluate(model, dataloader, device, num_classes, output_dir, sequence_length):
    model.eval()
    all_preds, all_labels = [], []
    # Liste zum Speichern eines zufälligen Samples
    random_sample = None
    random_batch_idx = random.randint(0, len(dataloader) - 1)

    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
        x, y = x.to(device), y.to(device).long()
        preds = sample_from_model(model, device, x)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().flatten().tolist())

        # Speichere Daten für das zufällige Sample
        if batch_idx == random_batch_idx:
            random_sample = {
                'signal': x.cpu().numpy()[0],
                'true_labels': y.cpu().numpy()[0],
                'pred_labels': np.array(preds[:sequence_length]),
                'batch_idx': batch_idx
            }

    # Visualisiere das zufällige Sample
    if random_sample is not None:
        plot_segmented_signal(
            signal=random_sample['signal'],
            ground_truth=random_sample['true_labels'],
            pred=random_sample['pred_labels'],
            output_dir=output_dir,
            sample_rate=250,
            filename_prefix=f"ecg_segment_random_batch_{random_sample['batch_idx']}"
        )

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[CLASS_NAMES.get(i, f"Class {i}") for i in range(num_classes)], digits=4))

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(),
                cbar_kws={'label': 'Percentage (%)'})
    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    evaluate_detection_metrics(all_preds, all_labels, sequence_length)
    return acc


def main():
    parser = argparse.ArgumentParser("Evaluate ECG Segmenter")
    parser.add_argument("--load_dir", type=str, default="MCG_segmentation/trained_models/UNet_1D_900k")
    parser.add_argument("--data_dir_eval", type=str, default="MCG_segmentation/Datasets/val")
    parser.add_argument("--output_dir", type=str, default="MCG_segmentation/trained_models/UNet_1D_900k/evaluation_results")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=500)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.load_dir, DEVICE)
    print(f"Model loaded from {args.load_dir}")

    # Load evaluation dataset
    eval_dataset = ECGFullDataset(args.data_dir_eval, sequence_length=args.sequence_length, augmentation_prob=0.00, baseline_wander_mag=0.0, gaussian_noise_std=0.00, overlap=400)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    # Evaluate
    print(f"Evaluating on {len(eval_dataset)} samples...")
    accuracy = evaluate(model, eval_dataloader, DEVICE, 4, args.output_dir, args.sequence_length)
    
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()