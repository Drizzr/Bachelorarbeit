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

from model.model import UNet1D
from model.data_loader import ECGFullDataset

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
    model_path = os.path.join(load_dir, "model.pth")
    param_path = os.path.join(load_dir, "params.json")

    if not os.path.exists(model_path) or not os.path.exists(param_path):
        raise FileNotFoundError("Model or params not found.")

    with open(param_path) as f:
        args = json.load(f).get("args", {})

    model = UNet1D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model, args.get("num_classes", 4)

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

    for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device).long()
        preds = sample_from_model(model, device, x)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().flatten().tolist())

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
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    evaluate_detection_metrics(all_preds, all_labels, sequence_length)
    return acc

def main():
    parser = argparse.ArgumentParser("Evaluate ECG Segmenter")
    parser.add_argument("--load_dir", type=str, default="MCG_segmentation/trained_models/UNet_1D_15M/checkpoints/best")
    parser.add_argument("--data_dir_eval", type=str, default="MCG_segmentation/Datasets/val")
    parser.add_argument("--output_dir", type=str, default="MCG_segmentation/evaluation_results")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=2000)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, num_classes = load_model(args.load_dir, DEVICE)
    print(f"Model loaded with {num_classes} classes")

    # Load evaluation dataset
    eval_dataset = ECGFullDataset(args.data_dir_eval, augmentation_prob=0.0, sequence_length=args.sequence_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False
    )

    # Evaluate
    print(f"Evaluating on {len(eval_dataset)} samples...")
    accuracy = evaluate(model, eval_dataloader, DEVICE, num_classes, args.output_dir, args.sequence_length)
    
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()