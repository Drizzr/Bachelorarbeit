import os, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from torch import nn

from model.model import ECGSegmenter, DENS_ECG_segmenter, UNet1D
from model.data_loader import ECGFullDataset

# Constants
CLASS_NAMES = {0: "No Wave", 1: "P Wave", 2: "QRS", 3: "T Wave"}
PLOT_COLORS = {0: "silver", 1: "blue", 2: "red", 3: "green"}
SEGMENT_COLORS = {0: "whitesmoke", 1: "lightblue", 2: "lightcoral", 3: "lightgreen"}

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

def extract_significant_points(labels):
    points = {1: {'onsets': [], 'offsets': []},
              2: {'onsets': [], 'offsets': []},
              3: {'onsets': [], 'offsets': []}}

    current = labels[0]
    start = 0 if current in points else None

    for i in range(1, len(labels)):
        if labels[i] != current:
            if current in points and start is not None:
                points[current]['onsets'].append(start)
                points[current]['offsets'].append(i - 1)
            if labels[i] in points:
                start = i
            else:
                start = None
            current = labels[i]

    if current in points and start is not None:
        points[current]['onsets'].append(start)
        points[current]['offsets'].append(len(labels) - 1)

    return points

def evaluate_detection_metrics(all_preds, all_labels, sequence_length, sample_rate=250, tolerance_ms=150):
    tolerance_samples = int((tolerance_ms / 1000) * sample_rate)
    stats = {
        wave: {
            'TP_onset': 0, 'FP_onset': 0, 'FN_onset': 0,
            'TP_offset': 0, 'FP_offset': 0, 'FN_offset': 0,
            'onset_errors': [], 'offset_errors': []
        } for wave in [1, 2, 3]
    }

    seq_len = len(all_labels)
    for i in range(0, seq_len, sequence_length):
        true_seq = all_labels[i:i + sequence_length]
        pred_seq = all_preds[i:i + sequence_length]

        true_pts = extract_significant_points(true_seq)
        pred_pts = extract_significant_points(pred_seq)

        for wave in [1, 2, 3]:
            # Onsets
            matched_true = [False] * len(true_pts[wave]['onsets'])
            for p in pred_pts[wave]['onsets']:
                found = False
                for j, t in enumerate(true_pts[wave]['onsets']):
                    if not matched_true[j] and abs(p - t) <= tolerance_samples:
                        stats[wave]['TP_onset'] += 1
                        stats[wave]['onset_errors'].append(p - t)
                        matched_true[j] = True
                        found = True
                        break
                if not found:
                    stats[wave]['FP_onset'] += 1
            stats[wave]['FN_onset'] += matched_true.count(False)

            # Offsets
            matched_true = [False] * len(true_pts[wave]['offsets'])
            for p in pred_pts[wave]['offsets']:
                found = False
                for j, t in enumerate(true_pts[wave]['offsets']):
                    if not matched_true[j] and abs(p - t) <= tolerance_samples:
                        stats[wave]['TP_offset'] += 1
                        stats[wave]['offset_errors'].append(p - t)
                        matched_true[j] = True
                        found = True
                        break
                if not found:
                    stats[wave]['FP_offset'] += 1
            stats[wave]['FN_offset'] += matched_true.count(False)

    print("\n=== Significant Point Detection Metrics (±{} ms) ===".format(tolerance_ms))
    for wave in [1, 2, 3]:
        name = CLASS_NAMES[wave]
        s = stats[wave]
        for point in ['onset', 'offset']:
            TP = s[f'TP_{point}']
            FP = s[f'FP_{point}']
            FN = s[f'FN_{point}']
            errors = np.array(s[f'{point}_errors'])

            mean_err = np.mean(errors) if len(errors) > 0 else 0
            std_err = np.std(errors) if len(errors) > 0 else 0
            Se = TP / (TP + FN) if (TP + FN) > 0 else 0
            PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
            F1 = 2 * PPV * Se / (PPV + Se) if (PPV + Se) > 0 else 0

            print(f"\n{name} {point.capitalize()}:")
            print(f"  TP={TP}, FP={FP}, FN={FN}")
            print(f"  Mean Error (m): {mean_err:.2f} samples ({mean_err/sample_rate*1000:.2f} ms)")
            print(f"  Std Dev (σ): {std_err:.2f} samples ({std_err/sample_rate*1000:.2f} ms)")
            print(f"  Sensitivity (Se): {Se:.4f}")
            print(f"  Precision (PPV): {PPV:.4f}")
            print(f"  F1 Score: {F1:.4f}")

def evaluate(model, dataloader, loss_fn, device, num_classes, output_dir, sequence_length, sample_info=None):
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

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\nEval Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[CLASS_NAMES.get(i, f"Class {i}") for i in range(num_classes)], digits=4))

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES.values(),
                yticklabels=CLASS_NAMES.values(),
                cbar_kws={'label': 'Percentage (%)'})
    plt.title("Confusion Matrix (%)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    evaluate_detection_metrics(all_preds, all_labels, sequence_length=sequence_length)

    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser("Evaluate ECG Segmenter")
    parser.add_argument("--load_dir", type=str, default="MCG_segmentation/checkpoints/best")
    parser.add_argument("--data_dir_eval", type=str, default="MCG_segmentation/Datasets/val")
    parser.add_argument("--output_dir", type=str, default="MCG_segmentation/evaluation_results")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plot_sample_index", type=int)
    parser.add_argument("--sequence_length", type=int, default=2000) 

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, num_classes = load_model(args.load_dir, DEVICE)

    dataset = ECGFullDataset(
            data_dir=args.data_dir_eval,
            sequence_length=args.sequence_length,
            augmentation_prob=0.0,  # Disable augmentation
            sinusoidal_noise_mag=0.0,
            gaussian_noise_std=0.0,
            baseline_wander_mag=0.0,
            powerline_mag=0.0,
            respiratory_artifact_prob=0.0,
            heart_rate_variability_prob=0.0,
            morphology_warp_prob=0.0,
        )

    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    sample_info = None
    if args.plot_sample_index is not None and 0 <= args.plot_sample_index < len(dataset):
        sample_info = {
            'batch_idx': args.plot_sample_index // args.eval_batch_size,
            'item_idx_in_batch': args.plot_sample_index % args.eval_batch_size,
            'dataset_idx': args.plot_sample_index
        }

    loss_fn = nn.CrossEntropyLoss()
    evaluate(model, dataloader, loss_fn, DEVICE, num_classes, args.output_dir, args.sequence_length, sample_info)

if __name__ == "__main__":
    main()
