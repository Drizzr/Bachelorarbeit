import torch
import time
import json
import os
import csv
import warnings

import numpy as np
# Matplotlib and Seaborn are no longer needed
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm


def sample_from_model(model, device, data: torch.Tensor, min_duration_sec: float = 0.00):
    if data.numel() == 0:
        warnings.warn("No data to segment.")
        return np.array([])

    data = data.to(device)
    with torch.no_grad():
        logits = model(data)
        probabilities = torch.softmax(logits, dim=-1)
        _, predicted_indices_pt = torch.max(probabilities, dim=-1)

    predicted_indices = predicted_indices_pt.cpu().numpy()
    # Flatten the predictions for processing
    return predicted_indices.flatten()


class Trainer(object):
    def __init__(self, model, train_loader, args, val_loader, optimizer, device,
                log_filepath: str,
                lr_scheduler=None,
                init_epoch=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.step = 0
        self.total_step = 0
        self.epoch = init_epoch
        self.last_epoch = args.num_epochs
        self.best_val_f1 = 0
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.log_filepath = log_filepath
        self.losses = []

        # Constants for metrics
        self.CLASS_NAMES = {0: "No Wave", 1: "P Wave", 2: "QRS", 3: "T Wave"}
        
        self._init_log_file()

        print("\n" + "="*50)
        print("🚀 Trainer Initialization (Text-Only Validation)")
        print("="*50)
        print(f"Model: {type(self.model).__name__}")
        print(f"Log File: {self.log_filepath}")
        print(f"Device: {self.device}")
        print("="*50 + "\n")

    def _init_log_file(self):
        log_dir = os.path.dirname(self.log_filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro", "learning_rate"])

    def _log_epoch_metrics(self, epoch_metrics: dict):
        header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro", "learning_rate"]
        row = [epoch_metrics.get(key, 'N/A') for key in header]
        with open(self.log_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _extract_significant_points(self, labels):
        points = {1: {'onsets': [], 'offsets': []}, 2: {'onsets': [], 'offsets': []}, 3: {'onsets': [], 'offsets': []}}
        if not labels: return points
        
        # Add a sentinel value to ensure the last segment is processed
        labels_with_sentinel = np.append(np.array(labels), -1)
        
        current_label = labels_with_sentinel[0]
        start_idx = 0
        
        for i in range(1, len(labels_with_sentinel)):
            if labels_with_sentinel[i] != current_label:
                if current_label in points:
                    points[current_label]['onsets'].append(start_idx)
                    points[current_label]['offsets'].append(i - 1)
                current_label = labels_with_sentinel[i]
                start_idx = i
        return points

    def _evaluate_detection_metrics(self, all_preds, all_labels, sequence_length, sample_rate=250, tolerance_ms=150):
        tolerance_samples = int((tolerance_ms / 1000) * sample_rate)
        stats = {wave: {f'{pt}_{metric}': 0 for pt in ['TP', 'FP', 'FN'] for metric in ['onset', 'offset']} |
                {f'{pt}_errors': [] for pt in ['onset', 'offset']} for wave in [1, 2, 3]}
        
        num_sequences = len(all_labels) // sequence_length
        for i in range(num_sequences):
            start = i * sequence_length
            end = start + sequence_length
            true_seq = all_labels[start:end]
            pred_seq = all_preds[start:end]
            
            true_pts = self._extract_significant_points(true_seq)
            pred_pts = self._extract_significant_points(pred_seq)
            
            for wave in [1, 2, 3]:
                for pt_type in ['onsets', 'offsets']:
                    gt_points = true_pts[wave][pt_type]
                    pred_points = pred_pts[wave][pt_type]
                    matched_gt = [False] * len(gt_points)
                    
                    for p_point in pred_points:
                        found_match = False
                        for j, t_point in enumerate(gt_points):
                            if not matched_gt[j] and abs(p_point - t_point) <= tolerance_samples:
                                stats[wave][f'TP_{pt_type[:-1]}'] += 1
                                stats[wave][f'{pt_type[:-1]}_errors'].append(p_point - t_point)
                                matched_gt[j] = True
                                found_match = True
                                break # Match found, move to next predicted point
                        if not found_match:
                            stats[wave][f'FP_{pt_type[:-1]}'] += 1
                    
                    stats[wave][f'FN_{pt_type[:-1]}'] += matched_gt.count(False)

        print(f"\n=== Validation Significant Point Detection Metrics (Tolerance: ±{tolerance_ms} ms) ===")
        for wave in [1, 2, 3]:
            name = self.CLASS_NAMES[wave]
            for point in ['onset', 'offset']:
                TP = stats[wave][f'TP_{point}']
                FP = stats[wave][f'FP_{point}']
                FN = stats[wave][f'FN_{point}']
                errors = np.array(stats[wave][f'{point}_errors'])
                
                mean_err = errors.mean() if len(errors) > 0 else 0
                std_err = errors.std() if len(errors) > 0 else 0
                Se = TP / (TP + FN) if (TP + FN) > 0 else 0
                PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
                F1 = 2 * PPV * Se / (PPV + Se) if (PPV + Se) > 0 else 0
                
                print(f"\n{name} {point.capitalize()}:")
                print(f"  TP={TP}, FP={FP}, FN={FN}")
                print(f"  Mean Error: {mean_err:.2f} samples ({mean_err / sample_rate * 1000:.2f} ms)")
                print(f"  Std Dev:    {std_err:.2f} samples ({std_err / sample_rate * 1000:.2f} ms)")
                print(f"  Sensitivity (Se): {Se:.4f}, Precision (PPV): {PPV:.4f}, F1 Score: {F1:.4f}")
        print("="*65)

    def train(self):
        print("\n" + "="*50)
        print("🏋️ Starting Training")
        print("="*50)
        self.model.to(self.device)

        while self.epoch <= self.last_epoch:
            epoch_start_time = time.time()
            self.model.train()
            epoch_losses, epoch_acc, epoch_steps = 0.0, 0.0, 0
            
            train_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.last_epoch} Training", leave=True)
            for signals, labels in train_bar:
                self.optimizer.zero_grad()
                signals, labels = signals.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True).long()
                
                logits = self.model(signals)
                loss = self.focal_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()
                if hasattr(self.args, 'clip') and self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                if self.lr_scheduler and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()
                
                with torch.no_grad():
                    epoch_acc += (torch.argmax(logits, dim=-1) == labels).float().mean().item()

                epoch_losses += loss.item()
                epoch_steps += 1
                train_bar.set_postfix({'Loss': f'{epoch_losses / epoch_steps:.4f}', 'Acc': f'{epoch_acc / epoch_steps:.4f}'})

            avg_epoch_loss = epoch_losses / len(self.train_loader)
            avg_epoch_acc = epoch_acc / len(self.train_loader)
            print(f"\nEpoch {self.epoch} Summary - Train Loss: {avg_epoch_loss:.4f}, Train Acc: {avg_epoch_acc:.4f}, Time: {time.time() - epoch_start_time:.2f}s")
            
            val_loss, val_acc, val_f1 = self.validate()

            if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)
            
            self._log_epoch_metrics({
                "epoch": self.epoch, "train_loss": avg_epoch_loss, "train_acc": avg_epoch_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1_macro": val_f1,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

            if val_f1 > self.best_val_f1:
                print(f"✅ Validation F1-score improved ({self.best_val_f1:.4f} -> {val_f1:.4f}). Saving best model...")
                self.best_val_f1 = val_f1
                self.save_model(is_best=True)
            if self.epoch % 5 == 0:
                self.save_model(is_best=False)

            self.epoch += 1
        print("🏁 Training Finished")

    def validate(self):
        print("\n" + "-"*50 + "\n🔍 Running Validation (Text-Stats Only)" + "\n" + "-"*50)
        val_start_time = time.time()
        self.model.eval()
        all_preds, all_labels = [], []
        
        val_bar = tqdm(self.val_loader, desc=f"Epoch {self.epoch}/{self.last_epoch} Validation", leave=False)
        
        with torch.no_grad():
            for signals, labels in val_bar:
                signals = signals.to(self.device)
                
                preds = sample_from_model(self.model, self.device, signals)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().flatten().tolist())

        # --- METRICS CALCULATION AND PRINTING ---
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"\nValidation Accuracy: {acc:.4f}")
        print(f"Validation Macro F1-Score: {f1:.4f}")
        
        print("\n--- Classification Report ---")
        print(classification_report(
            all_labels, all_preds, 
            target_names=[self.CLASS_NAMES.get(i, f"Class {i}") for i in range(len(self.CLASS_NAMES))], 
            digits=4, zero_division=0
        ))

        # --- DETECTION METRICS ---
        # Assuming sequence_length is accessible via the dataloader's dataset
        sequence_length = getattr(self.val_loader.dataset, 'sequence_length', 500)
        self._evaluate_detection_metrics(all_preds, all_labels, sequence_length=sequence_length)
        
        print(f"Validation finished in {time.time() - val_start_time:.2f}s")
        self.model.train() # Set model back to training mode
        
        # We return a dummy loss, as it's not the primary metric here
        return 0.0, acc, f1

    def save_model(self, is_best=False):
        save_dir = os.path.join(self.args.save_dir, "best" if is_best else f"checkpoint_epoch_{self.epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pth"))
        if self.lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(save_dir, "lr_scheduler.pth"))
        
        params = {"epoch": self.epoch, "best_val_f1": self.best_val_f1, "args": vars(self.args)}
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
        print(f"💾 Checkpoint saved to {save_dir}")

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        probs = torch.softmax(logits, dim=-1)
        # Use gather to select the probabilities of the true classes
        true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        # Add a small epsilon to prevent log(0)
        focal_loss = -alpha * ((1 - true_probs) ** gamma) * torch.log(true_probs + 1e-9)
        return focal_loss.mean()