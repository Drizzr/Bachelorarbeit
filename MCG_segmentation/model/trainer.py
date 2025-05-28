import torch
import time
import json
import os
import csv
from sklearn.metrics import f1_score
from tqdm import tqdm  # Added for progress bar

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
        self.val_losses = []

        self._init_log_file()

        # Improved print output for initialization
        print("\n" + "="*50)
        print("🚀 Trainer Initialization")
        print("="*50)
        print(f"Model: {type(self.model).__name__}")
        print(f"Starting Epoch: {self.epoch}/{self.last_epoch}")
        print(f"Log File: {self.log_filepath}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"LR Scheduler: {type(self.lr_scheduler).__name__ if self.lr_scheduler else 'None'}")
        print(f"Loss Function: FocalLoss")
        print(f"Device: {self.device}")
        print("="*50 + "\n")

    def _init_log_file(self):
        """Creates the log file and writes the header if it doesn't exist."""
        log_dir = os.path.dirname(self.log_filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"📁 Created log directory: {log_dir}")

        if not os.path.exists(self.log_filepath):
            print(f"📝 Creating new log file: {self.log_filepath}")
            with open(self.log_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "val_f1_macro",
                    "learning_rate"
                ])
        else:
            print(f"📝 Appending to existing log file: {self.log_filepath}")

    def _log_epoch_metrics(self, epoch_metrics: dict):
        """Appends a dictionary of epoch metrics to the CSV log file."""
        header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro", "learning_rate"]
        row = [epoch_metrics.get(key, 'N/A') for key in header]

        try:
            with open(self.log_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except IOError as e:
            print(f"❌ Error writing metrics to log file {self.log_filepath}: {e}")

    def train(self):
        print("\n" + "="*50)
        print("🏋️ Starting Training")
        print("="*50)

        self.model.to(self.device)

        while self.epoch <= self.last_epoch:
            epoch_start_time = time.time()
            self.model.train()
            epoch_losses = 0.0
            epoch_acc = 0.0
            epoch_steps = 0

            # Initialize tqdm progress bar for training
            train_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.last_epoch}", 
                             total=len(self.train_loader), leave=True)

            for i, (signals, labels) in enumerate(train_bar):
                self.optimizer.zero_grad()

                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                logits = self.model(signals)
                loss = self.focal_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                loss.backward()
                if hasattr(self.args, 'clip') and self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                if self.lr_scheduler and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    correct_preds = (preds == labels).float()
                    batch_acc = correct_preds.mean()
                    epoch_acc += batch_acc.item()

                current_loss = loss.item()
                epoch_losses += current_loss
                self.losses.append(current_loss)
                epoch_steps += 1
                self.total_step += 1

                # Update progress bar with current metrics
                if self.total_step % self.args.print_freq == 0:
                    avg_loss_print = sum(self.losses[-self.args.print_freq:]) / len(self.losses[-self.args.print_freq:])
                    avg_acc_epoch = epoch_acc / epoch_steps
                    current_lr = self.optimizer.param_groups[0]['lr']
                    train_bar.set_postfix({
                        'Loss': f'{avg_loss_print:.4f}',
                        'Acc': f'{avg_acc_epoch:.4f}',
                        'LR': f'{current_lr:.1e}'
                    })
                    self.losses = self.losses[-self.args.print_freq:]

            # Calculate epoch averages
            avg_epoch_loss = epoch_losses / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_epoch_acc = epoch_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0
            epoch_duration = time.time() - epoch_start_time

            # Improved epoch summary print
            print("\n" + "-"*50)
            print(f"📊 Epoch {self.epoch} Summary")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Train Acc:  {avg_epoch_acc:.4f}")
            print(f"  Duration:   {epoch_duration:.2f}s")
            print("-"*50)

            # Validation
            val_loss, val_acc, val_f1 = self.validate()

            # LR Scheduler Step
            if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)

            # Log Epoch Metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_metrics = {
                "epoch": self.epoch,
                "train_loss": avg_epoch_loss,
                "train_acc": avg_epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "learning_rate": current_lr
            }
            self._log_epoch_metrics(epoch_metrics)

            # Save Model
            if val_f1 > self.best_val_f1:
                print(f"✅ Validation F1-score improved ({self.best_val_f1:.4f} -> {val_f1:.4f}). Saving best model...")
                self.best_val_f1 = val_f1
                self.save_model(is_best=True)
            if self.epoch % 5 == 0:
                print(f"💾 Saving periodic checkpoint at epoch {self.epoch}...")
                self.save_model(is_best=False)

            print(f"\n🎉 End of Epoch {self.epoch}\n")
            self.epoch += 1

        print("="*50)
        print("🏁 Training Finished")
        print("="*50)

    def validate(self, limit=None):
        print("\n" + "-"*50)
        print("🔍 Running Validation")
        print("-"*50)

        val_start_time = time.time()
        val_total_loss = 0.0
        val_total_acc = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        self.model.eval()

        # Initialize tqdm progress bar for validation
        val_bar = tqdm(self.val_loader, desc="Validation", total=len(self.val_loader) if limit is None else limit, leave=False)

        with torch.no_grad():
            for i, (signals, labels) in enumerate(val_bar):
                if limit is not None and i >= limit:
                    print(f"⏳ Validation limited to {limit} batches.")
                    break

                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                logits = self.model(signals)
                loss = self.focal_loss(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                val_total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                correct_preds = (preds == labels).float()
                val_total_acc += correct_preds.mean().item()

                all_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())

                num_batches += 1

                # Update progress bar with current metrics
                val_bar.set_postfix({
                    'Loss': f'{val_total_loss / num_batches:.4f}',
                    'Acc': f'{val_total_acc / num_batches:.4f}'
                })

        avg_loss = val_total_loss / num_batches if num_batches > 0 else 0
        avg_acc = val_total_acc / num_batches if num_batches > 0 else 0
        val_duration = time.time() - val_start_time

        # Calculate F1 Score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels and all_preds else 0.0

        # Improved validation results print
        print("\n" + "-"*50)
        print(f"📈 Validation Results")
        print(f"  Avg Loss:   {avg_loss:.4f}")
        print(f"  Avg Acc:    {avg_acc:.4f}")
        print(f"  Macro F1:   {f1:.4f}")
        print(f"  Duration:   {val_duration:.2f}s")
        print("-"*50)

        self.model.train()
        return avg_loss, avg_acc, f1

    def save_model(self, is_best=False):
        """Saves model, optimizer, scheduler states and parameters."""
        save_dir = self.args.save_dir + "/" + f"checkpoint_epoch_{self.epoch}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_filename = os.path.join(save_dir, f"model.pth")
        optimizer_filename = os.path.join(save_dir, f"optimizer.pth")
        scheduler_filename = os.path.join(save_dir, f"lr_scheduler.pth")
        params_filename = os.path.join(save_dir, "train_params.json")

        torch.save(self.model.state_dict(), model_filename)
        torch.save(self.optimizer.state_dict(), optimizer_filename)
        if self.lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), scheduler_filename)

        params = {
            "epoch": self.epoch,
            "total_step": self.total_step,
            "best_val_f1": self.best_val_f1,
            "args": vars(self.args)
        }
        with open(params_filename, "w") as f:
            json.dump(params, f, indent=4)

        if is_best:
            print(f"🌟 Saving best model checkpoint...")
            save_dir = self.args.save_dir + "/" + f"best"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            best_model_path = os.path.join(save_dir, "model.pth")
            best_optim_path = os.path.join(save_dir, "optimizer.pth")
            best_sched_path = os.path.join(save_dir, "lr_scheduler.pth")
            best_params_path = os.path.join(save_dir, "params.json")

            torch.save(self.model.state_dict(), best_model_path)
            torch.save(self.optimizer.state_dict(), best_optim_path)
            if self.lr_scheduler:
                torch.save(self.lr_scheduler.state_dict(), best_sched_path)
            with open(best_params_path, "w") as f:
                json.dump(params, f, indent=4)

        print(f"💾 Checkpoint saved to {save_dir}")

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Focal Loss implementation.
        Args:
            logits: Model predictions (raw scores).
            labels: Ground truth labels.
            alpha: Weighting factor for the class.
            gamma: Focusing parameter to reduce loss for well-classified examples.
        Returns:
            Computed focal loss value.
        """
        probs = torch.softmax(logits, dim=-1)
        true_probs = probs[torch.arange(len(labels)), labels]
        focal_loss = -alpha * ((1 - true_probs) ** gamma) * torch.log(true_probs + 1e-8)
        return focal_loss.mean()