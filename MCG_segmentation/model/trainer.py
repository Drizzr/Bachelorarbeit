# model/trainer.py

import torch
import time
import json
import os
import csv 
from sklearn.metrics import f1_score 


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

        self.loss_fn = torch.nn.CrossEntropyLoss() 

        self.device = device
        self.log_filepath = log_filepath 
        self.losses = []
        self.val_losses = []

        self._init_log_file()


        print("--- Model Summary ---")
        print(model)
        print("---------------------")
        print(f"Trainer initialized. Starting from Epoch {self.epoch}.")
        print(f"Logging metrics to: {self.log_filepath}") # <-- Log file info
        print(f"Optimizer: {type(self.optimizer).__name__}")
        if self.lr_scheduler: 
            print(f"LR Scheduler: {type(self.lr_scheduler).__name__}")
        else: 
            print("LR Scheduler: None")
        print(f"Loss Function: {type(self.loss_fn).__name__}")
        print(f"Training on device: {self.device}")
        print("---------------------")

    def _init_log_file(self):
        """Creates the log file and writes the header if it doesn't exist."""
        # Check if directory exists, create if not
        log_dir = os.path.dirname(self.log_filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"Created log directory: {log_dir}")

        # Write header if file doesn't exist
        if not os.path.exists(self.log_filepath):
            print(f"Creating new log file: {self.log_filepath}")
            with open(self.log_filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "val_f1_macro", # <-- Added F1 score column
                    "learning_rate"
                ])
        else:
            print(f"Appending to existing log file: {self.log_filepath}")

    def _log_epoch_metrics(self, epoch_metrics: dict):
        """Appends a dictionary of epoch metrics to the CSV log file."""
        # Ensure order matches header
        header = ["epoch","train_loss","train_acc","val_loss","val_acc","val_f1_macro","learning_rate"]
        row = [epoch_metrics.get(key, 'N/A') for key in header] # Get value or 'N/A'

        try:
            with open(self.log_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except IOError as e:
            print(f"Error writing metrics to log file {self.log_filepath}: {e}")

    def train(self):
        epoch_start_time = time.time()
        mes = "Epoch {}, Step: {}/{}, Lr: {:.1E}, Loss: {:.4f}, Acc: {:.4f}, Step time (s): {:.2f}, ETA (Epoch min): {:.2f}"
        print(f"--- Starting Training ---")

        self.model.to(self.device)

        while self.epoch <= self.last_epoch:
            self.model.train()
            epoch_losses = 0.0
            epoch_acc = 0.0
            epoch_steps = 0
            step_start_time = time.time()

            for i, (signals, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                logits = self.model(signals)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.reshape(-1))

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

                if self.total_step % self.args.print_freq == 0:
                    avg_loss_print = sum(self.losses[-self.args.print_freq:]) / len(self.losses[-self.args.print_freq:])
                    avg_acc_epoch = epoch_acc / epoch_steps
                    current_lr = self.optimizer.param_groups[0]['lr']
                    step_time = time.time() - step_start_time

                    steps_left_epoch = len(self.train_loader) - (i + 1)
                    time_per_step = step_time / self.args.print_freq
                    eta_epoch_sec = steps_left_epoch * time_per_step
                    eta_epoch_min = eta_epoch_sec / 60

                    print(mes.format(
                        self.epoch, i + 1, len(self.train_loader),
                        current_lr, avg_loss_print, avg_acc_epoch,
                        step_time, eta_epoch_min,
                    ))

                    step_start_time = time.time()
                    self.losses = self.losses[-self.args.print_freq:]

            # --- End of Epoch ---
            avg_epoch_loss = epoch_losses / len(self.train_loader) if len(self.train_loader) > 0 else 0
            avg_epoch_acc = epoch_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0
            epoch_duration = time.time() - epoch_start_time

            print(f"\n--- Epoch {self.epoch} Summary ---")
            print(f"Avg Train Loss: {avg_epoch_loss:.4f}, Avg Train Acc: {avg_epoch_acc:.4f}, Duration: {epoch_duration:.2f}s")

            # --- Validation ---
            # Now returns loss, acc, and f1_score
            val_loss, val_acc, val_f1 = self.validate()
            self.val_losses.append(val_loss)

            # --- LR Scheduler Step (Epoch-wise or Plateau) ---
            if self.lr_scheduler and isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)

            # --- Log Epoch Metrics to CSV ---
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_metrics = {
                "epoch": self.epoch,
                "train_loss": avg_epoch_loss,
                "train_acc": avg_epoch_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1, # <-- Added F1
                "learning_rate": current_lr
            }
            self._log_epoch_metrics(epoch_metrics)
            # ---

            # --- Save Model ---
            if val_f1 > self.best_val_f1:
                print(f"Validation loss improved ({self.best_val_f1:.4f} -> {val_f1:.4f}). Saving best model...")
                self.best_val_f1 = val_f1
                self.save_model(is_best=True)
            if self.epoch % 5 == 0: # Save periodic checkpoint every 5 epochs
                print(f"Saving periodic checkpoint at epoch {self.epoch}...")
                self.save_model(is_best=False)

            print(f"--- End of Epoch {self.epoch} --- \n")
            self.epoch += 1
            epoch_start_time = time.time()

        print("--- Training Finished ---")


    def validate(self, limit=None):
        print("--- Running Validation ---")
        val_start_time = time.time()
        val_total_loss = 0.0
        val_total_acc = 0.0
        num_batches = 0
        all_preds = [] # <-- To calculate F1 score
        all_labels = [] # <-- To calculate F1 score
        self.model.eval()

        with torch.no_grad():
            for i, (signals, labels) in enumerate(self.val_loader):
                if limit is not None and i >= limit:
                    print(f"Validation limited to {limit} batches.")
                    break

                signals = signals.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                logits = self.model(signals)
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.reshape(-1))
                val_total_loss += loss.item() # Accumulate batch loss

                preds = torch.argmax(logits, dim=-1)
                correct_preds = (preds == labels).float()
                val_total_acc += correct_preds.mean().item() # Accumulate batch accuracy

                # Store predictions and labels for F1 score calculation
                all_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())

                num_batches += 1

        avg_loss = val_total_loss / num_batches if num_batches > 0 else 0
        avg_acc = val_total_acc / num_batches if num_batches > 0 else 0
        val_duration = time.time() - val_start_time

        # --- Calculate F1 Score ---
        if all_labels and all_preds:
            # Using macro average: calculates F1 for each class and finds unweighted mean.
            # Does not take label imbalance into account for the average score.
            # zero_division=0 prevents errors if a class has no predictions/true labels
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            f1 = 0.0 # Default if no data
        # ---

        # Print all metrics including F1
        print(f"Validation Results - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, Macro F1: {f1:.4f}, Duration: {val_duration:.2f}s")

        self.model.train()
        # Return all three metrics
        return avg_loss, avg_acc, f1


    def save_model(self, is_best=False):
        """Saves model, optimizer, scheduler states and parameters."""
        save_dir = self.args.save_dir + "/" + f"checkpoint_epoch_{self.epoch}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_filename = os.path.join(save_dir, f"model.pth")
        optimizer_filename = os.path.join(save_dir, f"optimizer.pth")
        scheduler_filename = os.path.join(save_dir, f"lr_scheduler.pth")
        params_filename = os.path.join(save_dir, "params.json")

        torch.save(self.model.state_dict(), model_filename)
        torch.save(self.optimizer.state_dict(), optimizer_filename)
        if self.lr_scheduler:
            torch.save(self.lr_scheduler.state_dict(), scheduler_filename) # <-- Save scheduler

        params = {
            "epoch": self.epoch,
            "total_step": self.total_step,
            "best_val_f1": self.best_val_f1,
            "model_args": self.model.__dict__,
            # Save args used for this run for reproducibility
            "args": vars(self.args) # Save command line args
        }
        with open(params_filename, "w") as f:
            json.dump(params, f, indent=4)

        # If it's the best model, also save it with a fixed 'best' name
        if is_best:
            print(f"Saving best model checkpoint...")
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
                json.dump(params, f, indent=4) # Save params for best model too

        print(f"Checkpoint saved successfully to {save_dir}")
