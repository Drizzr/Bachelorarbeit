# train.py

import argparse
from torch.utils.data import DataLoader
from model.model import * # Import necessary models
from model.trainer import Trainer # Import Trainer
from model.data_loader import * # Import dataset class
import torch
import sys
import json
import os
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


def load_from_checkpoint(args, train_loader, val_loader, device):
    """Load model from checkpoint"""
    print("loading model from checkpoint...")
    param_path = os.path.join(args.load_dir, "params.json")
    model_path = os.path.join(args.load_dir, "model.pth")
    optimizer_path = os.path.join(args.load_dir, "optimizer.pth")
    scheduler_path = os.path.join(args.load_dir, "lr_scheduler.pth") # Define path

    if not all(os.path.exists(p) for p in [param_path, model_path, optimizer_path]):
        raise FileNotFoundError(f"Missing required checkpoint files (model.pth, optimizer.pth, params.json) in: {args.load_dir}")
    else:
        print(f"Loading from latest found epoch checkpoint files in {args.load_dir}")

    with open(param_path, "r") as f:
        params = json.load(f)

    # Define model structure based on args (or potentially loaded args)
    model = Conv1D_Attention_Segmenter()
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters()) # LR will be overwritten

    # Define scheduler
    # Ensure train_loader has been initialized before calling this function if needed here
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                                    step_size_up=len(train_loader) * args.cycle_epochs_up,
                                    mode='triangular2', cycle_momentum=False)

    # Load states
    model.load_state_dict(torch.load(model_path, map_location=device))
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    # Load scheduler state if it exists
    if os.path.exists(scheduler_path):
        try:
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
            print("LR scheduler state loaded.")
        except Exception as e:
            print(f"Warning: Could not load LR scheduler state from {scheduler_path}: {e}")
    else:
        print(f"Warning: LR scheduler state file not found at {scheduler_path}")


    print("Model, optimizer, and scheduler (if found) loaded successfully.")

    # Restore epoch and step counts
    init_epoch = params.get("epoch", 1) + 1 # Start from the next epoch
    total_step = params.get("total_step", 0)
    best_val_loss = params.get("best_val_loss", float('inf'))

    trainer = Trainer(model, train_loader, args, val_loader, optimizer, device,
                    lr_scheduler=scheduler,
                    init_epoch=init_epoch, log_filepath=args.metrics_file) # Pass metrics file path

    trainer.total_step = total_step
    trainer.best_val_loss = best_val_loss # Restore best known loss

    print(f"Resuming training from Epoch {init_epoch}, Total Steps: {total_step}")

    return trainer, model


def main():

    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Train ECG Segmentation Model")

    # Training Process Args
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=64, help="Batch size for validation")
    parser.add_argument("--print_freq", type=int, default=50, help="Frequency of printing training stats (in steps)")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value (0 to disable)")

    # Checkpointing Args
    parser.add_argument("--from_check_point", action='store_true', default=False, help="Resume training from a checkpoint")
    parser.add_argument("--load_dir", type=str, default="MCG_segmentation/checkpoints", help="Directory containing checkpoint files (required if --from_check_point)")
    parser.add_argument("--save_dir", type=str, default="MCG_segmentation/checkpoints", help="Directory where model checkpoints will be saved")

    # Data Args
    parser.add_argument("--data_dir_train", type=str, default="MCG_segmentation/qtdb/processed/train", help="Path to training data directory")
    parser.add_argument("--data_dir_val", type=str, default="MCG_segmentation/qtdb/processed/val", help="Path to validation data directory")
    parser.add_argument("--sinusoidal_noise_mag", type=float, default=0.05, help="Magnitude of sinusoidal noise added during training")
    parser.add_argument("--sequence_length", type=int, default=500, help="Length of ECG sequence segments")
    parser.add_argument("--overlap", type=int, default=400, help="Overlap between consecutive sequence segments")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # LR Scheduler Args
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate (for Adam and CyclicLR)")
    parser.add_argument("--base_lr", type=float, default=1e-5, help="Base learning rate for CyclicLR")

    # Logging Arg
    parser.add_argument("--metrics_file", type=str, default="MCG_segmentation/logs/training_metrics.csv", help="Path to CSV file for saving epoch metrics")

    args = parser.parse_args()

    # Validate args
    if args.from_check_point and not args.load_dir:
        parser.error("--load_dir is required when --from_check_point is set.")

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)


    train_loader = None; val_loader = None
    try:
        print("Setting up datasets...")
        # Ensure wavelet_type is passed if needed by ECGFullDataset
        train_dataset = ECGFullDataset(
                data_dir=args.data_dir_train, overlap=args.overlap, sequence_length=args.sequence_length,
                sinusoidal_noise_mag=args.sinusoidal_noise_mag #, wavelet=wavelet_type
            )
        val_dataset = ECGFullDataset(
            data_dir=args.data_dir_val, overlap=args.overlap, sequence_length=args.sequence_length,
            sinusoidal_noise_mag=args.sinusoidal_noise_mag #, wavelet=wavelet_type
        )

        if len(train_dataset) == 0: 
            print(f"Warning: Training dataset is empty. Check path: {args.data_dir_train}")
        if len(val_dataset) == 0: 
            print(f"Warning: Validation dataset is empty. Check path: {args.data_dir_val}")
        
        print("Setting up dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=False, num_workers=args.num_workers, pin_memory=True)
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    except FileNotFoundError as e: 
        print(f"Error initializing datasets: {e}"); sys.exit(1)
    except ValueError as e: 
        print(f"Error initializing datasets: {e}"); sys.exit(1)
    # --- End Dataset & DataLoader Setup ---


    # --- Device Setup ---
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- End Device Setup ---

    # --- Model, Optimizer, Scheduler, Trainer Initialization ---
    if args.from_check_point:
        try:
            trainer, model = load_from_checkpoint(args, train_loader, val_loader, device)
        except FileNotFoundError as e: 
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred loading checkpoint: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

    else: # Start training from scratch
        print("Initializing new model, optimizer, and scheduler...")
        model = Conv1D_Attention_Segmenter()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=1e-4)
        if not train_loader: raise RuntimeError("Cannot initialize scheduler: train_loader is not available.")
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.num_epochs * len(train_loader), # T_max is usually specified in *steps* not epochs
                eta_min=args.base_lr # Anneal down to base_lr (or 0)
            )

        # Instantiate Trainer with log file path and STANDARD loss function
        trainer = Trainer(model, train_loader, args, val_loader, optimizer, device,
                        log_filepath=args.metrics_file, # Pass metrics file path
                        lr_scheduler=scheduler) 
        
    # --- Print Setup Summary ---
    # (Remains the same)
    print("_________________________________________________________________")
    print("CONFIGURATION:")
    for arg, value in vars(args).items(): 
        print(f"{arg:>20}: {value}")
    print(f"{'Device':>20}: {device}")
    print(f"{'Trainable Params':>20}: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("_________________________________________________________________")
    # --- End Print Setup Summary ---

    # --- Training Loop ---
    try:
        print("Starting training process...")
        trainer.train()
        print("Training finished successfully.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model state...")
        try: trainer.save_model(is_best=False); print("Model saved after interruption.")
        except Exception as save_e: print(f"Could not save model after interruption: {save_e}")
        print("Exiting."); sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during training: {e}"); import traceback; traceback.print_exc()
        print("Attempting to save model state after error...")
        try: trainer.save_model(is_best=False); print("Model saved after error.")
        except Exception as save_e: print(f"Could not save model after error: {save_e}")
        sys.exit(1)
    finally:
        print("Training script finished.")


if __name__ == "__main__":
    main()
