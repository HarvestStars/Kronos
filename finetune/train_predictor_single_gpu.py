import os
import sys
import json
import time
from time import gmtime, strftime

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# comet_ml is optional
try:
    import comet_ml
except Exception:
    comet_ml = None

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities (DDP helpers removed for single-GPU mode)
from utils.training_utils import (
    set_seed,
    get_model_size,
    format_time,
)


def create_dataloaders(config: dict):
    """
    Creates and returns dataloaders for training and validation (single GPU / single process).

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset).
    """
    print("[SingleGPU] Creating dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[SingleGPU] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, tokenizer, device, config, save_dir, logger):
    """
    The main training and validation loop for the predictor (single GPU).
    """
    start_time = time.time()
    effective_bs = config['batch_size']
    print(f"[SingleGPU] BATCHSIZE: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['predictor_learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        weight_decay=config['adam_weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['predictor_learning_rate'],
        steps_per_epoch=len(train_loader), epochs=config['epochs'],
        pct_start=0.03, div_factor=10
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()

        # keep dataset sampling deterministic if dataset exposes the hooks
        if hasattr(train_dataset, 'set_epoch_seed'):
            train_dataset.set_epoch_seed(epoch_idx * 10000)
        if hasattr(valid_dataset, 'set_epoch_seed'):
            valid_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            # Some datasets may add an extra leading dim; guard the squeeze
            if isinstance(batch_x, torch.Tensor) and batch_x.dim() == 4 and batch_x.size(0) == 1:
                batch_x = batch_x.squeeze(0)
            if isinstance(batch_x_stamp, torch.Tensor) and batch_x_stamp.dim() == 4 and batch_x_stamp.size(0) == 1:
                batch_x_stamp = batch_x_stamp.squeeze(0)

            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenize input data on-the-fly
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)  # (B,T) longs

            # Prepare inputs and targets for the language model
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

            # Forward pass and loss calculation
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            # Logging
            if (batch_idx_global + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(
                    f"[SingleGPU][Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {lr:.6f}, Loss: {loss.item():.4f}"
                )
            if logger:
                lr = optimizer.param_groups[0]['lr']
                logger.log_metric('train_predictor_loss_batch', loss.item(), step=batch_idx_global)
                logger.log_metric('train_S1_loss_each_batch', s1_loss.item(), step=batch_idx_global)
                logger.log_metric('train_S2_loss_each_batch', s2_loss.item(), step=batch_idx_global)
                logger.log_metric('predictor_learning_rate', lr, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum = 0.0
        val_batches_processed = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                if isinstance(batch_x, torch.Tensor) and batch_x.dim() == 4 and batch_x.size(0) == 1:
                    batch_x = batch_x.squeeze(0)
                if isinstance(batch_x_stamp, torch.Tensor) and batch_x_stamp.dim() == 4 and batch_x_stamp.size(0) == 1:
                    batch_x_stamp = batch_x_stamp.squeeze(0)

                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]

                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

                tot_val_loss_sum += val_loss.item()
                val_batches_processed += 1

        avg_val_loss = tot_val_loss_sum / max(val_batches_processed, 1)

        # --- End of Epoch Summary & Checkpointing ---
        print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
        print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
        if logger:
            logger.log_metric('val_predictor_loss_epoch', avg_val_loss, epoch=epoch_idx)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"[SingleGPU] Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")

    dt_result = {'best_val_loss': best_val_loss}
    return dt_result


def main(config: dict):
    """Main function to orchestrate the single-GPU training process."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'], rank=0)

    save_dir = os.path.join(config['save_path'], config['predictor_save_folder_name'])
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Logger and summary setup
    comet_logger, master_summary = None, {}
    master_summary = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'save_directory': save_dir,
        'world_size': 1,
    }
    if config.get('use_comet') and comet_ml is not None:
        comet_logger = comet_ml.Experiment(
            api_key=config['comet_config']['api_key'],
            project_name=config['comet_config']['project_name'],
            workspace=config['comet_config']['workspace'],
        )
        comet_logger.add_tag(config['comet_tag'])
        comet_logger.set_name(config['comet_name'])
        comet_logger.log_parameters(config)
        print("[SingleGPU] Comet Logger Initialized.")

    # Model Initialization
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path'])
    tokenizer.eval().to(device)

    model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    model.to(device)

    print(f"[SingleGPU] Predictor Model Size: {get_model_size(model)}")

    # Start Training
    dt_result = train_model(model, tokenizer, device, config, save_dir, comet_logger)

    # Finalize
    master_summary['final_result'] = dt_result
    with open(os.path.join(save_dir, 'summary_single_gpu.json'), 'w') as f:
        json.dump(master_summary, f, indent=4)
    print('Training finished. Summary file saved.')
    if comet_logger:
        comet_logger.end()


if __name__ == '__main__':
    # Single-GPU entrypoint, no torchrun needed.
    config_instance = Config()
    main(config_instance.__dict__)
