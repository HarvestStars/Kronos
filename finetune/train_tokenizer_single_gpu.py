import os
import sys
import json
import time
from time import gmtime, strftime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import comet_ml  # optional; only used if config['use_comet'] is True
except Exception:
    comet_ml = None

# Ensure project root is in path
sys.path.append("../")
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer

# Import shared utilities (no DDP helpers here)
from utils.training_utils import (
    set_seed,
    get_model_size,
    format_time,
)


def create_dataloaders(config: dict):
    """
    Creates and returns dataloaders for training and validation (single GPU / single process).

    Args:
        config (dict): A dictionary of configuration parameters.

    Returns:
        tuple: (train_loader, val_loader, train_dataset, valid_dataset)
    """
    print("[SingleGPU] Creating dataloaders...")
    train_dataset = QlibDataset("train")
    valid_dataset = QlibDataset("val")
    print(f"[SingleGPU] Train size: {len(train_dataset)}, Val size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=True,
        drop_last=False,
    )
    print(f"[SingleGPU] Dataloaders OK. Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset


def train_model(model, device, config, save_dir, logger):
    """
    Training and validation loop for single-GPU run.

    Args:
        model (nn.Module): The model to train.
        device (torch.device): CUDA or CPU device.
        config (dict): Configuration dictionary.
        save_dir (str): Directory to save checkpoints.
        logger (comet_ml.Experiment | None): Comet logger instance if enabled.

    Returns:
        tuple: (model, result_dict)
    """
    start_time = time.time()
    effective_bs = config["batch_size"] * config["accumulation_steps"]
    print(f"[SingleGPU] BATCHSIZE (per step): {config['batch_size']} | Effective batch size: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["tokenizer_learning_rate"],
        weight_decay=config["adam_weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config["tokenizer_learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.03,
        div_factor=10,
    )

    best_val_loss = float("inf")
    dt_result = {}
    batch_idx_global_train = 0

    for epoch_idx in range(config["epochs"]):
        epoch_start_time = time.time()
        model.train()

        # Set dataset seeds for reproducible sampling (if dataset exposes the helpers)
        if hasattr(train_dataset, "set_epoch_seed"):
            train_dataset.set_epoch_seed(epoch_idx * 10000)
        if hasattr(valid_dataset, "set_epoch_seed"):
            valid_dataset.set_epoch_seed(0)  # Keep validation sampling consistent

        for i, (ori_batch_x, _) in enumerate(train_loader):
            # Some datasets may return an extra leading dim; keep the original squeeze behavior but guard it
            if isinstance(ori_batch_x, torch.Tensor) and ori_batch_x.dim() >= 1:
                if ori_batch_x.dim() == 4 and ori_batch_x.size(0) == 1:
                    ori_batch_x = ori_batch_x.squeeze(0)
            ori_batch_x = ori_batch_x.to(device, non_blocking=True)  # (B, T, D)

            # --- Gradient Accumulation Loop ---
            current_batch_total_loss = 0.0
            for j in range(config["accumulation_steps"]):
                chunk = ori_batch_x.shape[0] // config["accumulation_steps"]
                start_idx = j * chunk
                end_idx = (j + 1) * chunk if j < config["accumulation_steps"] - 1 else ori_batch_x.shape[0]
                batch_x = ori_batch_x[start_idx:end_idx]

                # Forward pass
                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs

                # Loss calculation
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2  # Assuming w_1=w_2=1

                loss_scaled = loss / config["accumulation_steps"]
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            # --- Optimizer Step after Accumulation ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # --- Logging ---
            avg_loss = current_batch_total_loss / config["accumulation_steps"]
            if (batch_idx_global_train + 1) % config["log_interval"] == 0:
                print(
                    f"[SingleGPU][Epoch {epoch_idx + 1}/{config['epochs']} Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )
            if logger:
                logger.log_metric("train_tokenizer_loss_batch", avg_loss, step=batch_idx_global_train)
                logger.log_metric("train_vqvae_vq_loss_each_batch", bsq_loss.item(), step=batch_idx_global_train)
                logger.log_metric("train_recon_loss_pre_each_batch", recon_loss_pre.item(), step=batch_idx_global_train)
                logger.log_metric("train_recon_loss_each_batch", recon_loss_all.item(), step=batch_idx_global_train)
                logger.log_metric("tokenizer_learning_rate", optimizer.param_groups[0]["lr"], step=batch_idx_global_train)

            batch_idx_global_train += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                if isinstance(ori_batch_x, torch.Tensor) and ori_batch_x.dim() == 4 and ori_batch_x.size(0) == 1:
                    ori_batch_x = ori_batch_x.squeeze(0)
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)
                tot_val_loss_sum += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count += ori_batch_x.size(0)

        avg_val_loss = tot_val_loss_sum / val_sample_count if val_sample_count > 0 else 0.0

        # --- End of Epoch Summary & Checkpointing ---
        print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
        print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
        if logger:
            logger.log_metric("val_tokenizer_loss_epoch", avg_val_loss, epoch=epoch_idx)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"[SingleGPU] Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
            if logger:
                logger.log_model("best_model", save_path)

    return model, {"best_val_loss": best_val_loss}


def main(config: dict):
    """
    Orchestrates single-GPU training.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"], rank=0)  # reuse util for determinism

    save_dir = os.path.join(config["save_path"], config["tokenizer_save_folder_name"])
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    # Logger and summary
    comet_logger, master_summary = None, {}
    master_summary = {
        "start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        "save_directory": save_dir,
        "world_size": 1,
    }
    if config.get("use_comet") and comet_ml is not None:
        comet_logger = comet_ml.Experiment(
            api_key=config["comet_config"]["api_key"],
            project_name=config["comet_config"]["project_name"],
            workspace=config["comet_config"]["workspace"],
        )
        comet_logger.add_tag(config["comet_tag"])
        comet_logger.set_name(config["comet_name"])
        comet_logger.log_parameters(config)
        print("[SingleGPU] Comet Logger Initialized.")

    # Model Initialization
    model = KronosTokenizer.from_pretrained(config["pretrained_tokenizer_path"])
    model.to(device)

    print(f"[SingleGPU] Model Size: {get_model_size(model)}")

    # Start Training
    _, dt_result = train_model(model, device, config, save_dir, comet_logger)

    # Finalize and save summary
    master_summary["final_result"] = dt_result
    with open(os.path.join(save_dir, "summary_single_gpu.json"), "w") as f:
        json.dump(master_summary, f, indent=4)
    print("[SingleGPU] Training finished. Summary file saved.")
    if comet_logger:
        comet_logger.end()


if __name__ == "__main__":
    # Single-GPU entrypoint, no torchrun needed.
    config_instance = Config()
    main(config_instance.__dict__)
