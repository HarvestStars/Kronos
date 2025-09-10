
import os
import sys
import argparse
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

# Ensure project root is in the Python path
sys.path.append("../")
from config import Config
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference


# =================================================================================
# 1. Data Loading and Processing for Inference
# =================================================================================

class QlibTestDataset(Dataset):
    """
    Dataset for handling test data for inference.
    Iterates sequentially over all sliding windows and yields:
      - x:      context window (lookback_window) feature tensor
      - x_stamp: context time features
      - y_stamp: predict-window time features
      - y_vals: predict-window true OCLHV values (ground truth to compare with)
      - symbol: instrument id
      - anchor_ts: last time of the context window (t)
      - first_pred_ts: first time of the predict window (t+1)
    """

    def __init__(self, data: dict, config: Config):
        self.data = data
        self.config = config
        self.window_size = config.lookback_window + config.predict_window
        self.symbols = list(self.data.keys())
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.indices = []

        print("Preprocessing and building indices for test dataset...")
        for symbol in self.symbols:
            df = self.data[symbol].reset_index()
            # Generate time features on-the-fly
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df  # Store preprocessed dataframe

            num_samples = len(df) - self.window_size + 1
            if num_samples > 0:
                for i in range(num_samples):
                    anchor_ts = df.iloc[i + self.config.lookback_window - 1]['datetime']
                    first_pred_ts = df.iloc[i + self.config.lookback_window]['datetime']
                    self.indices.append((symbol, i, anchor_ts, first_pred_ts))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        symbol, start_idx, anchor_ts, first_pred_ts = self.indices[idx]
        df = self.data[symbol]

        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window

        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]

        x_raw = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)

        # true future OCLHV (ground truth) for the predict window
        y_vals = predict_df[self.feature_list].values.astype(np.float32)

        # time features for the predict window
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)

        # Instance-level normalization, consistent with training (on x only)
        x_mean_vec, x_std_vec = np.mean(x_raw, axis=0), np.std(x_raw, axis=0)
        x = (x_raw - x_mean_vec) / (x_std_vec + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(y_stamp),
            torch.from_numpy(y_vals),
            torch.from_numpy(x_mean_vec.astype(np.float32)),
            torch.from_numpy(x_std_vec.astype(np.float32)),
            symbol,
            anchor_ts,
            first_pred_ts,
        )


# =================================================================================
# 2. Inference: build per-symbol time series of predicted vs true close
# =================================================================================

def load_models(config: dict) -> tuple[KronosTokenizer, Kronos]:
    device = torch.device(config['device'])
    print(f"Loading models onto device: {device}...")
    tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_path']).to(device).eval()
    model = Kronos.from_pretrained(config['model_path']).to(device).eval()
    return tokenizer, model


def collate_fn_for_inference(batch):
    """
    Custom collate function to handle tensors + strings + timestamps.
    """
    x, x_stamp, y_stamp, y_vals, x_mean_vec, x_std_vec, symbols, anchor_ts, first_pred_ts = zip(*batch)
    x = torch.stack(x, dim=0)
    x_stamp = torch.stack(x_stamp, dim=0)
    y_stamp = torch.stack(y_stamp, dim=0)
    y_vals = torch.stack(y_vals, dim=0)
    x_mean_vec = torch.stack(x_mean_vec, dim=0)
    x_std_vec = torch.stack(x_std_vec, dim=0)
    return x, x_stamp, y_stamp, y_vals, x_mean_vec, x_std_vec, list(symbols), list(anchor_ts), list(first_pred_ts)


def generate_pred_truth_frames(config: dict, test_data: dict) -> dict[str, pd.DataFrame]:
    """
    Run inference and collect 1-step-ahead predicted close vs true close,
    aligned by the first predicted timestamp (t+1 for each window).

    Returns:
      dict[symbol] -> DataFrame(index=first_pred_ts, columns=['pred_close','true_close'])
    """
    tokenizer, model = load_models(config)
    device = torch.device(config['device'])

    base_cfg = Config()
    close_idx = base_cfg.feature_list.index('close')

    dataset = QlibTestDataset(data=test_data, config=base_cfg)
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'] // max(1, config['sample_count']),
        shuffle=False,
        num_workers=max(0, (os.cpu_count() or 2) // 2),
        collate_fn=collate_fn_for_inference,
        pin_memory=False,
    )

    per_symbol = defaultdict(list)

    with torch.no_grad():
        for x, x_stamp, y_stamp, y_vals, x_mean_vec, x_std_vec, symbols, anchor_ts, first_pred_ts in tqdm(loader, desc="Inference"):
            # x: [B, L, d], y_stamp: [B, P, tf_dim], y_vals: [B, P, d]
            preds = auto_regressive_inference(
                tokenizer, model, x.to(device), x_stamp.to(device), y_stamp.to(device),
                max_context=config['max_context'], pred_len=config['pred_len'], clip=config['clip'],
                T=config['T'], top_k=config['top_k'], top_p=config['top_p'], sample_count=config['sample_count']
            )
            # preds: [B, P, d]
            if isinstance(preds, torch.Tensor):
                preds_np = preds.detach().cpu().numpy()
            else:
                preds_np = preds

            # select 1-step ahead close in normalized space
            pred_norm_close = preds_np[:, -1, close_idx]

            # de-normalize to raw price space using per-item context mean/std
            x_mean_np = x_mean_vec.detach().cpu().numpy()
            x_std_np  = x_std_vec.detach().cpu().numpy()
            mean_close = x_mean_np[:, close_idx]
            std_close  = x_std_np[:, close_idx]
            pred_close_raw = pred_norm_close * (std_close + 1e-5) + mean_close

            true_step0_close = y_vals[:, 0, close_idx].detach().cpu().numpy()

            for i, sym in enumerate(symbols):
                per_symbol[sym].append((first_pred_ts[i], float(pred_close_raw[i]), float(true_step0_close[i])))

    # Build per-symbol DataFrames
    frames: dict[str, pd.DataFrame] = {}
    for sym, recs in per_symbol.items():
        df = pd.DataFrame(recs, columns=["datetime", "pred_close", "true_close"])
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        df = df.set_index("datetime")
        frames[sym] = df

    return frames


# =================================================================================
# 3. Plotting utilities
# =================================================================================

def plot_pred_vs_truth_per_symbol(frames: dict[str, pd.DataFrame], save_dir: str):
    """
    For each symbol, save a figure comparing predicted vs true close over time.
    One chart per figure. No custom colors.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_pngs = []

    for sym, df in frames.items():
        plt.figure(figsize=(12, 4))
        # Use matplotlib directly to comply with constraints
        plt.plot(df.index, df["true_close"], label="True Close")
        plt.plot(df.index, df["pred_close"], label="Pred Close")
        plt.title(f"{sym} — 1-step ahead Close: Prediction vs Truth")
        plt.xlabel("Datetime")
        plt.ylabel("Close")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fn = os.path.join(save_dir, f"pred_vs_truth_{sym}.png")
        plt.savefig(fn, dpi=200)
        plt.close()
        out_pngs.append(fn)

    return out_pngs


def save_frames_as_csv(frames: dict[str, pd.DataFrame], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    out_csvs = []
    for sym, df in frames.items():
        fn = os.path.join(save_dir, f"pred_vs_truth_{sym}.csv")
        df.to_csv(fn)
        out_csvs.append(fn)
    return out_csvs


def compute_basic_metrics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute MAE and RMSE per symbol between predicted and true close.
    """
    rows = []
    for sym, df in frames.items():
        pred = df["pred_close"].to_numpy(dtype=float)
        true = df["true_close"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(pred - true))) if len(pred) else np.nan
        rmse = float(np.sqrt(np.mean((pred - true) ** 2))) if len(pred) else np.nan
        rows.append({"symbol": sym, "MAE": mae, "RMSE": rmse, "n": len(pred)})
    return pd.DataFrame(rows).set_index("symbol")


# =================================================================================
# 4. Main Execution
# =================================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Kronos Inference and Plot Pred vs Truth")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference (e.g., 'cuda:0', 'cpu')")
    args = parser.parse_args()

    base_config = Config()
    run_config = {
        'device': args.device,
        'data_path': base_config.dataset_path,
        'result_save_path': base_config.backtest_result_path,  # reuse existing path
        'result_name': base_config.backtest_save_folder_name + "_pred_truth",
        'tokenizer_path': base_config.finetuned_tokenizer_path,
        'model_path': base_config.finetuned_predictor_path,
        'max_context': base_config.max_context,
        'pred_len': base_config.predict_window,
        'clip': base_config.clip,
        'T': base_config.inference_T,
        'top_k': base_config.inference_top_k,
        'top_p': base_config.inference_top_p,
        'sample_count': base_config.inference_sample_count,
        'batch_size': base_config.backtest_batch_size,
    }

    print("--- Running with Configuration ---")
    for key, val in run_config.items():
        print(f"{key:>20}: {val}")
    print("-" * 35)

    # --- Load Test Data ---
    test_data_path = os.path.join(run_config['data_path'], "test_data.pkl")
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    # --- Generate per-symbol frames ---
    frames = generate_pred_truth_frames(run_config, test_data)

    # --- Save results ---
    save_dir = os.path.join(run_config['result_save_path'], run_config['result_name'])
    os.makedirs(save_dir, exist_ok=True)

    # Save raw frames as CSV
    csv_paths = save_frames_as_csv(frames, os.path.join(save_dir, "csv"))
    print("Saved CSVs:")
    for p in csv_paths:
        print(" -", p)

    # Plot and save figures
    png_paths = plot_pred_vs_truth_per_symbol(frames, os.path.join(save_dir, "figs"))
    print("Saved figures:")
    for p in png_paths:
        print(" -", p)

    # Basic metrics
    metrics_df = compute_basic_metrics(frames)
    metrics_path = os.path.join(save_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path)
    print("Saved metrics to:", metrics_path)


if __name__ == '__main__':
    main()
