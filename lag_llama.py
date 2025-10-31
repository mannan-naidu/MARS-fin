# train_per_stock_lagllama_auto.py
import os
import json
import glob
import math
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoConfig

def coerce_numeric(series: pd.Series) -> pd.Series:
    # Convert to numeric and drop non-numeric rows
    s = pd.to_numeric(series, errors="coerce")
    return s

CONFIG = {
    "data_dir": "Nifty50_CSVs",
    "date_col": "Date",
    "value_col": "Close",
    "context_length": 64,
    "prediction_horizon": 1,  # next-step prediction
    "train_split": 0.9,
    "batch_size": 64,
    "num_epochs": 6,  # adjust as needed
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    "model_id": "time-series-foundation-models/lag-llama",
    "runs_root": "runs",
    "mixed_precision": True,
    "shuffle_windows": True,
    "max_windows_per_series": None,  # e.g., 5000 to cap
    "eval_every_steps": 200,
    "save_every_epochs": 1,
    "trust_remote_code": True,
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    return device

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def split_series_by_time(series: pd.Series, train_split: float) -> Tuple[pd.Series, pd.Series]:
    n = len(series)
    cut = int(n * train_split)
    return series.iloc[:cut].reset_index(drop=True), series.iloc[cut:].reset_index(drop=True)

def make_windows(values: np.ndarray, context_length: int, horizon: int):
    xs, ys = [], []
    max_start = len(values) - context_length - horizon + 1
    for i in range(max_start):
        x = values[i : i + context_length]
        y = values[i + context_length : i + context_length + horizon]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

class SingleSeriesDataset(Dataset):
    def __init__(
        self,
        series: pd.Series,
        context_length: int,
        horizon: int,
        train: bool,
        train_split: float,
        scaler: StandardScaler,
        max_windows: int = None,
        shuffle_windows: bool = True,
    ):
        if train:
            s_part, _ = split_series_by_time(series, train_split)
        else:
            _, s_part = split_series_by_time(series, train_split)

        vals = scaler.transform(s_part.values.reshape(-1, 1)).ravel()
        xs, ys = make_windows(vals, context_length, horizon)
        indices = list(range(len(xs)))
        if shuffle_windows:
            random.shuffle(indices)
        if max_windows is not None:
            indices = indices[:max_windows]
        self.xs = torch.tensor(xs[indices], dtype=torch.float32)
        self.ys = torch.tensor(ys[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

def load_single_csv(path: str, date_col: str, value_col: str) -> pd.Series:
    # Read with no type assumptions
    df = pd.read_csv(path, dtype=str)

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # If a second header-like row exists (e.g., contains 'SYMBOL' or ticker strings), drop it.
    # Heuristic: if the first non-header row has strings like '.NS' in numeric columns, drop such rows.
    # First, ensure required columns exist (case-insensitive fallback).
    col_map = {c.lower(): c for c in df.columns}
    if date_col not in df.columns and date_col.lower() in col_map:
        date_col = col_map[date_col.lower()]
    if value_col not in df.columns and value_col.lower() in col_map:
        value_col = col_map[value_col.lower()]

    if value_col not in df.columns:
        raise ValueError(f"Missing '{value_col}' in {path}. Found columns: {df.columns.tolist()}")

    # Parse date. Your sample uses dd-mm-YYYY like 01-01-2015.
    # Try dayfirst=True; if parse fails, fallback to default.
    if date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        except Exception:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Coerce Close to numeric; non-numeric become NaN and will be dropped
    df[value_col] = coerce_numeric(df[value_col])

    # Drop rows that are clearly the second header or any non-data lines:
    # - Close is NaN
    # - or Volume (if present) is non-numeric for those rows
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df = df[~df[value_col].isna()].copy()

    # Sort by date if present, else keep as-is
    if date_col in df.columns:
        df = df.sort_values(date_col)

    # Extract the value series
    s = pd.Series(df[value_col].astype(float).values).dropna().reset_index(drop=True)
    return s

def try_generate(model, xs, horizon):
    # Tries to use a generate-like path from remote code.
    # Expects output shape [B, context + horizon] or [B, horizon].
    out = model.generate(input_ids=xs, max_new_tokens=horizon, num_return_sequences=1)
    if out.dim() == 2 and out.shape[1] == horizon:
        return out
    # Assume [B, context + horizon]
    return out[:, -horizon:]

def forward_predict(model, xs, horizon):
    # Fallback path: if the remote code exposes a forward that returns next-step logits/values.
    # We attempt a simple one-step forecast repeated horizon times.
    preds = []
    ctx = xs
    for _ in range(horizon):
        y = model(ctx)  # expects output compatible with last-step prediction
        # If y is tuple or has logits, adapt here:
        if isinstance(y, (tuple, list)):
            y = y[0]
        # We expect y shape [B, T] or [B, 1]; take last step
        if y.dim() == 2:
            step = y[:, -1:]
        elif y.dim() == 3:
            step = y[:, -1, :]
            if step.dim() == 1:
                step = step.unsqueeze(-1)
        else:
            step = y
            if step.dim() == 1:
                step = step.unsqueeze(-1)
        preds.append(step)
        # Append to context along feature/time dimension
        ctx = torch.cat([ctx, step], dim=1)
    return torch.cat(preds, dim=1)

def evaluate(model, loader, device, loss_fn, mixed_precision=True) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="Eval", leave=False)
        for xs, ys in pbar:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=mixed_precision and device.type == "cuda"):
                try:
                    preds = try_generate(model, xs, ys.shape[1])
                except Exception:
                    preds = forward_predict(model, xs, ys.shape[1])
                loss = loss_fn(preds, ys)
            total += loss.item() * xs.size(0)
            n += xs.size(0)
            pbar.set_postfix({"mse": f"{loss.item():.4f}"})
    return total / max(1, n)

def evaluate_denorm_metrics(
    model,
    series: pd.Series,
    scaler: StandardScaler,
    device,
    context_length: int,
    horizon: int,
    train_split: float,
    batch_size: int = 512,
    mixed_precision: bool = True,
):
    model.eval()
    _, s_val = split_series_by_time(series, train_split)
    if len(s_val) < context_length + horizon + 1:
        return {"n_windows": 0, "mse_scaled": None, "rmse_scaled": None, "mse": None, "rmse": None, "mape_percent": None}

    vals_scaled = scaler.transform(s_val.values.reshape(-1, 1)).ravel()
    xs, ys = make_windows(vals_scaled, context_length, horizon)
    if len(xs) == 0:
        return {"n_windows": 0, "mse_scaled": None, "rmse_scaled": None, "mse": None, "rmse": None, "mape_percent": None}

    preds_all = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            xb = torch.tensor(xs[i : i + batch_size], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", enabled=mixed_precision and device.type == "cuda"):
                try:
                    pb = try_generate(model, xb, horizon)
                except Exception:
                    pb = forward_predict(model, xb, horizon)
            preds_all.append(pb.detach().float().cpu().numpy())

    preds_scaled = np.concatenate(preds_all, axis=0)
    y_true_scaled = ys

    preds_den = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    y_true_den = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(y_true_scaled.shape)

    mse_scaled = float(np.mean((preds_scaled - y_true_scaled) ** 2))
    rmse_scaled = float(np.sqrt(mse_scaled))
    mse = float(np.mean((preds_den - y_true_den) ** 2))
    rmse = float(np.sqrt(mse))
    eps = 1e-8
    denom = np.maximum(np.abs(y_true_den), eps)
    mape = float(np.mean(np.abs((preds_den - y_true_den) / denom)) * 100.0)

    return {"n_windows": int(len(xs)), "mse_scaled": mse_scaled, "rmse_scaled": rmse_scaled, "mse": mse, "rmse": rmse, "mape_percent": mape}

def train_one_stock(csv_path: str, device: torch.device):
    ticker = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n===== Training {ticker} =====")
    run_dir = os.path.join(CONFIG["runs_root"], ticker, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_json(CONFIG, os.path.join(run_dir, "config.json"))

    series = load_single_csv(csv_path, CONFIG["date_col"], CONFIG["value_col"])
    if len(series) < CONFIG["context_length"] + CONFIG["prediction_horizon"] + 10:
        print(f"Skipping {ticker}: series too short.")
        return

    s_train, _ = split_series_by_time(series, CONFIG["train_split"])
    scaler = StandardScaler().fit(s_train.values.reshape(-1, 1))

    train_ds = SingleSeriesDataset(series, CONFIG["context_length"], CONFIG["prediction_horizon"], True, CONFIG["train_split"], scaler, CONFIG["max_windows_per_series"], CONFIG["shuffle_windows"])
    val_ds = SingleSeriesDataset(series, CONFIG["context_length"], CONFIG["prediction_horizon"], False, CONFIG["train_split"], scaler, CONFIG["max_windows_per_series"], False)

    print(f"{ticker} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"Skipping {ticker}: not enough windows after split.")
        return

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=False)

    print("➡️ Loading model from Hub (trust_remote_code)...")
    cfg = AutoConfig.from_pretrained(CONFIG["model_id"], trust_remote_code=CONFIG["trust_remote_code"])
    model = AutoModel.from_pretrained(CONFIG["model_id"], trust_remote_code=CONFIG["trust_remote_code"]).to(device)
    print("✅ Model loaded.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    loss_fn = nn.MSELoss()
    scaler_amp = torch.cuda.amp.GradScaler(enabled=CONFIG["mixed_precision"] and device.type == "cuda")

    history = {"train_steps": [], "train_loss": [], "val_steps": [], "val_mse": []}
    global_step = 0
    best_val = math.inf

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{ticker} Epoch {epoch}/{CONFIG['num_epochs']}")
        running, count = 0.0, 0

        for xs, ys in pbar:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=CONFIG["mixed_precision"] and device.type == "cuda"):
                try:
                    preds = try_generate(model, xs, ys.shape[1])
                except Exception:
                    preds = forward_predict(model, xs, ys.shape[1])
                loss = loss_fn(preds, ys)

            if scaler_amp.is_enabled():
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss.backward()
                optimizer.step()

            running += loss.item() * xs.size(0)
            count += xs.size(0)
            global_step += 1
            avg_loss = running / max(1, count)
            pbar.set_postfix({"train_loss": f"{avg_loss:.4f}"})

            if CONFIG["eval_every_steps"] and (global_step % CONFIG["eval_every_steps"] == 0):
                val_mse = evaluate(model, val_loader, device, loss_fn, mixed_precision=CONFIG["mixed_precision"])
                history["val_steps"].append(global_step)
                history["val_mse"].append(val_mse)
                save_json(history, os.path.join(run_dir, "history.json"))
                print(f"🔎 {ticker} step {global_step} | Val MSE: {val_mse:.6f}")
                if val_mse < best_val:
                    best_val = val_mse
                    try:
                        model.save_pretrained(os.path.join(run_dir, "model_best"))
                    except Exception:
                        torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pt"))
                    print("💾 Saved new best model (step).")

        train_epoch_loss = running / max(1, count)
        history["train_steps"].append(global_step)
        history["train_loss"].append(train_epoch_loss)
        save_json(history, os.path.join(run_dir, "history.json"))

        val_mse = evaluate(model, val_loader, device, loss_fn, mixed_precision=CONFIG["mixed_precision"])
        print(f"📉 {ticker} Epoch {epoch} | Train Loss: {train_epoch_loss:.6f} | Val MSE: {val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            try:
                model.save_pretrained(os.path.join(run_dir, "model_best"))
            except Exception:
                torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pt"))
            print("💾 Saved new best model (epoch).")

        if (epoch % CONFIG["save_every_epochs"]) == 0:
            ckpt_dir = os.path.join(run_dir, f"model_epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            try:
                model.save_pretrained(ckpt_dir)
            except Exception:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.pt"))
            torch.save(
                {"epoch": epoch, "global_step": global_step, "optimizer": optimizer.state_dict(), "best_val": best_val},
                os.path.join(ckpt_dir, "trainer_state.pt"),
            )

    # Final save
    try:
        model.save_pretrained(os.path.join(run_dir, "model_final"))
    except Exception:
        torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pt"))
    print(f"✅ {ticker} training complete. Best Val MSE: {best_val:.6f}")

    den = evaluate_denorm_metrics(
        model=model,
        series=series,
        scaler=scaler,
        device=device,
        context_length=CONFIG["context_length"],
        horizon=CONFIG["prediction_horizon"],
        train_split=CONFIG["train_split"],
        batch_size=512,
        mixed_precision=CONFIG["mixed_precision"],
    )
    metrics_row = {"ticker": ticker, "best_val_mse_scaled": best_val, **den}
    pd.DataFrame([metrics_row]).to_csv(os.path.join(run_dir, "per_series_metrics_final.csv"), index=False)

    summary = {
        "ticker": ticker,
        "best_val_mse_scaled": best_val,
        "final_global_step": global_step,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "timestamp": datetime.now().isoformat(),
        "run_dir": run_dir,
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))
    print(f"🧾 {ticker} metrics saved to {run_dir}")

def main():
    set_seed(CONFIG["seed"])
    device = get_device()
    csv_paths = sorted(glob.glob(os.path.join(CONFIG["data_dir"], "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {CONFIG['data_dir']}")
    print(f"Found {len(csv_paths)} CSVs. Starting per-stock training...")
    for i, p in enumerate(csv_paths, 1):
        print(f"\n[{i}/{len(csv_paths)}] {os.path.basename(p)}")
        try:
            train_one_stock(p, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Error training {os.path.basename(p)}: {e}")

if __name__ == "__main__":
    main()